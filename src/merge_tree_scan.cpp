#include "merge_tree_scan.hpp"
#include "utils.hpp"
#include <cmath>
#include <algorithm>
#include <utils.hpp>
#include <thread>

#ifdef USE_CUDA
#include "resample.cuh"
#else
#include "resample.hpp"
#endif

#include <vtkImageData.h>
#include <vtkUnstructuredGrid.h>
#include <vtkSmartPointer.h>
#include <vtkDoubleArray.h>
#include <vtkPointData.h>
#include <vtkAlgorithmOutput.h>
#include <vtkXMLImageDataWriter.h>

#include <ttkMergeTree.h>

#include <vtkDataSet.h>
#include <vtkNew.h>

#include <fstream>

void MergeTreeScan::setParameters(const int target_res, const int nPoints, const float alpha, const KernelType kernelType)
{
    this->targetRes_ = target_res;
    this->alpha_ = alpha;
    this->nPoints_ = nPoints;
    this->kernel_ = kernelType;
}

void MergeTreeScan::normalizePoints(const float *points_ptr, const int targetRes, std::vector<float> &pointsNormalized)
{
    float x_max{}, x_min{}, y_max{}, y_min{};
    findAxisBound(points_ptr, this->nPoints_, x_max, x_min, y_max, y_min);

    float axisLength_x = x_max - x_min;
    float axisLength_y = y_max - y_min;

    x_max += 0.05 * axisLength_x;
    x_min -= 0.05 * axisLength_x;
    y_max += 0.05 * axisLength_y;
    y_min -= 0.05 * axisLength_y;

    axisLength_x = x_max - x_min;
    axisLength_y = y_max - y_min;

    float pixelSize = std::max(axisLength_x, axisLength_y) / targetRes;

    this->resX_ = static_cast<int>(std::ceil(axisLength_x / pixelSize));
    this->resY_ = static_cast<int>(std::ceil(axisLength_y / pixelSize));

    float scaleX = (float)this->resX_ / axisLength_x;
    float scaleY = (float)this->resY_ / axisLength_y;

    pointsNormalized.resize(2 * this->nPoints_);
    for (int i = 0; i < this->nPoints_; i++)
    {

        float x = points_ptr[2 * i];
        float y = points_ptr[2 * i + 1];
        x -= x_min;
        y -= y_min;
        x *= scaleX;
        y *= scaleY;
        pointsNormalized[2 * i] = x;
        pointsNormalized[2 * i + 1] = y;
    }
}

void MergeTreeScan::distributionFromSample(
    const float *points_ptr,
    const int targetRes,
    std::vector<float> &pointsNormalized,
    std::vector<double> &density)
{

    normalizePoints(points_ptr, this->targetRes_, pointsNormalized);
#ifdef USE_CUDA
    resample_gpu(pointsNormalized, this->resX_ + 1, this->resY_ + 1, this->alpha_, density, printLogs_);
#else
    resample_cpu(pointsNormalized, this->resX_ + 1, this->resY_ + 1, this->alpha_, this->kernel_, density);
#endif
}

void MergeTreeScan::execute(
    const float *points_ptr,
    int *labels_ptr,
    const float *distribution_ptr)

{
    using Clock = std::chrono::high_resolution_clock;
    using Ms = std::chrono::duration<double>;

    auto tock = [&](const char *step, Clock::time_point t0)
    {
        if (printLogs_)
            std::cout << "TIMER " << step << " " << std::fixed << std::setprecision(4) << Ms(Clock::now() - t0).count() << "\n";
        return Clock::now();
    };

    auto t0_total = Clock::now();
    auto t = Clock::now();

    std::vector<float> pointsNormalized{};
    std::vector<double> density;

    if (!distributionMode_)
    {
        distributionFromSample(points_ptr, targetRes_, pointsNormalized, density);
    }

    else
    {
        pointsNormalized.resize(2 * nPoints_);
        std::copy(points_ptr, points_ptr + 2 * nPoints_, pointsNormalized.begin());
        int nVertices = (resX_ + 1) * (resY_ + 1);
        density.resize(nVertices);
        std::copy(distribution_ptr, distribution_ptr + nVertices, density.begin());
    }

    t = tock("resample", t);

    std::vector<int> nodeVertexId{};
    std::vector<int> nodeCriticalType{};
    std::vector<std::pair<int, int>> edges{};
    std::vector<int> segmentationIds{};
    buildMergeTree(density, nodeVertexId, nodeCriticalType, edges, segmentationIds);

    std::vector<std::vector<int>> adjacencyList{};
    std::vector<int> bottomUpNodes{};
    int rootId{};
    buildClusterTree(edges, nodeCriticalType, adjacencyList, bottomUpNodes, rootId);
    t = tock("clusterTree", t);

    std::vector<float> nodeEnergy;
    computeNodeEnergies(adjacencyList, bottomUpNodes, segmentationIds, density, nodeVertexId, nodeEnergy);

    std::vector<bool> isCluster;
    std::vector<float> clusterWeight;
    selectClusters(adjacencyList, nodeEnergy, bottomUpNodes, rootId, isCluster, clusterWeight);

    int minClusterSize{};
    std::vector<int> labels{};
    computeLabels(adjacencyList, rootId, density, nodeVertexId, segmentationIds,
                  pointsNormalized, clusterWeight, isCluster, this->resX_,
                  minClusterSize, labels);

    for (int i = 0; i < nPoints_; i++)
        labels_ptr[i] = labels[i];

    tock("computeLabels", t);

    if (printLogs_)
        std::cout << "TIMER total " << " " << std::fixed << std::setprecision(4) << Ms(Clock::now() - t0_total).count() << "\n";
}

void MergeTreeScan::buildMergeTree(
    const std::vector<double> &density,
    std::vector<int> &nodeVertexId,
    std::vector<int> &nodeCriticalType,
    std::vector<std::pair<int, int>> &edges,
    std::vector<int> &segmentationId)
{
    const int nPoints_x = this->resX_ + 1;
    const int nPoints_y = this->resY_ + 1;
    const int nVertices = nPoints_x * nPoints_y;

    vtkNew<vtkImageData> image;
    image->SetDimensions(nPoints_x, nPoints_y, 1);
    image->SetExtent(0, nPoints_x - 1,
                     0, nPoints_y - 1,
                     0, 0);
    image->SetSpacing(1.0, 1.0, 1.0);
    image->SetOrigin(0.0, 0.0, 0.0);

    // vtkNew<vtkXMLImageDataWriter> writer;
    // writer->SetFileName("../tmp/output.vti");
    // writer->SetInputData(image);
    // writer->Write();

    vtkNew<vtkDoubleArray> scalars;
    scalars->SetName("density");
    scalars->SetNumberOfComponents(1);
    scalars->SetNumberOfTuples(nVertices);

    for (int i = 0; i < nVertices; i++)
    {
        scalars->SetValue(i, density[i]);
    }

    image->GetPointData()->SetScalars(scalars);
    image->GetPointData()->SetActiveScalars("density");

    vtkNew<ttkMergeTree> mergeTree{};
    mergeTree->SetInputDataObject(0, image);
    mergeTree->SetInputArrayToProcess(
        0, 0, 0, vtkDataObject::FIELD_ASSOCIATION_POINTS, "density");

    mergeTree->SetThreadNumber(std::thread::hardware_concurrency());
    mergeTree->SetBackend(static_cast<int>(ttkMergeTree::BACKEND::EXTREEM));
    mergeTree->SetTreeType(static_cast<int>(ttk::ftm::TreeType::Split));

    mergeTree->Update();

    vtkDataSet *outputNodes = vtkUnstructuredGrid::SafeDownCast(mergeTree->GetOutput(0));
    vtkLongLongArray *vertexIds = vtkLongLongArray::SafeDownCast(outputNodes->GetPointData()->GetArray("VertexId"));
    vtkIntArray *criticalType = vtkIntArray::SafeDownCast(outputNodes->GetPointData()->GetArray("CriticalType"));

    std::map<int, int> vertexNodeId{};
    int n_node = vertexIds->GetNumberOfTuples();
    nodeVertexId.resize(n_node);
    nodeCriticalType.resize(n_node);
    for (int i = 0; i < n_node; i++)
    {
        int vertexId = vertexIds->GetValue(i);
        vertexNodeId[vertexId] = i;
        nodeVertexId[i] = vertexId;
        nodeCriticalType[i] = criticalType->GetValue(i);
    }

    vtkDataSet *outputArcs = vtkDataSet::SafeDownCast(mergeTree->GetOutputDataObject(1));
    vtkIdTypeArray *downNodeId = vtkIdTypeArray::SafeDownCast(outputArcs->GetCellData()->GetArray("downNodeId"));
    vtkIdTypeArray *upNodeId = vtkIdTypeArray::SafeDownCast(outputArcs->GetCellData()->GetArray("upNodeId"));

    for (int i = 0; i < downNodeId->GetNumberOfTuples(); i++)
    {
        int downNodeId_local = vertexNodeId[downNodeId->GetValue(i)];
        int upNodeId_local = vertexNodeId[upNodeId->GetValue(i)];
        auto newEdge = std::make_pair(downNodeId_local, upNodeId_local);
        edges.push_back(newEdge);
    }

    vtkDataSet *outputSeg = vtkDataSet::SafeDownCast(mergeTree->GetOutputDataObject(2));
    vtkIntArray *segArray = vtkIntArray::SafeDownCast(outputSeg->GetPointData()->GetArray("SegmentationId"));

    segmentationId.resize(nVertices);
    for (int i = 0; i < segArray->GetNumberOfTuples(); i++)
    {
        int segId = segArray->GetValue(i);
        auto it = vertexNodeId.find(segId);
        if (it == vertexNodeId.end())
        {
            std::cout << "problem 3" << std::endl;
        }
        segmentationId[i] = vertexNodeId[segId];
    }
}

void MergeTreeScan::buildClusterTree(const std::vector<std::pair<int, int>> &edges,
                                     const std::vector<int> &nodeCriticalType,
                                     std::vector<std::vector<int>> &adjacencyList,
                                     std::vector<int> &bottomUpNodes,
                                     int &rootId)
{
    int nNodes = nodeCriticalType.size();
    adjacencyList.resize(nNodes, std::vector<int>());

    for (const auto &edge : edges)
    {
        int upNodeId = edge.second;
        if (nodeCriticalType[upNodeId] == 0)
        {
            rootId = upNodeId;
        }
        if (edge.first != edge.second)
        {
            adjacencyList[edge.second].push_back(edge.first);
        }
    }
    inverseBFS_Order(adjacencyList, rootId, bottomUpNodes);
}

void MergeTreeScan::computeNodeEnergies(
    const std::vector<std::vector<int>> &adjacencyList,
    const std::vector<int> &bottomUpNodes,
    const std::vector<int> &segmentationId,
    const std::vector<double> &density,
    const std::vector<int> &nodeVertexId,
    std::vector<float> &nodeEnergy)
{
    int nNodes = adjacencyList.size();
    nodeEnergy.resize(nNodes, 0.0f);

    std::vector<float> nodeMinScalarValue(nNodes, 0.0f);
    std::vector<int> node_nCells(nNodes, 0);

    for (int id : bottomUpNodes)
    {
        for (int child_id : adjacencyList[id])
        {
            nodeMinScalarValue[child_id] = density[id];
        }
    }

    for (size_t vertex_id = 0; vertex_id < segmentationId.size(); vertex_id++)
    {
        int nodeId = segmentationId[vertex_id];
        float value = density[vertex_id];

        nodeEnergy[nodeId] += value - nodeMinScalarValue[nodeId];
        node_nCells[nodeId] += 1;
    }

    for (size_t i = 0; i < bottomUpNodes.size() - 1; i++)
    {
        int nodeId = bottomUpNodes[i];

        if (adjacencyList[nodeId].empty())
            continue;

        int nCell = 0;

        for (int child : adjacencyList[nodeId])
        {

            nCell += node_nCells[child];
        }

        int vertexId = nodeVertexId[nodeId];

        nodeEnergy[nodeId] += nCell *
                              (density[vertexId] - nodeMinScalarValue[nodeId]);

        node_nCells[nodeId] += nCell;
    }
}

void MergeTreeScan::selectClusters(const std::vector<std::vector<int>> &adjacencyList,
                                   const std::vector<float> &nodeEnergy,
                                   const std::vector<int> &bottomUpNodes,
                                   const int &rootId,
                                   std::vector<bool> &isCluster,
                                   std::vector<float> &clusterWeight)
{

    int nNodes = adjacencyList.size();

    std::vector<float> costs(nNodes, 0.0f);
    std::vector<bool> deltas(nNodes, false);

    for (int nodeId = 0; nodeId < nNodes; nodeId++)
    {
        if (adjacencyList[nodeId].empty())
        {
            costs[nodeId] = nodeEnergy[nodeId];
            deltas[nodeId] = true;
        }
    }

    for (size_t i = 0; i < bottomUpNodes.size() - 1; i++)
    {
        int nodeId = bottomUpNodes[i];

        if (adjacencyList[nodeId].empty())
            continue;

        float costSum = 0.0f;

        for (int childId : adjacencyList[nodeId])
        {
            costSum += costs[childId];
        }

        if (nodeEnergy[nodeId] < costSum)
        {
            costs[nodeId] = costSum;
        }
        else
        {
            costs[nodeId] = nodeEnergy[nodeId];
            deltas[nodeId] = true;
        }
    }

    std::stack<int> stack;
    clusterWeight.resize(nNodes, 0.0f);
    isCluster.resize(nNodes, false);

    stack.push(adjacencyList[rootId][0]);

    while (!stack.empty())
    {
        int nodeId = stack.top();
        stack.pop();

        if (!deltas[nodeId])
        {
            for (int childId : adjacencyList[nodeId])
            {
                stack.push(childId);
            }
        }
        else
        {
            isCluster[nodeId] = true;
        }
    }

    for (int i = 0; i < nNodes; i++)
    {
        if (!isCluster[i])
            continue;

        std::stack<int> subTreeStack;
        subTreeStack.push(i);

        while (!subTreeStack.empty())
        {
            int nodeId = subTreeStack.top();
            subTreeStack.pop();
            clusterWeight[i] += nodeEnergy[nodeId];

            for (int childId : adjacencyList[nodeId])
            {
                subTreeStack.push(childId);
            }
        }
    }
}
void MergeTreeScan::computeLabels(
    const std::vector<std::vector<int>> &adjacencyList,
    const int &rootId,
    const std::vector<double> &density,
    const std::vector<int> &nodeVertexId,
    const std::vector<int> &segmentationIds,
    const std::vector<float> &pointsNormalized,
    const std::vector<float> &clusterWeight,
    const std::vector<bool> &isCluster,
    const int &resX,
    const int &minClusterSize,
    std::vector<int> &labels)
{
    int nNodes = adjacencyList.size();
    std::vector<int> nodeFlatMapId(nNodes, -1);

    auto assignSubTreeFlatmapId = [&](int clusterId)
    {
        std::vector<int> stack;
        stack.push_back(clusterId);
        while (!stack.empty())
        {
            int nodeId = stack.back();
            stack.pop_back();
            nodeFlatMapId[nodeId] = clusterId;
            for (int childId : adjacencyList[nodeId])
                stack.push_back(childId);
        }
    };

#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < nNodes; i++)
    {
        if (isCluster[i] && clusterWeight[i] > minClusterSize)
        {
            int flatMapId = segmentationIds[nodeVertexId[i]];
            if (flatMapId != i)
                std::cout << "problem : " << flatMapId << ", " << i << "\n";
            assignSubTreeFlatmapId(i);
        }
    }

    std::vector<int> segmentationId_cells{};
    vertexToCellValues(segmentationIds, density, this->resX_ + 1, this->resY_ + 1, segmentationId_cells);

    labels.resize(nPoints_, -1);

#pragma omp parallel for schedule(static)
    for (int i = 0; i < nPoints_; i++)
    {
        float x = pointsNormalized[2 * i];
        float y = pointsNormalized[2 * i + 1];
        int cellX = static_cast<int>(x);
        int cellY = static_cast<int>(y);
        int idx = cellX + cellY * resX;
        int parentId = nodeFlatMapId[segmentationId_cells[idx]];
        labels[i] = parentId;
    }
}