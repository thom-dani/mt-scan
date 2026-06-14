#pragma once
#include <map>
#include <vector>

class MergeTreeScan
{
public:
    MergeTreeScan() = default;

    void setParameters(const int target_res, const int alpha, const float n_points);
    void execute(const float *ptr, int *labels);

private:
    void normalizePoints(const float *points_ptr, const int targetRes, std::vector<float> &pointsNormalized);

    void buildMergeTree(
        const std::vector<double> &density,
        std::vector<int> &nodeVertexId,
        std::vector<int> &nodeCriticalType,
        std::vector<std::pair<int, int>> &edges,
        std::vector<int> &segmentationId);

    void buildClusterTree(const std::vector<std::pair<int, int>> &edges,
                          const std::vector<int> &nodeCriticalType,
                          std::vector<std::vector<int>> &adjacencyList,
                          std::vector<int> &bottomUpNodes,
                          int &rootId);

    void computeNodeEnergies(
        const std::vector<std::vector<int>> &adjacencyList,
        const std::vector<int> &bottomUpNodes,
        const std::vector<int> &segmentationId,
        const std::vector<double> &density,
        const std::vector<int> &nodeVertexId,
        std::vector<float> &nodeEnergy);

    void selectClusters(const std::vector<std::vector<int>> &adjacencyList,
                        const std::vector<float> &nodeEnergies,
                        const std::vector<int> &bottomUpNodes,
                        const int &rootId,
                        std::vector<bool> &isCluster,
                        std::vector<float> &clusterWeight);

    void computeLabels(
        const std::vector<std::vector<int>> &adjacencyList,
        const int &rootId,
        const std::vector<int> &nodeVertexId,
        const std::vector<int> &segmentationIds,
        const std::vector<float> &pointsNormalized,
        const std::vector<float> &clusterWeight,
        const std::vector<bool> &isCluster,
        const int &resX,
        const int &minClusterSize,
        std::vector<int> &labels);

    int targetRes_;
    int resX_;
    int resY_;
    int alpha_;
    int nPoints_;
};