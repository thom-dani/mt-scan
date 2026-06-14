#include <algorithm>
#include <cmath>
#include <vector>
#include <queue>
#include <map>

#include <iostream>
void findAxisBound(const float *points_ptr, const float n_points, float &x_max, float &x_min, float &y_max, float &y_min)
{

    x_min = x_max = points_ptr[0];
    y_min = y_max = points_ptr[1];

    for (int i = 1; i < n_points; i++)
    {
        float x = points_ptr[2 * i];
        float y = points_ptr[2 * i + 1];

        if (x < x_min)
            x_min = x;
        if (x > x_max)
            x_max = x;

        if (y < y_min)
            y_min = y;
        if (y > y_max)
            y_max = y;
    }
}

void buildOrderArray(const std::vector<double> &density, std::vector<int> &orderArray)
{
    int nPoints = density.size();
    std::vector<int> tmp(nPoints);
    orderArray.resize(nPoints);
    for (int i = 0; i < nPoints; i++)
    {
        tmp[i] = i;
    }

    sort(tmp.begin(), tmp.end(),
         [&](int a, int b)
         {
             if (density[a] != density[b])
                 return density[a] < density[b];
             return a < b; // tie-breaker
         });

    for (int i = 0; i < nPoints; i++)
    {
        orderArray[tmp[i]] = i;
    }
}

void inverseBFS_Order(const std::vector<std::vector<int>> &adjacencyList,
                      const int &rootId,
                      std::vector<int> &bottomUpNodes)
{
    int nNodes = adjacencyList.size();
    bottomUpNodes.reserve(nNodes);

    std::queue<int> q;
    q.push(rootId);

    while (!q.empty())
    {
        int nodeId = q.front();
        q.pop();

        bottomUpNodes.push_back(nodeId);

        for (int child : adjacencyList[nodeId])
        {
            q.push(child);
        }
    }
    std::reverse(bottomUpNodes.begin(), bottomUpNodes.end());
}

void vertexToCellValues(
    const std::vector<int> &vertexValues,
    const std::vector<double> &density,
    int nPointX,
    int nPointY,
    std::vector<int> &cellValues)
{
    int nCellX = nPointX - 1;
    int nCellY = nPointY - 1;

    cellValues.resize(nCellX * nCellY);

    for (int y = 0; y < nCellY; y++)
    {
        for (int x = 0; x < nCellX; x++)
        {
            int v0 = x + y * nPointX;
            int v1 = (x + 1) + y * nPointX;
            int v2 = x + (y + 1) * nPointX;
            int v3 = (x + 1) + (y + 1) * nPointX;

            std::vector<int> cornerIds = {v0, v1, v2, v3};

            int maxVertexId = *std::max_element(cornerIds.begin(), cornerIds.end(),
                                                [&](const auto &a, const auto &b)
                                                { return density[a] < density[b]; });

            int cellIndex = x + y * nCellX;

            cellValues[cellIndex] = vertexValues[maxVertexId];
        }
    }
}
