#pragma once

void findAxisBound(const float *points_ptr,
                   const float n_points_ptr,
                   float &x_max,
                   float &x_min,
                   float &y_max,
                   float &y_min);

void buildOrderArray(const std::vector<double> &density,
                     std::vector<int> &orderArray);

void inverseBFS_Order(const std::vector<std::vector<int>> &adjacencyList,
                      const int &rootId,
                      std::vector<int> &bottomUpNodes);

void vertexToCellValues(
    const std::vector<int> &vertexValues,
    int nPointX,
    int nPointY,
    std::vector<int> &cellValues);
