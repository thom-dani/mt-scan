
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>
#include <omp.h>
#include <iostream>
#include <cmath>

void resample_cpu(
    const std::vector<float> &points,
    const int nPointsX,
    const int nPointsY,
    const float alpha,
    std::vector<double> &density);