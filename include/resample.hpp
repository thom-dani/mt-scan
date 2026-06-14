#pragma once
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>
#include <omp.h>
#include <iostream>
#include <cmath>

enum class KernelType
{
    Gaussian,
    Cylinder,
    Linear
};

void resample_cpu(
    const std::vector<float> &points,
    const int nPointsX,
    const int nPointsY,
    const float alpha,
    const KernelType kernelType,
    std::vector<double> &density);

double kernel(const float x, const float parameter, const KernelType kernelType);
