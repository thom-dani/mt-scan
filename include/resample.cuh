#pragma once
#include <iostream>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <chrono>

// ── GPU kernel ────────────────────────────────────────────────────────────────

#define BLOCK_SIZE 256

__global__ void resampleKernelAtomic(
    const float *points,
    const int n_points,
    const int nPointsX,
    const int nPointsY,
    const float radius_abs_2,
    const float alpha,
    double *output);

void resample_gpu(
    const std::vector<float> &points,
    const int nPointsX,
    const int nPointsY,
    const float alpha,
    std::vector<double> &density,
    const bool printLogs=false);
