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
    double *output)
{
    int pid = blockIdx.x * blockDim.x + threadIdx.x;

    if (pid >= n_points)
        return;

    float px = points[2 * pid];
    float py = points[2 * pid + 1];

    float radius = sqrt(radius_abs_2);

    // bounding box on the grid for this point
    int x0 = max(0, (int)floor(px - radius));
    int x1 = min(nPointsX - 1, (int)ceil(px + radius));

    int y0 = max(0, (int)floor(py - radius));
    int y1 = min(nPointsY - 1, (int)ceil(py + radius));

    for (int vx = x0; vx <= x1; vx++)
    {
        for (int vy = y0; vy <= y1; vy++)
        {

            float dx = (float)vx - px;
            float dy = (float)vy - py;
            float d2 = dx * dx + dy * dy;

            if (d2 < radius_abs_2)
            {
                double val = exp(-(double)(alpha * d2));

                // atomic accumulation
                atomicAdd(&output[vx * nPointsY + vy], val);
            }
        }
    }
}

void resample_gpu(
    const std::vector<float> &points,
    const int nPointsX,
    const int nPointsY,
    const float alpha,
    std::vector<double> &density)
{
    auto t_start = std::chrono::high_resolution_clock::now();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    const float radius_abs_2 = -(float)std::log(0.2) / (float)alpha;
    const int n_verts = nPointsX * nPointsY;

    float *d_points;
    double *d_output;
    int n_points = points.size() / 2;

    cudaMalloc(&d_points, n_points * 2 * sizeof(float));
    cudaMalloc(&d_output, n_verts * sizeof(double));

    cudaMemcpy(d_points, points.data(),
               n_points * 2 * sizeof(float),
               cudaMemcpyHostToDevice);

    cudaMemset(d_output, 0, n_verts * sizeof(double));

    dim3 block(BLOCK_SIZE);
    dim3 grid((n_points + BLOCK_SIZE - 1) / BLOCK_SIZE);

    cudaEvent_t start_kernel, stop_kernel;
    cudaEventCreate(&start_kernel);
    cudaEventCreate(&stop_kernel);
    cudaEventRecord(start_kernel);

    std::cout << "before kernel" << std::endl;

    resampleKernelAtomic<<<grid, block>>>(
        d_points,
        n_points,
        nPointsX,
        nPointsY,
        radius_abs_2,
        alpha,
        d_output);

    std::cout << "after kernel" << std::endl;

    cudaGetLastError();
    cudaDeviceSynchronize();
    cudaEventRecord(stop_kernel);
    cudaEventSynchronize(stop_kernel);
    float t_kernel = 0.0;
    cudaEventElapsedTime(&t_kernel, start_kernel, stop_kernel);

    density.resize(nPointsX * nPointsY);
    cudaMemcpy(density.data(),
               d_output,
               n_verts * sizeof(double),
               cudaMemcpyDeviceToHost);

    cudaFree(d_points);
    cudaFree(d_output);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float t_total = 0.0;
    cudaEventElapsedTime(&t_total, start, stop);

    auto t_end = std::chrono::high_resolution_clock::now();

    double duration_total =
        std::chrono::duration<double>(
            t_end - t_start)
            .count();

    duration_total *= 1000;
    std::cout << "(" << nPointsX << " x "
              << nPointsY << "), "
              << n_points
              << alpha << ", "
              << t_kernel << ", "
              << duration_total << std::endl;
}