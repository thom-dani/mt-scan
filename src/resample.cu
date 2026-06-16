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
    float referenceLength = static_cast<float>(nPointsX > nPointsY ? nPointsY - 1 : nPointsX - 1);


    int x0 = max(0, (int)std::floor(px - radius * referenceLength));
    int x1 = min(nPointsX - 1, (int)std::ceil(px + radius * referenceLength));

    int y0 = max(0, (int)std::floor(py - radius * referenceLength));
    int y1 = min(nPointsY - 1, (int)std::ceil(py + radius * referenceLength));

    for (int vx = x0; vx <= x1; vx++)
    {
        for (int vy = y0; vy <= y1; vy++)
        {

            float dx = (float)(vx - px)/referenceLength;
            float dy = (float)(vy - py)/referenceLength;
            float d2 = dx * dx + dy * dy;

            if (d2 < radius_abs_2)
            {
                double val = exp(-(double)(alpha * alpha* d2));

                // atomic accumulation
                atomicAdd(&output[vx + vy * nPointsX], val);
            }
        }
    }
}

void resample_gpu(
    const std::vector<float> &points,
    const int nPointsX,
    const int nPointsY,
    const float alpha,
    std::vector<double> &density,
    const bool printLogs)
{
    auto t_start = std::chrono::high_resolution_clock::now();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    const float radius_abs_2 = -(float)std::log(0.05) / std::pow(alpha, 2);
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


    resampleKernelAtomic<<<grid, block>>>(
        d_points,
        n_points,
        nPointsX,
        nPointsY,
        radius_abs_2,
        alpha,
        d_output);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA kernel launch error: "
                  << cudaGetErrorString(err)
                  << std::endl;
    }
    
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
    if(printLogs){
        std::cout 
        <<"RESAMPLING "
        << n_points
        << alpha << ", "
        << t_kernel << ", "
        << duration_total << std::endl;
    }
}