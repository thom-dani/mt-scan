#include "resample.hpp"
#include <omp.h>

void resample_cpu(
    const std::vector<float> &points,
    const int nPointsX,
    const int nPointsY,
    const float alpha,
    const KernelType kernelType,
    std::vector<double> &density)
{

    density.resize(nPointsX * nPointsY, (double)0);
    float radius_abs_2 = -(float)std::log(0.05) / std::pow(alpha, 2);
    const float radius_abs = std::sqrt(radius_abs_2);

    const int n_threads = omp_get_max_threads();

    std::vector<std::vector<double>>
        local_grids(
            n_threads,
            std::vector<double>(nPointsX * nPointsY, double(0)));

    int nPoints = points.size() / 2;
    float referenceLength = static_cast<float>(nPointsX > nPointsY ? nPointsY - 1 : nPointsX - 1);

#pragma omp parallel
    {
        int tid = omp_get_thread_num();

#pragma omp for
        for (int p_id = 0; p_id < nPoints; p_id++)
        {
            float x = points[2 * p_id];
            float y = points[2 * p_id + 1];

            int x_min = std::max(0, (int)std::floor(x - radius_abs * referenceLength));
            int x_max = std::min(nPointsX - 1, (int)std::ceil(x + radius_abs * referenceLength));

            int y_min = std::max(0, (int)std::floor(y - radius_abs * referenceLength));
            int y_max = std::min(nPointsY - 1, (int)std::ceil(y + radius_abs * referenceLength));

            for (int i = x_min; i <= x_max; i++)
            {
                for (int j = y_min; j <= y_max; j++)
                {
                    float dx = (i - x) / referenceLength;
                    float dy = (j - y) / referenceLength;

                    float dist2 = dx * dx + dy * dy;

                    if (dist2 <= radius_abs_2)
                    {
                        double w = kernel(dist2, alpha, kernelType);
                        local_grids[tid][i + j * nPointsX] += w;
                    }
                }
            }
        }
    }

    for (int t = 0; t < n_threads; t++)
    {
        for (int idx = 0; idx < nPointsX * nPointsY; idx++)
        {
            density[idx] += local_grids[t][idx];
        }
    }
}

double kernel(const float x, const float parameter, const KernelType kernel)
{
    if (kernel == KernelType::Gaussian)
        return exp(-parameter * parameter * x);

    if (kernel == KernelType::Linear)
        return x < parameter ? 1 - x / parameter : 0;
    if (kernel == KernelType::Cylinder)
        return 1;
    return 1;
}