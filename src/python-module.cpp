#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "merge_tree_scan.hpp"
#include <iostream>

namespace py = pybind11;
py::array MergeTreeScanInterface(
    py::array points,
    float alpha,
    int target_res,
    KernelType kernel,
    bool print_logs,
    bool debugMode)
{

    auto buf = points.request();

    if (buf.ndim != 2 || buf.shape[1] != 2)
    {
        throw std::runtime_error("Input must be (N, 2)");
    }

    size_t n_points = buf.shape[0];
    float *points_ptr = static_cast<float *>(buf.ptr);

    MergeTreeScan scan;

    py::array_t<int> outputLabels(n_points);
    int *labels_ptr = outputLabels.mutable_data();

    scan.setParameters(target_res, n_points, alpha, kernel);
    scan.setPrintLogs(print_logs);
    scan.setDebugMode(debugMode);
    scan.execute(points_ptr, labels_ptr);

    return outputLabels;
}

namespace py = pybind11;
py::array MergeTreeScanDistributionInterface(
    py::array points,
    py::array distribution,
    bool print_logs,
    bool debugMode)
{

    auto buf = points.request();

    if (buf.ndim != 2 || buf.shape[1] != 2)
    {
        throw std::runtime_error("Input must be (N, 2)");
    }

    size_t n_points = buf.shape[0];
    float *points_ptr = static_cast<float *>(buf.ptr);

    auto buf_distrib = distribution.request();
    int resX = buf_distrib.shape[0] - 1;
    int resY = buf_distrib.shape[1] - 1;
    float *distrib_ptr = static_cast<float *>(buf_distrib.ptr);

    MergeTreeScan scan;

    py::array_t<int> outputLabels(n_points);
    int *labels_ptr = outputLabels.mutable_data();

    scan.setResolution(resX, resY);
    scan.setNumberOfPoints(n_points);
    scan.setDistributionMode(true);
    scan.setDebugMode(debugMode);
    scan.setPrintLogs(print_logs);

    scan.execute(points_ptr, labels_ptr, distrib_ptr);

    return outputLabels;
}

PYBIND11_MODULE(mt_scan, m)
{
    m.def("compute_labels", [](py::array points, float alpha, int target_res, std::string kernel, bool print_logs, bool debug)
          {
        KernelType k;
if      (kernel == "gaussian") k = KernelType::Gaussian;
else if (kernel == "cylinder") k = KernelType::Cylinder;
else if (kernel == "linear")   k = KernelType::Linear;
else throw std::invalid_argument("kernel must be one of: gaussian, cylinder, linear");
return MergeTreeScanInterface(points, alpha, target_res, k, print_logs, debug); }, py::arg("points"), py::arg("alpha"), py::arg("target_res"), py::arg("kernel") = "gaussian", py::arg("print_logs") = false, py::arg("debug") = false);

    m.def("compute_labels_distribution", [](py::array points, py::array distribution, bool print_logs, bool debug)
          { return MergeTreeScanDistributionInterface(points, distribution, print_logs, debug); }, py::arg("points"), py::arg("distribution"), py::arg("print_logs") = false, py::arg("debug") = false);
}