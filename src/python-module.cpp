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
    bool print_logs)
{

    auto buf = arr.request();

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
    scan.execute(points_ptr, labels_ptr);

    return outputLabels;
}

PYBIND11_MODULE(mt_scan, m)
{
    m.def("compute_labels", [](py::array points, float alpha, int target_res, std::string kernel, bool print_logs)
          {
        KernelType k;
if      (kernel == "gaussian") k = KernelType::Gaussian;
else if (kernel == "cylinder") k = KernelType::Cylinder;
else if (kernel == "linear")   k = KernelType::Linear;
else throw std::invalid_argument("kernel must be one of: gaussian, cylinder, linear");
return MergeTreeScanInterface(points, alpha, target_res, k, print_logs); }, py::arg("points"), py::arg("alpha"), py::arg("target_res"), py::arg("kernel") = "gaussian", py::arg("print_logs") = false);
}