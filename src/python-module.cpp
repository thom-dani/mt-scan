#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "merge_tree_scan.hpp"
#include <iostream>

namespace py = pybind11;
py::array MergeTreeScanInterface(
    py::array arr,
    int target_res,
    float alpha)
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

    scan.setParameters(target_res, n_points, alpha);
    scan.execute(points_ptr, labels_ptr);

    return outputLabels;
}

PYBIND11_MODULE(mt_scan, m)
{
    m.def("compute_labels", &MergeTreeScanInterface);
}
