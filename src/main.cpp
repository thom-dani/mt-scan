#include <ttkMergeTree.h>
#include <vtkDataSet.h>
#include <vtkDoubleArray.h>
#include <vtkImageData.h>
#include <vtkNew.h>
#include <vtkPointData.h>

#include <cmath>
#include <iostream>

int main(int argc, char **argv)
{

    // Create a 10x10 vtkImageData
    const int nx = 10, ny = 10;
    vtkNew<vtkImageData> image{};
    image->SetDimensions(nx, ny, 1);
    image->SetSpacing(1.0, 1.0, 1.0);
    image->SetOrigin(0.0, 0.0, 0.0);

    // Fill with a simple scalar field: two bumps (so the merge tree is
    // non-trivial)
    vtkNew<vtkDoubleArray> scalars{};
    scalars->SetName("density");
    scalars->SetNumberOfComponents(1);
    scalars->SetNumberOfTuples(nx * ny);

    for (int j = 0; j < ny; j++)
    {
        for (int i = 0; i < nx; i++)
        {
            double x = (double)i / (nx - 1);
            double y = (double)j / (ny - 1);
            // Two Gaussian bumps
            double bump1 = std::exp(-((x - 0.3) * (x - 0.3) + (y - 0.3) * (y - 0.3)) / 0.02);
            double bump2 = std::exp(-((x - 0.7) * (x - 0.7) + (y - 0.7) * (y - 0.7)) / 0.02);
            scalars->SetValue(j * nx + i, bump1 + bump2);
        }
    }

    image->GetPointData()->SetScalars(scalars);
    image->GetPointData()->SetActiveScalars("density");

    // Setup filter
    vtkNew<ttkMergeTree> mergeTree{};
    mergeTree->SetInputDataObject(0, image);
    mergeTree->SetInputArrayToProcess(
        0, 0, 0, vtkDataObject::FIELD_ASSOCIATION_POINTS, "density");
    mergeTree->SetTreeType(1); // 1 = split tree
    mergeTree->Update();

    // Access outputs
    auto *nodes = mergeTree->GetOutputDataObject(0);
    auto *arcs = mergeTree->GetOutputDataObject(1);
    auto *seg = mergeTree->GetOutputDataObject(2);

    std::cout << "nodes class: " << nodes->GetClassName() << std::endl;
    std::cout << "arcs class:  " << arcs->GetClassName() << std::endl;
    std::cout << "seg class:   " << seg->GetClassName() << std::endl;

    return 0;
}