cmake_minimum_required(VERSION 3.18)

option(USE_CUDA "Build with CUDA support" OFF)

if(USE_CUDA)
    project(mt-scan CUDA CXX C)
    set(CMAKE_CUDA_ARCHITECTURES 89)
    set(CMAKE_CUDA_STANDARD 17)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
else()
    project(mt-scan CXX C)
endif()

set(CMAKE_CXX_STANDARD 17)

find_package(OpenMP REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

execute_process(
    COMMAND ${Python3_EXECUTABLE} -m pybind11 --cmakedir
    OUTPUT_VARIABLE pybind11_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
find_package(pybind11 REQUIRED)

set(CMAKE_PREFIX_PATH /opt/paraview/install/lib/cmake/paraview-5.13/vtk/)
find_package(VTK REQUIRED COMPONENTS
    CommonCore
    CommonDataModel
    IOXML
)

set(CMAKE_PREFIX_PATH /usr/local/lib/cmake/ttkVTK/)
find_package(TTKVTK REQUIRED)



set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include_directories(${CMAKE_SOURCE_DIR}/include)

if(USE_CUDA)
    set(SOURCES
        src/merge_tree_scan.cpp
        src/utils.cpp
        src/resample.cu
    )
else()
    set(SOURCES
        src/merge_tree_scan.cpp
        src/utils.cpp
        src/resample.cpp
    )
endif()

pybind11_add_module(mt_scan
    src/python-module.cpp
    ${SOURCES}
)

if(USE_CUDA)
    set_target_properties(mt_scan PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()

target_include_directories(mt_scan PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(mt_scan PRIVATE
    ${VTK_LIBRARIES}
    OpenMP::OpenMP_CXX
    $<$<BOOL:${USE_CUDA}>:CUDA::cudart>
    ttk::vtk::ttkMergeTree
)

vtk_module_autoinit(
    TARGETS mt_scan
    MODULES ${VTK_LIBRARIES}
)

add_executable(testMergeTree src/main.cpp)

target_link_libraries(testMergeTree PRIVATE
    OpenMP::OpenMP_CXX
    ttk::vtk::ttkMergeTree
)
