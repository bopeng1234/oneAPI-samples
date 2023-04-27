cmake_minimum_required(VERSION 3.7)
project(dpcpp-bundle-adjustment)

set(CMAKE_CXX_COMPILER icpx)
set(CMAKE_CXX_FLAGS "-fsycl -O3 -Wno-tautological-constant-compare")

# OpenCV, Eigen
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIR})

add_subdirectory(src)
add_subdirectory(samples/sample_ba_from_file)

add_custom_target(run ONEAPI_DEVICE_SELECTOR=opencl:cpu ./samples/sample_ba_from_file/sample_ba_from_file ../data/ba_kitti_07.json)
add_custom_target(run_gpu ONEAPI_DEVICE_SELECTOR=opencl:gpu ./samples/sample_ba_from_file/sample_ba_from_file ../data/ba_kitti_07.json)
