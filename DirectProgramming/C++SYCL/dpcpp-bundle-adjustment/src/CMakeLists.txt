cmake_minimum_required(VERSION 3.7)

include_directories(${CMAKE_SOURCE_DIR}/include)
file(GLOB srcs ./*.cpp)
add_library(dpcpp_bundle_adjustment ${srcs})
target_link_libraries(dpcpp_bundle_adjustment sycl mkl_intel_lp64 mkl_intel_thread mkl_core iomp5 pthread m dl)
