cmake_minimum_required(VERSION 3.7)

include_directories(${CMAKE_SOURCE_DIR}/include)
add_executable(sample_ba_from_file main.cpp)
target_link_libraries(sample_ba_from_file ${OpenCV_LIBS} dpcpp_bundle_adjustment)
