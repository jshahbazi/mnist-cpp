cmake_minimum_required(VERSION 2.8.4)
project(mnist_cpp)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O2 -Wall")


include_directories(
    include
    src
    )

set(SOURCE_FILES
    src/main.cpp
    include/armadillo.h
    include/std_lib_facilities.h
    include/Eigen/
    include/armadillo_bits
    )

add_executable(mnist_cpp ${SOURCE_FILES})
target_link_libraries(mnist_cpp armadillo)