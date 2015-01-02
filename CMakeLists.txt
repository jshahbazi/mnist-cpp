cmake_minimum_required(VERSION 2.8.4)
project(mnist_cpp)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(SOURCE_FILES
    LICENSE
    main.cpp
    README.md)

add_executable(mnist_cpp ${SOURCE_FILES})