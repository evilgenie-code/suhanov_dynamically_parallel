cmake_minimum_required(VERSION 3.17)
project(untitled4 CUDA)

set(CMAKE_CUDA_STANDARD 14)

add_executable(untitled4 main.cu)

set_target_properties(
        untitled4
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)