cmake_minimum_required(VERSION 3.20)

project(hipcv
    VERSION 0.1.0
    DESCRIPTION "OpenCV CUDA-like image processing API for AMD GPUs"
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(hipcv INTERFACE)
target_include_directories(hipcv INTERFACE include)

