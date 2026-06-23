cmake_minimum_required(VERSION 3.20)

project(hipcv
    VERSION 0.1.0
    DESCRIPTION "OpenCV CUDA-like image processing API for AMD GPUs"
    LANGUAGES CXX
)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(HIPCV_ENABLE_HIP "Enable AMD HIP SDK support when available" ON)
option(HIPCV_BUILD_EXAMPLES "Build example executables" ON)
option(HIPCV_BUILD_TESTS "Build tests" ON)

set(HIPCV_HAS_HIP 0)

if(HIPCV_ENABLE_HIP)
    find_package(HIPSDK QUIET)
    if(HIPSDK_FOUND)
        set(HIPCV_HAS_HIP 1)
    else()
        message(STATUS "AMD HIP SDK was not found; building hipcv without GPU backend.")
    endif()
endif()

add_library(hipcv STATIC
    src/device.cpp
    src/gpu_mat.cpp
    src/imgproc.cpp
    src/version.cpp
)

target_include_directories(hipcv
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

target_compile_definitions(hipcv
    PUBLIC
        HIPCV_VERSION_MAJOR=${PROJECT_VERSION_MAJOR}
        HIPCV_VERSION_MINOR=${PROJECT_VERSION_MINOR}
        HIPCV_VERSION_PATCH=${PROJECT_VERSION_PATCH}
    PRIVATE
        HIPCV_HAS_HIP=${HIPCV_HAS_HIP}
)

if(HIPCV_HAS_HIP)
    target_link_libraries(hipcv PUBLIC HIPSDK::amdhip64 HIPSDK::hiprtc)
    target_compile_definitions(hipcv PRIVATE __HIP_PLATFORM_AMD__)
endif()

if(HIPCV_BUILD_EXAMPLES)
    add_executable(hipcv_windows_smoke examples/windows_smoke.cpp)
    target_link_libraries(hipcv_windows_smoke PRIVATE hipcv)

    add_executable(hipcv_cvt_color examples/cvt_color.cpp)
    target_link_libraries(hipcv_cvt_color PRIVATE hipcv)

    add_executable(hipcv_resize examples/resize.cpp)
    target_link_libraries(hipcv_resize PRIVATE hipcv)

    add_executable(hipcv_threshold examples/threshold.cpp)
    target_link_libraries(hipcv_threshold PRIVATE hipcv)

    add_executable(hipcv_blur examples/blur.cpp)
    target_link_libraries(hipcv_blur PRIVATE hipcv)

    add_executable(hipcv_gaussian_blur examples/gaussian_blur.cpp)
    target_link_libraries(hipcv_gaussian_blur PRIVATE hipcv)
endif()

if(HIPCV_BUILD_TESTS)
    enable_testing()

    function(hipcv_add_test name source)
        add_executable(${name} ${source})
        target_link_libraries(${name} PRIVATE hipcv)
        add_test(NAME ${name} COMMAND ${name})
    endfunction()

    if(TARGET hipcv_windows_smoke)
        add_test(NAME hipcv_windows_smoke COMMAND hipcv_windows_smoke)
    endif()

    hipcv_add_test(hipcv_test_status tests/test_status.cpp)
    hipcv_add_test(hipcv_test_gpu_mat_no_hip tests/test_gpu_mat_no_hip.cpp)
    hipcv_add_test(hipcv_test_cvt_color tests/test_cvt_color.cpp)
    hipcv_add_test(hipcv_test_resize tests/test_resize.cpp)
    hipcv_add_test(hipcv_test_threshold tests/test_threshold.cpp)
    hipcv_add_test(hipcv_test_blur tests/test_blur.cpp)
    hipcv_add_test(hipcv_test_gaussian_blur tests/test_gaussian_blur.cpp)
endif()
