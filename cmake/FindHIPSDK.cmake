include(FindPackageHandleStandardArgs)

set(_HIPSDK_HINTS)

foreach(_var HIP_PATH ROCM_PATH HIPSDK_ROOT ROCM_ROOT)
    if(DEFINED ENV{${_var}})
        list(APPEND _HIPSDK_HINTS "$ENV{${_var}}")
    endif()
endforeach()

if(WIN32)
    file(GLOB _HIPSDK_WINDOWS_ROOTS
        "C:/Program Files/AMD/ROCm/*"
        "C:/Program Files/AMD/HIP SDK/*"
    )
    list(APPEND _HIPSDK_HINTS ${_HIPSDK_WINDOWS_ROOTS})
else()
    list(APPEND _HIPSDK_HINTS
        "/opt/rocm"
        "/usr/local/rocm"
    )
endif()

find_program(HIPSDK_HIPCONFIG_EXECUTABLE
    NAMES hipconfig hipconfig.exe
    HINTS ${_HIPSDK_HINTS}
    PATH_SUFFIXES bin
)

find_program(HIPSDK_HIPCC_EXECUTABLE
    NAMES hipcc hipcc.exe
    HINTS ${_HIPSDK_HINTS}
    PATH_SUFFIXES bin
)

find_path(HIPSDK_INCLUDE_DIR
    NAMES hip/hip_runtime.h
    HINTS ${_HIPSDK_HINTS}
    PATH_SUFFIXES include
)

find_library(HIPSDK_AMDHIP64_LIBRARY
    NAMES amdhip64
    HINTS ${_HIPSDK_HINTS}
    PATH_SUFFIXES lib lib64
)

find_library(HIPSDK_HIPRTC_LIBRARY
    NAMES hiprtc
    HINTS ${_HIPSDK_HINTS}
    PATH_SUFFIXES lib lib64
)

find_package_handle_standard_args(HIPSDK
    REQUIRED_VARS
        HIPSDK_HIPCONFIG_EXECUTABLE
        HIPSDK_INCLUDE_DIR
        HIPSDK_AMDHIP64_LIBRARY
        HIPSDK_HIPRTC_LIBRARY
)

if(HIPSDK_FOUND AND NOT TARGET HIPSDK::amdhip64)
    add_library(HIPSDK::amdhip64 UNKNOWN IMPORTED)
    set_target_properties(HIPSDK::amdhip64 PROPERTIES
        IMPORTED_LOCATION "${HIPSDK_AMDHIP64_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${HIPSDK_INCLUDE_DIR}"
    )
endif()

if(HIPSDK_FOUND AND HIPSDK_HIPRTC_LIBRARY AND NOT TARGET HIPSDK::hiprtc)
    add_library(HIPSDK::hiprtc UNKNOWN IMPORTED)
    set_target_properties(HIPSDK::hiprtc PROPERTIES
        IMPORTED_LOCATION "${HIPSDK_HIPRTC_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${HIPSDK_INCLUDE_DIR}"
    )
endif()

mark_as_advanced(
    HIPSDK_HIPCONFIG_EXECUTABLE
    HIPSDK_HIPCC_EXECUTABLE
    HIPSDK_INCLUDE_DIR
    HIPSDK_AMDHIP64_LIBRARY
    HIPSDK_HIPRTC_LIBRARY
)
