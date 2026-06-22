#pragma once

#ifndef HIPCV_VERSION_MAJOR
#define HIPCV_VERSION_MAJOR 0
#endif

#ifndef HIPCV_VERSION_MINOR
#define HIPCV_VERSION_MINOR 1
#endif

#ifndef HIPCV_VERSION_PATCH
#define HIPCV_VERSION_PATCH 0
#endif

namespace hipcv {

constexpr int version_major = HIPCV_VERSION_MAJOR;
constexpr int version_minor = HIPCV_VERSION_MINOR;
constexpr int version_patch = HIPCV_VERSION_PATCH;

const char* version_string() noexcept;
bool built_with_hip() noexcept;

} // namespace hipcv
