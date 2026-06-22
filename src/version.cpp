#include "hipcv/version.hpp"

namespace hipcv {

const char* version_string() noexcept
{
    return "0.1.0";
}

bool built_with_hip() noexcept
{
#if HIPCV_HAS_HIP
    return true;
#else
    return false;
#endif
}

} // namespace hipcv
