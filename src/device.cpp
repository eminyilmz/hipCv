#include "hipcv/device.hpp"

#if HIPCV_HAS_HIP
#include <hip/hip_runtime.h>
#endif

namespace hipcv {

DeviceQuery query_devices() noexcept
{
    DeviceQuery query{};

#if HIPCV_HAS_HIP
    query.hip_backend_enabled = true;

    int count = 0;
    if (hipGetDeviceCount(&count) == hipSuccess) {
        query.device_count = count;
    }

    int runtime_version = 0;
    if (hipRuntimeGetVersion(&runtime_version) == hipSuccess) {
        query.runtime_version = runtime_version;
    }
#endif

    return query;
}

} // namespace hipcv
