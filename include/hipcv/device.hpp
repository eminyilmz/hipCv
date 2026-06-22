#pragma once

namespace hipcv {

struct DeviceQuery {
    bool hip_backend_enabled = false;
    int device_count = 0;
    int runtime_version = 0;
};

DeviceQuery query_devices() noexcept;

} // namespace hipcv
