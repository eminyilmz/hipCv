#include "hipcv/hipcv.hpp"

#include <iostream>

int main()
{
    const auto devices = hipcv::query_devices();

    std::cout << "hipcv " << hipcv::version_string() << '\n';
    std::cout << "HIP backend: " << (hipcv::built_with_hip() ? "enabled" : "disabled") << '\n';
    std::cout << "HIP runtime detected: " << (devices.hip_backend_enabled ? "yes" : "no") << '\n';
    std::cout << "Device count: " << devices.device_count << '\n';

    if (devices.runtime_version > 0) {
        std::cout << "HIP runtime version: " << devices.runtime_version << '\n';
    }

    hipcv::GpuMat mat;
    std::cout << "Empty GpuMat: " << (mat.empty() ? "yes" : "no") << '\n';

    return 0;
}
