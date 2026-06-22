#pragma once

namespace hipcv {

enum class StatusCode {
    ok,
    hip_not_enabled,
    invalid_argument,
    allocation_failed,
    copy_failed,
    runtime_error
};

class Status {
public:
    constexpr Status() = default;
    constexpr Status(StatusCode code, const char* message) noexcept
        : code_(code), message_(message)
    {
    }

    [[nodiscard]] constexpr bool ok() const noexcept { return code_ == StatusCode::ok; }
    [[nodiscard]] constexpr StatusCode code() const noexcept { return code_; }
    [[nodiscard]] constexpr const char* message() const noexcept { return message_; }

    static constexpr Status success() noexcept { return {}; }

private:
    StatusCode code_ = StatusCode::ok;
    const char* message_ = "ok";
};

} // namespace hipcv
