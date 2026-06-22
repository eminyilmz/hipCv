#include "hipcv/status.hpp"

#include <iostream>

namespace {

int failures = 0;

void expect(bool condition, const char* message)
{
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

} // namespace

int main()
{
    const auto ok = hipcv::Status::success();
    expect(ok.ok(), "success status should be ok");
    expect(ok.code() == hipcv::StatusCode::ok, "success status should use ok code");
    expect(ok.message() != nullptr, "success status should have a message");

    const hipcv::Status invalid{hipcv::StatusCode::invalid_argument, "bad input"};
    expect(!invalid.ok(), "invalid status should not be ok");
    expect(invalid.code() == hipcv::StatusCode::invalid_argument, "invalid status should keep its code");
    expect(invalid.message() != nullptr, "invalid status should have a message");

    return failures == 0 ? 0 : 1;
}
