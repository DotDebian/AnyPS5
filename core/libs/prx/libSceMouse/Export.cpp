#include <cstdint>
#include <cstddef>
#include <chrono>
#include <cstring>
#include <map>
#include <mutex>
#include "SceTypes.hpp"
#include "prx/libc/include/General.hpp"

namespace {

constexpr int MouseErrorInvalidArg = static_cast<int>(0x80DF0001);
constexpr int MouseErrorInvalidHandle = static_cast<int>(0x80DF0003);
constexpr int MouseErrorAlreadyOpened = static_cast<int>(0x80DF0004);
constexpr int MouseErrorNotInitialized = static_cast<int>(0x80DF0005);

struct MouseState {
    std::mutex mutex;
    bool initialized = false;
    int32_t nextHandle = 1;
    std::map<int32_t, int> userByHandle;
};

MouseState& State() {
    static MouseState state;
    return state;
}

}

extern "C" {

int APS5_VABI sceMouseClose(int32_t handle) {
 auto& state = State();
 std::lock_guard lock(state.mutex);
 return state.userByHandle.erase(handle) != 0 ? 0 : MouseErrorInvalidHandle;
}

int APS5_VABI sceMouseInit(void) {
 auto& state = State();
 std::lock_guard lock(state.mutex);
 state.initialized = true;
 return 0;
}

int APS5_VABI sceMouseOpen(int user_id, int32_t type, int32_t index, const void* param) {
 (void)param;
 auto& state = State();
 std::lock_guard lock(state.mutex);
 if (!state.initialized) return MouseErrorNotInitialized;
 if (type != 0 || index != 0) return MouseErrorInvalidArg;
 for (const auto& [handle, user] : state.userByHandle)
  if (user == user_id) return MouseErrorAlreadyOpened;
 const int32_t handle = state.nextHandle++;
 state.userByHandle.emplace(handle, user_id);
 return handle;
}

int APS5_VABI sceMouseRead(int32_t handle, MouseData* data, int32_t num) {
 if (data == nullptr || num < 1) return MouseErrorInvalidArg;
 auto& state = State();
 std::lock_guard lock(state.mutex);
 if (state.userByHandle.find(handle) == state.userByHandle.end()) return MouseErrorInvalidHandle;
 std::memset(data, 0, sizeof(MouseData));
 data->timestamp = static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now().time_since_epoch()).count());
 data->connected = false;
 return 1;
}

}
