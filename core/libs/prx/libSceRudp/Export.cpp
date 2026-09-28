#include <cstdint>
#include <cstddef>
#include <mutex>
#include "SceTypes.hpp"
#include "prx/libc/include/General.hpp"

namespace {

constexpr int RudpErrorNotInitialized = static_cast<int>(0x80770001);
constexpr int RudpErrorAlreadyInitialized = static_cast<int>(0x80770002);
constexpr int RudpErrorInvalidArgument = static_cast<int>(0x80770004);

struct RudpState {
 std::mutex mutex;
 bool initialized = false;
 bool internalIoThread = false;
 RudpEventHandler handler = nullptr;
 void* handlerArg = nullptr;
};

RudpState& State() {
 static RudpState state;
 return state;
}

}

extern "C" {

int APS5_VABI sceRudpEnableInternalIOThread(uint32_t stack_size, uint32_t priority) {
 (void)stack_size;
 (void)priority;
 auto& state = State();
 std::lock_guard lock(state.mutex);
 if (!state.initialized) return RudpErrorNotInitialized;
 state.internalIoThread = true;
 return 0;
}

int APS5_VABI sceRudpInit_nid_postfix(void* mem_pool, int mem_pool_size) {
 if (mem_pool == nullptr || mem_pool_size <= 0) return RudpErrorInvalidArgument;
 auto& state = State();
 std::lock_guard lock(state.mutex);
 if (state.initialized) return RudpErrorAlreadyInitialized;
 state.initialized = true;
 return 0;
}

int APS5_VABI sceRudpSetEventHandler(RudpEventHandler handler, void* arg) {
 auto& state = State();
 std::lock_guard lock(state.mutex);
 if (!state.initialized) return RudpErrorNotInitialized;
 state.handler = handler;
 state.handlerArg = arg;
 return 0;
}

}
