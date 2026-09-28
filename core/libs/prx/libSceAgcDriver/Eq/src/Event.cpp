#include "prx/libSceAgcDriver/Eq/include/Event.hpp"

#include <algorithm>
#include <cstdint>
#include <cstddef>
#include <mutex>
#include <vector>
#include "SceTypes.hpp"
#include "prx/libc/include/General.hpp"
#include "prx/libkernel/Equeue/Equeue.hpp"

namespace {

constexpr int16_t EvfiltGraphicsCore = -14;
constexpr int AgcErrorInvalidArgument = static_cast<int>(0x80000016);

struct Registration {
    KernelEqueue eq;
    int id;
};

std::mutex g_mutex;
std::vector<Registration> g_registrations;

}

void AgcDriverDeliverEopInterrupt() {
    constexpr int GraphicsEndOfPipe = 0x40;
    std::vector<Registration> registrations;
    {
        std::lock_guard lock(g_mutex);
        registrations = g_registrations;
    }
    for (const auto& registration : registrations)
        if (registration.id == GraphicsEndOfPipe)
            EqueueTriggerEvent_nid_postfix(registration.eq, static_cast<uintptr_t>(registration.id), EvfiltGraphicsCore, nullptr);
}

extern "C" {

int APS5_VABI sceAgcDriverAddEqEvent(KernelEqueue eq, int id, void* udata) {
 if (eq == 0 || !EqueuePin_nid_postfix(eq)) return AgcErrorInvalidArgument;
 KernelEqueueEvent event{};
 event.event.ident = static_cast<uintptr_t>(id);
 event.event.filter = EvfiltGraphicsCore;
 event.event.flags = EV_ADD | EV_CLEAR;
 event.event.udata = udata;
 event.filter.triggerFunc = [](KernelEqueueEvent* e, void*) {
  e->event.data = e->triggered ? e->event.data + 1 : 1;
  e->triggered = true;
 };
 event.filter.resetFunc = [](KernelEqueueEvent* e) {
  e->triggered = false;
  e->event.data = 0;
 };
 const int result = EqueueAddEvent_nid_postfix(eq, event);
 if (result != EQUEUE_OK) return result;
 std::lock_guard lock(g_mutex);
 if (std::none_of(g_registrations.begin(), g_registrations.end(), [&](const Registration& r) { return r.eq == eq && r.id == id; }))
  g_registrations.push_back({eq, id});
 return 0;
}

int APS5_VABI sceAgcDriverDeleteEqEvent(KernelEqueue eq, int id) {
 {
  std::lock_guard lock(g_mutex);
  std::erase_if(g_registrations, [&](const Registration& r) { return r.eq == eq && r.id == id; });
 }
 const int result = EqueueDeleteEvent_nid_postfix(eq, static_cast<uintptr_t>(id), EvfiltGraphicsCore);
 return result == EQUEUE_ERROR_ENOENT ? AgcErrorInvalidArgument : result;
}

}
