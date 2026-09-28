#include <cstdint>
#include "SceTypes.hpp"
#include "prx/libc/include/General.hpp"

#pragma GCC visibility push(default)
extern "C" {

int APS5_VABI sceAvPlayerGetStreamInfoEx(AvPlayerInternal* h, uint32_t stream_id, void* info) {
    (void)h;
    (void)stream_id;
    (void)info;
    NotImplemented_nid_no_patch(__func__);
    return 0;
}

int APS5_VABI sceAvPlayerSetAvailableBandwidth(AvPlayerInternal* h, uint32_t start_bandwidth, uint32_t minimum_bandwidth, uint32_t maximum_bandwidth) {
    (void)h;
    (void)start_bandwidth;
    (void)minimum_bandwidth;
    (void)maximum_bandwidth;
    NotImplemented_nid_no_patch(__func__);
    return 0;
}

int APS5_VABI sceAvPlayerStartEx(AvPlayerInternal* h, const void* start_info_ex) {
    (void)h;
    (void)start_info_ex;
    NotImplemented_nid_no_patch(__func__);
    return 0;
}

}
#pragma GCC visibility pop
