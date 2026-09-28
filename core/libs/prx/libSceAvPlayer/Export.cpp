#include <cstdint>
#include "SceTypes.hpp"
#include "prx/libSceAvPlayer/include/AvPlayer.hpp"
#include "prx/libc/include/General.hpp"

#pragma GCC visibility push(default)
extern "C" {

Bool APS5_VABI sceAvPlayerGetVideoData(AvPlayerInternal* h, AvPlayerFrameInfo* video_info) {
    if (!h || !video_info) return false;
    return static_cast<AvPlayer::Player*>(h)->GetVideoData(*video_info);
}

}
#pragma GCC visibility pop
