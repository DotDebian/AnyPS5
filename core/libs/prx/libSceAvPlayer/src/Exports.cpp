// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <cstdint>
#include <string_view>
#include "SceTypes.hpp"
#include "prx/libSceAvPlayer/include/AvPlayer.hpp"
#include "prx/libc/include/General.hpp"

using namespace AvPlayer;

namespace {

Player* ToPlayer(AvPlayerInternal* h) {
    return static_cast<Player*>(h);
}

bool HasAllocators(const AvPlayerMemAllocator& memory) {
    return memory.allocate && memory.deallocate && memory.allocate_texture && memory.deallocate_texture;
}

std::uint32_t VideoBufferCount(std::int32_t requested) {
    return static_cast<std::uint32_t>(std::clamp(requested, 2, 16));
}

}

#pragma GCC visibility push(default)
extern "C" {

int APS5_VABI sceAvPlayerAddSource(AvPlayerInternal* h, const char* filename) {
    if (!h || !filename) return SCE_AVPLAYER_ERROR_INVALID_PARAMS;
    return ToPlayer(h)->AddSource(filename, SourceTypeUnknown);
}

int APS5_VABI sceAvPlayerAddSourceEx(AvPlayerInternal* h, uint32_t uri_type, const AvPlayerSourceDetails* source_details) {
    if (!h || uri_type != 0 || !source_details || !source_details->uri.name) return SCE_AVPLAYER_ERROR_INVALID_PARAMS;
    return ToPlayer(h)->AddSource(std::string_view(source_details->uri.name, source_details->uri.length), source_details->source_type);
}

int APS5_VABI sceAvPlayerChangeStream(AvPlayerInternal* h, uint32_t old_stream_id, uint32_t new_stream_id) {
    (void)h;
    (void)old_stream_id;
    (void)new_stream_id;
    NotImplemented_nid_no_patch(__func__);
    return 0;
}

int APS5_VABI sceAvPlayerClose(AvPlayerInternal* h) {
    if (!h) return SCE_AVPLAYER_ERROR_INVALID_PARAMS;
    delete ToPlayer(h);
    return SCE_OK;
}

uint64_t APS5_VABI sceAvPlayerCurrentTime(AvPlayerInternal* h) {
    if (!h) return static_cast<uint64_t>(static_cast<int64_t>(SCE_AVPLAYER_ERROR_INVALID_PARAMS));
    return ToPlayer(h)->CurrentTime();
}

int APS5_VABI sceAvPlayerDisableStream(AvPlayerInternal* h, uint32_t stream_id) {
    if (!h) return SCE_AVPLAYER_ERROR_INVALID_PARAMS;
    return ToPlayer(h)->DisableStream(stream_id);
}

int APS5_VABI sceAvPlayerEnableStream(AvPlayerInternal* h, uint32_t stream_id) {
    if (!h) return SCE_AVPLAYER_ERROR_INVALID_PARAMS;
    return ToPlayer(h)->EnableStream(stream_id);
}

Bool APS5_VABI sceAvPlayerGetAudioData(AvPlayerInternal* h, AvPlayerFrameInfo* audio_info) {
    if (!h || !audio_info) return false;
    return ToPlayer(h)->GetAudioData(*audio_info);
}

int APS5_VABI sceAvPlayerGetStreamInfo(AvPlayerInternal* h, uint32_t stream_id, AvPlayerStreamInfo* info) {
    if (!h || !info) return SCE_AVPLAYER_ERROR_INVALID_PARAMS;
    return ToPlayer(h)->GetStreamInfo(stream_id, *info);
}

Bool APS5_VABI sceAvPlayerGetVideoDataEx(AvPlayerInternal* h, AvPlayerFrameInfoEx* video_info) {
    if (!h || !video_info) return false;
    return ToPlayer(h)->GetVideoData(*video_info);
}

AvPlayerInternal* APS5_VABI sceAvPlayerInit(AvPlayerInitData* init) {
    if (!init || !HasAllocators(init->memory_replacement)) return nullptr;
    return new Player(*init, VideoBufferCount(init->num_output_video_framebuffers));
}

int APS5_VABI sceAvPlayerInitEx(const AvPlayerInitDataEx* init_ex, AvPlayerInternal** handle) {
    if (!init_ex || !handle || !HasAllocators(init_ex->memory_replacement)) return SCE_AVPLAYER_ERROR_INVALID_PARAMS;
    AvPlayerInitData init{};
    init.memory_replacement = init_ex->memory_replacement;
    init.file_replacement = init_ex->file_replacement;
    init.event_replacement = init_ex->event_replacement;
    init.debug_level = init_ex->debug_level;
    init.num_output_video_framebuffers = init_ex->num_output_video_framebuffers;
    init.auto_start = init_ex->auto_start;
    init.default_language = init_ex->default_language;
    *handle = new Player(init, VideoBufferCount(init.num_output_video_framebuffers));
    return SCE_OK;
}

Bool APS5_VABI sceAvPlayerIsActive(AvPlayerInternal* h) {
    if (!h) return false;
    return ToPlayer(h)->IsActive();
}

int APS5_VABI sceAvPlayerJumpToTime(AvPlayerInternal* h, uint64_t time_ms) {
    if (!h) return SCE_AVPLAYER_ERROR_INVALID_PARAMS;
    return ToPlayer(h)->JumpToTime(time_ms);
}

int APS5_VABI sceAvPlayerPause(AvPlayerInternal* h) {
    if (!h) return SCE_AVPLAYER_ERROR_INVALID_PARAMS;
    return ToPlayer(h)->Pause();
}

int APS5_VABI sceAvPlayerPostInit(AvPlayerInternal* h, const AvPlayerPostInitData* post_init) {
    if (!h || !post_init) return SCE_AVPLAYER_ERROR_INVALID_PARAMS;
    return ToPlayer(h)->PostInit(*post_init);
}

int APS5_VABI sceAvPlayerResume(AvPlayerInternal* h) {
    if (!h) return SCE_AVPLAYER_ERROR_INVALID_PARAMS;
    return ToPlayer(h)->Resume();
}

int APS5_VABI sceAvPlayerSetAvSyncMode(AvPlayerInternal* h, uint32_t sync_mode) {
    if (!h) return SCE_AVPLAYER_ERROR_INVALID_PARAMS;
    return ToPlayer(h)->SetAvSyncMode(sync_mode);
}

int APS5_VABI sceAvPlayerSetLogCallback(void* callback, void* user_data) {
    (void)callback;
    (void)user_data;
    NotImplemented_nid_no_patch(__func__);
    return 0;
}

int APS5_VABI sceAvPlayerSetLooping(AvPlayerInternal* h, Bool loop) {
    if (!h) return SCE_AVPLAYER_ERROR_INVALID_PARAMS;
    return ToPlayer(h)->SetLooping(loop != 0);
}

int APS5_VABI sceAvPlayerSetTrickSpeed(AvPlayerInternal* h, int32_t trick_speed) {
    if (!h) return SCE_AVPLAYER_ERROR_INVALID_PARAMS;
    return ToPlayer(h)->SetTrickSpeed(trick_speed);
}

int APS5_VABI sceAvPlayerStart(AvPlayerInternal* h) {
    if (!h) return SCE_AVPLAYER_ERROR_INVALID_PARAMS;
    return ToPlayer(h)->Start();
}

int APS5_VABI sceAvPlayerStop(AvPlayerInternal* h) {
    if (!h) return SCE_AVPLAYER_ERROR_INVALID_PARAMS;
    return ToPlayer(h)->Stop();
}

int APS5_VABI sceAvPlayerStreamCount(AvPlayerInternal* h) {
    if (!h) return SCE_AVPLAYER_ERROR_INVALID_PARAMS;
    return ToPlayer(h)->StreamCount();
}

}
#pragma GCC visibility pop
