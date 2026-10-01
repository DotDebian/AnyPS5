#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include "SceTypes.hpp"
#include "prx/libc/include/general/VabiMacros.hpp"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/channel_layout.h>
#include <libavutil/opt.h>
}

extern "C" {
AvPlayerInternal* APS5_VABI sceAvPlayerInit(AvPlayerInitData*);
int APS5_VABI sceAvPlayerPostInit(AvPlayerInternal*, const AvPlayerPostInitData*);
int APS5_VABI sceAvPlayerAddSource(AvPlayerInternal*, const char*);
int APS5_VABI sceAvPlayerStreamCount(AvPlayerInternal*);
int APS5_VABI sceAvPlayerGetStreamInfo(AvPlayerInternal*, std::uint32_t, AvPlayerStreamInfo*);
int APS5_VABI sceAvPlayerEnableStream(AvPlayerInternal*, std::uint32_t);
int APS5_VABI sceAvPlayerStart(AvPlayerInternal*);
int APS5_VABI sceAvPlayerStop(AvPlayerInternal*);
int APS5_VABI sceAvPlayerPause(AvPlayerInternal*);
int APS5_VABI sceAvPlayerResume(AvPlayerInternal*);
int APS5_VABI sceAvPlayerJumpToTime(AvPlayerInternal*, std::uint64_t);
int APS5_VABI sceAvPlayerSetLooping(AvPlayerInternal*, Bool);
int APS5_VABI sceAvPlayerSetAvSyncMode(AvPlayerInternal*, std::uint32_t);
int APS5_VABI sceAvPlayerSetTrickSpeed(AvPlayerInternal*, std::int32_t);
Bool APS5_VABI sceAvPlayerGetVideoDataEx(AvPlayerInternal*, AvPlayerFrameInfoEx*);
Bool APS5_VABI sceAvPlayerGetVideoData(AvPlayerInternal*, AvPlayerFrameInfo*);
Bool APS5_VABI sceAvPlayerGetAudioData(AvPlayerInternal*, AvPlayerFrameInfo*);
Bool APS5_VABI sceAvPlayerIsActive(AvPlayerInternal*);
std::uint64_t APS5_VABI sceAvPlayerCurrentTime(AvPlayerInternal*);
int APS5_VABI sceAvPlayerClose(AvPlayerInternal*);
}

namespace {

constexpr int Width = 100;
constexpr int Height = 60;
constexpr int Pitch = 256;
constexpr int FrameRate = 30;
constexpr int FrameCount = 30;
constexpr int SampleRate = 48000;
constexpr int SkipTest = 77;
constexpr int InvalidParams = static_cast<int>(0x806A0001u);
constexpr int OperationFailed = static_cast<int>(0x806A0002u);
constexpr std::int32_t LoopingBack = static_cast<std::int32_t>(0x806A00A1u);
constexpr std::int32_t JumpComplete = static_cast<std::int32_t>(0x806A00A3u);
constexpr std::int32_t EventStop = 1;
constexpr std::int32_t EventReady = 2;
constexpr std::int32_t EventPlay = 3;
constexpr std::int32_t EventPause = 4;
constexpr std::int32_t EventWarning = 0x20;
const char* const MoviePath = "app0/avplayer.mp4";

struct Skip {};

void Check(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

int LumaFor(int frame) {
    return 20 + frame * 7;
}

class Encoder {
public:
    ~Encoder() {
        avcodec_free_context(&m_video);
        avcodec_free_context(&m_audio);
        av_frame_free(&m_frame);
        av_packet_free(&m_packet);
        if (m_output) {
            if (m_output->pb) avio_closep(&m_output->pb);
            avformat_free_context(m_output);
        }
    }

    void Write(const char* path) {
        const auto* h264 = avcodec_find_encoder_by_name("libx264");
        const auto* aac = avcodec_find_encoder(AV_CODEC_ID_AAC);
        if (!h264 || !aac) throw Skip{};
        Check(avformat_alloc_output_context2(&m_output, nullptr, "mp4", path) >= 0, "mp4 muxer unavailable");
        m_video = avcodec_alloc_context3(h264);
        m_video->width = Width;
        m_video->height = Height;
        m_video->time_base = {1, FrameRate};
        m_video->framerate = {FrameRate, 1};
        m_video->pix_fmt = AV_PIX_FMT_YUV420P;
        m_video->gop_size = 10;
        m_video->max_b_frames = 2;
        av_opt_set(m_video->priv_data, "crf", "4", 0);
        m_audio = avcodec_alloc_context3(aac);
        m_audio->sample_fmt = AV_SAMPLE_FMT_FLTP;
        m_audio->sample_rate = SampleRate;
        m_audio->bit_rate = 128000;
        m_audio->time_base = {1, SampleRate};
        const AVChannelLayout stereo = AV_CHANNEL_LAYOUT_STEREO;
        av_channel_layout_copy(&m_audio->ch_layout, &stereo);
        for (auto* context : {m_video, m_audio}) {
            if (m_output->oformat->flags & AVFMT_GLOBALHEADER) context->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
            Check(avcodec_open2(context, nullptr, nullptr) >= 0, "encoder open failed");
            auto* stream = avformat_new_stream(m_output, nullptr);
            avcodec_parameters_from_context(stream->codecpar, context);
            stream->time_base = context->time_base;
        }
        av_dict_set(&m_output->streams[0]->metadata, "language", "eng", 0);
        av_dict_set(&m_output->streams[1]->metadata, "language", "eng", 0);
        Check(avio_open(&m_output->pb, path, AVIO_FLAG_WRITE) >= 0, "cannot create the movie");
        Check(avformat_write_header(m_output, nullptr) >= 0, "header write failed");
        m_frame = av_frame_alloc();
        m_packet = av_packet_alloc();
        for (int index = 0; index < FrameCount; ++index) {
            av_frame_unref(m_frame);
            m_frame->format = AV_PIX_FMT_YUV420P;
            m_frame->width = Width;
            m_frame->height = Height;
            Check(av_frame_get_buffer(m_frame, 0) >= 0, "frame allocation failed");
            for (int row = 0; row < Height; ++row) {
                for (int column = 0; column < Width; ++column) m_frame->data[0][row * m_frame->linesize[0] + column] = static_cast<std::uint8_t>(column < 48 ? LumaFor(index) : 255 - LumaFor(index));
            }
            for (int plane = 1; plane < 3; ++plane) {
                for (int row = 0; row < Height / 2; ++row) std::memset(m_frame->data[plane] + row * m_frame->linesize[plane], 128, Width / 2);
            }
            m_frame->pts = index;
            Encode(m_video, m_frame, 0);
        }
        Encode(m_video, nullptr, 0);
        const int samples = m_audio->frame_size;
        for (int offset = 0; offset < SampleRate; offset += samples) {
            av_frame_unref(m_frame);
            m_frame->format = AV_SAMPLE_FMT_FLTP;
            m_frame->nb_samples = samples;
            m_frame->sample_rate = SampleRate;
            av_channel_layout_copy(&m_frame->ch_layout, &stereo);
            Check(av_frame_get_buffer(m_frame, 0) >= 0, "audio frame allocation failed");
            for (int channel = 0; channel < 2; ++channel) {
                auto* data = reinterpret_cast<float*>(m_frame->data[channel]);
                for (int sample = 0; sample < samples; ++sample) data[sample] = 0.25f * static_cast<float>(std::sin(2.0 * 3.14159265358979 * 440.0 * (offset + sample) / SampleRate));
            }
            m_frame->pts = offset;
            Encode(m_audio, m_frame, 1);
        }
        Encode(m_audio, nullptr, 1);
        Check(av_write_trailer(m_output) >= 0, "trailer write failed");
    }

private:
    void Encode(AVCodecContext* context, AVFrame* frame, int stream) {
        Check(avcodec_send_frame(context, frame) >= 0, "encoding failed");
        while (avcodec_receive_packet(context, m_packet) >= 0) {
            av_packet_rescale_ts(m_packet, context->time_base, m_output->streams[stream]->time_base);
            m_packet->stream_index = stream;
            Check(av_interleaved_write_frame(m_output, m_packet) >= 0, "packet write failed");
        }
    }

    AVFormatContext* m_output = nullptr;
    AVCodecContext* m_video = nullptr;
    AVCodecContext* m_audio = nullptr;
    AVFrame* m_frame = nullptr;
    AVPacket* m_packet = nullptr;
};

struct Allocations {
    std::mutex mutex;
    std::map<void*, bool> blocks;
    int textures = 0;
};

Allocations allocations;

void* Allocate(std::uint32_t alignment, std::uint32_t size, bool texture) {
    alignment = std::max<std::uint32_t>(alignment, 16);
    void* memory = std::aligned_alloc(alignment, (size + alignment - 1) / alignment * alignment);
    std::lock_guard lock(allocations.mutex);
    allocations.blocks[memory] = texture;
    if (texture) ++allocations.textures;
    return memory;
}

void Release(void* memory, bool texture) {
    {
        std::lock_guard lock(allocations.mutex);
        const auto found = allocations.blocks.find(memory);
        if (found == allocations.blocks.end() || found->second != texture) std::abort();
        allocations.blocks.erase(found);
    }
    std::free(memory);
}

void* APS5_VABI AllocateMemory(void*, std::uint32_t alignment, std::uint32_t size) { return Allocate(alignment, size, false); }
void APS5_VABI DeallocateMemory(void*, void* memory) { Release(memory, false); }
void* APS5_VABI AllocateTexture(void*, std::uint32_t alignment, std::uint32_t size) { return Allocate(alignment, size, true); }
void APS5_VABI DeallocateTexture(void*, void* memory) { Release(memory, true); }

bool IsAllocation(const void* memory, bool texture) {
    std::lock_guard lock(allocations.mutex);
    const auto found = allocations.blocks.find(const_cast<void*>(memory));
    return found != allocations.blocks.end() && found->second == texture;
}

struct Events {
    std::mutex mutex;
    std::vector<std::pair<std::int32_t, std::int32_t>> received;

    bool Seen(std::int32_t id, std::int32_t warning = 0) {
        std::lock_guard lock(mutex);
        return std::any_of(received.begin(), received.end(), [&](const auto& event) { return event.first == id && event.second == warning; });
    }

    void Clear() {
        std::lock_guard lock(mutex);
        received.clear();
    }
};

void APS5_VABI OnEvent(void* object, std::int32_t id, std::int32_t source, void* data) {
    auto* events = static_cast<Events*>(object);
    if (source != 0 || (id == EventWarning) != (data != nullptr)) std::abort();
    std::lock_guard lock(events->mutex);
    events->received.emplace_back(id, id == EventWarning ? *static_cast<std::int32_t*>(data) : 0);
}

struct HostFile {
    std::ifstream stream;
    std::uint64_t size = 0;
};

std::int32_t APS5_VABI OpenFile(void* object, const char* path) {
    auto* file = static_cast<HostFile*>(object);
    if (std::strcmp(path, "/app0/replaced.mp4") != 0) return -1;
    file->stream.open(MoviePath, std::ios::binary);
    file->size = std::filesystem::file_size(MoviePath);
    return file->stream ? 0 : -1;
}

std::int32_t APS5_VABI CloseFile(void* object) {
    static_cast<HostFile*>(object)->stream.close();
    return 0;
}

std::int32_t APS5_VABI ReadFile(void* object, std::uint8_t* buffer, std::uint64_t position, std::uint32_t length) {
    auto* file = static_cast<HostFile*>(object);
    file->stream.clear();
    file->stream.seekg(static_cast<std::streamoff>(position));
    file->stream.read(reinterpret_cast<char*>(buffer), length);
    return static_cast<std::int32_t>(file->stream.gcount());
}

std::uint64_t APS5_VABI FileSize(void* object) {
    return static_cast<HostFile*>(object)->size;
}

bool WaitFor(const std::function<bool()>& condition, int milliseconds = 5000) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(milliseconds);
    while (std::chrono::steady_clock::now() < deadline) {
        if (condition()) return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return condition();
}

AvPlayerInitData InitData(Events* events) {
    AvPlayerInitData init{};
    init.memory_replacement = {nullptr, AllocateMemory, DeallocateMemory, AllocateTexture, DeallocateTexture};
    init.event_replacement = {events, events ? OnEvent : nullptr};
    init.num_output_video_framebuffers = 4;
    init.default_language = "eng";
    return init;
}

int FrameIndex(std::uint64_t timestamp) {
    return static_cast<int>((timestamp * FrameRate + 500) / 1000);
}

void CheckVideoFrame(const AvPlayerFrameInfoEx& info) {
    const auto& video = info.details.video;
    Check(video.width == 112 && video.height == 64 && video.pitch == Pitch, "unexpected frame geometry");
    Check(video.crop_left_offset == 0 && video.crop_top_offset == 0 && video.crop_right_offset == Pitch - Width && video.crop_bottom_offset == 4, "unexpected crop offsets");
    Check(video.luma_bit_depth == 8 && video.chroma_bit_depth == 8, "unexpected bit depth");
    Check(std::fabs(video.aspect_ratio - static_cast<float>(Width) / Height) < 0.01f, "unexpected aspect ratio");
    Check(std::memcmp(video.language_code, "eng", 4) == 0, "missing video language");
    Check(IsAllocation(info.p_data, true), "video frame is not a texture allocation");
    const int index = FrameIndex(info.timestamp);
    Check(index >= 0 && index < FrameCount, "frame timestamp out of range");
    const auto* luma = static_cast<const std::uint8_t*>(info.p_data);
    const auto* chroma = luma + Pitch * 64;
    for (const int row : {2, 30, 57, 63}) {
        Check(std::abs(luma[row * Pitch + 10] - LumaFor(index)) <= 6, "left luma mismatch for frame " + std::to_string(index));
        Check(std::abs(luma[row * Pitch + 90] - (255 - LumaFor(index))) <= 6, "right luma mismatch for frame " + std::to_string(index));
    }
    for (const int row : {0, 15, 31}) {
        Check(std::abs(chroma[row * Pitch + 20] - 128) <= 6 && std::abs(chroma[row * Pitch + 21] - 128) <= 6, "chroma mismatch");
    }
}

void CheckAudioFrame(const AvPlayerFrameInfo& info) {
    Check(info.details.audio.channel_count == 2 && info.details.audio.sample_rate == SampleRate, "unexpected audio format");
    Check(info.details.audio.size > 0 && info.details.audio.size <= 1024 * 2 * 2 && info.details.audio.size % 4 == 0, "unexpected audio size");
    Check(IsAllocation(info.p_data, false), "audio frame is not a plain allocation");
}

void TestPlayback() {
    Events events;
    AvPlayerInitData invalid = InitData(&events);
    invalid.memory_replacement.allocate_texture = nullptr;
    Check(sceAvPlayerInit(&invalid) == nullptr, "init accepted missing allocators");
    AvPlayerInitData init = InitData(&events);
    auto* player = sceAvPlayerInit(&init);
    Check(player != nullptr, "init failed");
    AvPlayerPostInitData post{};
    post.demux_video_buffer_size = 64 * 1024;
    Check(sceAvPlayerPostInit(player, &post) == 0, "post init failed");
    Check(sceAvPlayerPostInit(player, nullptr) == InvalidParams, "post init accepted null data");
    Check(sceAvPlayerStreamCount(player) == OperationFailed, "stream count without a source");
    Check(!sceAvPlayerIsActive(player), "active without a source");
    Check(sceAvPlayerAddSource(player, "/app0/missing.mp4") == OperationFailed, "missing file accepted");
    Check(sceAvPlayerAddSource(player, "/app0/avplayer.mp4") == 0, "add source failed");
    Check(sceAvPlayerAddSource(player, "/app0/avplayer.mp4") == OperationFailed, "second source accepted");
    Check(WaitFor([&] { return events.Seen(EventReady); }), "ready event missing");

    Check(sceAvPlayerStreamCount(player) == 2, "unexpected stream count");
    int video = -1;
    int audio = -1;
    for (std::uint32_t index = 0; index < 2; ++index) {
        AvPlayerStreamInfo info{};
        Check(sceAvPlayerGetStreamInfo(player, index, &info) == 0, "stream info failed");
        Check(info.duration >= 950 && info.duration <= 1100, "unexpected stream duration");
        if (info.type == 1) {
            video = static_cast<int>(index);
            Check(info.details.video.width == 112 && info.details.video.height == 64, "unexpected video stream size");
            Check(std::strcmp(info.details.video.language_code, "eng") == 0, "unexpected video language");
        } else if (info.type == 0) {
            audio = static_cast<int>(index);
            Check(info.details.audio.channel_count == 2 && info.details.audio.sample_rate == SampleRate, "unexpected audio stream");
        }
    }
    Check(video >= 0 && audio >= 0, "streams missing");
    AvPlayerStreamInfo unused{};
    Check(sceAvPlayerGetStreamInfo(player, 2, &unused) == OperationFailed, "invalid stream accepted");
    Check(sceAvPlayerGetStreamInfo(player, 0, nullptr) == InvalidParams, "null stream info accepted");
    Check(sceAvPlayerEnableStream(player, 7) == OperationFailed, "invalid stream enabled");
    Check(sceAvPlayerEnableStream(player, static_cast<std::uint32_t>(video)) == 0, "enable video failed");
    Check(sceAvPlayerEnableStream(player, static_cast<std::uint32_t>(audio)) == 0, "enable audio failed");
    Check(sceAvPlayerIsActive(player), "not active before start");
    Check(sceAvPlayerPause(player) == OperationFailed, "pause accepted before start");
    Check(sceAvPlayerStart(player) == 0, "start failed");
    Check(WaitFor([&] { return events.Seen(EventPlay); }), "play event missing");

    AvPlayerFrameInfoEx frame{};
    AvPlayerFrameInfo sound{};
    int frames = 0;
    int sounds = 0;
    std::uint64_t last = 0;
    Check(WaitFor([&] {
        if (sceAvPlayerGetAudioData(player, &sound)) {
            CheckAudioFrame(sound);
            ++sounds;
        }
        if (sceAvPlayerGetVideoDataEx(player, &frame)) {
            CheckVideoFrame(frame);
            Check(frames == 0 || frame.timestamp > last, "video timestamps not increasing");
            last = frame.timestamp;
            ++frames;
        }
        return frames >= 8 && sounds >= 8;
    }), "playback stalled");
    Check(sceAvPlayerCurrentTime(player) > 0, "clock not running");
    Check(sceAvPlayerSetTrickSpeed(player, 0) == InvalidParams, "zero trick speed accepted");
    Check(sceAvPlayerSetTrickSpeed(player, 100) == 0, "normal trick speed rejected");

    Check(sceAvPlayerPause(player) == 0, "pause failed");
    Check(WaitFor([&] { return events.Seen(EventPause); }), "pause event missing");
    const auto paused = sceAvPlayerCurrentTime(player);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    Check(!sceAvPlayerGetVideoDataEx(player, &frame) && !sceAvPlayerGetAudioData(player, &sound), "data delivered while paused");
    Check(sceAvPlayerCurrentTime(player) == paused, "clock moved while paused");
    Check(sceAvPlayerIsActive(player), "inactive while paused");
    events.Clear();
    Check(sceAvPlayerResume(player) == 0, "resume failed");
    Check(sceAvPlayerResume(player) == OperationFailed, "resume accepted while playing");
    Check(WaitFor([&] { return events.Seen(EventPlay); }), "resume event missing");

    Check(sceAvPlayerJumpToTime(player, 700) == 0, "jump failed");
    Check(WaitFor([&] { return events.Seen(EventWarning, JumpComplete); }), "jump completion missing");
    Check(WaitFor([&] { return sceAvPlayerGetVideoDataEx(player, &frame) != 0; }), "no frame after the jump");
    CheckVideoFrame(frame);
    Check(frame.timestamp >= 700 && frame.timestamp <= 767, "jump landed at " + std::to_string(frame.timestamp));

    Check(sceAvPlayerSetAvSyncMode(player, 2) == InvalidParams, "invalid sync mode accepted");
    Check(sceAvPlayerSetAvSyncMode(player, 1) == 0, "sync mode rejected");
    Check(WaitFor([&] {
        if (sceAvPlayerGetVideoDataEx(player, &frame)) {
            CheckVideoFrame(frame);
            last = frame.timestamp;
        }
        while (sceAvPlayerGetAudioData(player, &sound)) CheckAudioFrame(sound);
        return !sceAvPlayerIsActive(player);
    }), "playback never ended");
    Check(FrameIndex(last) >= FrameCount - 2, "last frame missing");
    Check(WaitFor([&] { return events.Seen(EventStop); }), "end of stream stop event missing");
    Check(sceAvPlayerCurrentTime(player) >= 950, "clock not at the end");
    Check(sceAvPlayerStop(player) == 0, "stop after the end failed");

    events.Clear();
    Check(sceAvPlayerSetLooping(player, 1) == 0, "looping rejected");
    Check(sceAvPlayerStart(player) == 0, "restart failed");
    bool wrapped = false;
    last = 0;
    frames = 0;
    Check(WaitFor([&] {
        while (sceAvPlayerGetAudioData(player, &sound)) CheckAudioFrame(sound);
        if (sceAvPlayerGetVideoDataEx(player, &frame)) {
            CheckVideoFrame(frame);
            if (frames > 0 && frame.timestamp < last) wrapped = true;
            last = frame.timestamp;
            ++frames;
        }
        return wrapped && frames > FrameCount + 5;
    }), "looping never wrapped");
    Check(events.Seen(EventWarning, LoopingBack), "looping warning missing");
    Check(sceAvPlayerIsActive(player), "looping player went inactive");
    Check(sceAvPlayerStop(player) == 0, "stop failed");
    Check(sceAvPlayerStop(player) == OperationFailed, "second stop accepted");
    Check(!sceAvPlayerIsActive(player), "active after stop");
    Check(WaitFor([&] { return events.Seen(EventStop); }), "stop event missing");
    Check(sceAvPlayerClose(player) == 0, "close failed");
    Check(sceAvPlayerClose(nullptr) == InvalidParams, "null close accepted");
    std::lock_guard lock(allocations.mutex);
    Check(allocations.blocks.empty(), "allocations leaked");
    Check(allocations.textures >= 4, "video buffers not allocated as textures");
}

void TestFileReplacementAutoStart() {
    HostFile file;
    AvPlayerInitData init = InitData(nullptr);
    init.file_replacement = {&file, OpenFile, CloseFile, ReadFile, FileSize};
    auto* player = sceAvPlayerInit(&init);
    Check(player != nullptr, "init failed");
    Check(sceAvPlayerAddSource(player, "/app0/replaced.mp4") == 0, "replaced source failed");
    AvPlayerFrameInfo frame{};
    Check(WaitFor([&] { return sceAvPlayerGetVideoData(player, &frame) != 0; }), "auto start never produced a frame");
    Check(frame.details.video.width == 112 && frame.details.video.height == 64 && IsAllocation(frame.p_data, true), "unexpected auto start frame");
    Check(sceAvPlayerClose(player) == 0, "close failed");
    Check(!file.stream.is_open(), "replaced file left open");
}

void TestHandedOutFramesStayIntact() {
    constexpr int Buffers = 6;
    constexpr int Retained = Buffers - 2;
    AvPlayerInitData init = InitData(nullptr);
    init.num_output_video_framebuffers = Buffers;
    auto* player = sceAvPlayerInit(&init);
    Check(player != nullptr, "init failed");
    Check(sceAvPlayerSetAvSyncMode(player, 1) == 0, "sync mode rejected");
    Check(sceAvPlayerAddSource(player, "/app0/avplayer.mp4") == 0, "add source failed");
    struct Taken {
        const std::uint8_t* luma;
        int index;
    };
    std::deque<Taken> taken;
    for (int round = 0; round < 16; ++round) {
        AvPlayerFrameInfoEx frame{};
        Check(WaitFor([&] { return sceAvPlayerGetVideoDataEx(player, &frame) != 0; }), "no frame for round " + std::to_string(round));
        CheckVideoFrame(frame);
        taken.push_back({static_cast<const std::uint8_t*>(frame.p_data), FrameIndex(frame.timestamp)});
        if (taken.size() > Retained) taken.pop_front();
        std::this_thread::sleep_for(std::chrono::milliseconds(30));
        for (std::size_t k = 0; k < taken.size(); ++k) {
            for (std::size_t other = k + 1; other < taken.size(); ++other) Check(taken[k].luma != taken[other].luma, "one buffer handed out twice among the last " + std::to_string(Retained) + " frames");
            const int index = taken[k].index;
            Check(std::abs(taken[k].luma[30 * Pitch + 10] - LumaFor(index)) <= 6, "frame " + std::to_string(index) + " was overwritten " + std::to_string(taken.size() - 1 - k) + " frames after it was handed out");
        }
    }
    Check(sceAvPlayerClose(player) == 0, "close failed");
}

}

int main() {
    try {
        std::filesystem::create_directories("app0");
        Encoder().Write(MoviePath);
        TestPlayback();
        TestFileReplacementAutoStart();
        TestHandedOutFramesStayIntact();
        std::puts("AvPlayer tests passed");
        return 0;
    } catch (const Skip&) {
        std::puts("libx264 or the AAC encoder is unavailable");
        return SkipTest;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
