#include <cstdint>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <vector>
#include "SceTypes.hpp"
#include "prx/libc/include/General.hpp"

#if APS5_HAVE_FFMPEG
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/frame.h>
#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>
}
#endif

// Tech debt: without FFmpeg (APS5_HAVE_FFMPEG unset) decoders return black NV12 pictures that
// carry the access unit timestamps, so players still advance through a movie.

namespace {

constexpr int SCE_VIDEODEC2_ERROR_ARGUMENT_POINTER = static_cast<int>(0x811d0103u);
constexpr int SCE_VIDEODEC2_ERROR_DECODER_INSTANCE = static_cast<int>(0x811d0106u);
constexpr std::uint64_t WorkMemoryBytes = 1u << 20u;
constexpr std::uint32_t PitchAlignment = 256;
constexpr std::uint32_t HeightAlignment = 32;

struct ComputeMemoryInfo {
    std::uint64_t thisSize;
    std::uint64_t cpuGpuMemorySize;
    void* cpuGpuMemory;
};

struct ComputeConfigInfo {
    std::uint64_t thisSize;
    std::uint16_t computePipeId;
    std::uint16_t computeQueueId;
    bool checkMemoryType;
};

struct DecoderConfigInfo {
    std::uint64_t thisSize;
    std::uint32_t resourceType;
    std::uint32_t codecType;
    std::uint32_t profile;
    std::uint32_t maxLevel;
    std::int32_t maxFrameWidth;
    std::int32_t maxFrameHeight;
    std::int32_t maxDpbFrameCount;
    std::uint32_t decodePipelineDepth;
};

struct DecoderMemoryInfo {
    std::uint64_t thisSize;
    std::uint64_t cpuMemorySize;
    void* cpuMemory;
    std::uint64_t gpuMemorySize;
    void* gpuMemory;
    std::uint64_t cpuGpuMemorySize;
    void* cpuGpuMemory;
    std::uint64_t maxFrameBufferSize;
    std::uint32_t frameBufferAlignment;
};

struct InputData {
    std::uint64_t thisSize;
    const std::uint8_t* auData;
    std::uint64_t auSize;
    std::uint64_t ptsData;
    std::uint64_t dtsData;
    std::uint64_t attachedData;
};

struct FrameBuffer {
    std::uint64_t thisSize;
    void* frameBuffer;
    std::uint64_t frameBufferSize;
    bool isAccepted;
};

struct OutputInfo {
    std::uint64_t thisSize;
    bool isValid;
    bool isErrorFrame;
    std::uint8_t pictureCount;
    std::uint32_t codecType;
    std::uint32_t frameWidth;
    std::uint32_t framePitch;
    std::uint32_t frameHeight;
    void* frameBuffer;
    std::uint64_t frameBufferSize;
    std::uint32_t frameFormat;
    std::uint32_t framePitchInBytes;
};

struct AvcPictureInfo {
    std::uint64_t thisSize;
    bool isValid;
    std::uint64_t ptsData;
    std::uint64_t dtsData;
    std::uint64_t attachedData;
    std::uint8_t idrPictureFlag;
    std::uint8_t profileIdc;
    std::uint8_t levelIdc;
    std::uint32_t picWidthInLumaSamples;
    std::uint32_t picHeightInLumaSamples;
};

struct Picture {
    std::uint64_t pts = 0;
    std::uint64_t dts = 0;
    std::uint64_t attached = 0;
    bool idr = false;
};

#if APS5_HAVE_FFMPEG
struct FrameDeleter {
    void operator()(AVFrame* frame) const { av_frame_free(&frame); }
};
using FramePointer = std::unique_ptr<AVFrame, FrameDeleter>;
#endif

struct Decoder {
    std::uint32_t codec = 0;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint32_t profile = 0;
    std::uint32_t level = 0;
    std::map<const void*, Picture> pictures;
    std::uint64_t decoded = 0;
    std::mutex mutex;
#if APS5_HAVE_FFMPEG
    AVCodecContext* context = nullptr;
    SwsContext* scaler = nullptr;
    std::map<std::int64_t, Picture> inputs;
    std::int64_t nextKey = 0;
    std::deque<FramePointer> ready;
    std::vector<std::uint8_t> annexB;
    bool flushing = false;

    ~Decoder() {
        sws_freeContext(scaler);
        avcodec_free_context(&context);
    }
#endif
};

std::mutex lock;
std::map<std::uint64_t, std::shared_ptr<Decoder>> decoders;
std::uint64_t nextDecoder = 1;
std::uint64_t nextQueue = 1;

bool Trace() {
    static const bool trace = std::getenv("APS5_TRACE_VIDEODEC") != nullptr;
    return trace;
}

std::uint32_t AlignUp(std::uint32_t value, std::uint32_t alignment) {
    return (value + alignment - 1) / alignment * alignment;
}

std::uint64_t ChromaOffset(std::uint32_t width, std::uint32_t height) {
    return static_cast<std::uint64_t>(AlignUp(width, PitchAlignment)) * height;
}

std::uint64_t LumaBytes(std::uint32_t width, std::uint32_t height) {
    return static_cast<std::uint64_t>(AlignUp(width, PitchAlignment)) * AlignUp(height, HeightAlignment);
}

template <typename T>
bool Fits(const T* object, std::size_t end) {
    return object->thisSize >= end;
}

class BitReader {
public:
    BitReader(const std::uint8_t* data, std::size_t size) : data(data), size(size) {}

    std::uint32_t Bit() {
        if (position >= size * 8) { failed = true; return 0; }
        const auto bit = (data[position / 8] >> (7 - position % 8)) & 1u;
        ++position;
        return bit;
    }

    std::uint32_t Bits(unsigned count) {
        std::uint32_t value = 0;
        for (unsigned index = 0; index < count; ++index) value = (value << 1u) | Bit();
        return value;
    }

    std::uint32_t Ue() {
        unsigned zeros = 0;
        while (Bit() == 0 && !failed && zeros < 32) ++zeros;
        if (zeros >= 32) { failed = true; return 0; }
        return ((1u << zeros) - 1u) + Bits(zeros);
    }

    std::int32_t Se() {
        const auto value = Ue();
        return (value & 1u) != 0 ? static_cast<std::int32_t>((value + 1) / 2) : -static_cast<std::int32_t>(value / 2);
    }

    bool failed = false;

private:
    const std::uint8_t* data;
    std::size_t size;
    std::size_t position = 0;
};

void SkipScalingList(BitReader& reader, int count) {
    int last = 8;
    int next = 8;
    for (int index = 0; index < count; ++index) {
        if (next != 0) next = (last + reader.Se() + 256) % 256;
        last = next == 0 ? last : next;
    }
}

bool ParseSps(const std::uint8_t* nal, std::size_t size, Decoder& decoder) {
    std::uint8_t rbsp[256];
    std::size_t length = 0;
    for (std::size_t index = 1; index < size && length < sizeof(rbsp); ++index) {
        if (index >= 3 && nal[index] == 3 && nal[index - 1] == 0 && nal[index - 2] == 0) continue;
        rbsp[length++] = nal[index];
    }
    BitReader reader(rbsp, length);
    const auto profile = reader.Bits(8);
    reader.Bits(8);
    const auto level = reader.Bits(8);
    reader.Ue();
    std::uint32_t chromaFormat = 1;
    if (profile == 100 || profile == 110 || profile == 122 || profile == 244 || profile == 44 || profile == 83 || profile == 86 || profile == 118 || profile == 128) {
        chromaFormat = reader.Ue();
        if (chromaFormat == 3) reader.Bit();
        reader.Ue();
        reader.Ue();
        reader.Bit();
        if (reader.Bit() != 0) {
            for (int index = 0; index < (chromaFormat != 3 ? 8 : 12); ++index) {
                if (reader.Bit() != 0) SkipScalingList(reader, index < 6 ? 16 : 64);
            }
        }
    }
    reader.Ue();
    const auto pocType = reader.Ue();
    if (pocType == 0) {
        reader.Ue();
    } else if (pocType == 1) {
        reader.Bit();
        reader.Se();
        reader.Se();
        const auto cycle = reader.Ue();
        for (std::uint32_t index = 0; index < cycle && !reader.failed; ++index) reader.Se();
    }
    reader.Ue();
    reader.Bit();
    const auto widthMbs = reader.Ue() + 1;
    const auto heightMapUnits = reader.Ue() + 1;
    const auto frameMbsOnly = reader.Bit();
    if (frameMbsOnly == 0) reader.Bit();
    reader.Bit();
    std::uint32_t cropLeft = 0, cropRight = 0, cropTop = 0, cropBottom = 0;
    if (reader.Bit() != 0) {
        cropLeft = reader.Ue();
        cropRight = reader.Ue();
        cropTop = reader.Ue();
        cropBottom = reader.Ue();
    }
    if (reader.failed) return false;
    const std::uint32_t cropX = chromaFormat == 0 || chromaFormat == 3 ? 1 : 2;
    const std::uint32_t cropY = (chromaFormat == 1 ? 2 : 1) * (2 - frameMbsOnly);
    decoder.width = widthMbs * 16 - (cropLeft + cropRight) * cropX;
    decoder.height = (2 - frameMbsOnly) * heightMapUnits * 16 - (cropTop + cropBottom) * cropY;
    decoder.profile = profile;
    decoder.level = level;
    return true;
}

void ScanAccessUnit(const std::uint8_t* data, std::size_t size, Decoder& decoder) {
    for (std::size_t index = 0; index + 4 < size; ++index) {
        if (data[index] == 0 && data[index + 1] == 0 && data[index + 2] == 1 && (data[index + 3] & 0x1fu) == 7) {
            ParseSps(data + index + 3, size - index - 3, decoder);
            return;
        }
    }
    for (std::size_t index = 0; index + 5 <= size;) {
        const std::size_t length = (static_cast<std::size_t>(data[index]) << 24u) | (static_cast<std::size_t>(data[index + 1]) << 16u) | (static_cast<std::size_t>(data[index + 2]) << 8u) | data[index + 3];
        if (length == 0 || length > size - index - 4) return;
        if ((data[index + 4] & 0x1fu) == 7) {
            ParseSps(data + index + 4, length, decoder);
            return;
        }
        index += 4 + length;
    }
}

void FillBlack(void* buffer, std::uint64_t bytes, std::uint32_t width, std::uint32_t height) {
    const auto luma = ChromaOffset(width, height);
    const auto lumaBytes = luma < bytes ? luma : bytes;
    std::memset(buffer, 0x10, lumaBytes);
    if (bytes > lumaBytes) std::memset(static_cast<std::uint8_t*>(buffer) + lumaBytes, 0x80, bytes - lumaBytes);
}

void Emit(Decoder& decoder, FrameBuffer* frame, OutputInfo* output, const Picture* picture) {
    output->isValid = picture != nullptr;
    output->isErrorFrame = false;
    output->pictureCount = picture != nullptr ? 1 : 0;
    output->codecType = decoder.codec;
    if (!picture) return;
    const auto width = decoder.width != 0 ? decoder.width : 1920u;
    const auto height = decoder.height != 0 ? decoder.height : 1080u;
    const auto pitch = AlignUp(width, PitchAlignment);
    void* target = frame ? frame->frameBuffer : nullptr;
    const auto bytes = frame ? frame->frameBufferSize : 0;
    if (target && bytes != 0) FillBlack(target, bytes, width, height);
    if (frame) frame->isAccepted = true;
    output->frameWidth = width;
    output->framePitch = pitch;
    output->frameHeight = height;
    output->frameBuffer = target;
    output->frameBufferSize = bytes;
    if (Fits(output, offsetof(OutputInfo, framePitchInBytes) + 4)) {
        output->frameFormat = 0;
        output->framePitchInBytes = pitch;
    }
    decoder.pictures[target] = *picture;
}

std::shared_ptr<Decoder> FindDecoder(std::uint64_t handle) {
    std::lock_guard guard(lock);
    const auto found = decoders.find(handle);
    return found != decoders.end() ? found->second : nullptr;
}

#if APS5_HAVE_FFMPEG

bool OpenCodec(Decoder& decoder) {
    const AVCodec* codec = avcodec_find_decoder(AV_CODEC_ID_H264);
    if (codec == nullptr) return false;
    decoder.context = avcodec_alloc_context3(codec);
    if (decoder.context == nullptr) return false;
    decoder.context->thread_count = 0;
    decoder.context->thread_type = FF_THREAD_SLICE;
    if (avcodec_open2(decoder.context, codec, nullptr) < 0) {
        avcodec_free_context(&decoder.context);
        return false;
    }
    return true;
}

const std::vector<std::uint8_t>* ToAnnexB(Decoder& decoder, const std::uint8_t* data, std::size_t size) {
    if (size >= 4 && data[0] == 0 && data[1] == 0 && (data[2] == 1 || (data[2] == 0 && data[3] == 1))) {
        decoder.annexB.assign(data, data + size);
        return &decoder.annexB;
    }
    decoder.annexB.clear();
    for (std::size_t index = 0; index + 4 <= size;) {
        const std::size_t length = (static_cast<std::size_t>(data[index]) << 24u) | (static_cast<std::size_t>(data[index + 1]) << 16u) | (static_cast<std::size_t>(data[index + 2]) << 8u) | data[index + 3];
        if (length == 0 || length > size - index - 4) return nullptr;
        static constexpr std::uint8_t startCode[4] = {0, 0, 0, 1};
        decoder.annexB.insert(decoder.annexB.end(), startCode, startCode + 4);
        decoder.annexB.insert(decoder.annexB.end(), data + index + 4, data + index + 4 + length);
        index += 4 + length;
    }
    return decoder.annexB.empty() ? nullptr : &decoder.annexB;
}

void Receive(Decoder& decoder) {
    for (;;) {
        FramePointer frame(av_frame_alloc());
        if (!frame || avcodec_receive_frame(decoder.context, frame.get()) < 0) return;
        decoder.ready.push_back(std::move(frame));
    }
}

bool Send(Decoder& decoder, const AVPacket* packet) {
    const int result = avcodec_send_packet(decoder.context, packet);
    Receive(decoder);
    return result >= 0 || result == AVERROR(EAGAIN) || result == AVERROR_EOF;
}

bool WriteNv12(Decoder& decoder, const AVFrame& frame, std::uint8_t* target, std::uint64_t bytes) {
    const auto width = static_cast<std::uint32_t>(frame.width);
    const auto height = static_cast<std::uint32_t>(frame.height);
    const auto pitch = AlignUp(width, PitchAlignment);
    const auto luma = ChromaOffset(width, height);
    if (luma + static_cast<std::uint64_t>(pitch) * (height / 2) > bytes) return false;
    std::uint8_t* planes[2] = {target, target + luma};
    const int strides[2] = {static_cast<int>(pitch), static_cast<int>(pitch)};
    const auto format = static_cast<AVPixelFormat>(frame.format);
    if (format == AV_PIX_FMT_YUV420P || format == AV_PIX_FMT_YUVJ420P) {
        for (std::uint32_t row = 0; row < height; ++row) std::memcpy(planes[0] + static_cast<std::size_t>(row) * pitch, frame.data[0] + static_cast<std::ptrdiff_t>(row) * frame.linesize[0], width);
        for (std::uint32_t row = 0; row < height / 2; ++row) {
            const auto* u = frame.data[1] + static_cast<std::ptrdiff_t>(row) * frame.linesize[1];
            const auto* v = frame.data[2] + static_cast<std::ptrdiff_t>(row) * frame.linesize[2];
            auto* uv = planes[1] + static_cast<std::size_t>(row) * pitch;
            for (std::uint32_t column = 0; column < width / 2; ++column) {
                uv[column * 2] = u[column];
                uv[column * 2 + 1] = v[column];
            }
        }
        return true;
    }
    decoder.scaler = sws_getCachedContext(decoder.scaler, frame.width, frame.height, format, frame.width, frame.height, AV_PIX_FMT_NV12, SWS_BILINEAR, nullptr, nullptr, nullptr);
    if (decoder.scaler == nullptr) return false;
    return sws_scale(decoder.scaler, frame.data, frame.linesize, 0, frame.height, planes, strides) == frame.height;
}

void EmitDecoded(Decoder& decoder, FrameBuffer* frame, OutputInfo* output) {
    output->isValid = false;
    output->isErrorFrame = false;
    output->pictureCount = 0;
    output->codecType = decoder.codec;
    if (frame) frame->isAccepted = false;
    if (decoder.ready.empty() || !frame || !frame->frameBuffer) return;
    const auto picture = std::move(decoder.ready.front());
    decoder.ready.pop_front();
    Picture timestamps{};
    const auto key = picture->pts != AV_NOPTS_VALUE ? picture->pts : picture->best_effort_timestamp;
    const auto found = decoder.inputs.find(key);
    if (found != decoder.inputs.end()) {
        timestamps = found->second;
        decoder.inputs.erase(found);
    }
    decoder.inputs.erase(decoder.inputs.begin(), decoder.inputs.lower_bound(key - 64));
    timestamps.idr = (picture->flags & AV_FRAME_FLAG_KEY) != 0;
    decoder.width = static_cast<std::uint32_t>(picture->width);
    decoder.height = static_cast<std::uint32_t>(picture->height);
    if (decoder.context->profile > 0) decoder.profile = static_cast<std::uint32_t>(decoder.context->profile) & 0xffu;
    if (decoder.context->level > 0) decoder.level = static_cast<std::uint32_t>(decoder.context->level);
    const bool written = WriteNv12(decoder, *picture, static_cast<std::uint8_t*>(frame->frameBuffer), frame->frameBufferSize);
    const auto pitch = AlignUp(decoder.width, PitchAlignment);
    frame->isAccepted = true;
    output->isValid = true;
    output->isErrorFrame = !written;
    output->pictureCount = 1;
    output->frameWidth = decoder.width;
    output->framePitch = pitch;
    output->frameHeight = decoder.height;
    output->frameBuffer = frame->frameBuffer;
    output->frameBufferSize = frame->frameBufferSize;
    if (Fits(output, offsetof(OutputInfo, framePitchInBytes) + 4)) {
        output->frameFormat = 0;
        output->framePitchInBytes = pitch;
    }
    decoder.pictures[frame->frameBuffer] = timestamps;
}

#endif

}

extern "C" {

int APS5_VABI sceVideodec2QueryComputeMemoryInfo_nid_postfix(ComputeMemoryInfo* info) {
    if (!info) return SCE_VIDEODEC2_ERROR_ARGUMENT_POINTER;
    if (Trace()) std::fprintf(stderr, "[videodec2] QueryComputeMemoryInfo size=%llu\n", static_cast<unsigned long long>(info->thisSize));
    info->cpuGpuMemorySize = WorkMemoryBytes;
    return 0;
}

int APS5_VABI sceVideodec2AllocateComputeQueue_nid_postfix(const ComputeConfigInfo* config, const ComputeMemoryInfo* memory, std::uint64_t* queue) {
    if (!config || !memory || !queue) return SCE_VIDEODEC2_ERROR_ARGUMENT_POINTER;
    if (Trace()) std::fprintf(stderr, "[videodec2] AllocateComputeQueue config=%llu memory=%llu\n", static_cast<unsigned long long>(config->thisSize), static_cast<unsigned long long>(memory->thisSize));
    std::lock_guard guard(lock);
    *queue = nextQueue++;
    return 0;
}

int APS5_VABI sceVideodec2ReleaseComputeQueue_nid_postfix(std::uint64_t queue) {
    (void)queue;
    return 0;
}

int APS5_VABI sceVideodec2QueryDecoderMemoryInfo_nid_postfix(const DecoderConfigInfo* config, DecoderMemoryInfo* memory) {
    if (!config || !memory) return SCE_VIDEODEC2_ERROR_ARGUMENT_POINTER;
    if (Trace()) std::fprintf(stderr, "[videodec2] QueryDecoderMemoryInfo config=%llu memory=%llu codec=%u profile=%u %dx%d dpb=%d\n", static_cast<unsigned long long>(config->thisSize), static_cast<unsigned long long>(memory->thisSize), config->codecType, config->profile, config->maxFrameWidth, config->maxFrameHeight, config->maxDpbFrameCount);
    memory->cpuMemorySize = WorkMemoryBytes;
    memory->gpuMemorySize = WorkMemoryBytes;
    memory->cpuGpuMemorySize = WorkMemoryBytes;
    const auto width = config->maxFrameWidth > 0 ? static_cast<std::uint32_t>(config->maxFrameWidth) : 1920u;
    const auto height = config->maxFrameHeight > 0 ? static_cast<std::uint32_t>(config->maxFrameHeight) : 1080u;
    memory->maxFrameBufferSize = LumaBytes(width, height) * 3 / 2;
    memory->frameBufferAlignment = PitchAlignment;
    return 0;
}

int APS5_VABI sceVideodec2CreateDecoder_nid_postfix(const DecoderConfigInfo* config, const DecoderMemoryInfo* memory, std::uint64_t* handle) {
    if (!config || !memory || !handle) return SCE_VIDEODEC2_ERROR_ARGUMENT_POINTER;
    auto decoder = std::make_shared<Decoder>();
    decoder->codec = config->codecType;
#if APS5_HAVE_FFMPEG
    const bool decoding = OpenCodec(*decoder);
#else
    const bool decoding = false;
#endif
    std::lock_guard guard(lock);
    *handle = nextDecoder++;
    decoders.emplace(*handle, decoder);
    std::fprintf(stderr, "[videodec2] decoder %llu: codec=%u max %dx%d (%s)\n", static_cast<unsigned long long>(*handle), config->codecType, config->maxFrameWidth, config->maxFrameHeight, decoding ? "H.264 through FFmpeg" : "black frames, no H.264 decoder");
    return 0;
}

int APS5_VABI sceVideodec2DeleteDecoder_nid_postfix(std::uint64_t handle) {
    std::lock_guard guard(lock);
    return decoders.erase(handle) != 0 ? 0 : SCE_VIDEODEC2_ERROR_DECODER_INSTANCE;
}

int APS5_VABI sceVideodec2Decode_nid_postfix(std::uint64_t handle, const InputData* input, FrameBuffer* frame, OutputInfo* output) {
    if (!input || !output) return SCE_VIDEODEC2_ERROR_ARGUMENT_POINTER;
    const auto found = FindDecoder(handle);
    if (!found) return SCE_VIDEODEC2_ERROR_DECODER_INSTANCE;
    auto& decoder = *found;
    std::lock_guard guard(decoder.mutex);
    if (input->auData && input->auSize != 0 && decoder.width == 0) ScanAccessUnit(input->auData, static_cast<std::size_t>(input->auSize), decoder);
    if (Trace() && decoder.decoded < 3) std::fprintf(stderr, "[videodec2] Decode input=%llu frame=%llu output=%llu au=%llu pts=%llu size=%ux%u fb=%p/%llu\n", static_cast<unsigned long long>(input->thisSize), static_cast<unsigned long long>(frame ? frame->thisSize : 0), static_cast<unsigned long long>(output->thisSize), static_cast<unsigned long long>(input->auSize), static_cast<unsigned long long>(input->ptsData), decoder.width, decoder.height, frame ? frame->frameBuffer : nullptr, static_cast<unsigned long long>(frame ? frame->frameBufferSize : 0));
    ++decoder.decoded;
    const Picture picture{input->ptsData, input->dtsData, input->attachedData};
#if APS5_HAVE_FFMPEG
    if (decoder.context != nullptr) {
        if (decoder.flushing) {
            avcodec_flush_buffers(decoder.context);
            decoder.flushing = false;
        }
        const auto* unit = input->auData && input->auSize != 0 ? ToAnnexB(decoder, input->auData, static_cast<std::size_t>(input->auSize)) : nullptr;
        if (unit != nullptr) {
            AVPacket* packet = av_packet_alloc();
            if (!packet) return SCE_VIDEODEC2_ERROR_ARGUMENT_POINTER;
            packet->data = const_cast<std::uint8_t*>(unit->data());
            packet->size = static_cast<int>(unit->size());
            packet->pts = decoder.nextKey;
            decoder.inputs.emplace(decoder.nextKey++, picture);
            const bool sent = Send(decoder, packet);
            av_packet_free(&packet);
            if (!sent && Trace()) std::fprintf(stderr, "[videodec2] decoder %llu rejected access unit %llu\n", static_cast<unsigned long long>(handle), static_cast<unsigned long long>(decoder.decoded));
        }
        EmitDecoded(decoder, frame, output);
        return 0;
    }
#endif
    Emit(decoder, frame, output, &picture);
    return 0;
}

int APS5_VABI sceVideodec2Flush_nid_postfix(std::uint64_t handle, FrameBuffer* frame, OutputInfo* output) {
    if (!output) return SCE_VIDEODEC2_ERROR_ARGUMENT_POINTER;
    const auto found = FindDecoder(handle);
    if (!found) return SCE_VIDEODEC2_ERROR_DECODER_INSTANCE;
    auto& decoder = *found;
    std::lock_guard guard(decoder.mutex);
#if APS5_HAVE_FFMPEG
    if (decoder.context != nullptr) {
        if (!decoder.flushing) {
            decoder.flushing = true;
            Send(decoder, nullptr);
        }
        EmitDecoded(decoder, frame, output);
        return 0;
    }
#endif
    if (frame) frame->isAccepted = false;
    Emit(decoder, frame, output, nullptr);
    return 0;
}

int APS5_VABI sceVideodec2Reset_nid_postfix(std::uint64_t handle) {
    const auto found = FindDecoder(handle);
    if (!found) return SCE_VIDEODEC2_ERROR_DECODER_INSTANCE;
    auto& decoder = *found;
    std::lock_guard guard(decoder.mutex);
    decoder.pictures.clear();
#if APS5_HAVE_FFMPEG
    if (decoder.context != nullptr) avcodec_flush_buffers(decoder.context);
    decoder.ready.clear();
    decoder.inputs.clear();
    decoder.flushing = false;
#endif
    return 0;
}

int APS5_VABI sceVideodec2GetPictureInfo_nid_postfix(const OutputInfo* output, AvcPictureInfo* first, void* second) {
    (void)second;
    if (!output || !first) return SCE_VIDEODEC2_ERROR_ARGUMENT_POINTER;
    std::vector<std::shared_ptr<Decoder>> all;
    {
        std::lock_guard guard(lock);
        for (const auto& [handle, decoder] : decoders) all.push_back(decoder);
    }
    for (const auto& decoder : all) {
        std::lock_guard guard(decoder->mutex);
        const auto found = decoder->pictures.find(output->frameBuffer);
        if (found == decoder->pictures.end()) continue;
        const auto& picture = found->second;
        first->isValid = true;
        first->ptsData = picture.pts;
        first->dtsData = picture.dts;
        first->attachedData = picture.attached;
        if (Fits(first, offsetof(AvcPictureInfo, picHeightInLumaSamples) + 4)) {
#if APS5_HAVE_FFMPEG
            first->idrPictureFlag = decoder->context != nullptr ? (picture.idr ? 1 : 0) : 1;
#else
            first->idrPictureFlag = 1;
#endif
            first->profileIdc = static_cast<std::uint8_t>(decoder->profile);
            first->levelIdc = static_cast<std::uint8_t>(decoder->level);
            first->picWidthInLumaSamples = output->frameWidth;
            first->picHeightInLumaSamples = output->frameHeight;
        }
        return 0;
    }
    first->isValid = false;
    return 0;
}

}
