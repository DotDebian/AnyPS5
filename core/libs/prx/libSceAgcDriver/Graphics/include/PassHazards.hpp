#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_PASSHAZARDS_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_PASSHAZARDS_HPP

#ifndef VK_NO_PROTOTYPES
#define VK_NO_PROTOTYPES
#endif
#include <vulkan/vulkan.h>
#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <utility>
#include <vector>

namespace AgcDriver::Graphics {

using GuestRangeList = std::span<const std::pair<std::uint64_t, std::uint64_t>>;

struct PassAccess {
    GuestRangeList reads;
    GuestRangeList writes;
    std::span<const std::pair<VkImage, bool>> images;
    std::span<const std::pair<VkImage, bool>> attachments;
    bool anyReads = false;
    bool anyWrites = false;
    bool gds = false;
};

enum class PassHazard : std::uint8_t { None, ReadAfterWrite, WriteAfterWrite, WriteAfterRead, Image, UnknownRead, UnknownWrite, Gds, Count };
inline constexpr std::array<const char*, static_cast<std::size_t>(PassHazard::Count)> PassHazardNames{"none", "raw", "waw", "war", "image", "unknown-read", "unknown-write", "gds"};

enum class PassBreak : std::uint8_t { Capture, ReadsTarget, GpuIndirect, MeshIndirect, DepthClear, FirstInBatch, OtherWork, Key, PreviousWrote, Hazard, SampleSlots, None, Count = None };
inline constexpr std::array<const char*, static_cast<std::size_t>(PassBreak::Count)> PassBreakNames{"capture", "reads-target", "gpu-indirect", "mesh-indirect", "depth-clear", "first-in-batch", "other-work", "key", "previous-wrote", "hazard", "sample-slots"};

enum class PassBlock : std::uint8_t { None, Lease, Completion, Buffers, Images, Gds, Count };
inline constexpr std::array<const char*, static_cast<std::size_t>(PassBlock::Count)> PassBlockNames{"none", "lease", "completion", "buffers", "images", "gds"};

class GuestRangeSet {
public:
    void Insert(std::uint64_t begin, std::uint64_t end);
    bool Overlaps(std::uint64_t begin, std::uint64_t end) const;
    bool Empty() const { return ranges.empty(); }
    std::size_t Size() const { return ranges.size(); }
    void Clear() { ranges.clear(); }

private:
    std::vector<std::pair<std::uint64_t, std::uint64_t>> ranges;
};

struct QueuedWrite {
    std::uint64_t begin;
    std::uint64_t end;
    bool computed;
    std::uint8_t kind = 0;
};

std::vector<std::size_t> QueuedWriteGroups(std::span<const QueuedWrite> writes);

class PassHazards {
public:
    PassHazard Check(const PassAccess& access, bool samePass) const;
    void Add(const PassAccess& access);
    void BeginPass() { ++pass; }
    void Clear();
    bool Empty() const;

private:
    struct Attachment {
        VkImage image;
        bool written;
        std::uint64_t pass;
    };
    bool overlapsAny(const GuestRangeSet& set, GuestRangeList ranges) const;
    GuestRangeSet reads;
    GuestRangeSet writes;
    std::vector<std::pair<VkImage, bool>> images;
    std::vector<Attachment> attachments;
    std::uint64_t pass = 0;
    bool anyReads = false;
    bool anyWrites = false;
    bool gds = false;
};

}

#endif
