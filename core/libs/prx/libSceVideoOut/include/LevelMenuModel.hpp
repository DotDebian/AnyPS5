#ifndef CORE_LIBS_PRX_LIBSCEVIDEOOUT_INCLUDE_LEVELMENUMODEL_HPP
#define CORE_LIBS_PRX_LIBSCEVIDEOOUT_INCLUDE_LEVELMENUMODEL_HPP

#include <chrono>
#include <cstddef>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace LevelMenuModel {

struct LevelEntry {
    std::string category;
    std::string name;
    std::string file;

    bool operator==(const LevelEntry&) const = default;
};

struct AutoSelect {
    std::chrono::milliseconds delay{};
    std::string level;
};

std::vector<LevelEntry> ParseLevelList(std::string_view xml);
void MoveCategoryToEnd(std::vector<LevelEntry>& entries, std::string_view category);
std::vector<std::size_t> FilterLevels(const std::vector<LevelEntry>& entries, std::string_view query);
std::vector<std::string> SplitNulList(std::string_view data);
std::vector<std::string> BuildRelaunchArgs(const std::vector<std::string>& argv, std::string_view level);
std::vector<std::string> BuildRelaunchEnv(const std::vector<std::string>& environment);
std::optional<AutoSelect> ParseAutoSelect(std::string_view text);

}

#endif
