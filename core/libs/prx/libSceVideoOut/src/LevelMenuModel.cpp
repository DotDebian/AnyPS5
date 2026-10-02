#include "prx/libSceVideoOut/include/LevelMenuModel.hpp"

#include <algorithm>
#include <cctype>
#include <charconv>
#include <cmath>

namespace LevelMenuModel {

namespace {

constexpr std::string_view AutoSelectVariable = "APS5_LEVEL_MENU_AUTOSELECT=";

std::string lower(std::string_view text) {
    std::string result(text);
    std::transform(result.begin(), result.end(), result.begin(), [](unsigned char character) {
        return static_cast<char>(std::tolower(character));
    });
    return result;
}

std::string_view trim(std::string_view text) {
    while (!text.empty() && std::isspace(static_cast<unsigned char>(text.front()))) text.remove_prefix(1);
    while (!text.empty() && std::isspace(static_cast<unsigned char>(text.back()))) text.remove_suffix(1);
    return text;
}

std::string decodeEntities(std::string_view text) {
    static constexpr std::pair<std::string_view, char> entities[] = {
        {"&amp;", '&'}, {"&lt;", '<'}, {"&gt;", '>'}, {"&quot;", '"'}, {"&apos;", '\''}
    };
    std::string result;
    result.reserve(text.size());
    for (std::size_t index = 0; index < text.size();) {
        bool replaced = false;
        if (text[index] == '&') {
            for (const auto& [entity, character] : entities) {
                if (text.substr(index, entity.size()) != entity) continue;
                result.push_back(character);
                index += entity.size();
                replaced = true;
                break;
            }
        }
        if (!replaced) result.push_back(text[index++]);
    }
    return result;
}

std::string childText(std::string_view block, std::string_view tag) {
    const std::string open = "<" + std::string(tag) + ">";
    const std::string close = "</" + std::string(tag) + ">";
    const auto start = block.find(open);
    if (start == std::string_view::npos) return {};
    const auto contentStart = start + open.size();
    const auto end = block.find(close, contentStart);
    if (end == std::string_view::npos) return {};
    return decodeEntities(trim(block.substr(contentStart, end - contentStart)));
}

struct Element {
    std::string_view content;
    std::size_t next;
};

std::optional<Element> nextElement(std::string_view text, std::string_view tag, std::size_t from) {
    const std::string open = "<" + std::string(tag) + ">";
    const std::string close = "</" + std::string(tag) + ">";
    const auto start = text.find(open, from);
    if (start == std::string_view::npos) return std::nullopt;
    const auto contentStart = start + open.size();
    const auto end = text.find(close, contentStart);
    if (end == std::string_view::npos) return std::nullopt;
    return Element{text.substr(contentStart, end - contentStart), end + close.size()};
}

}

std::vector<LevelEntry> ParseLevelList(std::string_view xml) {
    std::vector<LevelEntry> entries;
    for (auto category = nextElement(xml, "Category", 0); category; category = nextElement(xml, "Category", category->next)) {
        const auto body = category->content;
        const auto firstLevel = body.find("<Level>");
        const auto categoryName = childText(body.substr(0, firstLevel), "Name");
        for (auto level = nextElement(body, "Level", 0); level; level = nextElement(body, "Level", level->next)) {
            auto file = childText(level->content, "File");
            if (file.empty()) continue;
            auto name = childText(level->content, "Name");
            if (name.empty()) name = file;
            entries.push_back({categoryName, std::move(name), std::move(file)});
        }
    }
    return entries;
}

void MoveCategoryToEnd(std::vector<LevelEntry>& entries, std::string_view category) {
    std::stable_partition(entries.begin(), entries.end(), [category](const LevelEntry& entry) { return entry.category != category; });
}

std::vector<std::size_t> FilterLevels(const std::vector<LevelEntry>& entries, std::string_view query) {
    std::vector<std::string> tokens;
    const auto lowered = lower(query);
    std::size_t start = 0;
    while (start < lowered.size()) {
        const auto end = std::min(lowered.find(' ', start), lowered.size());
        if (end > start) tokens.push_back(lowered.substr(start, end - start));
        start = end + 1;
    }
    std::vector<std::size_t> result;
    for (std::size_t index = 0; index < entries.size(); ++index) {
        const auto& entry = entries[index];
        const auto haystack = lower(entry.category + " " + entry.name + " " + entry.file);
        const bool matches = std::all_of(tokens.begin(), tokens.end(), [&haystack](const std::string& token) {
            return haystack.find(token) != std::string::npos;
        });
        if (matches) result.push_back(index);
    }
    return result;
}

std::vector<std::string> SplitNulList(std::string_view data) {
    std::vector<std::string> result;
    std::size_t start = 0;
    while (start < data.size()) {
        const auto end = std::min(data.find('\0', start), data.size());
        result.emplace_back(data.substr(start, end - start));
        start = end + 1;
    }
    return result;
}

std::vector<std::string> BuildRelaunchArgs(const std::vector<std::string>& argv, std::string_view level) {
    std::vector<std::string> result;
    for (std::size_t index = 0; index < argv.size(); ++index) {
        if (index > 0 && argv[index] == "-lvl") {
            ++index;
            continue;
        }
        result.push_back(argv[index]);
    }
    result.emplace_back("-lvl");
    result.emplace_back(level);
    return result;
}

std::vector<std::string> BuildRelaunchEnv(const std::vector<std::string>& environment) {
    std::vector<std::string> result;
    for (const auto& variable : environment) {
        if (variable.starts_with(AutoSelectVariable)) continue;
        result.push_back(variable);
    }
    return result;
}

std::optional<AutoSelect> ParseAutoSelect(std::string_view text) {
    const auto colon = text.find(':');
    if (colon == std::string_view::npos) return std::nullopt;
    const auto secondsText = trim(text.substr(0, colon));
    const auto level = trim(text.substr(colon + 1));
    if (level.empty() || std::any_of(level.begin(), level.end(), [](unsigned char character) { return std::isspace(character); })) return std::nullopt;
    double seconds = 0.0;
    const auto [end, error] = std::from_chars(secondsText.data(), secondsText.data() + secondsText.size(), seconds);
    if (error != std::errc{} || end != secondsText.data() + secondsText.size() || !std::isfinite(seconds) || seconds < 0.0) return std::nullopt;
    return AutoSelect{std::chrono::milliseconds(static_cast<long long>(seconds * 1000.0)), std::string(level)};
}

}
