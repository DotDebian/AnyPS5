#ifndef CORE_LIBS_PRX_LIBSCEVIDEOOUT_INCLUDE_LEVELMENU_HPP
#define CORE_LIBS_PRX_LIBSCEVIDEOOUT_INCLUDE_LEVELMENU_HPP

#include "SDL.h"
#include "prx/libSceVideoOut/include/LevelMenuModel.hpp"

#include <bitset>
#include <chrono>
#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <vector>

class LevelMenuFont;

class LevelMenu {
public:
    LevelMenu();
    ~LevelMenu();
    LevelMenu(const LevelMenu&) = delete;
    LevelMenu& operator=(const LevelMenu&) = delete;

    bool HandleEvent(const SDL_Event& event, SDL_Window* gameWindow);
    std::optional<std::string> Update(SDL_Window* gameWindow);
    bool IsOpen() const;
    void Close(SDL_Window* gameWindow);

private:
    struct Row {
        bool header;
        std::size_t entry;
    };

    bool loadLevels();
    void open(SDL_Window* gameWindow);
    void handleKey(const SDL_KeyboardEvent& key, SDL_Window* gameWindow);
    void handleText(const char* text);
    void handleMouse(const SDL_Event& event);
    void refilter();
    void moveSelection(long delta);
    void choose();
    std::vector<Row> rows() const;
    int visibleRows(int height) const;
    void draw();

    bool enabled = false;
    std::vector<LevelMenuModel::LevelEntry> entries;
    std::string listPath;
    bool listLoaded = false;
    SDL_Window* window = nullptr;
    std::unique_ptr<LevelMenuFont> font;
    std::string filter;
    std::vector<std::size_t> filtered;
    std::size_t selected = 0;
    long scrollTop = 0;
    bool dirty = false;
    std::optional<std::string> pendingLevel;
    std::chrono::steady_clock::time_point launchAt{};
    std::bitset<SDL_NUM_SCANCODES> swallowKeyUp;
    std::optional<LevelMenuModel::AutoSelect> autoSelect;
    std::chrono::steady_clock::time_point firstUpdate{};
    std::chrono::steady_clock::time_point autoEnterAt{};
    bool autoOpened = false;
    int listTop = 0;
    int lineHeight = 0;
};

#endif
