#include "prx/libSceVideoOut/include/LevelMenu.hpp"
#include "prx/libc/include/General.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string_view>

#define STBTT_STATIC
#define STB_TRUETYPE_IMPLEMENTATION
#include "stb_truetype.h"
#include "stb_easy_font.h"

namespace {

constexpr int FontPixelHeight = 18;
constexpr int Padding = 10;
constexpr auto LaunchDelay = std::chrono::milliseconds(250);
constexpr auto AutoEnterDelay = std::chrono::milliseconds(1500);
constexpr const char* LevelListGuestPath = "/app0/data/prein/product_levels.xml";

constexpr SDL_Color Background{24, 26, 32, 255};
constexpr SDL_Color FilterBackground{42, 46, 58, 255};
constexpr SDL_Color SelectionBackground{52, 88, 150, 255};
constexpr SDL_Color TextColor{224, 224, 228, 255};
constexpr SDL_Color DimColor{132, 136, 150, 255};
constexpr SDL_Color AccentColor{120, 182, 255, 255};

constexpr std::array<const char*, 9> FontCandidates{
    "/usr/share/fonts/TTF/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/noto/NotoSans-Regular.ttf",
    "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
    "/usr/share/fonts/google-noto/NotoSans-Regular.ttf",
    "/usr/share/fonts/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
};

bool envFlag(const char* name) {
    const char* value = std::getenv(name);
    return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
}

std::string readWholeFile(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) return {};
    return std::string((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
}

void fillRect(SDL_Surface* surface, int x, int y, int width, int height, SDL_Color color) {
    const SDL_Rect rect{x, y, width, height};
    SDL_FillRect(surface, &rect, SDL_MapRGB(surface->format, color.r, color.g, color.b));
}

void blendPixel(SDL_Surface* surface, int x, int y, SDL_Color color, unsigned alpha) {
    if (x < 0 || y < 0 || x >= surface->w || y >= surface->h || alpha == 0) return;
    if (surface->format->BytesPerPixel != 4) {
        if (alpha >= 128) fillRect(surface, x, y, 1, 1, color);
        return;
    }
    auto* pixel = reinterpret_cast<Uint32*>(static_cast<Uint8*>(surface->pixels) + y * surface->pitch) + x;
    Uint8 red = 0;
    Uint8 green = 0;
    Uint8 blue = 0;
    SDL_GetRGB(*pixel, surface->format, &red, &green, &blue);
    const auto mix = [alpha](Uint8 from, Uint8 to) { return static_cast<Uint8>((from * (255u - alpha) + to * alpha + 127u) / 255u); };
    *pixel = SDL_MapRGB(surface->format, mix(red, color.r), mix(green, color.g), mix(blue, color.b));
}

}

class LevelMenuFont {
public:
    static std::unique_ptr<LevelMenuFont> Load(int pixelHeight) {
        auto font = std::unique_ptr<LevelMenuFont>(new LevelMenuFont());
        std::vector<std::string> paths;
        if (const char* configured = std::getenv("APS5_LEVEL_MENU_FONT"); configured != nullptr && configured[0] != '\0') paths.emplace_back(configured);
        paths.insert(paths.end(), FontCandidates.begin(), FontCandidates.end());
        for (const auto& path : paths) {
            const auto bytes = readWholeFile(path);
            if (bytes.empty()) continue;
            font->data.assign(bytes.begin(), bytes.end());
            const int offset = stbtt_GetFontOffsetForIndex(font->data.data(), 0);
            if (offset < 0 || stbtt_InitFont(&font->info, font->data.data(), offset) == 0) continue;
            font->truetype = true;
            font->scale = stbtt_ScaleForPixelHeight(&font->info, static_cast<float>(pixelHeight));
            int ascent = 0;
            int descent = 0;
            int gap = 0;
            stbtt_GetFontVMetrics(&font->info, &ascent, &descent, &gap);
            font->ascent = static_cast<int>(std::lround(ascent * font->scale));
            font->lineHeight = static_cast<int>(std::lround((ascent - descent + gap) * font->scale)) + 4;
            std::fprintf(stderr, "[levelmenu] font: %s\n", path.c_str());
            return font;
        }
        font->data.clear();
        font->lineHeight = 12 * EasyFontScale;
        std::fprintf(stderr, "[levelmenu] no TrueType font found (set APS5_LEVEL_MENU_FONT), using the built-in bitmap font\n");
        return font;
    }

    int LineHeight() const {
        return lineHeight;
    }

    int Width(std::string_view text) {
        if (!truetype) {
            std::string copy(text);
            return stb_easy_font_width(copy.data()) * EasyFontScale;
        }
        float width = 0.0f;
        for (const char character : text) width += glyph(character).advance;
        return static_cast<int>(std::ceil(width));
    }

    void Draw(SDL_Surface* surface, int x, int y, std::string_view text, SDL_Color color, int right) {
        if (!truetype) {
            drawEasy(surface, x, y, text, color, right);
            return;
        }
        float penX = static_cast<float>(x);
        const int baseline = y + ascent + 2;
        for (const char character : text) {
            const auto& shape = glyph(character);
            const int left = static_cast<int>(std::lround(penX)) + shape.xoff;
            if (left + shape.width > right) break;
            for (int row = 0; row < shape.height; ++row) {
                for (int column = 0; column < shape.width; ++column) {
                    blendPixel(surface, left + column, baseline + shape.yoff + row, color, shape.alpha[static_cast<std::size_t>(row * shape.width + column)]);
                }
            }
            penX += shape.advance;
        }
    }

private:
    static constexpr int EasyFontScale = 2;

    struct Glyph {
        int width = 0;
        int height = 0;
        int xoff = 0;
        int yoff = 0;
        float advance = 0.0f;
        std::vector<unsigned char> alpha;
    };

    const Glyph& glyph(char character) {
        const int code = static_cast<unsigned char>(character) >= 32 && static_cast<unsigned char>(character) < 127 ? character : '?';
        auto& slot = cache[static_cast<std::size_t>(code)];
        if (slot) return *slot;
        Glyph result;
        int advance = 0;
        int bearing = 0;
        stbtt_GetCodepointHMetrics(&info, code, &advance, &bearing);
        result.advance = static_cast<float>(advance) * scale;
        unsigned char* bitmap = stbtt_GetCodepointBitmap(&info, 0.0f, scale, code, &result.width, &result.height, &result.xoff, &result.yoff);
        if (bitmap != nullptr) {
            result.alpha.assign(bitmap, bitmap + static_cast<std::size_t>(result.width) * static_cast<std::size_t>(result.height));
            stbtt_FreeBitmap(bitmap, nullptr);
        } else {
            result.width = 0;
            result.height = 0;
        }
        slot = std::move(result);
        return *slot;
    }

    void drawEasy(SDL_Surface* surface, int x, int y, std::string_view text, SDL_Color color, int right) {
        struct Vertex {
            float x;
            float y;
            float z;
            unsigned char color[4];
        };
        std::string copy(text);
        std::vector<Vertex> vertices(copy.size() * 70 + 4);
        unsigned char rgba[4]{color.r, color.g, color.b, 255};
        const int quads = stb_easy_font_print(0.0f, 0.0f, copy.data(), rgba, vertices.data(), static_cast<int>(vertices.size() * sizeof(Vertex)));
        for (int quad = 0; quad < quads; ++quad) {
            const auto& topLeft = vertices[static_cast<std::size_t>(quad) * 4];
            const auto& bottomRight = vertices[static_cast<std::size_t>(quad) * 4 + 2];
            const int left = x + static_cast<int>(topLeft.x) * EasyFontScale;
            if (left >= right) continue;
            fillRect(surface, left, y + 4 + static_cast<int>(topLeft.y) * EasyFontScale,
                static_cast<int>(bottomRight.x - topLeft.x) * EasyFontScale, static_cast<int>(bottomRight.y - topLeft.y) * EasyFontScale, color);
        }
    }

    LevelMenuFont() = default;

    std::vector<unsigned char> data;
    stbtt_fontinfo info{};
    bool truetype = false;
    float scale = 1.0f;
    int ascent = 0;
    int lineHeight = 0;
    std::array<std::optional<Glyph>, 128> cache;
};

LevelMenu::LevelMenu() {
    enabled = envFlag("APS5_LEVEL_MENU");
    if (!enabled) return;
    std::fprintf(stderr, "[levelmenu] enabled: F2 opens the level select\n");
    if (const char* text = std::getenv("APS5_LEVEL_MENU_AUTOSELECT"); text != nullptr && text[0] != '\0') {
        autoSelect = LevelMenuModel::ParseAutoSelect(text);
        if (autoSelect) std::fprintf(stderr, "[levelmenu] auto-select %s after %lld ms\n", autoSelect->level.c_str(), static_cast<long long>(autoSelect->delay.count()));
        else std::fprintf(stderr, "[levelmenu] ignoring APS5_LEVEL_MENU_AUTOSELECT='%s' (expected <seconds>:<level file>)\n", text);
    }
}

LevelMenu::~LevelMenu() {
    if (window != nullptr) SDL_DestroyWindow(window);
}

bool LevelMenu::IsOpen() const {
    return window != nullptr;
}

bool LevelMenu::loadLevels() {
    if (listLoaded) return !entries.empty();
    listLoaded = true;
    listPath = ResolvePath_nid_no_patch(LevelListGuestPath).string();
    entries = LevelMenuModel::ParseLevelList(readWholeFile(listPath));
    LevelMenuModel::MoveCategoryToEnd(entries, "Meta");
    if (entries.empty()) std::fprintf(stderr, "[levelmenu] no levels in %s, the menu stays closed\n", listPath.c_str());
    return !entries.empty();
}

void LevelMenu::open(SDL_Window* gameWindow) {
    if (window != nullptr || !loadLevels()) return;
    if (!font) font = LevelMenuFont::Load(FontPixelHeight);
    lineHeight = font->LineHeight();
    int display = gameWindow != nullptr ? SDL_GetWindowDisplayIndex(gameWindow) : 0;
    if (display < 0) display = 0;
    SDL_Rect usable{0, 0, 1280, 720};
    SDL_GetDisplayUsableBounds(display, &usable);
    const int width = std::min(680, std::max(320, usable.w - 40));
    const int height = std::min(lineHeight * 32 + 4 * Padding, std::max(240, usable.h * 85 / 100));
    window = SDL_CreateWindow("AnyPS5 level select", SDL_WINDOWPOS_CENTERED_DISPLAY(display), SDL_WINDOWPOS_CENTERED_DISPLAY(display), width, height,
        SDL_WINDOW_SHOWN | SDL_WINDOW_RESIZABLE | SDL_WINDOW_ALWAYS_ON_TOP);
    if (window == nullptr) {
        std::fprintf(stderr, "[levelmenu] cannot create the menu window: %s\n", SDL_GetError());
        return;
    }
    SDL_SetWindowMinimumSize(window, 320, 240);
    SDL_RaiseWindow(window);
    SDL_StartTextInput();
    refilter();
    dirty = true;
    std::fprintf(stderr, "[levelmenu] opened with %zu levels from %s\n", entries.size(), listPath.c_str());
}

void LevelMenu::Close(SDL_Window* gameWindow) {
    if (window == nullptr) return;
    SDL_DestroyWindow(window);
    window = nullptr;
    pendingLevel.reset();
    autoEnterAt = {};
    if (gameWindow != nullptr) SDL_RaiseWindow(gameWindow);
    std::fprintf(stderr, "[levelmenu] closed\n");
}

void LevelMenu::refilter() {
    filtered = LevelMenuModel::FilterLevels(entries, filter);
    selected = 0;
    scrollTop = 0;
    dirty = true;
}

void LevelMenu::moveSelection(long delta) {
    if (filtered.empty()) return;
    const long last = static_cast<long>(filtered.size()) - 1;
    selected = static_cast<std::size_t>(std::clamp(static_cast<long>(selected) + delta, 0L, last));
    dirty = true;
}

void LevelMenu::choose() {
    if (filtered.empty() || pendingLevel) return;
    const auto& entry = entries[filtered[selected]];
    std::fprintf(stderr, "[levelmenu] selected %s (%s)\n", entry.name.c_str(), entry.file.c_str());
    pendingLevel = entry.file;
    launchAt = std::chrono::steady_clock::now() + LaunchDelay;
    dirty = true;
}

int LevelMenu::visibleRows(int height) const {
    if (lineHeight <= 0) return 1;
    return std::max(1, (height - listTop - lineHeight - 2 * Padding) / lineHeight);
}

std::vector<LevelMenu::Row> LevelMenu::rows() const {
    std::vector<Row> result;
    const std::string* category = nullptr;
    for (const auto index : filtered) {
        if (category == nullptr || *category != entries[index].category) {
            category = &entries[index].category;
            result.push_back({true, index});
        }
        result.push_back({false, index});
    }
    return result;
}

void LevelMenu::handleKey(const SDL_KeyboardEvent& key, SDL_Window* gameWindow) {
    const auto code = key.keysym.scancode;
    if (key.type == SDL_KEYUP) {
        if (code >= 0 && code < SDL_NUM_SCANCODES) swallowKeyUp[static_cast<std::size_t>(code)] = false;
        return;
    }
    if (code >= 0 && code < SDL_NUM_SCANCODES) swallowKeyUp[static_cast<std::size_t>(code)] = true;
    if (pendingLevel) return;
    int height = 0;
    if (window != nullptr) SDL_GetWindowSize(window, nullptr, &height);
    const long page = visibleRows(height) - 1;
    switch (code) {
        case SDL_SCANCODE_ESCAPE: Close(gameWindow); break;
        case SDL_SCANCODE_F2: if (key.repeat == 0) Close(gameWindow); break;
        case SDL_SCANCODE_RETURN:
        case SDL_SCANCODE_KP_ENTER: choose(); break;
        case SDL_SCANCODE_UP: moveSelection(-1); break;
        case SDL_SCANCODE_DOWN: moveSelection(1); break;
        case SDL_SCANCODE_PAGEUP: moveSelection(-std::max(1L, page)); break;
        case SDL_SCANCODE_PAGEDOWN: moveSelection(std::max(1L, page)); break;
        case SDL_SCANCODE_HOME: moveSelection(-static_cast<long>(filtered.size())); break;
        case SDL_SCANCODE_END: moveSelection(static_cast<long>(filtered.size())); break;
        case SDL_SCANCODE_BACKSPACE:
            if (filter.empty()) break;
            if ((key.keysym.mod & KMOD_CTRL) != 0) filter.clear();
            else filter.pop_back();
            refilter();
            break;
        default: break;
    }
}

void LevelMenu::handleText(const char* text) {
    if (pendingLevel) return;
    bool changed = false;
    for (const char* character = text; *character != '\0'; ++character) {
        const auto value = static_cast<unsigned char>(*character);
        if (value < 32 || value >= 127) continue;
        filter.push_back(static_cast<char>(value));
        changed = true;
    }
    if (changed) refilter();
}

void LevelMenu::handleMouse(const SDL_Event& event) {
    if (window == nullptr || pendingLevel) return;
    const auto menuId = SDL_GetWindowID(window);
    if (event.type == SDL_MOUSEWHEEL && event.wheel.windowID == menuId) {
        int direction = (event.wheel.y > 0) - (event.wheel.y < 0);
        if (event.wheel.direction == SDL_MOUSEWHEEL_FLIPPED) direction = -direction;
        moveSelection(-3L * direction);
        return;
    }
    if (event.type != SDL_MOUSEBUTTONDOWN || event.button.windowID != menuId || event.button.button != SDL_BUTTON_LEFT || lineHeight <= 0) return;
    if (event.button.y < listTop) return;
    const auto all = rows();
    const long index = scrollTop + (event.button.y - listTop) / lineHeight;
    if (index < 0 || index >= static_cast<long>(all.size()) || all[static_cast<std::size_t>(index)].header) return;
    const auto found = std::find(filtered.begin(), filtered.end(), all[static_cast<std::size_t>(index)].entry);
    if (found == filtered.end()) return;
    selected = static_cast<std::size_t>(found - filtered.begin());
    dirty = true;
    if (event.button.clicks >= 2) choose();
}

void LevelMenu::draw() {
    SDL_Surface* surface = SDL_GetWindowSurface(window);
    if (surface == nullptr) {
        std::fprintf(stderr, "[levelmenu] no window surface: %s\n", SDL_GetError());
        return;
    }
    if (SDL_MUSTLOCK(surface) && SDL_LockSurface(surface) != 0) return;
    const int width = surface->w;
    const int height = surface->h;
    const int right = width - Padding;
    fillRect(surface, 0, 0, width, height, Background);

    int y = Padding;
    font->Draw(surface, Padding, y, "Level select", AccentColor, right);
    const std::string hint = "Enter: restart into level   Esc: close";
    font->Draw(surface, std::max(Padding, right - font->Width(hint)), y, hint, DimColor, right);
    y += lineHeight + 4;

    fillRect(surface, Padding - 4, y - 2, width - 2 * Padding + 8, lineHeight + 4, FilterBackground);
    const std::string count = std::to_string(filtered.size()) + " / " + std::to_string(entries.size());
    const int countLeft = right - font->Width(count);
    font->Draw(surface, Padding, y, "Filter: " + filter + "_", TextColor, countLeft - Padding);
    font->Draw(surface, countLeft, y, count, DimColor, right + 1);
    y += lineHeight + Padding;
    listTop = y;

    if (pendingLevel) {
        const auto& entry = entries[filtered[selected]];
        font->Draw(surface, Padding, y + lineHeight, "Restarting into " + entry.name + " (" + entry.file + ") ...", AccentColor, right);
    } else if (filtered.empty()) {
        font->Draw(surface, Padding, y, "No level matches the filter", DimColor, right);
    } else {
        const auto all = rows();
        const int visible = visibleRows(height);
        long selectedRow = 0;
        for (std::size_t index = 0; index < all.size(); ++index) {
            if (!all[index].header && all[index].entry == filtered[selected]) {
                selectedRow = static_cast<long>(index);
                break;
            }
        }
        const long wantedTop = selectedRow > 0 && all[static_cast<std::size_t>(selectedRow - 1)].header ? selectedRow - 1 : selectedRow;
        if (wantedTop < scrollTop) scrollTop = wantedTop;
        if (selectedRow >= scrollTop + visible) scrollTop = selectedRow - visible + 1;
        scrollTop = std::clamp(scrollTop, 0L, std::max(0L, static_cast<long>(all.size()) - visible));
        for (long index = scrollTop; index < static_cast<long>(all.size()) && index < scrollTop + visible; ++index) {
            const auto& row = all[static_cast<std::size_t>(index)];
            const int rowY = listTop + static_cast<int>(index - scrollTop) * lineHeight;
            const auto& entry = entries[row.entry];
            if (row.header) {
                font->Draw(surface, Padding, rowY, entry.category, AccentColor, right);
                continue;
            }
            if (index == selectedRow) fillRect(surface, Padding - 4, rowY, width - 2 * Padding + 8, lineHeight, SelectionBackground);
            const int fileLeft = std::max(width / 2, right - font->Width(entry.file));
            font->Draw(surface, Padding + 16, rowY, entry.name, TextColor, fileLeft - Padding);
            font->Draw(surface, fileLeft, rowY, entry.file, index == selectedRow ? TextColor : DimColor, right + 1);
        }
    }
    font->Draw(surface, Padding, height - lineHeight - Padding / 2, "Up/Down, PgUp/PgDn, Home/End, type to filter, Backspace, F2 or Esc closes", DimColor, right);

    if (SDL_MUSTLOCK(surface)) SDL_UnlockSurface(surface);
    SDL_UpdateWindowSurface(window);
}

bool LevelMenu::HandleEvent(const SDL_Event& event, SDL_Window* gameWindow) {
    if (!enabled) return false;
    switch (event.type) {
        case SDL_KEYDOWN:
        case SDL_KEYUP: {
            const auto code = event.key.keysym.scancode;
            if (IsOpen()) {
                handleKey(event.key, gameWindow);
                return true;
            }
            if (code == SDL_SCANCODE_F2) {
                if (event.type == SDL_KEYDOWN && event.key.repeat == 0) {
                    swallowKeyUp[static_cast<std::size_t>(code)] = true;
                    open(gameWindow);
                } else if (event.type == SDL_KEYUP) {
                    swallowKeyUp[static_cast<std::size_t>(code)] = false;
                }
                return true;
            }
            if (event.type == SDL_KEYUP && code >= 0 && code < SDL_NUM_SCANCODES && swallowKeyUp[static_cast<std::size_t>(code)]) {
                swallowKeyUp[static_cast<std::size_t>(code)] = false;
                return true;
            }
            return false;
        }
        case SDL_TEXTINPUT:
            if (!IsOpen()) return false;
            handleText(event.text.text);
            return true;
        case SDL_TEXTEDITING:
            return IsOpen();
        case SDL_MOUSEMOTION:
        case SDL_MOUSEBUTTONDOWN:
        case SDL_MOUSEBUTTONUP:
        case SDL_MOUSEWHEEL:
            if (!IsOpen()) return false;
            handleMouse(event);
            return true;
        case SDL_CONTROLLERBUTTONDOWN:
        case SDL_CONTROLLERBUTTONUP:
        case SDL_CONTROLLERAXISMOTION:
            return IsOpen();
        case SDL_WINDOWEVENT:
            if (!IsOpen()) return false;
            if (event.window.windowID == SDL_GetWindowID(window)) {
                if (event.window.event == SDL_WINDOWEVENT_CLOSE) Close(gameWindow);
                else dirty = true;
                return true;
            }
            if (gameWindow != nullptr && event.window.windowID == SDL_GetWindowID(gameWindow) && event.window.event == SDL_WINDOWEVENT_CLOSE) {
                Close(gameWindow);
                SDL_Event quit{};
                quit.type = SDL_QUIT;
                SDL_PushEvent(&quit);
            }
            return false;
        default:
            return false;
    }
}

std::optional<std::string> LevelMenu::Update(SDL_Window* gameWindow) {
    if (!enabled) return std::nullopt;
    const auto now = std::chrono::steady_clock::now();
    if (firstUpdate == std::chrono::steady_clock::time_point{}) firstUpdate = now;
    if (autoSelect && !autoOpened && now >= firstUpdate + autoSelect->delay) {
        autoOpened = true;
        std::fprintf(stderr, "[levelmenu] auto-select: opening the menu and typing %s\n", autoSelect->level.c_str());
        open(gameWindow);
        if (IsOpen()) {
            filter.clear();
            refilter();
            handleText(autoSelect->level.c_str());
            autoEnterAt = now + AutoEnterDelay;
        }
    }
    if (autoEnterAt != std::chrono::steady_clock::time_point{} && now >= autoEnterAt) {
        autoEnterAt = {};
        SDL_KeyboardEvent enter{};
        enter.type = SDL_KEYDOWN;
        enter.state = SDL_PRESSED;
        enter.keysym.scancode = SDL_SCANCODE_RETURN;
        enter.keysym.sym = SDLK_RETURN;
        std::fprintf(stderr, "[levelmenu] auto-select: pressing Enter with %zu match(es) for '%s'\n", filtered.size(), filter.c_str());
        if (const char* snapshot = std::getenv("APS5_LEVEL_MENU_SNAPSHOT"); snapshot != nullptr && snapshot[0] != '\0' && window != nullptr) {
            draw();
            if (SDL_Surface* surface = SDL_GetWindowSurface(window); surface != nullptr && SDL_SaveBMP(surface, snapshot) == 0) std::fprintf(stderr, "[levelmenu] menu snapshot: %s\n", snapshot);
        }
        handleKey(enter, gameWindow);
        swallowKeyUp[SDL_SCANCODE_RETURN] = false;
    }
    if (window != nullptr && dirty) {
        dirty = false;
        draw();
    }
    if (pendingLevel && now >= launchAt) {
        auto level = std::move(*pendingLevel);
        pendingLevel.reset();
        return level;
    }
    return std::nullopt;
}
