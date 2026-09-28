#include <cstdint>
#include <cstddef>
#include "SceTypes.hpp"
#include "prx/libc/include/General.hpp"
#include <map>
#include <mutex>
#include <stdexcept>

namespace {

constexpr int ImeErrorBusy = static_cast<int>(0x80BC0001);
constexpr int ImeErrorNotOpened = static_cast<int>(0x80BC0002);

struct KeyboardListener {
    EventHandler handler;
    void* arg;
};

std::mutex g_keyboardMutex;
std::map<int32_t, KeyboardListener> g_keyboards;

}

extern "C" {

int APS5_VABI sceImeClose_nid_postfix(void) {
 NotImplemented_nid_no_patch(__func__);
 return 0;
}

int APS5_VABI sceImeGetPanelSize(const Param* param, uint32_t* width, uint32_t* height) {
 (void)param;
 (void)width;
 (void)height;
 NotImplemented_nid_no_patch(__func__);
 return 0;
}

int APS5_VABI sceImeKeyboardClose(int32_t user_id) {
 std::lock_guard lock(g_keyboardMutex);
 return g_keyboards.erase(user_id) != 0 ? 0 : ImeErrorNotOpened;
}

int APS5_VABI sceImeKeyboardGetInfo(uint32_t resource_id, KeyboardInfo* info) {
 (void)resource_id;
 (void)info;
 NotImplemented_nid_no_patch(__func__);
 return 0;
}

int APS5_VABI sceImeKeyboardGetResourceId(int32_t user_id, KeyboardResourceIdArray* resource_ids) {
 (void)user_id;
 (void)resource_ids;
 NotImplemented_nid_no_patch(__func__);
 return 0;
}

int APS5_VABI sceImeKeyboardOpen(int32_t user_id, const KeyboardParam* param) {
 if (param == nullptr || param->handler == nullptr) throw std::invalid_argument("sceImeKeyboardOpen: missing parameter or event handler");
 std::lock_guard lock(g_keyboardMutex);
 if (!g_keyboards.emplace(user_id, KeyboardListener{param->handler, param->arg}).second) return ImeErrorBusy;
 return 0;
}

int APS5_VABI sceImeKeyboardSetMode(int32_t user_id, uint32_t mode) {
 (void)user_id;
 (void)mode;
 NotImplemented_nid_no_patch(__func__);
 return 0;
}

int APS5_VABI sceImeOpen_nid_postfix(const Param* param, const ExtendedParam* extended) {
 (void)param;
 (void)extended;
 NotImplemented_nid_no_patch(__func__);
 return 0;
}

void APS5_VABI sceImeParamInit(Param* param) {
 (void)param;
 NotImplemented_nid_no_patch(__func__);
}

int APS5_VABI sceImeSetCaret(const Caret* caret) {
 (void)caret;
 NotImplemented_nid_no_patch(__func__);
 return 0;
}

int APS5_VABI sceImeSetText(const char16_t* text, uint32_t length) {
 (void)text;
 (void)length;
 NotImplemented_nid_no_patch(__func__);
 return 0;
}

int APS5_VABI sceImeSetTextGeometry(TextAreaMode mode, const TextGeometry* geometry) {
 (void)mode;
 (void)geometry;
 NotImplemented_nid_no_patch(__func__);
 return 0;
}

int APS5_VABI sceImeUpdate(EventHandler handler) {
 if (handler == nullptr) throw std::invalid_argument("sceImeUpdate: null event handler");
 std::lock_guard lock(g_keyboardMutex);
 return g_keyboards.empty() ? ImeErrorNotOpened : 0;
}

}
