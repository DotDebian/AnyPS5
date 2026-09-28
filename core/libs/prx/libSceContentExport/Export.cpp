#include <cstdint>
#include <cstddef>
#include <atomic>
#include <stdexcept>
#include "SceTypes.hpp"
#include "prx/libc/include/General.hpp"

namespace {

std::atomic<bool> g_initialized{false};

}

extern "C" {

int APS5_VABI sceContentExportInit2(const ContentExportInitParam2* init_param) {
 if (init_param == nullptr) throw std::invalid_argument("sceContentExportInit2: null parameter");
 if (g_initialized.exchange(true)) throw std::logic_error("sceContentExportInit2: already initialized");
 return 0;
}

int APS5_VABI sceContentExportTerm(void) {
 if (!g_initialized.exchange(false)) throw std::logic_error("sceContentExportTerm: not initialized");
 return 0;
}

int APS5_VABI sceContentExportStart(void) {
 NotImplemented_nid_no_patch(__func__);
 return 0;
}

int APS5_VABI sceContentExportFinish(void) {
 NotImplemented_nid_no_patch(__func__);
 return 0;
}

int APS5_VABI sceContentExportFromData(void) {
 NotImplemented_nid_no_patch(__func__);
 return 0;
}

}
