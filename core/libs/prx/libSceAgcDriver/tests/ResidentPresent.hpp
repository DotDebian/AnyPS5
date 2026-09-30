#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_TESTS_RESIDENTPRESENT_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_TESTS_RESIDENTPRESENT_HPP

#include "prx/libSceAgcDriver/Graphics/include/Context.hpp"

// The display format decision (DisplayFormat.hpp) for both 8-bit and both 10-bit channel orders,
// and a resident image presented through GpuColorTransfer::DetileImage showing the same pixels as
// the guest-memory path (ResidentPresent.cpp).
void RunResidentPresentTests(const AgcDriver::Graphics::Context& context);

#endif
