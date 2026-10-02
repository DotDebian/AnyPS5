#include "prx/libSceVideoOut/include/LevelRelaunch.hpp"
#include "prx/libSceVideoOut/include/LevelMenuModel.hpp"

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <utility>
#include <vector>

#ifdef __linux__
#include <dirent.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <signal.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace {

#ifdef __linux__

struct InheritedFd {
    int fd;
    bool closeOnExec;
};

std::vector<int> openFds() {
    std::vector<int> result;
    DIR* directory = opendir("/proc/self/fd");
    if (directory == nullptr) return result;
    const int own = dirfd(directory);
    while (const dirent* entry = readdir(directory)) {
        if (entry->d_name[0] < '0' || entry->d_name[0] > '9') continue;
        const int fd = std::atoi(entry->d_name);
        if (fd != own) result.push_back(fd);
    }
    closedir(directory);
    return result;
}

std::vector<InheritedFd> snapshotInheritedFds() {
    std::vector<InheritedFd> result;
    for (const int fd : openFds()) {
        const int flags = fcntl(fd, F_GETFD);
        if (flags >= 0) result.push_back({fd, (flags & FD_CLOEXEC) != 0});
    }
    return result;
}

const std::vector<InheritedFd> inheritedFds = snapshotInheritedFds();

std::string readWholeFile(const char* path) {
    std::ifstream file(path, std::ios::binary);
    return std::string((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
}

std::vector<char*> pointers(std::vector<std::string>& strings) {
    std::vector<char*> result;
    result.reserve(strings.size() + 1);
    for (auto& value : strings) result.push_back(value.data());
    result.push_back(nullptr);
    return result;
}

void closeEverythingButInheritedOnExec() {
#ifdef SYS_close_range
    constexpr unsigned closeRangeCloexec = 1u << 2;
    if (syscall(SYS_close_range, 3u, ~0u, closeRangeCloexec) != 0)
#endif
    {
        for (const int fd : openFds()) {
            if (fd < 3) continue;
            const int flags = fcntl(fd, F_GETFD);
            if (flags >= 0) fcntl(fd, F_SETFD, flags | FD_CLOEXEC);
        }
    }
    for (const auto& inherited : inheritedFds) {
        if (inherited.fd < 3 || inherited.closeOnExec) continue;
        const int flags = fcntl(inherited.fd, F_GETFD);
        if (flags >= 0) fcntl(inherited.fd, F_SETFD, flags & ~FD_CLOEXEC);
    }
}

#endif

}

namespace LevelRelaunch {

void RestartIntoLevel(const std::string& level) {
#ifdef __linux__
    auto argv = LevelMenuModel::BuildRelaunchArgs(LevelMenuModel::SplitNulList(readWholeFile("/proc/self/cmdline")), level);
    auto environment = LevelMenuModel::BuildRelaunchEnv(LevelMenuModel::SplitNulList(readWholeFile("/proc/self/environ")));
    if (argv.size() < 3) {
        std::fprintf(stderr, "[levelmenu] cannot read this process's command line, not restarting\n");
        return;
    }
    std::string commandLine;
    for (const auto& argument : argv) commandLine += (commandLine.empty() ? "" : " ") + argument;
    std::string kept;
    for (const auto& inherited : inheritedFds) {
        if (inherited.fd >= 3 && !inherited.closeOnExec) kept += (kept.empty() ? "" : ",") + std::to_string(inherited.fd);
    }
    std::fprintf(stderr, "[levelmenu] restarting into %s: %s (pid %d, %zu environment variables, inherited fds kept: %s)\n",
        level.c_str(), commandLine.c_str(), static_cast<int>(getpid()), environment.size(), kept.empty() ? "none" : kept.c_str());

    using HoldWrites = bool (*)(bool, int);
    const auto holdWrites = reinterpret_cast<HoldWrites>(dlsym(RTLD_DEFAULT, "SaveDataHoldWrites_nid_no_patch"));
    const bool writesHeld = holdWrites != nullptr && holdWrites(true, 2000);
    if (holdWrites != nullptr && !writesHeld) std::fprintf(stderr, "[levelmenu] a save-data write is still running after 2 s, restarting anyway\n");
    std::fflush(nullptr);

    auto argvPointers = pointers(argv);
    auto environmentPointers = pointers(environment);
    sigset_t previousMask;
    sigset_t emptyMask;
    sigemptyset(&emptyMask);
    pthread_sigmask(SIG_SETMASK, &emptyMask, &previousMask);
    closeEverythingButInheritedOnExec();
    execve("/proc/self/exe", argvPointers.data(), environmentPointers.data());
    const int error = errno;
    pthread_sigmask(SIG_SETMASK, &previousMask, nullptr);
    if (writesHeld) holdWrites(false, 0);
    std::fprintf(stderr, "[levelmenu] restart failed: execve: %s\n", std::strerror(error));
#else
    std::fprintf(stderr, "[levelmenu] restarting into %s is only supported on Linux\n", level.c_str());
#endif
}

}
