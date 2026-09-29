add_library(libSceVideodec2 SHARED EXCLUDE_FROM_ALL
        ${CMAKE_CURRENT_SOURCE_DIR}/Export.cpp
)

target_include_directories(libSceVideodec2 PRIVATE ${LIBS_INCLUDE_DIR})
configure_windows_unwind(libSceVideodec2)

if(TARGET anyps5_ffmpeg)
    target_link_libraries(libSceVideodec2 PRIVATE anyps5_ffmpeg)
    set(VIDEODEC2_HAS_FFMPEG ON PARENT_SCOPE)
    message(STATUS "libSceVideodec2: H.264 decoding with FFmpeg from 3rdparty/FFmpeg")
else()
    message(STATUS "libSceVideodec2: FFmpeg disabled or 3rdparty/FFmpeg missing, decoders return black pictures")
endif()
