add_library(libSceVideodec2 SHARED EXCLUDE_FROM_ALL
        ${CMAKE_CURRENT_SOURCE_DIR}/Export.cpp
)

target_include_directories(libSceVideodec2 PRIVATE ${LIBS_INCLUDE_DIR})
configure_windows_unwind(libSceVideodec2)

find_package(PkgConfig QUIET)
if(PKG_CONFIG_FOUND)
    pkg_check_modules(VIDEODEC2_FFMPEG QUIET IMPORTED_TARGET libavcodec libavutil libswscale)
endif()
if(VIDEODEC2_FFMPEG_FOUND)
    target_link_libraries(libSceVideodec2 PRIVATE PkgConfig::VIDEODEC2_FFMPEG)
    target_compile_definitions(libSceVideodec2 PRIVATE APS5_HAVE_FFMPEG=1)
    set(VIDEODEC2_HAS_FFMPEG ON PARENT_SCOPE)
    message(STATUS "libSceVideodec2: H.264 decoding with FFmpeg ${VIDEODEC2_FFMPEG_libavcodec_VERSION}")
else()
    message(STATUS "libSceVideodec2: FFmpeg not found, decoders return black pictures")
endif()
