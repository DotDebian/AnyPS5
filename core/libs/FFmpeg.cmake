# FFmpeg from the 3rdparty/FFmpeg submodule, built once with its own configure and make
# (ExternalProject) for libSceAvPlayer (MP4 files, opened by path or through the title's file
# callbacks: the file protocol, the mov demuxer, H.264 and AAC, scaled and resampled for the title).
# avcodec, avformat, avutil, swscale and swresample are static PIC archives with only those
# components; each prx links them with their symbols hidden (--exclude-libs), so no FFmpeg runtime
# is needed and none is exported. Defines the INTERFACE target anyps5_ffmpeg when the submodule
# and sh/make are there and ANYPS5_ENABLE_FFMPEG is on (the default outside Windows).
if(WIN32)
    set(ANYPS5_FFMPEG_DEFAULT OFF)
else()
    set(ANYPS5_FFMPEG_DEFAULT ON)
endif()
option(ANYPS5_ENABLE_FFMPEG "Build FFmpeg from 3rdparty/FFmpeg for libSceAvPlayer" ${ANYPS5_FFMPEG_DEFAULT})

set(ANYPS5_FFMPEG_SOURCE_DIR ${CMAKE_SOURCE_DIR}/3rdparty/FFmpeg)
find_program(ANYPS5_FFMPEG_SH sh)
find_program(ANYPS5_FFMPEG_MAKE NAMES make mingw32-make)

if(ANYPS5_ENABLE_FFMPEG AND EXISTS ${ANYPS5_FFMPEG_SOURCE_DIR}/configure AND ANYPS5_FFMPEG_SH AND ANYPS5_FFMPEG_MAKE)
    include(ExternalProject)

    set(ANYPS5_FFMPEG_PREFIX ${CMAKE_CURRENT_BINARY_DIR}/ffmpeg)
    set(ANYPS5_FFMPEG_INCLUDE_DIR ${ANYPS5_FFMPEG_PREFIX}/include)
    set(ANYPS5_FFMPEG_LIB_DIR ${ANYPS5_FFMPEG_PREFIX}/lib)
    # Link order: each archive before the ones it uses.
    set(ANYPS5_FFMPEG_LIBS
            ${ANYPS5_FFMPEG_LIB_DIR}/libavformat.a
            ${ANYPS5_FFMPEG_LIB_DIR}/libavcodec.a
            ${ANYPS5_FFMPEG_LIB_DIR}/libswscale.a
            ${ANYPS5_FFMPEG_LIB_DIR}/libswresample.a
            ${ANYPS5_FFMPEG_LIB_DIR}/libavutil.a
    )

    set(ANYPS5_FFMPEG_OPTIONS
            --prefix=${ANYPS5_FFMPEG_PREFIX}
            --enable-static
            --disable-shared
            --enable-pic
            --disable-autodetect
            --disable-everything
            --disable-programs
            --disable-doc
            --disable-network
            --disable-avdevice
            --disable-avfilter
            --enable-avcodec
            --enable-avformat
            --enable-avutil
            --enable-swscale
            --enable-swresample
            --enable-decoder=h264,aac
            --enable-parser=h264,aac
            --enable-demuxer=mov
            --enable-protocol=file
    )
    if(CMAKE_C_COMPILER)
        list(APPEND ANYPS5_FFMPEG_OPTIONS --cc=${CMAKE_C_COMPILER})
    endif()
    find_program(ANYPS5_FFMPEG_NASM nasm)
    if(NOT ANYPS5_FFMPEG_NASM)
        list(APPEND ANYPS5_FFMPEG_OPTIONS --disable-x86asm)
    endif()

    cmake_host_system_information(RESULT ANYPS5_FFMPEG_JOBS QUERY NUMBER_OF_LOGICAL_CORES)

    ExternalProject_Add(anyps5_ffmpeg_build
            SOURCE_DIR ${ANYPS5_FFMPEG_SOURCE_DIR}
            PREFIX ${ANYPS5_FFMPEG_PREFIX}
            BINARY_DIR ${ANYPS5_FFMPEG_PREFIX}/build
            CONFIGURE_COMMAND ${ANYPS5_FFMPEG_SH} ${ANYPS5_FFMPEG_SOURCE_DIR}/configure ${ANYPS5_FFMPEG_OPTIONS}
            BUILD_COMMAND ${ANYPS5_FFMPEG_MAKE} -j${ANYPS5_FFMPEG_JOBS}
            INSTALL_COMMAND ${ANYPS5_FFMPEG_MAKE} install
            BUILD_BYPRODUCTS ${ANYPS5_FFMPEG_LIBS}
            EXCLUDE_FROM_ALL ON
    )
    file(MAKE_DIRECTORY ${ANYPS5_FFMPEG_INCLUDE_DIR})

    add_library(anyps5_ffmpeg INTERFACE)
    add_dependencies(anyps5_ffmpeg anyps5_ffmpeg_build)
    target_include_directories(anyps5_ffmpeg SYSTEM INTERFACE ${ANYPS5_FFMPEG_INCLUDE_DIR})
    target_link_libraries(anyps5_ffmpeg INTERFACE ${ANYPS5_FFMPEG_LIBS})
    if(NOT WIN32)
        target_link_libraries(anyps5_ffmpeg INTERFACE m)
    endif()
    target_link_options(anyps5_ffmpeg INTERFACE -Wl,--exclude-libs,ALL)
    target_compile_definitions(anyps5_ffmpeg INTERFACE APS5_HAVE_FFMPEG=1)
    message(STATUS "FFmpeg: built from 3rdparty/FFmpeg (file protocol, mov demuxer, h264 and aac decoders, swscale, swresample)")
else()
    message(STATUS "FFmpeg: disabled or 3rdparty/FFmpeg missing")
endif()
