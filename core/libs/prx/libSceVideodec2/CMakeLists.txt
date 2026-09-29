add_library(libSceVideodec2 SHARED EXCLUDE_FROM_ALL
        ${CMAKE_CURRENT_SOURCE_DIR}/Export.cpp
)

target_include_directories(libSceVideodec2 PRIVATE ${LIBS_INCLUDE_DIR})
configure_windows_unwind(libSceVideodec2)

if(WIN32)
    set(VIDEODEC2_FFMPEG_DEFAULT OFF)
else()
    set(VIDEODEC2_FFMPEG_DEFAULT ON)
endif()
option(ANYPS5_ENABLE_FFMPEG "Build FFmpeg from 3rdparty/FFmpeg for H.264 decoding in libSceVideodec2" ${VIDEODEC2_FFMPEG_DEFAULT})

set(VIDEODEC2_FFMPEG_SOURCE_DIR ${CMAKE_SOURCE_DIR}/3rdparty/FFmpeg)
find_program(VIDEODEC2_SH sh)
find_program(VIDEODEC2_MAKE NAMES make mingw32-make)

if(ANYPS5_ENABLE_FFMPEG AND EXISTS ${VIDEODEC2_FFMPEG_SOURCE_DIR}/configure AND VIDEODEC2_SH AND VIDEODEC2_MAKE)
    include(ExternalProject)

    set(VIDEODEC2_FFMPEG_PREFIX ${CMAKE_CURRENT_BINARY_DIR}/ffmpeg)
    set(VIDEODEC2_FFMPEG_INCLUDE_DIR ${VIDEODEC2_FFMPEG_PREFIX}/include)
    set(VIDEODEC2_FFMPEG_LIB_DIR ${VIDEODEC2_FFMPEG_PREFIX}/lib)
    set(VIDEODEC2_FFMPEG_LIBS
            ${VIDEODEC2_FFMPEG_LIB_DIR}/libavcodec.a
            ${VIDEODEC2_FFMPEG_LIB_DIR}/libswscale.a
            ${VIDEODEC2_FFMPEG_LIB_DIR}/libavutil.a
    )

    set(VIDEODEC2_FFMPEG_OPTIONS
            --prefix=${VIDEODEC2_FFMPEG_PREFIX}
            --enable-static
            --disable-shared
            --enable-pic
            --disable-autodetect
            --disable-everything
            --disable-programs
            --disable-doc
            --disable-network
            --disable-avdevice
            --disable-avformat
            --disable-avfilter
            --disable-swresample
            --enable-avcodec
            --enable-avutil
            --enable-swscale
            --enable-decoder=h264
    )
    if(CMAKE_C_COMPILER)
        list(APPEND VIDEODEC2_FFMPEG_OPTIONS --cc=${CMAKE_C_COMPILER})
    endif()
    find_program(VIDEODEC2_NASM nasm)
    if(NOT VIDEODEC2_NASM)
        list(APPEND VIDEODEC2_FFMPEG_OPTIONS --disable-x86asm)
    endif()

    cmake_host_system_information(RESULT VIDEODEC2_JOBS QUERY NUMBER_OF_LOGICAL_CORES)

    ExternalProject_Add(ffmpeg_h264
            SOURCE_DIR ${VIDEODEC2_FFMPEG_SOURCE_DIR}
            PREFIX ${VIDEODEC2_FFMPEG_PREFIX}
            BINARY_DIR ${VIDEODEC2_FFMPEG_PREFIX}/build
            CONFIGURE_COMMAND ${VIDEODEC2_SH} ${VIDEODEC2_FFMPEG_SOURCE_DIR}/configure ${VIDEODEC2_FFMPEG_OPTIONS}
            BUILD_COMMAND ${VIDEODEC2_MAKE} -j${VIDEODEC2_JOBS}
            INSTALL_COMMAND ${VIDEODEC2_MAKE} install
            BUILD_BYPRODUCTS ${VIDEODEC2_FFMPEG_LIBS}
            EXCLUDE_FROM_ALL ON
    )
    file(MAKE_DIRECTORY ${VIDEODEC2_FFMPEG_INCLUDE_DIR})

    add_dependencies(libSceVideodec2 ffmpeg_h264)
    target_include_directories(libSceVideodec2 SYSTEM PRIVATE ${VIDEODEC2_FFMPEG_INCLUDE_DIR})
    target_link_libraries(libSceVideodec2 PRIVATE ${VIDEODEC2_FFMPEG_LIBS})
    if(NOT WIN32)
        target_link_libraries(libSceVideodec2 PRIVATE m)
    endif()
    target_link_options(libSceVideodec2 PRIVATE -Wl,--exclude-libs,ALL)
    target_compile_definitions(libSceVideodec2 PRIVATE APS5_HAVE_FFMPEG=1)
    set(VIDEODEC2_HAS_FFMPEG ON PARENT_SCOPE)
    message(STATUS "libSceVideodec2: H.264 decoding with FFmpeg from 3rdparty/FFmpeg")
else()
    message(STATUS "libSceVideodec2: FFmpeg disabled or 3rdparty/FFmpeg missing, decoders return black pictures")
endif()
