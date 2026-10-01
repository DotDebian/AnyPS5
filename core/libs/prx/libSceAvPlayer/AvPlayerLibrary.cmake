function(add_sce_avplayer_library target)
    set(avPlayerDir ${CMAKE_CURRENT_FUNCTION_LIST_DIR})
    add_library(${target} SHARED EXCLUDE_FROM_ALL
            ${CMAKE_CURRENT_SOURCE_DIR}/Export.cpp
            ${avPlayerDir}/src/Exports.cpp
            ${avPlayerDir}/src/Player.cpp
            ${avPlayerDir}/src/Source.cpp
    )
    target_include_directories(${target} PRIVATE ${LIBS_INCLUDE_DIR})
    target_link_libraries(${target} PRIVATE libc)
    set_target_properties(${target} PROPERTIES
            CXX_VISIBILITY_PRESET hidden
            VISIBILITY_INLINES_HIDDEN ON
    )
    configure_windows_unwind(${target})

    # FFmpeg from 3rdparty/FFmpeg (FFmpeg.cmake); a system FFmpeg found
    # through pkg-config when the submodule is not there.
    if(TARGET anyps5_ffmpeg)
        target_link_libraries(${target} PRIVATE anyps5_ffmpeg)
        message(STATUS "${target}: playback with FFmpeg from 3rdparty/FFmpeg")
        set(AVPLAYER_HAS_FFMPEG ON PARENT_SCOPE)
        return()
    endif()
    find_package(PkgConfig QUIET)
    if(PKG_CONFIG_FOUND)
        pkg_check_modules(AVPLAYER_FFMPEG QUIET IMPORTED_TARGET libavformat libavcodec libavutil libswscale libswresample)
    endif()
    if(AVPLAYER_FFMPEG_FOUND)
        target_link_libraries(${target} PRIVATE PkgConfig::AVPLAYER_FFMPEG)
        target_compile_definitions(${target} PRIVATE APS5_HAVE_FFMPEG=1)
        message(STATUS "${target}: playback with FFmpeg ${AVPLAYER_FFMPEG_libavformat_VERSION}")
    else()
        message(STATUS "${target}: FFmpeg not found, adding sources throws not implemented")
    endif()
    set(AVPLAYER_HAS_FFMPEG ${AVPLAYER_FFMPEG_FOUND} PARENT_SCOPE)
endfunction()
