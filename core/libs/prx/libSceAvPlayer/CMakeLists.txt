include(${CMAKE_CURRENT_SOURCE_DIR}/AvPlayerLibrary.cmake)
add_sce_avplayer_library(libSceAvPlayer)
set(AVPLAYER_HAS_FFMPEG ${AVPLAYER_HAS_FFMPEG} PARENT_SCOPE)

if(BUILD_TESTING)
    if(AVPLAYER_HAS_FFMPEG)
        find_package(PkgConfig QUIET)
        if(PKG_CONFIG_FOUND)
            pkg_check_modules(AVPLAYER_TEST_FFMPEG QUIET IMPORTED_TARGET libavformat libavcodec libavutil)
        endif()
    endif()
    if(AVPLAYER_HAS_FFMPEG AND AVPLAYER_TEST_FFMPEG_FOUND)
        add_test_executable(guest_avplayer_tests ${LIBS_INCLUDE_DIR}/tests/GuestAvPlayer.cpp)
        target_include_directories(guest_avplayer_tests PRIVATE ${LIBS_INCLUDE_DIR})
        target_link_libraries(guest_avplayer_tests PRIVATE libSceAvPlayer PkgConfig::AVPLAYER_TEST_FFMPEG)
        configure_windows_unwind(guest_avplayer_tests)
        file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/guest_avplayer)
        add_test(NAME guest_avplayer COMMAND guest_avplayer_tests)
        set_tests_properties(guest_avplayer PROPERTIES
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/guest_avplayer
            SKIP_RETURN_CODE 77
            TIMEOUT 60)
        if(WIN32)
            set_tests_properties(guest_avplayer PROPERTIES ENVIRONMENT_MODIFICATION
                "PATH=path_list_prepend:$<TARGET_FILE_DIR:libSceAvPlayer>")
        endif()
    endif()
endif()
