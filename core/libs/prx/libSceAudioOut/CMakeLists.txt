add_library(libSceAudioOut SHARED EXCLUDE_FROM_ALL
        ${CMAKE_CURRENT_SOURCE_DIR}/src/AudioOut.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/AudioOut2Context.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/AudioOut2PadMix.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/AudioOut2Port.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/AudioOut2SpeakerArray.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/AudioOut2User.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/AudioOut2Mastering.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/AudioOut2System.cpp
)

target_include_directories(libSceAudioOut PRIVATE
        ${LIBS_INCLUDE_DIR}
        ${CMAKE_SOURCE_DIR}/3rdparty/SDL2/include
)
target_link_libraries(libSceAudioOut PRIVATE libc libkernel SDL2-static)
configure_windows_unwind(libSceAudioOut)

if(BUILD_TESTING)
    add_executable(audio_out2_pad_mix_tests tests/AudioOut2PadMix.cpp src/AudioOut2PadMix.cpp)
    target_include_directories(audio_out2_pad_mix_tests PRIVATE ${LIBS_INCLUDE_DIR})
    target_link_libraries(audio_out2_pad_mix_tests PRIVATE libc)
    configure_windows_unwind(audio_out2_pad_mix_tests)
    add_test(NAME audio_out2_pad_mix COMMAND audio_out2_pad_mix_tests)
    if(WIN32)
        set_tests_properties(audio_out2_pad_mix PROPERTIES ENVIRONMENT_MODIFICATION
            "PATH=path_list_prepend:$<TARGET_FILE_DIR:libc>")
    endif()
endif()
