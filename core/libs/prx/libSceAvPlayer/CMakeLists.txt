include(${CMAKE_CURRENT_SOURCE_DIR}/AvPlayerLibrary.cmake)
add_sce_avplayer_library(libSceAvPlayer)
set(AVPLAYER_HAS_FFMPEG ${AVPLAYER_HAS_FFMPEG} PARENT_SCOPE)
