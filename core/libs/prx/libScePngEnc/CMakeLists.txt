add_library(libScePngEnc SHARED EXCLUDE_FROM_ALL ${CMAKE_CURRENT_SOURCE_DIR}/Export.cpp)
target_include_directories(libScePngEnc PRIVATE ${LIBS_INCLUDE_DIR} ${CMAKE_SOURCE_DIR}/3rdparty/stb)
set_target_properties(libScePngEnc PROPERTIES CXX_EXTENSIONS OFF)
configure_windows_unwind(libScePngEnc)
