find_package(Freetype REQUIRED)

add_library(libSceFontFt SHARED EXCLUDE_FROM_ALL
        ${CMAKE_CURRENT_SOURCE_DIR}/Export.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/Driver.cpp
)

target_include_directories(libSceFontFt PRIVATE ${LIBS_INCLUDE_DIR})
target_link_libraries(libSceFontFt PRIVATE Freetype::Freetype libc)
set_target_properties(libSceFontFt PROPERTIES
        CXX_EXTENSIONS OFF
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN ON
)
configure_windows_unwind(libSceFontFt)
