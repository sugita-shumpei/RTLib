add_subdirectory(tinyobjloader)
if (NOT UFBX)
    add_library(UFBX STATIC
     ${CMAKE_CURRENT_SOURCE_DIR}/ufbx/ufbx.h
     ${CMAKE_CURRENT_SOURCE_DIR}/ufbx/ufbx.c
    )
    target_include_directories(UFBX PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/ufbx)
endif()