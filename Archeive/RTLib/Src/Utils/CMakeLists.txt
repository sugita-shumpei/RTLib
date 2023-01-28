set(RTLIB_CORE_SOURCES 
    ${CMAKE_CURRENT_SOURCE_DIR}/Include/RTLib/Utils/Math.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Include/RTLib/Utils/Camera.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Src/Camera.cpp
)

add_library(RTLib-Utils ${RTLIB_CORE_SOURCES}
)
target_include_directories(RTLib-Utils PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/Include)
target_link_libraries(RTLib-Utils PUBLIC nlohmann_json::nlohmann_json)
set_target_properties(RTLib-Utils PROPERTIES FOLDER RTLib/RTLib/Utils)