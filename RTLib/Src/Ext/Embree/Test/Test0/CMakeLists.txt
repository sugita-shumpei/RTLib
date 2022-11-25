add_executable(RTLib-Ext-Embree-Lib-Test0 ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)
target_link_libraries(RTLib-Ext-Embree-Lib-Test0 PRIVATE embree RTLib-Core)

set_target_properties(RTLib-Ext-Embree-Lib-Test0 PROPERTIES FOLDER RTLib/RTLib/Ext/Embree)