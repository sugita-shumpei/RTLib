set(RTLIB_EXT_EMBREE_TEST_TEST0_DATA_PATH "${RTLIB_TEST_DATA_PATH}")
set(RTLIB_EXT_EMBREE_TEST_TEST0_JSON_PATH ${CMAKE_CURRENT_BINARY_DIR})
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/RTLibExtEmbreeTest0Config.h.in ${CMAKE_CURRENT_BINARY_DIR}/RTLibExtEmbreeTest0Config.h)
add_executable(
	RTLib-Ext-Embree-Lib-Test0 ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/RTLibExtEmbreeTest0Config.h.in 
	${CMAKE_CURRENT_BINARY_DIR}/RTLibExtEmbreeTest0Config.h)
target_link_libraries(RTLib-Ext-Embree-Lib-Test0 PRIVATE embree RTLib-Core  stb::stb)
target_include_directories(RTLib-Ext-Embree-Lib-Test0 PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

set_target_properties(RTLib-Ext-Embree-Lib-Test0 PROPERTIES FOLDER RTLib/RTLib/Ext/Embree)