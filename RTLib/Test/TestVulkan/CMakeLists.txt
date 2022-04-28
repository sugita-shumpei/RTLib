add_library(TestVulkan-Lib
${CMAKE_CURRENT_SOURCE_DIR}/Include/TestVulkan.h
${CMAKE_CURRENT_SOURCE_DIR}/Src/TestVulkan.cpp
)
target_include_directories(TestVulkan-Lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/Include)
target_link_libraries(TestVulkan-Lib PUBLIC Vulkan::Vulkan)
set_target_properties(TestVulkan-Lib PROPERTIES FOLDER RTLib/Test/TestVulkan)
