add_library(TestLib 
${CMAKE_CURRENT_SOURCE_DIR}/Include/TestApplication.h
${CMAKE_CURRENT_SOURCE_DIR}/Src/TestApplication.cpp
${CMAKE_CURRENT_SOURCE_DIR}/Include/TestAppInitDelegate.h
${CMAKE_CURRENT_SOURCE_DIR}/Src/TestAppInitDelegate.cpp
${CMAKE_CURRENT_SOURCE_DIR}/Include/TestAppMainDelegate.h
${CMAKE_CURRENT_SOURCE_DIR}/Src/TestAppMainDelegate.cpp
${CMAKE_CURRENT_SOURCE_DIR}/Include/TestAppFreeDelegate.h
${CMAKE_CURRENT_SOURCE_DIR}/Src/TestAppFreeDelegate.cpp
${CMAKE_CURRENT_SOURCE_DIR}/Include/TestAppExtendedData.h
${CMAKE_CURRENT_SOURCE_DIR}/Src/TestAppExtendedData.cpp
)
target_include_directories(TestLib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/Include)
set_target_properties(TestLib PROPERTIES FOLDER RTLib/Test/TestLib)
