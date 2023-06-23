find_package(Imath REQUIRED)
find_package(glm   REQUIRED)
find_package(nlohmann_json REQUIRED)

add_library(RTLib-Core STATIC 
	Include/RTLib/Core/DataTypes.h 
	Include/RTLib/Core/Vector.h 
	Include/RTLib/Core/Matrix.h 
	Include/RTLib/Core/Math.h
	Include/RTLib/Core/Json.h 
	Src/Json.cpp
)
target_include_directories(RTLib-Core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/Include)
target_link_libraries(RTLib-Core PUBLIC Imath::Imath nlohmann_json::nlohmann_json)

add_subdirectory(Test)