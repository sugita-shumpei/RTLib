find_package(Imath REQUIRED)
find_package(glm   REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(assimp REQUIRED)
find_package(directxtex REQUIRED)

add_library(RTLib-Core STATIC 
	Include/RTLib/Core/DataTypes.h 
	Include/RTLib/Core/Math.h

	Include/RTLib/Core/Vector.h 
	Src/Vector.cpp

	Include/RTLib/Core/Matrix.h 
	Src/Matrix.cpp

	Include/RTLib/Core/Json.h 
	Src/Json.cpp

	Include/RTLib/Core/Camera.h
	Src/Camera.cpp
	
	Include/RTLib/Core/Quaternion.h
	Src/Quaternion.cpp

	Include/RTLib/Core/Texture.h
	Src/Texture.cpp

	Include/RTLib/Core/Sampler.h
	Src/Sampler.cpp

	Include/RTLib/Core/Mesh.h	
	Src/Mesh.cpp

	Include/RTLib/Core/Aabb.h
)

target_include_directories(RTLib-Core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/Include)
target_link_libraries(RTLib-Core PUBLIC Imath::Imath nlohmann_json::nlohmann_json glm::glm assimp::assimp Microsoft::DirectXTex OTK-ShaderUtil)

add_subdirectory(Test)