find_package(Imath REQUIRED)
find_package(glm   REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(assimp REQUIRED)
find_package(directxtex REQUIRED)
find_package(stduuid REQUIRED)

add_library(RTLib-Core    STATIC )
target_sources(RTLib-Core PRIVATE
	Include/RTLib/Core/Animation.h
	Include/RTLib/Core/AnimateTransform.h
	Include/RTLib/Core/Camera.h
	Include/RTLib/Core/DataTypes.h
	Include/RTLib/Core/Json.h
	Include/RTLib/Core/Math.h
	Include/RTLib/Core/Matrix.h
	Include/RTLib/Core/Mesh.h
	Include/RTLib/Core/Object.h
	Include/RTLib/Core/ObjectTypeID.h
	Include/RTLib/Core/Preprocessor.h
	Include/RTLib/Core/Quaternion.h
	Include/RTLib/Core/Transform.h
	Include/RTLib/Core/Uuid.h
	Include/RTLib/Core/Vector.h
	
	Src/Animation.cpp
	Src/AnimateTransform.cpp
	Src/Camera.cpp
	Src/DataTypes.cpp
	Src/Json.cpp
	Src/Math.cpp
	Src/Matrix.cpp
	Src/Mesh.cpp
	Src/Object.cpp
	Src/Preprocessor.cpp
	Src/Quaternion.cpp
	Src/Transform.cpp
	Src/Uuid.cpp
	Src/Vector.cpp
)
target_include_directories(RTLib-Core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/Include)
target_link_libraries(RTLib-Core PUBLIC Imath::Imath nlohmann_json::nlohmann_json glm::glm assimp::assimp Microsoft::DirectXTex OTK-ShaderUtil stduuid)

add_subdirectory(Test)