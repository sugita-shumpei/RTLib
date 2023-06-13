find_package(OptiX REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(glm REQUIRED)
find_package(assimp REQUIRED)
find_package(glfw3 REQUIRED)
add_library(RTLib-Test-TestLib STATIC
	${CMAKE_CURRENT_SOURCE_DIR}/Include/TestLib/Context.h
	${CMAKE_CURRENT_SOURCE_DIR}/Include/TestLib/Camera.h
	${CMAKE_CURRENT_SOURCE_DIR}/Include/TestLib/CUGL.h
	${CMAKE_CURRENT_SOURCE_DIR}/Include/TestLib/Module.h
	${CMAKE_CURRENT_SOURCE_DIR}/Include/TestLib/Pipeline.h
	${CMAKE_CURRENT_SOURCE_DIR}/Include/TestLib/PipelineGroup.h
	${CMAKE_CURRENT_SOURCE_DIR}/Include/TestLib/ProgramGroup.h
	${CMAKE_CURRENT_SOURCE_DIR}/Include/TestLib/ShaderBindingTable.h
	${CMAKE_CURRENT_SOURCE_DIR}/Include/TestLib/OGL4Renderer.h
	${CMAKE_CURRENT_SOURCE_DIR}/Include/TestLib/Utils.h

	${CMAKE_CURRENT_SOURCE_DIR}/Src/Camera.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/Src/Context.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/Src/CUGL.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/Src/Module.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/Src/Pipeline.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/Src/PipelineGroup.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/Src/ProgramGroup.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/Src/ShaderBindingTable.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/Src/OGL4Renderer.cpp
)
target_include_directories(RTLib-Test-TestLib PUBLIC Include)
target_link_libraries(RTLib-Test-TestLib PUBLIC 
	OTK-Error 
	OTK-Memory 
	OTK-ShaderUtil
	OTK-OptiXMemory 

	OptiX::OptiX 
	CUDA::cuda_driver 
	nlohmann_json::nlohmann_json 
	glfw 
	glm::glm 
	assimp::assimp 
	glad_gl_core_46 
)