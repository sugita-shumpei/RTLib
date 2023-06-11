find_package(CUDAToolkit REQUIRED)
# find_package(glad        REQUIRED)
find_package(glfw3 REQUIRED)
find_package(glm REQUIRED)
find_package(assimp REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(OptiX REQUIRED)
include(embed_optixir)
embed_ptx(CONST RELOCATABLE 
	OUTPUT_TARGET RTLib-Test-Test1-OPX7-PTX 
	HEADER RTLib-Test-Test1-OPX7-ptx-generated.h
	PTX_INCLUDE_DIRECTORIES ${CMAKE_CURRENT_SOURCE_DIR}/Include
	PTX_LINK_LIBRARIES OptiX::OptiX  CUDA::cudart OTK-ShaderUtil 
	SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/Src/Test1_OPX7.cu
)
embed_optixir(CONST RELOCATABLE 
	OUTPUT_TARGET RTLib-Test-Test1-OPX7-OPTIXIR 
	HEADER RTLib-Test-Test1-OPX7-optixir-generated.h
	OPTIXIR_INCLUDE_DIRECTORIES ${CMAKE_CURRENT_SOURCE_DIR}/Include
	OPTIXIR_LINK_LIBRARIES OptiX::OptiX  CUDA::cudart OTK-ShaderUtil 
	SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/Src/Test1_OPX7.cu
)

add_executable(RTLib-Test-Test1 
	${CMAKE_CURRENT_SOURCE_DIR}/Include/Test1_Main.h
	${CMAKE_CURRENT_SOURCE_DIR}/Include/Test1_Context.h
	${CMAKE_CURRENT_SOURCE_DIR}/Include/Test1_Camera.h
	${CMAKE_CURRENT_SOURCE_DIR}/Include/Test1_OPX7.h
	${CMAKE_CURRENT_SOURCE_DIR}/Include/Test1_CUGL.h
	${CMAKE_CURRENT_SOURCE_DIR}/Include/Test1_Module.h
	${CMAKE_CURRENT_SOURCE_DIR}/Include/Test1_Pipeline.h
	${CMAKE_CURRENT_SOURCE_DIR}/Include/Test1_PipelineGroup.h
	${CMAKE_CURRENT_SOURCE_DIR}/Include/Test1_ProgramGroup.h
	${CMAKE_CURRENT_SOURCE_DIR}/Include/Test1_ShaderBindingTable.h
	${CMAKE_CURRENT_SOURCE_DIR}/Include/Test1_OGL4Renderer.h

	${CMAKE_CURRENT_SOURCE_DIR}/Src/Test1_Main.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/Src/Test1_Camera.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/Src/Test1_Context.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/Src/Test1_CUGL.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/Src/Test1_Module.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/Src/Test1_Pipeline.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/Src/Test1_PipelineGroup.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/Src/Test1_ProgramGroup.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/Src/Test1_ShaderBindingTable.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/Src/Test1_OGL4Renderer.cpp
)
target_include_directories(RTLib-Test-Test1 PUBLIC Include)
target_link_libraries(RTLib-Test-Test1 PUBLIC 
	OTK-Error 
	OTK-Memory 
	OTK-ShaderUtil
	OTK-OptiXMemory 
	RTLib-Test-Test1-OPX7-PTX 
	RTLib-Test-Test1-OPX7-OPTIXIR

	OptiX::OptiX 
	CUDA::cuda_driver 
	nlohmann_json::nlohmann_json 
	glfw 
	glm::glm 
	assimp::assimp 
	glad_gl_core_46 
)
