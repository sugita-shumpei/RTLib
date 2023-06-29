find_package(CUDAToolkit REQUIRED)
# find_package(glad        REQUIRED)
find_package(glfw3 REQUIRED)
find_package(glm REQUIRED)
find_package(assimp REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(OptiX REQUIRED)
find_package(stduuid REQUIRED)

embed_ptx(CONST RELOCATABLE 
	OUTPUT_TARGET RTLib-Test-Test0-OPX7 
	HEADER Test0_OPX7_ptx_generated.h
	PTX_INCLUDE_DIRECTORIES ${CMAKE_CURRENT_SOURCE_DIR}/Include
	PTX_LINK_LIBRARIES OptiX::OptiX  CUDA::cudart OTK-ShaderUtil 
	SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/Src/Test0_OPX7.cu
)

add_executable(RTLib-Test-Test0 Include/Test0.h Src/Test0.cpp)
target_include_directories(RTLib-Test-Test0 PUBLIC Include)
target_link_libraries(RTLib-Test-Test0 PUBLIC 
	OTK-Error 
	OTK-Memory 
	OTK-ShaderUtil
	OTK-OptiXMemory 
	RTLib-Test-Test0-OPX7

	OptiX::OptiX 
	CUDA::cuda_driver 
	nlohmann_json::nlohmann_json 
	glfw 
	glm::glm 
	stduuid
	assimp::assimp 
	glad_gl_core_46 
)
