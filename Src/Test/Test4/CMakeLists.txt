find_package(CUDAToolkit REQUIRED)
# find_package(glad        REQUIRED)
find_package(glfw3 REQUIRED)
find_package(glm REQUIRED)
find_package(assimp REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(OptiX REQUIRED)
include(embed_optixir)
embed_ptx(CONST RELOCATABLE 
	OUTPUT_TARGET RTLib-Test-Test4-OPX7-PTX 
	HEADER RTLib-Test-Test4-OPX7-ptx-generated.h
	PTX_INCLUDE_DIRECTORIES ${CMAKE_CURRENT_SOURCE_DIR}/Include
	PTX_LINK_LIBRARIES OptiX::OptiX  CUDA::cudart OTK-ShaderUtil 
	SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/Src/Test4_OPX7.cu
)
embed_optixir(CONST RELOCATABLE 
	OUTPUT_TARGET RTLib-Test-Test4-OPX7-OPTIXIR 
	HEADER RTLib-Test-Test4-OPX7-optixir-generated.h
	OPTIXIR_INCLUDE_DIRECTORIES ${CMAKE_CURRENT_SOURCE_DIR}/Include
	OPTIXIR_LINK_LIBRARIES OptiX::OptiX  CUDA::cudart OTK-ShaderUtil 
	SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/Src/Test4_OPX7.cu
)

add_executable(RTLib-Test-Test4 
	${CMAKE_CURRENT_SOURCE_DIR}/Include/Test4_Main.h
	${CMAKE_CURRENT_SOURCE_DIR}/Include/Test4_OPX7.h

	${CMAKE_CURRENT_SOURCE_DIR}/Src/Test4_Main.cpp
)
target_include_directories(RTLib-Test-Test4 PUBLIC Include)
target_link_libraries(RTLib-Test-Test4 PUBLIC 
	RTLib-Test-TestLib
	RTLib-Test-Test4-OPX7-PTX 
	RTLib-Test-Test4-OPX7-OPTIXIR
)
