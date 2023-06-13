find_package(CUDAToolkit REQUIRED)
# find_package(glad        REQUIRED)
find_package(glfw3 REQUIRED)
find_package(glm REQUIRED)
find_package(assimp REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(OptiX REQUIRED)
include(embed_optixir)
embed_ptx(CONST RELOCATABLE 
	OUTPUT_TARGET RTLib-Test-Test2-OPX7-PTX 
	HEADER RTLib-Test-Test2-OPX7-ptx-generated.h
	PTX_INCLUDE_DIRECTORIES ${CMAKE_CURRENT_SOURCE_DIR}/Include
	PTX_LINK_LIBRARIES OptiX::OptiX  CUDA::cudart OTK-ShaderUtil 
	SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/Src/Test2_OPX7.cu
)
embed_optixir(CONST RELOCATABLE 
	OUTPUT_TARGET RTLib-Test-Test2-OPX7-OPTIXIR 
	HEADER RTLib-Test-Test2-OPX7-optixir-generated.h
	OPTIXIR_INCLUDE_DIRECTORIES ${CMAKE_CURRENT_SOURCE_DIR}/Include
	OPTIXIR_LINK_LIBRARIES OptiX::OptiX  CUDA::cudart OTK-ShaderUtil 
	SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/Src/Test2_OPX7.cu
)

add_executable(RTLib-Test-Test2 
	${CMAKE_CURRENT_SOURCE_DIR}/Include/Test2_Main.h
	${CMAKE_CURRENT_SOURCE_DIR}/Include/Test2_OPX7.h

	${CMAKE_CURRENT_SOURCE_DIR}/Src/Test2_Main.cpp
)
target_include_directories(RTLib-Test-Test2 PUBLIC Include)
target_link_libraries(RTLib-Test-Test2 PUBLIC 
	RTLib-Test-TestLib
	RTLib-Test-Test2-OPX7-PTX 
	RTLib-Test-Test2-OPX7-OPTIXIR
)
