set(OptiXToolkit_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../Lib/optix-toolkit)
include(${OptiXToolkit_ROOT}/CMake/embed_ptx.cmake)
add_subdirectory(Memory)
add_subdirectory(ShaderUtil)