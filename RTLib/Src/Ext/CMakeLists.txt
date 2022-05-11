add_subdirectory(GL)

add_subdirectory(GLFW)

if (CUDAToolkit_FOUND)
    add_subdirectory(CUDA)
    add_subdirectory(CUGL)
endif()

if (OptiX_FOUND AND CUDAToolkit_FOUND)
    add_subdirectory(OPX7)
endif()