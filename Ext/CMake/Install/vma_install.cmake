set(VMA_SOURCE_DIR     ${CMAKE_CURRENT_LIST_DIR}/../../VulkanMemoryAllocator)
set(VMA_BINARY_DIR     ${CMAKE_CURRENT_LIST_DIR}/../../Build/VulkanMemoryAllocator)
set(VMA_INSTALL_PREFIX ${CMAKE_CURRENT_LIST_DIR}/../../Install)


message(STATUS VMA_SOURCE_DIR=${VMA_SOURCE_DIR})
message(STATUS VMA_BINARY_DIR=${VMA_BINARY_DIR})
message(STATUS VMA_BINARY_DIR=${VMA_INSTALL_PREFIX})

execute_process(COMMAND 
    ${CMAKE_COMMAND} -S ${VMA_SOURCE_DIR} -B ${VMA_BINARY_DIR} -DCMAKE_INSTALL_PREFIX=${VMA_INSTALL_PREFIX}
)
execute_process(COMMAND
    ${CMAKE_COMMAND} --build ${VMA_BINARY_DIR} --config Release
)
execute_process(COMMAND
    ${CMAKE_COMMAND} --install ${VMA_BINARY_DIR} --config Release
)