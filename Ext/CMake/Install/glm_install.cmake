set(GLM_SOURCE_DIR     ${CMAKE_CURRENT_LIST_DIR}/../../Lib/glm)
set(GLM_BINARY_DIR     ${CMAKE_CURRENT_LIST_DIR}/../../Build/glm)
set(GLM_INSTALL_PREFIX ${CMAKE_CURRENT_LIST_DIR}/../../Install)

if (NOT EXISTS ${GLM_BINARY_DIR})
    make_directory(${GLM_BINARY_DIR})
endif()

message(STATUS GLM_SOURCE_DIR=${GLM_SOURCE_DIR})
message(STATUS GLM_BINARY_DIR=${GLM_BINARY_DIR})
message(STATUS GLM_BINARY_DIR=${GLM_INSTALL_PREFIX})

execute_process(COMMAND 
    ${CMAKE_COMMAND} -S ${GLM_SOURCE_DIR} -B ${GLM_BINARY_DIR} -DCMAKE_INSTALL_PREFIX=${GLM_INSTALL_PREFIX} 
)
execute_process(COMMAND
    ${CMAKE_COMMAND} --build ${GLM_BINARY_DIR} --config Release
)
execute_process(COMMAND
    ${CMAKE_COMMAND} --install ${GLM_BINARY_DIR} --config Release
)