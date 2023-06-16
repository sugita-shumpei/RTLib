set(GLI_SOURCE_DIR     ${CMAKE_CURRENT_LIST_DIR}/../../Lib/gli)
set(GLI_BINARY_DIR     ${CMAKE_CURRENT_LIST_DIR}/../../Build/gli)
set(GLI_INSTALL_PREFIX ${CMAKE_CURRENT_LIST_DIR}/../../Install)

if (NOT EXISTS ${GLI_BINARY_DIR})
    make_directory(${GLI_BINARY_DIR})
endif()


message(STATUS GLI_SOURCE_DIR=${GLI_SOURCE_DIR})
message(STATUS GLI_BINARY_DIR=${GLI_BINARY_DIR})
message(STATUS GLI_BINARY_DIR=${GLI_INSTALL_PREFIX})

execute_process(COMMAND 
    ${CMAKE_COMMAND} -S ${GLI_SOURCE_DIR} -B ${GLI_BINARY_DIR} -DCMAKE_INSTALL_PREFIX=${GLI_INSTALL_PREFIX} 
)
execute_process(COMMAND
    ${CMAKE_COMMAND} --build ${GLI_BINARY_DIR} --config Release
)
execute_process(COMMAND
    ${CMAKE_COMMAND} --install ${GLI_BINARY_DIR} --config Release
)