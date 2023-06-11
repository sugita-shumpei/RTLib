set(SDL3_SOURCE_DIR     ${CMAKE_CURRENT_LIST_DIR}/../sdl)
set(SDL3_BINARY_DIR     ${CMAKE_CURRENT_LIST_DIR}/../../Build/sdl)
set(SDL3_INSTALL_PREFIX ${CMAKE_CURRENT_LIST_DIR}/../../Install)


message(STATUS SDL3_SOURCE_DIR=${SDL3_SOURCE_DIR})
message(STATUS SDL3_BINARY_DIR=${SDL3_BINARY_DIR})
message(STATUS SDL3_BINARY_DIR=${SDL3_INSTALL_PREFIX})

execute_process(COMMAND 
    ${CMAKE_COMMAND} -S ${SDL3_SOURCE_DIR} -B ${SDL3_BINARY_DIR} -DCMAKE_INSTALL_PREFIX=${SDL3_INSTALL_PREFIX} -DSDL_STATIC=ON
)
execute_process(COMMAND
    ${CMAKE_COMMAND} --build ${SDL3_BINARY_DIR} --config Release
)
execute_process(COMMAND
    ${CMAKE_COMMAND} --install ${SDL3_BINARY_DIR} --config Release
)