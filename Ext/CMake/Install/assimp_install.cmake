set(ASSIMP_SOURCE_DIR     ${CMAKE_CURRENT_LIST_DIR}/../../Lib/assimp)
set(ASSIMP_BINARY_DIR     ${CMAKE_CURRENT_LIST_DIR}/../../Build/assimp)
set(ASSIMP_INSTALL_PREFIX ${CMAKE_CURRENT_LIST_DIR}/../../Install)

if (NOT EXISTS ${ASSIMP_BINARY_DIR})
    make_directory(${ASSIMP_BINARY_DIR})
endif()

if (NOT EXISTS ${ASSIMP_INSTALL_PREFIX}/lib/Debug)
    make_directory(${ASSIMP_INSTALL_PREFIX}/lib/Debug)
endif()

if (NOT EXISTS ${ASSIMP_INSTALL_PREFIX}/lib/Release)
    make_directory(${ASSIMP_INSTALL_PREFIX}/lib/Release)
endif()

if (NOT EXISTS ${ASSIMP_INSTALL_PREFIX}/bin/Debug)
    make_directory(${ASSIMP_INSTALL_PREFIX}/bin/Debug)
endif()

if (NOT EXISTS ${ASSIMP_INSTALL_PREFIX}/bin/Release)
    make_directory(${ASSIMP_INSTALL_PREFIX}/bin/Release)
endif()


message(STATUS ASSIMP_SOURCE_DIR=${ASSIMP_SOURCE_DIR})
message(STATUS ASSIMP_BINARY_DIR=${ASSIMP_BINARY_DIR})
message(STATUS ASSIMP_INSTALL_PREFIX=${ASSIMP_INSTALL_PREFIX})

execute_process(COMMAND 
    ${CMAKE_COMMAND} -S ${ASSIMP_SOURCE_DIR} -B ${ASSIMP_BINARY_DIR} -DCMAKE_INSTALL_PREFIX=${ASSIMP_INSTALL_PREFIX}
)
execute_process(COMMAND
    ${CMAKE_COMMAND} --build ${ASSIMP_BINARY_DIR} --config Debug
)
execute_process(COMMAND
    ${CMAKE_COMMAND} --install ${ASSIMP_BINARY_DIR} --config Debug
)
execute_process(COMMAND
    ${CMAKE_COMMAND} --build ${ASSIMP_BINARY_DIR} --config Release
)
execute_process(COMMAND
    ${CMAKE_COMMAND} --install ${ASSIMP_BINARY_DIR} --config Release
)
