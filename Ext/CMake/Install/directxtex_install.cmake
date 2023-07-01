set(DIRECTXTEX_SOURCE_DIR     ${CMAKE_CURRENT_LIST_DIR}/../../Lib/DirectXTex)
set(DIRECTXTEX_BINARY_DIR     ${CMAKE_CURRENT_LIST_DIR}/../../build/DirectXTex)
set(DIRECTXTEX_INSTALL_PREFIX ${CMAKE_CURRENT_LIST_DIR}/../../install)

if (NOT EXISTS ${DIRECTXTEX_BINARY_DIR})
    make_directory(${DIRECTXTEX_BINARY_DIR})
endif()

message(STATUS DIRECTXTEX_SOURCE_DIR=${DIRECTXTEX_SOURCE_DIR})
message(STATUS DIRECTXTEX_BINARY_DIR=${DIRECTXTEX_BINARY_DIR})
message(STATUS DIRECTXTEX_INSTALL_PREFIX=${DIRECTXTEX_INSTALL_PREFIX})

if (NOT EXISTS ${DIRECTXTEX_INSTALL_PREFIX}/lib/Debug)
    make_directory(${DIRECTXTEX_INSTALL_PREFIX}/lib/Debug)
endif()

if (NOT EXISTS ${DIRECTXTEX_INSTALL_PREFIX}/lib/Release)
    make_directory(${DIRECTXTEX_INSTALL_PREFIX}/lib/Release)
endif()

if (NOT EXISTS ${DIRECTXTEX_INSTALL_PREFIX}/lib/RelWithDebInfo)
    make_directory(${DIRECTXTEX_INSTALL_PREFIX}/lib/RelWithDebInfo)
endif()

execute_process(COMMAND 
    ${CMAKE_COMMAND} -S ${DIRECTXTEX_SOURCE_DIR} -B ${DIRECTXTEX_BINARY_DIR} -DCMAKE_INSTALL_PREFIX=${DIRECTXTEX_INSTALL_PREFIX} 
)
execute_process(COMMAND
    ${CMAKE_COMMAND} --build ${DIRECTXTEX_BINARY_DIR}   --config Debug
)
execute_process(COMMAND
    ${CMAKE_COMMAND} --install ${DIRECTXTEX_BINARY_DIR} --config Debug
)
## PATCH: デフォルトだとDebug用のlibとRelease用のlibが衝突するので名前を変更
execute_process(COMMAND 
    ${CMAKE_COMMAND} -E rename ${DIRECTXTEX_INSTALL_PREFIX}/lib/DirectXTex.lib ${DIRECTXTEX_INSTALL_PREFIX}/lib/Debug/DirectXTex.lib 
)

execute_process(COMMAND
    ${CMAKE_COMMAND} --build ${DIRECTXTEX_BINARY_DIR} --config Release
)
execute_process(COMMAND
    ${CMAKE_COMMAND} --install ${DIRECTXTEX_BINARY_DIR} --config Release
)
execute_process(COMMAND 
    ${CMAKE_COMMAND} -E rename ${DIRECTXTEX_INSTALL_PREFIX}/lib/DirectXTex.lib ${DIRECTXTEX_INSTALL_PREFIX}/lib/Release/DirectXTex.lib 
)

execute_process(COMMAND
    ${CMAKE_COMMAND} --build ${DIRECTXTEX_BINARY_DIR} --config RelWithDebInfo
)
execute_process(COMMAND
    ${CMAKE_COMMAND} --install ${DIRECTXTEX_BINARY_DIR} --config RelWithDebInfo
)
execute_process(COMMAND 
    ${CMAKE_COMMAND} -E rename ${DIRECTXTEX_INSTALL_PREFIX}/lib/DirectXTex.lib ${DIRECTXTEX_INSTALL_PREFIX}/lib/RelWithDebInfo/DirectXTex.lib 
)

## PATCH: 偏光した名前に対応して, ターゲット名も変更
set(DIRECTXTEX_TARGETS_DEBUG_CMAKE "")
file(READ ${DIRECTXTEX_INSTALL_PREFIX}/share/directxtex/DirectXTex-targets-debug.cmake DIRECTXTEX_TARGETS_DEBUG_CMAKE)
string(REPLACE "DirectXTex.lib" "Debug/DirectXTex.lib" DIRECTXTEX_TARGETS_DEBUG_CMAKE ${DIRECTXTEX_TARGETS_DEBUG_CMAKE})
file(WRITE ${DIRECTXTEX_INSTALL_PREFIX}/share/directxtex/DirectXTex-targets-debug.cmake ${DIRECTXTEX_TARGETS_DEBUG_CMAKE})

set(DIRECTXTEX_TARGETS_RELEASE_CMAKE "")
file(READ ${DIRECTXTEX_INSTALL_PREFIX}/share/directxtex/DirectXTex-targets-release.cmake DIRECTXTEX_TARGETS_RELEASE_CMAKE)
string(REPLACE "DirectXTex.lib" "Release/DirectXTex.lib" DIRECTXTEX_TARGETS_RELEASE_CMAKE ${DIRECTXTEX_TARGETS_RELEASE_CMAKE})
file(WRITE ${DIRECTXTEX_INSTALL_PREFIX}/share/directxtex/DirectXTex-targets-release.cmake ${DIRECTXTEX_TARGETS_RELEASE_CMAKE})


set(DIRECTXTEX_TARGETS_RELWITHDEBINFO_CMAKE "")
file(READ ${DIRECTXTEX_INSTALL_PREFIX}/share/directxtex/DirectXTex-targets-relwithdebinfo.cmake DIRECTXTEX_TARGETS_RELWITHDEBINFO_CMAKE)
string(REPLACE "DirectXTex.lib" "RelWithDebInfo/DirectXTex.lib" DIRECTXTEX_TARGETS_RELWITHDEBINFO_CMAKE ${DIRECTXTEX_TARGETS_RELWITHDEBINFO_CMAKE})
file(WRITE ${DIRECTXTEX_INSTALL_PREFIX}/share/directxtex/DirectXTex-targets-relwithdebinfo.cmake ${DIRECTXTEX_TARGETS_RELWITHDEBINFO_CMAKE})