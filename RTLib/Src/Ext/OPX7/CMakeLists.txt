set(RTLIB_EXT_OPX7_SOURCES 
    ${CMAKE_CURRENT_SOURCE_DIR}/Include/RTLib/Ext/OPX7/OPX7Utility.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Include/RTLib/Ext/OPX7/OPX7Common.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Include/RTLib/Ext/OPX7/OPX7Context.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Src/OPX7Context.cpp

    ${CMAKE_CURRENT_SOURCE_DIR}/Include/RTLib/Ext/CUDA/OPX7Module.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Src/OPX7Module.cpp 

    ${CMAKE_CURRENT_SOURCE_DIR}/Include/RTLib/Ext/CUDA/OPX7ProgramGroup.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Src/OPX7ProgramGroup.cpp   

    ${CMAKE_CURRENT_SOURCE_DIR}/Include/RTLib/Ext/CUDA/OPX7Pipeline.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Src/OPX7Pipeline.cpp   

    ${CMAKE_CURRENT_SOURCE_DIR}/Include/RTLib/Ext/CUDA/OPX7AccelerationStructure.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Src/OPX7AccelerationStructure.cpp  

    ${CMAKE_CURRENT_SOURCE_DIR}/Include/RTLib/Ext/CUDA/OPX7AccelerationStructureInstance.h  
    ${CMAKE_CURRENT_SOURCE_DIR}/Src/OPX7AccelerationStructureInstance.cpp   

    ${CMAKE_CURRENT_SOURCE_DIR}/Include/RTLib/Ext/CUDA/OPX7ShaderTable.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Src/OPX7ShaderTable.cpp 
)


add_library(RTLib-Ext-OPX7 
    ${RTLIB_EXT_OPX7_SOURCES}
)
target_include_directories(RTLib-Ext-OPX7 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/Include)
target_link_libraries(RTLib-Ext-OPX7 PUBLIC RTLib-Ext-CUDA)
set_target_properties(RTLib-Ext-OPX7 PROPERTIES FOLDER RTLib/RTLib/Ext/OPX7)
