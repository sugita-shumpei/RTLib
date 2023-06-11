add_library(OTK-ShaderUtil INTERFACE)
target_sources(OTK-ShaderUtil INTERFACE 
	# ${CMAKE_CURRENT_SOURCE_DIR}/.. = Ext/CMake/opitx-toolkit
	# ${CMAKE_CURRENT_SOURCE_DIR}/../../.. = Ext

	${OptiXToolkit_ROOT}/ShaderUtil/include/OptiXToolkit/ShaderUtil/vec_math.h
	${OptiXToolkit_ROOT}/ShaderUtil/include/OptiXToolkit/ShaderUtil/SelfIntersectionAvoidance.h
	${OptiXToolkit_ROOT}/ShaderUtil/include/OptiXToolkit/ShaderUtil/SelfIntersectionAvoidanceTypes.h
	${OptiXToolkit_ROOT}/ShaderUtil/include/OptiXToolkit/ShaderUtil/ray_cone.h
	${OptiXToolkit_ROOT}/ShaderUtil/include/OptiXToolkit/ShaderUtil/Preprocessor.h
	${OptiXToolkit_ROOT}/ShaderUtil/include/OptiXToolkit/ShaderUtil/OptixSelfIntersectionAvoidance.h
	${OptiXToolkit_ROOT}/ShaderUtil/include/OptiXToolkit/ShaderUtil/CudaSelfIntersectionAvoidance.h
	${OptiXToolkit_ROOT}/ShaderUtil/include/OptiXToolkit/ShaderUtil/color.h
)
target_include_directories(OTK-ShaderUtil INTERFACE ${OptiXToolkit_ROOT}/ShaderUtil/include)