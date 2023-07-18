
find_package(boost_smart_ptr REQUIRED)
add_library(RTLib-Scene    STATIC )
target_sources(RTLib-Scene PRIVATE
	Include/RTLib/Scene/AnimateTransform.h
	Include/RTLib/Scene/Animator.h
	Include/RTLib/Scene/Component.h
	Include/RTLib/Scene/Camera.h
	Include/RTLib/Scene/Mesh.h
	Include/RTLib/Scene/Object.h
	Include/RTLib/Scene/ObjectTypeID.h
	Include/RTLib/Scene/Scene.h
	Include/RTLib/Scene/Transform.h
	
	Src/Animator.cpp
	Src/AnimateTransform.cpp
	Src/Component.cpp
	Src/Camera.cpp
	Src/Mesh.cpp
	Src/Object.cpp
	Src/Scene.cpp
	Src/Transform.cpp

	Src/Internals/TransformGraph.h
	Src/Internals/TransformGraph.cpp

	Src/Internals/TransformGraphNode.h
	Src/Internals/TransformGraphNode.cpp
	Src/Internals/Container.h
)

target_include_directories(RTLib-Scene PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/Include)
target_link_libraries(RTLib-Scene PUBLIC RTLib-Core Boost::smart_ptr)