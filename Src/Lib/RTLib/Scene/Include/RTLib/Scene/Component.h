#ifndef RTLIB_SCENE_COMPONENT__H
#define RTLIB_SCENE_COMPONENT__H
#include <RTLib/Core/Object.h>
#include <RTLib/Scene/ObjectTypeID.h>
namespace RTLib
{
	RTLIB_SCENE_DEFINE_OBJECT_TYPE_ID(Component, "5EFF206F-F556-4615-B1B5-603F71779BCF");
	namespace Scene
	{
		struct Object;
		struct Transform;

		struct Component :public RTLib::Core::Object
		{
			Component() noexcept : RTLib::Core::Object() {}
			virtual ~Component() noexcept {}

			virtual auto get_transform() -> std::shared_ptr<RTLib::Scene::Transform> = 0;
			virtual auto get_object() -> std::shared_ptr<RTLib::Scene::Object> = 0;
		};
	}
}
#endif
