#ifndef RTLIB_CORE_SCENE_COMPONENT__H
#define RTLIB_CORE_SCENE_COMPONENT__H
#include <RTLib/Core/Object.h>
namespace RTLib
{
	inline namespace Core
	{
		struct SceneObject;
		struct SceneTransform;

		RTLIB_CORE_DEFINE_OBJECT_TYPE_ID(SceneComponent, "5EFF206F-F556-4615-B1B5-603F71779BCF");
		struct SceneComponent :public Object
		{
			SceneComponent() noexcept : Object() {}
			virtual ~SceneComponent() noexcept {}

			virtual auto get_transform() -> std::shared_ptr<SceneTransform> = 0;
			virtual auto get_object() -> std::shared_ptr<SceneObject> = 0;
		};
	}
}
#endif
