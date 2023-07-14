#ifndef RTLIB_SCENE_SCENE__H
#define RTLIB_SCENE_SCENE__H
#include <RTLib/Core/DataTypes.h>
#include <memory>
namespace RTLib {
	namespace Scene
	{
		struct Transform;
		struct Scene
		{
			 Scene() noexcept;
			~Scene() noexcept;

			void attach_child(std::shared_ptr<RTLib::Scene::Transform>     transform);

			auto get_num_children() const noexcept    -> UInt32;
			auto get_child(UInt32 idx) const noexcept -> std::shared_ptr<RTLib::Scene::Transform>;

		private:
			struct Impl;
			std::unique_ptr<Impl> m_Impl;
		};
	}
}
#endif
