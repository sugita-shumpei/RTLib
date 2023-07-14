#ifndef RTLIB_SCENE_TRANSFORM_GRAPH__H
#define RTLIB_SCENE_TRANSFORM_GRAPH__H
#include <RTLib/Scene/Component.h>
#include <RTLib/Core/Transform.h>
#include <vector>
namespace RTLib
{
	namespace Scene
	{
		struct Transform;
		namespace Internals
		{
			struct TransformGraphNode;
			struct TransformGraph
			{
				 TransformGraph() noexcept ;
				~TransformGraph() noexcept ;

				void attach_child(std::shared_ptr<RTLib::Scene::Transform>     transform);

				auto get_num_children() const noexcept    -> UInt32 ;
				auto get_child(UInt32 idx) const noexcept -> std::shared_ptr<RTLib::Scene::Transform>;

			private:
				std::shared_ptr<RTLib::Scene::Internals::TransformGraphNode> m_Root;
			};
		}

	}
}
#endif