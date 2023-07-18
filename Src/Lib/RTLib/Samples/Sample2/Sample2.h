#ifndef RTLIB_SAMPLE_SAMPLE2__H
#define RTLIB_SAMPLE_SAMPLE2__H
#include <memory>
#include <glm/glm.hpp>
#include <vector>
namespace RTLib
{
	namespace Sample
	{
		struct ObjectBase : public std::enable_shared_from_this<ObjectBase> {
			virtual ~ObjectBase() {}


		};
		struct Component:public ObjectBase {
			virtual ~Component() {}

		};
		struct Transform;
		struct Object :public ObjectBase
		{
			static auto New() -> std::shared_ptr<Object> {
				return std::shared_ptr<Object>(new Object());
			}
			virtual ~Object() {}

			std::shared_ptr<Transform> transform;
		};
		struct TransformGraphNode;
		struct Scene : public ObjectBase
		{
			static auto New() -> std::shared_ptr<Scene> {
				return std::shared_ptr<Scene>(new Scene());
			}
			virtual ~Scene() {}

			std::shared_ptr<TransformGraphNode> m_Root;
		};
		struct Transform:public Component
		{
			static auto New() -> std::shared_ptr<Transform> {
				return std::shared_ptr<Transform>(new Transform());
			}
			virtual ~Transform() {}

			glm::vec3 m_Position;
			glm::vec3 m_Scaling;

			std::weak_ptr<TransformGraphNode> m_Graph;
		};
		struct TransformGraphNode : public std::enable_shared_from_this<TransformGraphNode> 
		{
			static auto New() -> std::shared_ptr<TransformGraphNode> {
				return std::shared_ptr<TransformGraphNode>(new TransformGraphNode());
			}

			virtual ~TransformGraphNode() {}

			std::weak_ptr<TransformGraphNode>                m_Parent   = {};
			std::vector<std::shared_ptr<TransformGraphNode>> m_Children = {};
			std::shared_ptr<Object>                          m_Object   = {};
		};
	
	}
}
#endif
