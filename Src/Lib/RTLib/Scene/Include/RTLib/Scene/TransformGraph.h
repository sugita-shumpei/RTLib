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
		struct TransformGraphNode :public std::enable_shared_from_this<TransformGraphNode>
		{
			static auto New(
				std::shared_ptr<Scene::Transform>       transform = nullptr,
				std::shared_ptr<Scene::TransformGraphNode> parent = nullptr
			) -> std::shared_ptr<Scene::TransformGraphNode>;

			virtual ~TransformGraphNode() noexcept;

			auto get_transform() const noexcept -> std::shared_ptr<Scene::Transform>;

			auto attach_child(std::shared_ptr<Scene::Transform>           transform) -> std::shared_ptr<Scene::TransformGraphNode>;
			auto remove_child(std::shared_ptr<Scene::TransformGraphNode>       node) -> std::shared_ptr<Scene::Transform>;

			auto get_num_children() const noexcept -> UInt32 { return m_Children.size(); }
			auto get_child(UInt32 idx) const noexcept -> std::shared_ptr<Scene::TransformGraphNode>;

			auto get_parent() const noexcept -> std::shared_ptr<Scene::TransformGraphNode>;

			auto get_local_to_parent_matrix() const noexcept -> Matrix4x4;

			auto get_local_position() const noexcept -> Vector3;
			auto get_local_rotation() const noexcept -> Quat;
			auto get_local_scaling()  const noexcept -> Vector3;

			auto get_cache_transform_point_matrix()     const noexcept -> Matrix4x4;
			auto get_cache_transform_vector_matrix()    const noexcept -> Matrix4x4;
			auto get_cache_transform_direction_matrix() const noexcept -> Matrix4x4;
			auto get_cache_transform_scaling()          const noexcept -> Vector3;

			void update_cache_transform() noexcept;
		private:
			TransformGraphNode(
				std::shared_ptr<Scene::Transform>       transform = nullptr,
				std::shared_ptr<Scene::TransformGraphNode> parent = nullptr
			);
		private:
			Matrix4x4                                                m_CacheTransformPointMatrix;
			Matrix4x4                                                m_CacheTransformVectorMatrix;
			Matrix4x4                                                m_CacheTransformDirectionMatrix;
			Vector3                                                  m_CacheTransformScaling;
			std::shared_ptr<Scene::Object>                           m_Object;
			std::vector<std::shared_ptr<Scene::TransformGraphNode>>  m_Children;
			std::weak_ptr<Scene::TransformGraphNode>                 m_Parent;
		};

		using  TransformGraphNodePtr = std::shared_ptr<TransformGraphNode>;
		struct TransformGraph
		{
			TransformGraph() noexcept :m_Root{ TransformGraphNode::New() } {}
			~TransformGraph() noexcept {}

			void attach_child(std::shared_ptr<Scene::Transform>     transform)
			{
				if (!transform) { return; }
				m_Root->attach_child(transform);
			}

			auto get_num_children() const noexcept    -> UInt32 { return m_Root->get_num_children(); }
			auto get_child(UInt32 idx) const noexcept -> std::shared_ptr<Scene::Transform>
			{
				auto child = m_Root->get_child(idx);
				if (child) {
					return child->get_transform();
				}
				else {
					return nullptr;
				}
			}
		private:
			TransformGraphNodePtr m_Root;
		};

	}
}
#endif