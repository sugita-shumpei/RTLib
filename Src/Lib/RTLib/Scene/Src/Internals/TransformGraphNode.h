#ifndef RTLIB_SCENE_INTERNAL_TRANSFORM_GRAPH__H
#define RTLIB_SCENE_INTERNAL_TRANSFORM_GRAPH__H
#include <RTLib/Scene/Component.h>
#include <RTLib/Core/Transform.h>
#include <vector>
namespace RTLib
{
	namespace Scene
	{
		namespace Internals
		{

			struct TransformGraphNode :public std::enable_shared_from_this<RTLib::Scene::Internals::TransformGraphNode>
			{
				static auto New(
					std::shared_ptr<RTLib::Scene::Transform>       transform = nullptr,
					std::shared_ptr<RTLib::Scene::Internals::TransformGraphNode> parent = nullptr
				) -> std::shared_ptr<RTLib::Scene::Internals::TransformGraphNode>;

				virtual ~TransformGraphNode() noexcept;

				auto get_transform() const noexcept -> std::shared_ptr<Scene::Transform>;

				auto attach_child(std::shared_ptr<RTLib::Scene::Transform> transform) -> std::shared_ptr<RTLib::Scene::Internals::TransformGraphNode>;
				auto remove_child(std::shared_ptr<RTLib::Scene::Internals::TransformGraphNode>       node) -> std::shared_ptr<RTLib::Scene::Transform>;

				auto get_num_children() const noexcept -> UInt32 { return m_Children.size(); }
				auto get_child(UInt32 idx) const noexcept -> std::shared_ptr<RTLib::Scene::Internals::TransformGraphNode>;

				auto get_parent() const noexcept -> std::shared_ptr<RTLib::Scene::Internals::TransformGraphNode>;

				auto get_local_to_parent_matrix() const noexcept -> Matrix4x4;

				auto get_local_position() const noexcept -> Vector3;
				auto get_local_rotation() const noexcept -> Quat;
				auto get_local_scaling()  const noexcept -> Vector3;

				auto get_cache_transform_point_matrix()     const noexcept -> Matrix4x4;
				auto get_cache_transform_vector_matrix()    const noexcept -> Matrix4x4;
				auto get_cache_transform_direction_matrix() const noexcept -> Matrix4x4;
				auto get_cache_transform_scaling()          const noexcept -> Vector3;

				void update_cache_transform() noexcept;

				bool contain_in_graph(const std::shared_ptr<RTLib::Scene::Internals::TransformGraphNode>& node) const noexcept;
			private:
				TransformGraphNode(
					std::shared_ptr<RTLib::Scene::Transform>       transform = nullptr,
					std::shared_ptr<RTLib::Scene::Internals::TransformGraphNode> parent = nullptr
				);
			private:
				Matrix4x4 m_CacheTransformPointMatrix;
				Matrix4x4 m_CacheTransformVectorMatrix;
				Matrix4x4 m_CacheTransformDirectionMatrix;
				Vector3 m_CacheTransformScaling;
				std::shared_ptr<RTLib::Scene::Object> m_Object;
				std::vector<std::shared_ptr<RTLib::Scene::Internals::TransformGraphNode>>  m_Children;
				std::weak_ptr<RTLib::Scene::Internals::TransformGraphNode> m_Parent;
			};

			using  TransformGraphNodePtr = std::shared_ptr<TransformGraphNode>;
		}
	}
}
#endif
