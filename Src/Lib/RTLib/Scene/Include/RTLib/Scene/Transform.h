#ifndef RTLIB_SCENE_TRANSFORM__H
#define RTLIB_SCENE_TRANSFORM__H
#include <RTLib/Scene/Component.h>
#include <RTLib/Core/Transform.h>
#include <vector>
namespace RTLib
{
	RTLIB_SCENE_DEFINE_OBJECT_TYPE_ID(Transform, "F334BA1D-67D7-4047-A09F-0CE07F4D3B78");
	namespace Scene
	{
		namespace Internals
		{
			struct TransformGraphNode;
		}
		struct AnimateTransform;
		struct Animator;
		// 参照カウント
		// SceneTransformNodeが割り当てられていない場合: 0 or 1
		// SceneTransformNodeが割り当てられている場合  : 2 以上
		struct Transform : public RTLib::Scene::Component
		{
			friend class Internals::TransformGraphNode;
			friend class AnimateTransform;
			friend class Animator;

			Transform(
				std::shared_ptr<RTLib::Scene::Object> object,
				const RTLib::Core::Transform& localTransform = {}
			);
			virtual ~Transform() noexcept;
			// Derive From RTLib::Core::Object
			virtual auto query_object(const TypeID& typeID) -> std::shared_ptr<Core::Object> override;
			virtual auto get_type_id() const noexcept -> TypeID override;
			virtual auto get_name() const noexcept -> String override;
			// Derive From RTLib::Scene::Component
			virtual auto get_transform() -> std::shared_ptr<RTLib::Scene::Transform> override;
			virtual auto get_object() -> std::shared_ptr<RTLib::Scene::Object>    override;

			auto get_child_count() const noexcept -> UInt32;
			auto get_child(UInt32 idx) const noexcept -> std::shared_ptr<RTLib::Scene::Transform>;

			auto get_parent() const noexcept -> std::shared_ptr<RTLib::Scene::Transform>;
			void set_parent(std::shared_ptr<RTLib::Scene::Transform> parent);

			auto get_local_position() const noexcept -> Vector3 { return m_Local.position; }
			auto get_local_rotation() const noexcept -> Quat { return m_Local.rotation; }
			auto get_local_scaling()  const noexcept -> Vector3 { return m_Local.scaling; }

			void set_local_position(const Vector3& localPosition) noexcept;
			void set_local_rotation(const Quat& localRotation) noexcept;
			void set_local_scaling(const Vector3& localScaling) noexcept;

			auto get_position() const noexcept -> Vector3;
			auto get_rotation() const noexcept -> Quat;
			auto get_scaling() const noexcept -> Vector3;

			void set_position(const Vector3& position) noexcept;
			void set_rotation(const Quat& rotation) noexcept;
			void set_scaling(const Vector3& scaling) noexcept;

			auto get_local_to_parent_matrix() const noexcept -> Matrix4x4;
			auto get_parent_to_local_matrix() const noexcept -> Matrix4x4;

			auto get_local_to_world_matrix() const noexcept -> Matrix4x4;
			auto get_wolrd_to_local_matrix() const noexcept -> Matrix4x4;

			bool contain_in_graph(const std::shared_ptr<RTLib::Scene::Transform>& transform) const noexcept;
		private:
			auto internal_get_node() const noexcept-> std::shared_ptr<RTLib::Scene::Internals::TransformGraphNode>;
			auto internal_get_parent_node() const noexcept-> std::shared_ptr<RTLib::Scene::Internals::TransformGraphNode>;
			void internal_set_node(std::shared_ptr<RTLib::Scene::Internals::TransformGraphNode> node);
			auto internal_get_parent_cache_point_matrix() const noexcept -> Matrix4x4;
			auto internal_get_parent_cache_vector_matrix() const noexcept -> Matrix4x4;
			auto internal_get_parent_cache_direction_matrix() const noexcept -> Matrix4x4;
			auto internal_get_parent_cache_scaling() const noexcept -> Vector3;
			void internal_set_local_position_without_update(const Vector3& localPosition)noexcept;
			void internal_set_local_rotation_without_update(const Quat& localRotation)noexcept;
			void internal_set_local_scaling_without_update(const Vector3& localScaling)noexcept;
			void internal_update_cache();
		private:
			std::weak_ptr<RTLib::Scene::Object> m_Object;
			std::weak_ptr<RTLib::Scene::Internals::TransformGraphNode>   m_Node;
			Core::Transform m_Local;
		};
		// SCENE TRANSFORM GRAPH -> SCENE OBJECT ->
		using TransformPtr = std::shared_ptr<RTLib::Scene::Transform>;
	}
}
#endif
