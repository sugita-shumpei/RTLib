#ifndef RTLIB_SCENE_CAMERA__H
#define RTLIB_SCENE_CAMERA__H
#include <RTLib/Scene/ObjectTypeID.h>
#include <RTLib/Scene/Component.h>
#include <RTLib/Scene/Transform.h>
#include <RTLib/Core/Camera.h>
namespace RTLib
{
	//namespace Scene
	//{
	//	struct Camera;
	//}
	//inline namespace Core {
	//	
	//	RTLIB_CORE_DEFINE_OBJECT_TYPE_ID_2(Scene::Camera, SceneCamera, "129D57C2-A275-4956-BB72-F7FF87ACCFCF");
	//}
	RTLIB_SCENE_DEFINE_OBJECT_TYPE_ID(Camera, "129D57C2-A275-4956-BB72-F7FF87ACCFCF");
	namespace Scene
	{
		struct Object;
		struct Camera : public RTLib::Scene::Component
		{
			static auto New(std::shared_ptr<RTLib::Scene::Object> object) -> std::shared_ptr < RTLib::Scene::Camera>;
			virtual ~Camera() noexcept;
			// Derive From RTLib::Core::Object
			virtual auto query_object(const TypeID& typeID) -> std::shared_ptr<RTLib::Core::Object> override;
			virtual auto get_type_id() const noexcept -> TypeID override;
			virtual auto get_name() const noexcept -> String override;
			// Derive From RTLib::Scene::Component
			virtual auto get_transform() -> std::shared_ptr<RTLib::Scene::Transform> override;
			virtual auto get_object() -> std::shared_ptr<RTLib::Scene::Object> override;

			auto get_position() const noexcept -> Vector3;
			auto get_rotation() const noexcept -> Quat;
			auto get_scaling() const noexcept  -> Vector3;

			auto get_proj_matrix() const noexcept -> Matrix4x4;
			auto get_view_matrix() const noexcept -> Matrix4x4;

			auto get_type() const noexcept -> CameraType { return m_Camera.get_type(); }
			void set_type(CameraType type) noexcept { m_Camera.set_type(type); }

			auto get_aspect() const noexcept -> Float32 { return m_Camera.get_aspect(); }
			void set_aspect(Float32 aspect) noexcept { m_Camera.set_aspect(aspect); }

			auto get_field_of_view() const noexcept -> Float32 { return m_Camera.get_field_of_view(); }
			void set_field_of_view(Float32 fieldOfView) noexcept { m_Camera.set_field_of_view(fieldOfView); }

			auto get_orthographics_size() const noexcept -> Float32 { return m_Camera.get_orthographics_size(); }
			void set_orthographics_size(Float32 orthographicsSize) noexcept { m_Camera.set_orthographics_size(orthographicsSize); }

			auto get_near_clip_plane() const noexcept -> Float32 { return m_Camera.get_near_clip_plane(); }
			void set_near_clip_plane(Float32 nearClipPlane)noexcept { m_Camera.set_near_clip_plane(nearClipPlane); }

			auto get_far_clip_plane() const noexcept -> Float32 { return m_Camera.get_far_clip_plane(); }
			void set_far_clip_plane(Float32 farClipPlane)noexcept { return m_Camera.set_far_clip_plane(farClipPlane); }
		private:
			auto internal_get_transform() const noexcept -> std::shared_ptr<RTLib::Scene::Transform>;
			auto internal_get_object() const noexcept -> std::shared_ptr<RTLib::Scene::Object>;
		private:
			Camera(std::shared_ptr<RTLib::Scene::Object> object);
		private:
			std::weak_ptr<RTLib::Scene::Object> m_Object;
			Core::Camera m_Camera = {};
		};
	}
}
#endif
