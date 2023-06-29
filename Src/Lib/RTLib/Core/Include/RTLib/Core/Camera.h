#ifndef RTLIB_CORE_CAMERA__H
#define RTLIB_CORE_CAMERA__H

#ifndef __CUDACC__
#include <RTLib/Core/Matrix.h>
#include <RTLib/Core/Vector.h>
#include <RTLib/Core/Quaternion.h>
#include <RTLib/Core/Json.h>
namespace RTLib
{
	inline namespace Core
	{
		struct Camera
		{
			 Camera(const Vector3& scaling_ = Vector3(1.0f), const Quat& rotation_ = Quat(), const Vector3& position_ = Vector3(0.0f), Float32 fovy_ = 30.0f, Float32 aspect_ = 1.0f, Float32 zNear_ = 0.01f, Float32 zFa_r = 100.0f) noexcept;
			~Camera() noexcept {}

			Camera(const Camera&)noexcept = default;
			Camera& operator=(const Camera&)noexcept = default;
			// set transform
			auto set_position(const Vector3& position_)  noexcept-> Camera& { position = position_; return *this; }
			auto set_scaling(const Vector3& scale) noexcept -> Camera& { scaling = scale; return *this; }
			auto set_rotation(Quat rotation_) noexcept -> Camera& { rotation = rotation_; return *this; }
			// get transfrom
			auto get_position()const noexcept -> Vector3 { return position; }
			auto get_scaling() const noexcept -> Vector3 { return scaling; }
			auto get_rotation()const noexcept -> Quat { return rotation; }
			// set camera
			auto set_fovy(Float32 fovy)  noexcept-> Camera& { fovY = fovy; return *this; }
			auto set_aspect(Float32 aspect_)  noexcept-> Camera& { aspect = aspect_; return *this; }
			auto set_znear(Float32 znear_)  noexcept-> Camera& { zNear = znear_; return *this; }
			auto set_zfar(Float32 zfar_)  noexcept-> Camera& { zFar = zfar_; return *this; }
			// get camera
			auto get_fovy() const noexcept -> Float32 { return fovY; }
			auto get_aspect() const noexcept -> Float32 { return aspect; }
			auto get_znear() const noexcept -> Float32 { return zNear; }
			auto get_zfar() const noexcept -> Float32 { return zFar; }
			// Proj * View * Model
			Matrix4x4 get_view_matrix() const noexcept;
			Matrix4x4 get_proj_matrix() const noexcept;
			// camera coordinate -> world coordinate
			auto transform_camera_to_world_point(const Vector3& p) const noexcept -> Vector3;
			auto transform_camera_to_world_vector(const Vector3& v) const noexcept -> Vector3;
			auto transform_camera_to_world_direction(const Vector3& d) const noexcept -> Vector3;
			// world coordinate -> camera coordinate
			auto transform_world_to_camera_point(const Vector3& p) const noexcept -> Vector3;
			auto transform_world_to_camera_vector(const Vector3& v) const noexcept -> Vector3;
			auto transform_world_to_camera_direction(const Vector3& d) const noexcept -> Vector3;
			// front
			auto set_front(Vector3 front, const Vector3& vup = Vector3(0.0f, 1.0f, 0.0f)) noexcept-> Camera&;
			// transform
			Vector3 get_right() const noexcept { return transform_camera_to_world_vector(Vector3( 1.0f, 0.0f, 0.0f)); }
			Vector3 get_left () const noexcept { return transform_camera_to_world_vector(Vector3(-1.0f, 0.0f, 0.0f)); }

			Vector3 get_up()    const noexcept { return transform_camera_to_world_vector(Vector3(0.0f, 1.0f, 0.0f)); }
			Vector3 get_down()  const noexcept { return transform_camera_to_world_vector(Vector3(0.0f,-1.0f, 0.0f)); }

			Vector3 get_front() const noexcept { return transform_camera_to_world_vector(Vector3(0.0f, 0.0f, 1.0f)); }
			Vector3 get_back()  const noexcept { return transform_camera_to_world_vector(Vector3(0.0f, 0.0f,-1.0f)); }

			Vector3 scaling;
			Quat    rotation;
			Vector3 position;

			Float32 fovY  ;
			Float32 aspect;
			Float32 zNear ;
			Float32 zFar  ;
		};

		void to_json(Json& json, const Camera& camera);
		void from_json(const Json& json, Camera& camera);
	}
}
#endif

#endif
