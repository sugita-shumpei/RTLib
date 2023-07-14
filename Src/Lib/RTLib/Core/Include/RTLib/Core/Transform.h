#ifndef RTLIB_CORE_TRANSFORM__H
#define RTLIB_CORE_TRANSFORM__H
#include <RTLib/Core/Vector.h>
#include <RTLib/Core/Matrix.h>
#include <RTLib/Core/Quaternion.h>
#include <RTLib/Core/Transform.h>
namespace RTLib
{
	namespace Core
	{
		struct Transform
		{
			Transform(const Vector3& position_ = Vector3(0.0f), const Quat& rotation_ = Quat(1.0f, 0.0f, 0.0f, 0.0f), const Vector3& scaling_ = Vector3(1.0f)) noexcept
				:position{ position_ }, rotation{ rotation_ }, scaling{ scaling_ } {}

			Transform(const Transform& transform) noexcept = default;
			Transform& operator=(const Transform& transform) noexcept = default;


			static auto from_translation(const Vector3& offset_) noexcept -> Transform {
				Transform tran; tran.position = offset_; return tran;
			}
			static auto from_rotation(const Quat& rotation_ )    noexcept -> Transform {
				Transform tran; tran.rotation = rotation_; return tran;
			}
			static auto from_scaling(const Vector3& scaling_)    noexcept -> Transform {
				Transform tran; tran.scaling  = scaling_; return tran;
			}

			auto get_position() const noexcept -> const Vector3& { return position; }
			auto get_rotation() const noexcept -> const Quat   & { return rotation; }
			auto get_scaling () const noexcept -> const Vector3& { return scaling ; }

			void set_position(const Vector3& position_) noexcept { position = position_; }
			void set_rotation(const Quat   & rotation_) noexcept { rotation = rotation_; }
			void set_scaling (const Vector3& scaling_ ) noexcept { position = scaling_; }

			auto get_local_to_parent_matrix() const noexcept -> Matrix4x4;
			auto get_parent_to_local_matrix() const noexcept -> Matrix4x4;

			Vector3 position;
			Quat    rotation;
			Vector3 scaling;
		};
	}
}
#endif
