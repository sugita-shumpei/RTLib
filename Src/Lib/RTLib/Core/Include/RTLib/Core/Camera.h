#ifndef RTLIB_CORE_CAMERA__H
#define RTLIB_CORE_CAMERA__H
#include <RTLib/Core/DataTypes.h>
#include <RTLib/Core/Matrix.h>
namespace RTLib
{
	namespace Core
	{
		enum class CameraType
		{
			ePerspective,
			eOrthogonal
		};
		struct Camera
		{
			Camera(
				CameraType type = CameraType::ePerspective,
				Float32 aspect = 1.0f,
				Float32 fieldOfViewOrorthographicsSize = 30.0f,
				Float32 nearClipPlane = 0.01f,
				Float32 farClipPlane = 1000.f
			) noexcept :
				m_Type{ type },
				m_Aspect{ aspect },
				m_FieldOfViewOrOrthographicsSize{ fieldOfViewOrorthographicsSize },
				m_NearClipPlane{ nearClipPlane },
				m_FarClipPlane{ farClipPlane } 
			{}

			Camera(const Camera& camera) noexcept = default;
			Camera& operator=(const Camera& camera) noexcept = default;

			auto get_proj_matrix() const noexcept -> Matrix4x4;

			auto get_type() const noexcept -> CameraType { return m_Type; }
			void set_type(CameraType type) noexcept { m_Type = type; }

			auto get_aspect() const noexcept -> Float32 { return m_Aspect; }
			void set_aspect(Float32 aspect) noexcept {
				m_Aspect = aspect;
			}

			auto get_field_of_view() const noexcept -> Float32 { return m_FieldOfViewOrOrthographicsSize; }
			void set_field_of_view(Float32 fieldOfView) noexcept {
				if (m_Type == CameraType::ePerspective) {
					m_FieldOfViewOrOrthographicsSize = fieldOfView;
				}
			}

			auto get_orthographics_size() const noexcept -> Float32 { return m_FieldOfViewOrOrthographicsSize; }
			void set_orthographics_size(Float32 orthographicsSize) noexcept
			{
				if (m_Type == CameraType::eOrthogonal) {
					m_FieldOfViewOrOrthographicsSize = orthographicsSize;
				}
			}

			auto get_near_clip_plane() const noexcept -> Float32 { return m_NearClipPlane; }
			void set_near_clip_plane(Float32 nearClipPlane)noexcept {
				m_NearClipPlane = nearClipPlane;
			}

			auto get_far_clip_plane() const noexcept -> Float32 { return m_FarClipPlane; }
			void set_far_clip_plane(Float32 farClipPlane)noexcept {
				m_FarClipPlane = farClipPlane;
			}
		private:
			CameraType m_Type;
			Float32 m_Aspect;
			Float32 m_FieldOfViewOrOrthographicsSize;
			Float32 m_NearClipPlane;
			Float32 m_FarClipPlane;
		};
	}
}
#endif
