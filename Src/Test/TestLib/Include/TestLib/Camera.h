#ifndef TEST_TESTLIB_CAMERA__H
#define TEST_TESTLIB_CAMERA__H
#include <OptiXToolkit/ShaderUtil/vec_math.h>
#include <glm/glm.hpp>
#include <tuple>
#include <array>
#include <string>
namespace TestLib
{
	struct Camera
	{
		Camera(
			const float3& eye    = make_float3(0.0f, 0.0f, 0.0f),
			const float3& lookat = make_float3(0.0f, 0.0f,-1.0f),
			const float3& vup    = make_float3(0.0f, 1.0f, 0.0f),
			float aspect = 1.0f,
			float fovy   = 45.0f,
			float znear  = 1e-3f,
			float zfar   = 1000.0f
		) noexcept;
		~Camera();

		auto get_uvw() const noexcept -> std::tuple<float3, float3, float3>;

		auto get_eye   () const noexcept -> float3;
		auto get_lookat() const noexcept -> float3;
		auto get_vup   () const noexcept -> float3;
		// (front, up)
		// 
		// vup * front -> right
		// front * right->up

		void set_eye   (float3    eye) noexcept;
		void set_lookat(float3 lookat) noexcept;
		void set_vup   (float3    vup) noexcept;

		auto get_front () const noexcept -> float3;
		auto get_right () const noexcept -> float3;
		auto get_up    () const noexcept -> float3;
		
		void set_front (float3  front) noexcept;
		void set_right (float3  right) noexcept;
		void set_up    (float3     up) noexcept;

		auto get_aspect() const noexcept -> float ;
		auto get_fovy  () const noexcept -> float ;
		auto get_znear () const noexcept -> float ;
		auto get_zfar  () const noexcept -> float ;

		void set_aspect(float aspect) noexcept;
		void set_fovy  (float fovy  ) noexcept;
		void set_znear (float znear ) noexcept;
		void set_zfar  (float zfar  ) noexcept;

	private:
	private:
		float3 m_Eye   ;
		float3 m_Lookat;

		float3 m_Right ;
		float3 m_Up    ;
		float3 m_Vup   ;

		float  m_Aspect;
		float  m_Fovy  ;
		float  m_ZNear ;
		float  m_ZFar  ;
	};
}
#endif
