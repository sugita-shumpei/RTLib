#include <Test1_Camera.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

Test1::Camera::Camera(const float3& eye, const float3& lookat, const float3& vup, float aspect, float fovy, float znear, float zfar) noexcept
	:m_Eye{eye},m_Lookat{lookat},m_Vup{vup},m_Aspect{aspect},m_Fovy{fovy},m_ZNear{znear},m_ZFar{zfar}
{}

Test1::Camera::~Camera()
{
}

auto Test1::Camera::get_uvw() const noexcept -> std::tuple<float3, float3, float3>
{
	using namespace otk;
	auto front = m_Lookat - m_Eye;   // z
	auto right = cross(m_Vup, front);// x
	auto top   = cross(front, right);// y
	
	auto range = m_ZNear * std::tanf(glm::radians(m_Fovy * 0.5f));

	auto w = m_Aspect * range;
	auto h = range;

	right *= w;
	top   *= h;
	front *= m_ZNear;

	return std::make_tuple(right, top, front);
}

auto Test1::Camera::get_eye() const noexcept -> float3
{
	return m_Eye;
}

auto Test1::Camera::get_lookat() const noexcept -> float3
{
	return m_Lookat;
}

auto Test1::Camera::get_vup() const noexcept -> float3
{
	return m_Vup;
}

void Test1::Camera::set_eye(float3 eye) noexcept
{
	m_Eye = eye;
}

void Test1::Camera::set_lookat(float3 lookat) noexcept
{
	m_Lookat = lookat;
}

void Test1::Camera::set_vup(float3 up) noexcept
{
	m_Vup = up;
}

auto Test1::Camera::get_aspect() const noexcept -> float
{
	return m_Aspect;
}

auto Test1::Camera::get_fovy() const noexcept -> float
{
	return m_Fovy;
}

auto Test1::Camera::get_znear() const noexcept -> float
{
	return m_ZNear;
}

auto Test1::Camera::get_zfar() const noexcept -> float
{
	return m_ZFar;
}

void Test1::Camera::set_aspect(float aspect) noexcept
{
	m_Aspect = aspect;
}

void Test1::Camera::set_fovy(float fovy) noexcept
{
	m_Fovy = fovy;
}

void Test1::Camera::set_znear(float znear) noexcept
{
	m_ZNear = znear;
}

void Test1::Camera::set_zfar(float zfar) noexcept
{
	m_ZFar = zfar;
}
