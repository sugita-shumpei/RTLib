#include <TestLib/Camera.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

TestLib::Camera::Camera(const float3& eye, const float3& lookat, const float3& vup, float aspect, float fovy, float znear, float zfar) noexcept
	:m_Eye{eye},m_Lookat{lookat},m_Aspect{aspect},m_Fovy{fovy},m_ZNear{znear},m_ZFar{zfar},m_Right{1,0,0},m_Up{0,1,0}
{
	set_vup(vup);
}

TestLib::Camera::~Camera()
{
}

auto TestLib::Camera::get_uvw() const noexcept -> std::tuple<float3, float3, float3>
{
	using namespace otk;
	auto front = get_front();   // z
	auto right = get_right();// x
	auto up    = get_up();// y
	
	auto range = m_ZNear * std::tanf(glm::radians(m_Fovy * 0.5f));

	auto w = m_Aspect * range;
	auto h = range;

	right *= w;
	up    *= h;
	front *= m_ZNear;

	return std::make_tuple(right, up, front);
}

auto TestLib::Camera::get_eye() const noexcept -> float3
{
	return m_Eye;
}

auto TestLib::Camera::get_lookat() const noexcept -> float3
{
	return m_Lookat;
}

void TestLib::Camera::set_eye(float3 eye) noexcept
{
	m_Eye = eye;
}

void TestLib::Camera::set_lookat(float3 lookat) noexcept
{
	m_Lookat = lookat;
}

void TestLib::Camera::set_vup(float3 vup) noexcept
{
	using namespace otk;

	auto lenR = length(m_Right);
	auto lenU = length(m_Up);
	auto front = get_front();

	m_Right = normalize(cross(vup, front));
	m_Up = normalize(cross(front, m_Right));
	m_Right *= lenR;
	m_Up *= lenU;
}

auto TestLib::Camera::get_front() const noexcept -> float3
{
	using namespace otk;
	return m_Lookat - m_Eye;
}

auto TestLib::Camera::get_right() const noexcept -> float3
{
	return m_Right;
}

auto TestLib::Camera::get_up() const noexcept -> float3
{
	return m_Up;
}

void TestLib::Camera::set_front(float3 front) noexcept
{
	using namespace otk;
	m_Lookat = m_Eye + front;

}

void TestLib::Camera::set_right(float3 right) noexcept
{
}

void TestLib::Camera::set_up(float3 up) noexcept
{
}

auto TestLib::Camera::get_aspect() const noexcept -> float
{
	return m_Aspect;
}

auto TestLib::Camera::get_fovy() const noexcept -> float
{
	return m_Fovy;
}

auto TestLib::Camera::get_znear() const noexcept -> float
{
	return m_ZNear;
}

auto TestLib::Camera::get_zfar() const noexcept -> float
{
	return m_ZFar;
}

void TestLib::Camera::set_aspect(float aspect) noexcept
{
	m_Aspect = aspect;
}

void TestLib::Camera::set_fovy(float fovy) noexcept
{
	m_Fovy = fovy;
}

void TestLib::Camera::set_znear(float znear) noexcept
{
	m_ZNear = znear;
}

void TestLib::Camera::set_zfar(float zfar) noexcept
{
	m_ZFar = zfar;
}
