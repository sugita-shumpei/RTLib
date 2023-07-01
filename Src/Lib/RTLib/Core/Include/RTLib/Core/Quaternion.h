#ifndef RTLIB_CORE_QUATERNION__H
#define RTLIB_CORE_QUATERNION__H

#include <RTLib/Core/Vector.h>
#include <RTLib/Core/Json.h>

#ifndef __CUDACC__
#include <glm/glm.hpp>
#include <glm/gtx/quaternion.hpp>
#endif

namespace RTLib
{
	namespace Core
	{
#ifndef __CUDACC__
		using Quat     = glm::quat;
		using Quat_F32 = glm::f32quat;
		using Quat_F64 = glm::f64quat;
#else
		template<typename T>
		struct BasicQuatTraits;
		template<>
		struct BasicQuatTraits<RTLib::Core::Float32>
		{
			using Vec3 = RTLib::Core::Vector3_F32;
			using Vec4 = RTLib::Core::Vector3_F64;
		};
		template<>
		struct BasicQuatTraits<RTLib::Core::Float32>
		{
			using Vec3 = RTLib::Core::Vector3_F64;
			using Vec4 = RTLib::Core::Vector3_F64;
		};

		template<typename T>
		struct BasicQuat {
			using Vec3 = typename BasicQuatTraits<T>::Vec3;
			using Vec4 = typename BasicQuatTraits<T>::Vec4;

			BasicQuat(const BasicQuat&) = default;
			BasicQuat& operator=(const BasicQuat&) = default;

			auto operator+(const BasicQuat& q) const noexcept -> BasicQuat
			{
				return BasicQuat{ x + q.x,y + q.y,z + q.z,w + q.w };
			}

			auto operator-(const BasicQuat& q) const noexcept -> BasicQuat
			{
				return BasicQuat{ x - q.x,y - q.y,z - q.z,w - q.w };
			}

			auto operator*(const BasicQuat& q) const noexcept -> BasicQuat {
				// i * j = k
				// j * k = i
				// k * i = j
				BasicQuat res;
				res.x = (w * q.x + x * q.w) + (y * q.z - z * q.y);
				res.y = (w * q.y + y * q.w) + (z * q.x - x * q.z);
				res.z = (w * q.z + z * q.w) + (x * q.y - y * q.x);
				res.w = (w * q.w - x * q.x - y * q.y - z * q.z);
				return res;
			}

			auto operator*(const Vec3& v) const noexcept-> Vec3 {
				using namespace otk;
				// i * j = k
				// j * k = i
				// k * i = j
				auto r = Vec3{ x,y,z };
				return (w * w - dot(r,r)) * v + static_cast<T>(2) * dot(v,r) * v + w * cross(v,r);
			}

			auto operator*(T s) const noexcept -> BasicQuat {
				return { w * s, x * s, y * s, z * s };
			}

			auto operator/(const BasicQuat& q) const noexcept -> BasicQuat {
				// i * j = k
				// j * k = i
				// k * i = j
				T invDet = static_cast<T>(1)/(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
				BasicQuat res;
				res.x = ((-w * q.x + x * q.w) + (-y * q.z + z * q.y)) * invDet;
				res.y = ((-w * q.y + y * q.w) + (-z * q.x + x * q.z)) * invDet;
				res.z = ((-w * q.z + z * q.w) + (-x * q.y + y * q.x)) * invDet;
				res.w = (( w * q.w + x * q.x  +   y * q.y + z * q.z)) * invDet;
				return res;
			}

			auto operator/(T s) const noexcept -> BasicQuat {
				return { w / s, x / s, y / s, z / s };
			}

			auto conjugate() const noexcept -> BasicQuat
			{
				return { w , -x, -y, -z };
			}

			auto inverse() const noexcept -> BasicQuat
			{
				T invDet = static_cast<T>(1) / (q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
				return { w* invDet, -x* invDet, -y* invDet, -z* invDet };
			}

			T w; T x; T y; T z;
		};
#endif

	}
}
#ifndef __CUDACC__
namespace glm
{
	void to_json(RTLib::Core::Json& json, const RTLib::Core::Quat& q);
	void from_json(const RTLib::Core::Json& json, RTLib::Core::Quat& q);
}
#endif

#endif
