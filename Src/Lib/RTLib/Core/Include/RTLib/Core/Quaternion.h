#ifndef RTLIB_CORE_QUATERNION__H
#define RTLIB_CORE_QUATERNION__H
#include <RTLib/Core/DataTypes.h>
#include <RTLib/Core/Preprocessor.h>
#include <RTLib/Core/Vector.h>
#ifndef __CUDACC__
#include <glm/glm.hpp>
#include <glm/gtx/quaternion.hpp>
#else
#include <OptiXToolkit/ShaderUtil/vec_math.h>
#endif
namespace RTLib
{
	inline namespace Core
	{
#ifndef __CUDACC__
		using Quat_F32 = glm::f32quat;
		using Quat_F64 = glm::f64quat;
#else
		namespace internals
		{
			template<typename T>
			struct Quat
			{
				RTLIB_DEVICE Quat(T w_ = static_cast<T>(0), T x_ = static_cast<T>(0), T y_ = static_cast<T>(0), T z_ = static_cast<T>(0))noexcept
					:x{x_},y{y_},z{z_},w{w_}{}

				RTLIB_DEVICE Quat(const Quat&) noexcept = default;
				RTLIB_DEVICE Quat& operator=(const Quat&) noexcept = default;

				RTLIB_DEVICE Quat operator+(const Quat& q) const noexcept
				{
					return Quat(x + q.x, y + q.y, z + q.z, w + q.w);
				}

				RTLIB_DEVICE Quat operator-(const Quat& q) const noexcept
				{
					return Quat(x - q.x, y - q.y, z - q.z, w - q.w);
				}

				RTLIB_DEVICE Quat operator*(const Quat& q) const noexcept
				{
					return Quat(
						w * q.w - x * q.x - y * q.y - z * q.z,
						y * q.z - z * q.y + w * q.x + x * q.w,
						z * q.x - x * q.z + w * q.y + y * q.w,
						x * q.y - y * q.x + w * q.z + z * q.w
					);
				}

				RTLIB_DEVICE Quat operator/(const Quat& q) const noexcept
				{
					T invDet = static_cast<T>(1) / (q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
					return Quat(
						( w * q.w + x * q.x + y * q.y + z * q.z)* invDet,
						(-y * q.z + z * q.y - w * q.x + x * q.w)* invDet,
						(-z * q.x + x * q.z - w * q.y + y * q.w)* invDet,
						(-x * q.y + y * q.x - w * q.z + z * q.w)* invDet
					);
				}

				RTLIB_DEVICE auto operator*(const MathVector3<T>& v) const noexcept -> MathVector3<T>
				{
					auto qv = make_vector3(x, y,z);
					return (w * w - x * x - y * y - z * z) * v + 2 * dot(qv, v) * qv + 2 * w * otk::cross(qv, v);
				}

				RTLIB_DEVICE Quat operator*(T s) const noexcept
				{
					return Quat(x * s, y * s, z * s, w * s);
				}

				RTLIB_DEVICE Quat operator/(T s) const noexcept
				{
					return Quat(x / s, y / s, z / s, w / s);
				}

				T w;
				T x;
				T y;
				T z;
			};
		}
		using Quat_F32 = internals::Quat<Float32>;
		using Quat_F64 = internals::Quat<Float64>;
#endif

		using Quat = Quat_F32;

		template<typename T>
		struct MathQuatTraits;

		template<>
		struct MathQuatTraits<Float32> {
			using type = Quat_F32;
		};

		template<>
		struct MathQuatTraits<Float64> {
			using type = Quat_F64;
		};

		template<typename T>
		using MathQuat = typename MathQuatTraits<T>::type;
	}

}
#endif
