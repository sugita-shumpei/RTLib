#ifndef RTLIB_CORE_AABB__H
#define RTLIB_CORE_AABB__H
#include <RTLib/Core/Vector.h>
namespace RTLib
{
	inline namespace Core
	{
		template <typename T>
		struct BaseAabb
		{
			BaseAabb() noexcept :
				min{ {FLT_MAX, FLT_MAX, FLT_MAX} }
				max{ {FLT_MIN, FLT_MIN, FLT_MIN} }
			{}
			BaseAabb(const BaseAabb&) noexcept = default;
			BaseAabb& operator=(const BaseAabb&) noexcept = default;

			RTLib::Coer::Bool operator!() const noexcept
			{
#ifdef __CUDACC__
				using namespace otk;
#else
				using namespace glm;
#endif
				return min(max, min) != min;
			}

			BaseAabb<T> operator|(const BaseAabb<T>& v)const noexcept
			{

				BaseAabb<T> res;
#ifdef __CUDACC__
				using namespace otk;
#else
				using namespace glm;
#endif
				res.min = min(v.min, min);
				res.max = max(v.max, max);
				return *this;
			}
			BaseAabb<T> operator&(const BaseAabb<T>& v)const noexcept
			{

				BaseAabb<T> res;
#ifdef __CUDACC__
				using namespace otk;
#else
				using namespace glm;
#endif
				res.min = max(v.min, min);
				res.max = min(v.max, max);
				return *this;
			}

			RTLib::Core::Vector3<T> min;
			RTLib::Core::Vector3<T> max;
		};

		using Aabb     = BaseAabb<RTLib::Core::Float32>;
		using Aabb_F32 = BaseAabb<RTLib::Core::Float32>;
		using Aabb_F64 = BaseAabb<RTLib::Core::Float64>;

	}
}
#endif
