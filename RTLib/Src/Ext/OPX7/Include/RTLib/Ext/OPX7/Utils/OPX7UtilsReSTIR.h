#ifndef RTLIB_EXT_OPX7_UTILS_OPX7_UTILS_RESTIR_H
#define RTLIB_EXT_OPX7_UTILS_OPX7_UTILS_RESTIR_H
#include <RTLib/Ext/CUDA/Math/VectorFunction.h>
namespace RTLib
{
	namespace Ext
	{
		namespace OPX7
		{
			namespace Utils
			{
				template<typename T>
				struct Reservoir
				{
					float        w     = 0.0f;
					float        w_sum = 0.0f;
					unsigned int m     = 0;
					T            y     = {};
					RTLIB_INLINE RTLIB_HOST_DEVICE bool Update(T x_i, float w_i, float rnd01)
					{
						w = 0.0f;
						w_sum += w_i;
						++m;
						if ((w_i / w_sum) >= rnd01)
						{
							y = x_i;
							return true;
						}
						return false;
					}

					static auto Combine(size_t N, const Reservoir<T>* pReservoirs, const float* pNewTargets, const float* rnd01s)->Reservoir<T>
					{
						Reservoir<T> s;
						size_t k = 0;
						for (size_t i = 0; i < N; ++i)
						{
							if (s.Update(pReservoirs[i].y, pNewTargets[i] * pReservoirs[i].w * pReservoirs[i].w, rnd01s[i])) {
								k = i;
							}
							s.m += pReservoirs[i].base.m;
						}
						s.w = (s.w_sum) / (s.m * pNewTargets[k]);
						return s;
					}
				};

			}
		}
	}
}
#endif