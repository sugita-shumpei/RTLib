#ifndef RTLIB_UTILS_MATH_H
#define RTLIB_UTILS_MATH_H
#include <cmath>
#include <array>
#define RTLIB_UTILS_MATH_CONSTANTS_PI 3.14159265358979323846264338327950288
namespace RTLib {
	namespace Utils
	{
		inline auto Add(const std::array<float, 2>& v0, const std::array<float, 2>& v1)noexcept->std::array<float, 2> {
			return std::array<float, 2>{
				v0[0] + v1[0],
				v0[1] + v1[1],
			};
		}
		inline auto Sub(const std::array<float,2>& v0, const std::array<float, 2>& v1)noexcept->std::array<float, 2> {
			return std::array<float, 2>{
				v0[0] - v1[0],
				v0[1] - v1[1],
			};
		}
		inline auto Mul(const std::array<float, 2>& v0, const std::array<float, 2>& v1)noexcept->std::array<float, 2> {
			return std::array<float, 2>{
				v0[0] * v1[0],
				v0[1] * v1[1],
			};
		}
		inline auto Div(const std::array<float, 2>& v0, const std::array<float, 2>& v1)noexcept->std::array<float, 2> {
			return std::array<float, 2>{
				v0[0] / v1[0],
				v0[1] / v1[1],
			};
		}
		inline auto Len(const std::array<float, 2>& v)noexcept -> float { return std::sqrt(v[0]* v[0]+ v[1]* v[1]); }
		inline auto Normalize(const std::array<float, 2>& v)->std::remove_reference<decltype(v)>::type { auto len = Len(v); return Div(v, std::array<float, 2>{len, len}); }

		inline auto Add(const std::array<float, 3>& v0, const std::array<float, 3>& v1)noexcept->std::array<float, 3> {
			return std::array<float, 3>{
				v0[0] + v1[0],
				v0[1] + v1[1],
				v0[2] + v1[2],
			};
		}
		inline auto Sub(const std::array<float, 3>& v0, const std::array<float, 3>& v1)noexcept->std::array<float, 3> {
			return std::array<float, 3>{
				v0[0] - v1[0],
				v0[1] - v1[1],
				v0[2] - v1[2],
			};
		}
		inline auto Mul(const std::array<float, 3>& v0, const std::array<float, 3>& v1)noexcept->std::array<float, 3> {
			return std::array<float, 3>{
				v0[0] * v1[0],
				v0[1] * v1[1],
				v0[2] * v1[2],
			};
		}
		inline auto Div(const std::array<float, 3>& v0, const std::array<float, 3>& v1)noexcept->std::array<float, 3> {
			return std::array<float, 3>{
				v0[0] / v1[0],
				v0[1] / v1[1],
				v0[2] / v1[2],
			};
		}
		inline auto Len(const std::array<float, 3>& v)noexcept -> float { return std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]); }
		inline auto Normalize(const std::array<float, 3>& v)->std::remove_reference<decltype(v)>::type { auto len = Len(v); return Div(v, std::array<float, 3>{len, len, len}); }
		inline auto Cross(const std::array<float, 3>& v0, const std::array<float, 3>& v1)noexcept->std::array<float, 3> {
			return std::array<float, 3>{
				v0[1] * v1[2] - v0[2] * v1[1],
				v0[2] * v1[0] - v0[0] * v1[2],
				v0[0] * v1[1] - v0[1] * v1[0]
			};
		}

		inline auto Add(const std::array<float, 4>&   v0, const std::array<float, 4>& v1)noexcept->std::array<float, 4> {
			return std::array<float, 4>{
				v0[0] + v1[0],
				v0[1] + v1[1],
				v0[2] + v1[2],
				v0[3] + v1[3],
			};
		}
		inline auto Sub(const std::array<float, 4>& v0, const std::array<float, 4>& v1)noexcept->std::array<float, 4> {
			return std::array<float, 4>{
				v0[0] - v1[0],
				v0[1] - v1[1],
				v0[2] - v1[2],
				v0[3] - v1[3],
			};
		}
		inline auto Mul(const std::array<float, 4>& v0, const std::array<float, 4>& v1)noexcept->std::array<float, 4> {
			return std::array<float, 4>{
				v0[0] * v1[0],
				v0[1] * v1[1],
				v0[2] * v1[2],
				v0[3] * v1[3],
			};
		}
		inline auto Div(const std::array<float, 4>& v0, const std::array<float, 4>& v1)noexcept->std::array<float, 4> {
			return std::array<float, 4>{
				v0[0] / v1[0],
				v0[1] / v1[1],
				v0[2] / v1[2],
				v0[3] / v1[3],
			};
		}
		inline auto Len(const std::array<float, 4>& v)noexcept -> float { return std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2] + v[3] * v[3]); }
		inline auto Normalize(const std::array<float, 4>& v)->std::remove_reference<decltype(v)>::type { auto len = Len(v); return Div(v, std::array<float, 4>{len, len, len, len}); }

	}
}
#endif