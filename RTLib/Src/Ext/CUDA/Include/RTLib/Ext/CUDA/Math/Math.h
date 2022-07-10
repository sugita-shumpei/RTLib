#ifndef RTLIB_RTLIB_EXT_CUDA_MATH_RTLIB_MATH_H
#define RTLIB_RTLIB_EXT_CUDA_MATH_RTLIB_MATH_H
#include <cuda.h>

#include <RTLib/Ext/CUDA/Math/VectorFunction.h>
#include <RTLib/Ext/CUDA/Math/Preprocessors.h>

#if defined(__cplusplus) && !defined(__CUDA_ARCH__)
#include <bitset>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#endif

#ifndef RTLIB_M_PI
#define RTLIB_M_PI     3.14159265358979323846264338327950288
#endif
#ifndef RTLIB_M_PI2
#define RTLIB_M_PI2    1.5707963267948966192313216916398
#endif
#ifndef RTLIB_M_2PI
#define RTLIB_M_2PI    6.283185307179586476925286766559
#endif
#ifndef RTLIB_M_INV_PI
#define RTLIB_M_INV_PI 0.318309886183790691216444201927515678107738494873046875
#endif
namespace RTLib
{
    namespace Ext
    {
        namespace CUDA
        {
            namespace Math
            {
                // max
                RTLIB_INLINE RTLIB_HOST_DEVICE signed char max(const signed char v0, const signed char v1)
                {
#if defined(__CUDA_ARCH__) || !defined(__cplusplus)
                    return v0 >= v1 ? v0 : v1;
#elif defined(__cplusplus)
                    return std::max(v0, v1);
#else
                    return v0 >= v1 ? v0 : v1;
#endif
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE short max(const short v0, const short v1)
                {
#if defined(__CUDA_ARCH__) || !defined(__cplusplus)
                    return ::max(v0, v1);
#elif defined(__cplusplus)
                    return std::max(v0, v1);
#else
                    return v0 >= v1 ? v0 : v1;
#endif
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE int max(const int v0, const int v1)
                {
#if defined(__CUDA_ARCH__) || !defined(__cplusplus)
                    return ::max(v0, v1);
#elif defined(__cplusplus)
                    return std::max(v0, v1);
#else
                    return v0 >= v1 ? v0 : v1;
#endif
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE long max(const long v0, const long v1)
                {
#if defined(__CUDA_ARCH__) || !defined(__cplusplus)
                    return ::max(v0, v1);
#elif defined(__cplusplus)
                    return std::max(v0, v1);
#else
                    return v0 >= v1 ? v0 : v1;
#endif
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE long long max(const long long v0, const long long v1)
                {
#if defined(__CUDA_ARCH__) || !defined(__cplusplus)
                    return ::max(v0, v1);
#elif defined(__cplusplus)
                    return std::max(v0, v1);
#else
                    return v0 >= v1 ? v0 : v1;
#endif
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE unsigned char max(const unsigned char v0, const unsigned char v1)
                {
#if defined(__CUDA_ARCH__) || !defined(__cplusplus)
                    return v0 >= v1 ? v0 : v1;
#elif defined(__cplusplus)
                    return std::max(v0, v1);
#else
                    return v0 >= v1 ? v0 : v1;
#endif
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE unsigned short max(const unsigned short v0, const unsigned short v1)
                {
#if defined(__CUDA_ARCH__) || !defined(__cplusplus)
                    return ::max(v0, v1);
#elif defined(__cplusplus)
                    return std::max(v0, v1);
#else
                    return v0 >= v1 ? v0 : v1;
#endif
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE unsigned int max(const unsigned int v0, const unsigned int v1)
                {
#if defined(__CUDA_ARCH__) || !defined(__cplusplus)
                    return ::max(v0, v1);
#elif defined(__cplusplus)
                    return std::max(v0, v1);
#else
                    return v0 >= v1 ? v0 : v1;
#endif
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE unsigned long max(const unsigned long v0, const unsigned long v1)
                {
#if defined(__CUDA_ARCH__) || !defined(__cplusplus)
                    return ::max(v0, v1);
#elif defined(__cplusplus)
                    return std::max(v0, v1);
#else
                    return v0 >= v1 ? v0 : v1;
#endif
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE unsigned long long max(const unsigned long long v0, const unsigned long long v1)
                {
#if defined(__CUDA_ARCH__) || !defined(__cplusplus)
                    return ::max(v0, v1);
#elif defined(__cplusplus)
                    return std::max(v0, v1);
#else
                    return v0 >= v1 ? v0 : v1;
#endif
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE float max(const float v0, const float v1)
                {
#if defined(__CUDA_ARCH__) || !defined(__cplusplus)
                    return ::max(v0, v1);
#elif defined(__cplusplus)
                    return std::max(v0, v1);
#else
                    return v0 >= v1 ? v0 : v1;
#endif
                }
                // min
                RTLIB_INLINE RTLIB_HOST_DEVICE signed char min(const signed char v0, const signed char v1)
                {
#if defined(__CUDA_ARCH__) || !defined(__cplusplus)
                    return v0 <= v1 ? v0 : v1;
#elif defined(__cplusplus)
                    return std::min(v0, v1);
#else
                    return v0 <= v1 ? v0 : v1;
#endif
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE short min(const short v0, const short v1)
                {
#if defined(__CUDA_ARCH__) || !defined(__cplusplus)
                    return ::min(v0, v1);
#elif defined(__cplusplus)
                    return std::min(v0, v1);
#else
                    return v0 <= v1 ? v0 : v1;
#endif
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE int min(const int v0, const int v1)
                {
#if defined(__CUDA_ARCH__) || !defined(__cplusplus)
                    return ::min(v0, v1);
#elif defined(__cplusplus)
                    return std::min(v0, v1);
#else
                    return v0 <= v1 ? v0 : v1;
#endif
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE long min(const long v0, const long v1)
                {
#if defined(__CUDA_ARCH__) || !defined(__cplusplus)
                    return ::min(v0, v1);
#elif defined(__cplusplus)
                    return std::min(v0, v1);
#else
                    return v0 <= v1 ? v0 : v1;
#endif
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE long long min(const long long v0, const long long v1)
                {
#if defined(__CUDA_ARCH__) || !defined(__cplusplus)
                    return ::min(v0, v1);
#elif defined(__cplusplus)
                    return std::min(v0, v1);
#else
                    return v0 <= v1 ? v0 : v1;
#endif
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE unsigned char min(const unsigned char v0, const unsigned char v1)
                {
#if defined(__CUDA_ARCH__) || !defined(__cplusplus)
                    return v0 <= v1 ? v0 : v1;
#elif defined(__cplusplus)
                    return std::min(v0, v1);
#else
                    return v0 <= v1 ? v0 : v1;
#endif
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE unsigned short min(const unsigned short v0, const unsigned short v1)
                {
#if defined(__CUDA_ARCH__) || !defined(__cplusplus)
                    return ::min(v0, v1);
#elif defined(__cplusplus)
                    return std::min(v0, v1);
#else
                    return v0 <= v1 ? v0 : v1;
#endif
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE unsigned int min(const unsigned int v0, const unsigned int v1)
                {
#if defined(__CUDA_ARCH__) || !defined(__cplusplus)
                    return ::min(v0, v1);
#elif defined(__cplusplus)
                    return std::min(v0, v1);
#else
                    return v0 <= v1 ? v0 : v1;
#endif
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE unsigned long min(const unsigned long v0, const unsigned long v1)
                {
#if defined(__CUDA_ARCH__) || !defined(__cplusplus)
                    return ::min(v0, v1);
#elif defined(__cplusplus)
                    return std::min(v0, v1);
#else
                    return v0 <= v1 ? v0 : v1;
#endif
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE unsigned long long min(const unsigned long long v0, const unsigned long long v1)
                {
#if defined(__CUDA_ARCH__) || !defined(__cplusplus)
                    return ::min(v0, v1);
#elif defined(__cplusplus)
                    return std::min(v0, v1);
#else
                    return v0 <= v1 ? v0 : v1;
#endif
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE float min(const float v0, const float v1)
                {
#if defined(__CUDA_ARCH__) || !defined(__cplusplus)
                    return ::min(v0, v1);
#elif defined(__cplusplus)
                    return std::min(v0, v1);
#else
                    return v0 <= v1 ? v0 : v1;
#endif
                }
                // mix
                RTLIB_INLINE RTLIB_HOST_DEVICE signed char mix(const signed char v0, const signed char v1, const signed char a)
                {
                    return v0 * (1 - a) + v1 * a;
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE short mix(const short v0, const short v1, const short a)
                {
                    return v0 * (1 - a) + v1 * a;
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE int mix(const int v0, const int v1, const int a)
                {
                    return v0 * (1 - a) + v1 * a;
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE long mix(const long v0, const long v1, const long a)
                {
                    return v0 * (1 - a) + v1 * a;
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE long long mix(const long long v0, const long long v1, const long long a)
                {
                    return v0 * (1 - a) + v1 * a;
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE unsigned char mix(const unsigned char v0, const unsigned char v1, const unsigned char a)
                {
                    return v0 * (1 - a) + v1 * a;
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE unsigned short mix(const unsigned short v0, const unsigned short v1, const unsigned short a)
                {
                    return v0 * (1 - a) + v1 * a;
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE unsigned int mix(const unsigned int v0, const unsigned int v1, const unsigned int a)
                {
                    return v0 * (1 - a) + v1 * a;
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE unsigned long mix(const unsigned long v0, const unsigned long v1, const unsigned long a)
                {
                    return v0 * (1 - a) + v1 * a;
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE unsigned long long mix(const unsigned long long v0, const unsigned long long v1, const unsigned long long a)
                {
                    return v0 * (1 - a) + v1 * a;
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE float mix(const float v0, const float v1, const float a)
                {
                    return v0 * (1 - a) + v1 * a;
                }
                // clamp
                RTLIB_INLINE RTLIB_HOST_DEVICE signed char clamp(const signed char v, const signed char low, const signed char high)
                {
#if defined(__CUDA_ARCH__) || defined(__CUDACC__)
                    return min(max(v, low), high);
#elif defined(__cplusplus)
                    return std::clamp(v, low, high);
#endif
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE short clamp(const short v, const short low, const short high)
                {
#if defined(__CUDA_ARCH__) || defined(__CUDACC__)
                    return min(max(v, low), high);
#elif defined(__cplusplus)
                    return std::clamp(v, low, high);
#endif
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE int clamp(const int v, const int low, const int high)
                {
#if defined(__CUDA_ARCH__) || defined(__CUDACC__)
                    return min(max(v, low), high);
#elif defined(__cplusplus)
                    return std::clamp(v, low, high);
#endif
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE long clamp(const long v, const long low, const long high)
                {
#if defined(__CUDA_ARCH__) || defined(__CUDACC__)
                    return min(max(v, low), high);
#elif defined(__cplusplus)
                    return std::clamp(v, low, high);
#endif
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE long long clamp(const long long v, const long long low, const long long high)
                {
#if defined(__CUDA_ARCH__) || defined(__CUDACC__)
                    return min(max(v, low), high);
#elif defined(__cplusplus)
                    return std::clamp(v, low, high);
#endif
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE unsigned char clamp(const unsigned char v, const unsigned char low, const unsigned char high)
                {
#if defined(__CUDA_ARCH__) || defined(__CUDACC__)
                    return min(max(v, low), high);
#elif defined(__cplusplus)
                    return std::clamp(v, low, high);
#endif
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE unsigned short clamp(const unsigned short v, const unsigned short low, const unsigned short high)
                {
#if defined(__CUDA_ARCH__) || defined(__CUDACC__)
                    return min(max(v, low), high);
#elif defined(__cplusplus)
                    return std::clamp(v, low, high);
#endif
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE unsigned int clamp(const unsigned int v, const unsigned int low, const unsigned int high)
                {
#if defined(__CUDA_ARCH__) || defined(__CUDACC__)
                    return min(max(v, low), high);
#elif defined(__cplusplus)
                    return std::clamp(v, low, high);
#endif
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE unsigned long clamp(const unsigned long v, const unsigned long low, const unsigned long high)
                {
#if defined(__CUDA_ARCH__) || defined(__CUDACC__)
                    return min(max(v, low), high);
#elif defined(__cplusplus)
                    return std::clamp(v, low, high);
#endif
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE unsigned long long clamp(const unsigned long long v, const unsigned long long low, const unsigned long long high)
                {
#if defined(__CUDA_ARCH__) || defined(__CUDACC__)
                    return min(max(v, low), high);
#elif defined(__cplusplus)
                    return std::clamp(v, low, high);
#endif
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE float clamp(const float v, const float low, const float high)
                {
#if defined(__CUDA_ARCH__) || defined(__CUDACC__)
                    return min(max(v, low), high);
#elif defined(__cplusplus)
                    return std::clamp(v, low, high);
#endif
                }
                // step
                RTLIB_INLINE RTLIB_HOST_DEVICE signed char step(const signed char edge, const signed char x)
                {
                    return x < edge ? 0 : 1;
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE short step(const short edge, const short x)
                {
                    return x < edge ? 0 : 1;
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE int step(const int edge, const int x)
                {
                    return x < edge ? 0 : 1;
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE long step(const long edge, const long x)
                {
                    return x < edge ? 0 : 1;
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE long long step(const long long edge, const long long x)
                {
                    return x < edge ? 0 : 1;
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE unsigned char step(const unsigned char edge, const unsigned char x)
                {
                    return x < edge ? 0 : 1;
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE unsigned short step(const unsigned short edge, const unsigned short x)
                {
                    return x < edge ? 0 : 1;
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE unsigned int step(const unsigned int edge, const unsigned int x)
                {
                    return x < edge ? 0 : 1;
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE unsigned long step(const unsigned long edge, const unsigned long x)
                {
                    return x < edge ? 0 : 1;
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE unsigned long long step(const unsigned long long edge, const unsigned long long x)
                {
                    return x < edge ? 0 : 1;
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE float step(const float edge, const float x)
                {
                    return x < edge ? 0 : 1;
                }
                // powN
                RTLIB_INLINE RTLIB_HOST_DEVICE signed char pow2(const signed char v)
                {
                    return v * v;
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE signed char pow3(const signed char v)
                {
                    return v * v * v;
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE signed char pow4(const signed char v)
                {
                    return v * v * v * v;
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE signed char pow5(const signed char v)
                {
                    return v * v * v * v * v;
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE short pow2(const short v)
                {
                    return v * v;
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE short pow3(const short v)
                {
                    return v * v * v;
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE short pow4(const short v)
                {
                    return v * v * v * v;
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE short pow5(const short v)
                {
                    return v * v * v * v * v;
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE int pow2(const int v)
                {
                    return v * v;
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE int pow3(const int v)
                {
                    return v * v * v;
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE int pow4(const int v)
                {
                    return v * v * v * v;
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE int pow5(const int v)
                {
                    return v * v * v * v * v;
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE long pow2(const long v)
                {
                    return v * v;
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE long pow3(const long v)
                {
                    return v * v * v;
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE long pow4(const long v)
                {
                    return v * v * v * v;
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE long pow5(const long v)
                {
                    return v * v * v * v * v;
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE long long pow2(const long long v)
                {
                    return v * v;
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE long long pow3(const long long v)
                {
                    return v * v * v;
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE long long pow4(const long long v)
                {
                    return v * v * v * v;
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE long long pow5(const long long v)
                {
                    return v * v * v * v * v;
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE unsigned char pow2(const unsigned char v)
                {
                    return v * v;
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE unsigned char pow3(const unsigned char v)
                {
                    return v * v * v;
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE unsigned char pow4(const unsigned char v)
                {
                    return v * v * v * v;
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE unsigned char pow5(const unsigned char v)
                {
                    return v * v * v * v * v;
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE unsigned short pow2(const unsigned short v)
                {
                    return v * v;
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE unsigned short pow3(const unsigned short v)
                {
                    return v * v * v;
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE unsigned short pow4(const unsigned short v)
                {
                    return v * v * v * v;
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE unsigned short pow5(const unsigned short v)
                {
                    return v * v * v * v * v;
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE unsigned int pow2(const unsigned int v)
                {
                    return v * v;
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE unsigned int pow3(const unsigned int v)
                {
                    return v * v * v;
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE unsigned int pow4(const unsigned int v)
                {
                    return v * v * v * v;
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE unsigned int pow5(const unsigned int v)
                {
                    return v * v * v * v * v;
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE unsigned long pow2(const unsigned long v)
                {
                    return v * v;
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE unsigned long pow3(const unsigned long v)
                {
                    return v * v * v;
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE unsigned long pow4(const unsigned long v)
                {
                    return v * v * v * v;
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE unsigned long pow5(const unsigned long v)
                {
                    return v * v * v * v * v;
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE unsigned long long pow2(const unsigned long long v)
                {
                    return v * v;
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE unsigned long long pow3(const unsigned long long v)
                {
                    return v * v * v;
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE unsigned long long pow4(const unsigned long long v)
                {
                    return v * v * v * v;
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE unsigned long long pow5(const unsigned long long v)
                {
                    return v * v * v * v * v;
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE float pow2(const float v)
                {
                    return v * v;
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE float pow3(const float v)
                {
                    return v * v * v;
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE float pow4(const float v)
                {
                    return v * v * v * v;
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE float pow5(const float v)
                {
                    return v * v * v * v * v;
                }
                // rgb<->adobe srgb
                RTLIB_INLINE RTLIB_HOST_DEVICE float linear_to_gamma(const float v)
                {
#if defined(__cplusplus) && !defined(__CUDA_ARCH__)
                    using std::powf;
#endif
                    return (v <= 0.00174f) ? (32.0f * v) : (::powf(v, 1.0f / 2.2f));
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE float gamma_to_linear(const float v)
                {
#if defined(__cplusplus) && !defined(__CUDA_ARCH__)
                    using std::powf;
#endif
                    return (v <= 0.0556f) ? (v / 32.f) : (::powf(v, 2.2f));
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE unsigned short to_combine(unsigned char upper, unsigned char lower)
                {
                    return (static_cast<unsigned short>(upper) << 8) | static_cast<unsigned short>(lower);
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE unsigned int to_combine(unsigned short upper, unsigned short lower)
                {
                    return (static_cast<unsigned int>(upper) << 16) | static_cast<unsigned int>(lower);
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE unsigned long long to_combine(unsigned int upper, unsigned int lower)
                {
                    return (static_cast<unsigned long long>(upper) << 32) | static_cast<unsigned long long>(lower);
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE unsigned char to_upper(unsigned short v)
                {
                    return static_cast<unsigned char>(v >> 8);
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE unsigned short to_upper(unsigned int v)
                {
                    return static_cast<unsigned short>(v >> 16);
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE unsigned int to_upper(unsigned long long v)
                {
                    return static_cast<unsigned int>(v >> 32);
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE unsigned char to_lower(unsigned short v)
                {
                    return static_cast<unsigned char>(v & 0x00FF);
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE unsigned short to_lower(unsigned int v)
                {
                    return static_cast<unsigned short>(v & 0x0000FFFF);
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE unsigned int to_lower(unsigned long long v)
                {
                    return static_cast<unsigned int>(v & 0x00000000FFFFFFFF);
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE double to_uniform_double(unsigned long long v)
                {
                    unsigned long long value = (v >> 12) | (static_cast<unsigned long long>(0x3FF) << 52);
#if defined(__CUDA_ARCH__)
                    return __ull2double_rn(value) - 1.0;
#elif defined(__cplusplus)
                    double ans = 0.0;
                    memcpy(&ans, &value, sizeof(unsigned long long));
                    return ans - 1.0;
#endif
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE float to_uniform_float(unsigned int v)
                {
                    unsigned int value = (v >> 9) | (static_cast<unsigned int>(0x7F) << 23);
#if defined(__CUDA_ARCH__)
                    return __int_as_float(value) - 1.0f;
#elif defined(__cplusplus)
                    float ans = 0.0f;
                    memcpy(&ans, &value, sizeof(unsigned int));
                    return ans - 1.0f;
#endif
#if defined(__CUDA_ARCH__)

#endif
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE float3 canonical_to_dir(const float2 &p)
                {
                    const float cosTheta = 2.0f * p.x - 1.0f;
                    const float sinTheta = sqrtf(1.0f - cosTheta * cosTheta);
                    const float phi = RTLIB_M_2PI * p.y;
                    const float cosPhi = ::cosf(phi);
                    const float sinPhi = ::sinf(phi);
                    return make_float3(sinTheta * cosPhi, sinTheta * sinPhi, cosTheta);
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE float2 dir_to_canonical(const float3 &d)
                {
                    if (!isfinite(d.x) || !isfinite(d.y) || !isfinite(d.z))
                    {
                        return make_float2(0, 0);
                    }
                    const float z = RTLib::Ext::CUDA::Math::clamp(d.z, -1.0f, 1.0f);
                    float phi = atan2f(d.y, d.x);
                    while (phi < 0.0f)
                    {
                        phi += RTLIB_M_2PI;
                    }
                    return make_float2((z + 1.0f) / 2.0f, phi / RTLIB_M_2PI);
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE float3 spherical_to_dir(const float2 &p)
                {
                    const float tht = RTLIB_M_PI * p.x;
                    const float cosTht = ::cosf(tht);
                    const float sinTht = ::sinf(tht);
                    const float phi = RTLIB_M_2PI * p.y;
                    const float cosPhi = ::cosf(phi);
                    const float sinPhi = ::sinf(phi);
                    return make_float3(sinTht * cosPhi, sinTht * sinPhi, cosTht);
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE float2 dir_to_spherical(const float3 &d)
                {
                    const float tht = ::acosf(d.z);
                    float phi = atan2f(d.y, d.x);
                    while (phi < 0.0f)
                    {
                        phi += RTLIB_M_2PI;
                    }
                    return make_float2(tht / RTLIB_M_PI, phi / RTLIB_M_2PI);
                }
                RTLIB_INLINE RTLIB_DEVICE int pop_count32(unsigned int ui32)
                {
#ifdef __CUDA_ARCH__
                    return __popc(ui32);
#else
                    return std::bitset<32>(ui32).count();
#endif
                }
                RTLIB_INLINE RTLIB_DEVICE float to_average_rgb(const float3 &rgb)
                {
                    return (rgb.x + rgb.y + rgb.z) / 3.0f;
                }
                RTLIB_INLINE RTLIB_DEVICE float to_luminance(const float3 &rgb)
                {
                    return 0.2126f * rgb.x + 0.7152f * rgb.y + 0.0722f * rgb.z;
                }
                // max
                RTLIB_INLINE RTLIB_HOST_DEVICE char2 max(const char2& v0, const char2& v1)
                {
#if defined(__cplusplus) && !defined(__CUDA_ARCH__)
                    using std::max;
#endif
                    return make_char2(max(v0.x, v1.x), max(v0.y, v1.y));
                }
                // min
                RTLIB_INLINE RTLIB_HOST_DEVICE char2 min(const char2& v0, const char2& v1)
                {
#if defined(__cplusplus) && !defined(__CUDA_ARCH__)
                    using std::min;
#endif
                    return make_char2(min(v0.x, v1.x), min(v0.y, v1.y));
                }
                // mix
                RTLIB_INLINE RTLIB_HOST_DEVICE char2 mix(const char2& v0, const char2& v1, const char2& a)
                {
                    return make_char2(mix(v0.x, v1.x, a.x), mix(v0.y, v1.y, a.y));
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE char2 mix(const char2& v0, const char2& v1, const signed char a)
                {
                    return make_char2(mix(v0.x, v1.x, a), mix(v0.y, v1.y, a));
                }
                // clamp
                RTLIB_INLINE RTLIB_HOST_DEVICE char2 clamp(const char2& v, const char2& low, const char2& high)
                {
                    return make_char2(clamp(v.x, low.x, high.x), clamp(v.y, low.y, high.y));
                }
                // powN
                RTLIB_INLINE RTLIB_HOST_DEVICE char2 pow2(const char2& v)
                {
                    return make_char2(pow2(v.x), pow2(v.y));
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE char2 pow3(const char2& v)
                {
                    return make_char2(pow3(v.x), pow3(v.y));
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE char2 pow4(const char2& v)
                {
                    return make_char2(pow4(v.x), pow4(v.y));
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE char2 pow5(const char2& v)
                {
                    return make_char2(pow5(v.x), pow5(v.y));
                }
                // max
                RTLIB_INLINE RTLIB_HOST_DEVICE char3 max(const char3& v0, const char3& v1)
                {
#if defined(__cplusplus) && !defined(__CUDA_ARCH__)
                    using std::max;
#endif
                    return make_char3(max(v0.x, v1.x), max(v0.y, v1.y), max(v0.z, v1.z));
                }
                // min
                RTLIB_INLINE RTLIB_HOST_DEVICE char3 min(const char3& v0, const char3& v1)
                {
#if defined(__cplusplus) && !defined(__CUDA_ARCH__)
                    using std::min;
#endif
                    return make_char3(min(v0.x, v1.x), min(v0.y, v1.y), min(v0.z, v1.z));
                }
                // mix
                RTLIB_INLINE RTLIB_HOST_DEVICE char3 mix(const char3& v0, const char3& v1, const char3& a)
                {
                    return make_char3(mix(v0.x, v1.x, a.x), mix(v0.y, v1.y, a.y), mix(v0.z, v1.z, a.z));
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE char3 mix(const char3& v0, const char3& v1, const signed char a)
                {
                    return make_char3(mix(v0.x, v1.x, a), mix(v0.y, v1.y, a), mix(v0.z, v1.z, a));
                }
                // clamp
                RTLIB_INLINE RTLIB_HOST_DEVICE char3 clamp(const char3& v, const char3& low, const char3& high)
                {
                    return make_char3(clamp(v.x, low.x, high.x), clamp(v.y, low.y, high.y), clamp(v.z, low.z, high.z));
                }
                // powN
                RTLIB_INLINE RTLIB_HOST_DEVICE char3 pow2(const char3& v)
                {
                    return make_char3(pow2(v.x), pow2(v.y), pow2(v.z));
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE char3 pow3(const char3& v)
                {
                    return make_char3(pow3(v.x), pow3(v.y), pow3(v.z));
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE char3 pow4(const char3& v)
                {
                    return make_char3(pow4(v.x), pow4(v.y), pow4(v.z));
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE char3 pow5(const char3& v)
                {
                    return make_char3(pow5(v.x), pow5(v.y), pow5(v.z));
                }
                // max
                RTLIB_INLINE RTLIB_HOST_DEVICE char4 max(const char4& v0, const char4& v1)
                {
#if defined(__cplusplus) && !defined(__CUDA_ARCH__)
                    using std::max;
#endif
                    return make_char4(max(v0.x, v1.x), max(v0.y, v1.y), max(v0.z, v1.z), max(v0.w, v1.w));
                }
                // min
                RTLIB_INLINE RTLIB_HOST_DEVICE char4 min(const char4& v0, const char4& v1)
                {
#if defined(__cplusplus) && !defined(__CUDA_ARCH__)
                    using std::min;
#endif
                    return make_char4(min(v0.x, v1.x), min(v0.y, v1.y), min(v0.z, v1.z), min(v0.w, v1.w));
                }
                // mix
                RTLIB_INLINE RTLIB_HOST_DEVICE char4 mix(const char4& v0, const char4& v1, const char4& a)
                {
                    return make_char4(mix(v0.x, v1.x, a.x), mix(v0.y, v1.y, a.y), mix(v0.z, v1.z, a.z), mix(v0.w, v1.w, a.w));
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE char4 mix(const char4& v0, const char4& v1, const signed char a)
                {
                    return make_char4(mix(v0.x, v1.x, a), mix(v0.y, v1.y, a), mix(v0.z, v1.z, a), mix(v0.w, v1.w, a));
                }
                // clamp
                RTLIB_INLINE RTLIB_HOST_DEVICE char4 clamp(const char4& v, const char4& low, const char4& high)
                {
                    return make_char4(clamp(v.x, low.x, high.x), clamp(v.y, low.y, high.y), clamp(v.z, low.z, high.z), clamp(v.w, low.w, high.w));
                }
                // powN
                RTLIB_INLINE RTLIB_HOST_DEVICE char4 pow2(const char4& v)
                {
                    return make_char4(pow2(v.x), pow2(v.y), pow2(v.z), pow2(v.w));
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE char4 pow3(const char4& v)
                {
                    return make_char4(pow3(v.x), pow3(v.y), pow3(v.z), pow3(v.w));
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE char4 pow4(const char4& v)
                {
                    return make_char4(pow4(v.x), pow4(v.y), pow4(v.z), pow4(v.w));
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE char4 pow5(const char4& v)
                {
                    return make_char4(pow5(v.x), pow5(v.y), pow5(v.z), pow5(v.w));
                }
                // max
                RTLIB_INLINE RTLIB_HOST_DEVICE short2 max(const short2& v0, const short2& v1)
                {
#if defined(__cplusplus) && !defined(__CUDA_ARCH__)
                    using std::max;
#endif
                    return make_short2(max(v0.x, v1.x), max(v0.y, v1.y));
                }
                // min
                RTLIB_INLINE RTLIB_HOST_DEVICE short2 min(const short2& v0, const short2& v1)
                {
#if defined(__cplusplus) && !defined(__CUDA_ARCH__)
                    using std::min;
#endif
                    return make_short2(min(v0.x, v1.x), min(v0.y, v1.y));
                }
                // mix
                RTLIB_INLINE RTLIB_HOST_DEVICE short2 mix(const short2& v0, const short2& v1, const short2& a)
                {
                    return make_short2(mix(v0.x, v1.x, a.x), mix(v0.y, v1.y, a.y));
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE short2 mix(const short2& v0, const short2& v1, const short a)
                {
                    return make_short2(mix(v0.x, v1.x, a), mix(v0.y, v1.y, a));
                }
                // clamp
                RTLIB_INLINE RTLIB_HOST_DEVICE short2 clamp(const short2& v, const short2& low, const short2& high)
                {
                    return make_short2(clamp(v.x, low.x, high.x), clamp(v.y, low.y, high.y));
                }
                // powN
                RTLIB_INLINE RTLIB_HOST_DEVICE short2 pow2(const short2& v)
                {
                    return make_short2(pow2(v.x), pow2(v.y));
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE short2 pow3(const short2& v)
                {
                    return make_short2(pow3(v.x), pow3(v.y));
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE short2 pow4(const short2& v)
                {
                    return make_short2(pow4(v.x), pow4(v.y));
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE short2 pow5(const short2& v)
                {
                    return make_short2(pow5(v.x), pow5(v.y));
                }
                // max
                RTLIB_INLINE RTLIB_HOST_DEVICE short3 max(const short3& v0, const short3& v1)
                {
#if defined(__cplusplus) && !defined(__CUDA_ARCH__)
                    using std::max;
#endif
                    return make_short3(max(v0.x, v1.x), max(v0.y, v1.y), max(v0.z, v1.z));
                }
                // min
                RTLIB_INLINE RTLIB_HOST_DEVICE short3 min(const short3& v0, const short3& v1)
                {
#if defined(__cplusplus) && !defined(__CUDA_ARCH__)
                    using std::min;
#endif
                    return make_short3(min(v0.x, v1.x), min(v0.y, v1.y), min(v0.z, v1.z));
                }
                // mix
                RTLIB_INLINE RTLIB_HOST_DEVICE short3 mix(const short3& v0, const short3& v1, const short3& a)
                {
                    return make_short3(mix(v0.x, v1.x, a.x), mix(v0.y, v1.y, a.y), mix(v0.z, v1.z, a.z));
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE short3 mix(const short3& v0, const short3& v1, const short a)
                {
                    return make_short3(mix(v0.x, v1.x, a), mix(v0.y, v1.y, a), mix(v0.z, v1.z, a));
                }
                // clamp
                RTLIB_INLINE RTLIB_HOST_DEVICE short3 clamp(const short3& v, const short3& low, const short3& high)
                {
                    return make_short3(clamp(v.x, low.x, high.x), clamp(v.y, low.y, high.y), clamp(v.z, low.z, high.z));
                }
                // powN
                RTLIB_INLINE RTLIB_HOST_DEVICE short3 pow2(const short3& v)
                {
                    return make_short3(pow2(v.x), pow2(v.y), pow2(v.z));
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE short3 pow3(const short3& v)
                {
                    return make_short3(pow3(v.x), pow3(v.y), pow3(v.z));
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE short3 pow4(const short3& v)
                {
                    return make_short3(pow4(v.x), pow4(v.y), pow4(v.z));
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE short3 pow5(const short3& v)
                {
                    return make_short3(pow5(v.x), pow5(v.y), pow5(v.z));
                }
                // max
                RTLIB_INLINE RTLIB_HOST_DEVICE short4 max(const short4& v0, const short4& v1)
                {
#if defined(__cplusplus) && !defined(__CUDA_ARCH__)
                    using std::max;
#endif
                    return make_short4(max(v0.x, v1.x), max(v0.y, v1.y), max(v0.z, v1.z), max(v0.w, v1.w));
                }
                // min
                RTLIB_INLINE RTLIB_HOST_DEVICE short4 min(const short4& v0, const short4& v1)
                {
#if defined(__cplusplus) && !defined(__CUDA_ARCH__)
                    using std::min;
#endif
                    return make_short4(min(v0.x, v1.x), min(v0.y, v1.y), min(v0.z, v1.z), min(v0.w, v1.w));
                }
                // mix
                RTLIB_INLINE RTLIB_HOST_DEVICE short4 mix(const short4& v0, const short4& v1, const short4& a)
                {
                    return make_short4(mix(v0.x, v1.x, a.x), mix(v0.y, v1.y, a.y), mix(v0.z, v1.z, a.z), mix(v0.w, v1.w, a.w));
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE short4 mix(const short4& v0, const short4& v1, const short a)
                {
                    return make_short4(mix(v0.x, v1.x, a), mix(v0.y, v1.y, a), mix(v0.z, v1.z, a), mix(v0.w, v1.w, a));
                }
                // clamp
                RTLIB_INLINE RTLIB_HOST_DEVICE short4 clamp(const short4& v, const short4& low, const short4& high)
                {
                    return make_short4(clamp(v.x, low.x, high.x), clamp(v.y, low.y, high.y), clamp(v.z, low.z, high.z), clamp(v.w, low.w, high.w));
                }
                // powN
                RTLIB_INLINE RTLIB_HOST_DEVICE short4 pow2(const short4& v)
                {
                    return make_short4(pow2(v.x), pow2(v.y), pow2(v.z), pow2(v.w));
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE short4 pow3(const short4& v)
                {
                    return make_short4(pow3(v.x), pow3(v.y), pow3(v.z), pow3(v.w));
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE short4 pow4(const short4& v)
                {
                    return make_short4(pow4(v.x), pow4(v.y), pow4(v.z), pow4(v.w));
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE short4 pow5(const short4& v)
                {
                    return make_short4(pow5(v.x), pow5(v.y), pow5(v.z), pow5(v.w));
                }
                // max
                RTLIB_INLINE RTLIB_HOST_DEVICE int2 max(const int2& v0, const int2& v1)
                {
#if defined(__cplusplus) && !defined(__CUDA_ARCH__)
                    using std::max;
#endif
                    return make_int2(max(v0.x, v1.x), max(v0.y, v1.y));
                }
                // min
                RTLIB_INLINE RTLIB_HOST_DEVICE int2 min(const int2& v0, const int2& v1)
                {
#if defined(__cplusplus) && !defined(__CUDA_ARCH__)
                    using std::min;
#endif
                    return make_int2(min(v0.x, v1.x), min(v0.y, v1.y));
                }
                // mix
                RTLIB_INLINE RTLIB_HOST_DEVICE int2 mix(const int2& v0, const int2& v1, const int2& a)
                {
                    return make_int2(mix(v0.x, v1.x, a.x), mix(v0.y, v1.y, a.y));
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE int2 mix(const int2& v0, const int2& v1, const int a)
                {
                    return make_int2(mix(v0.x, v1.x, a), mix(v0.y, v1.y, a));
                }
                // clamp
                RTLIB_INLINE RTLIB_HOST_DEVICE int2 clamp(const int2& v, const int2& low, const int2& high)
                {
                    return make_int2(clamp(v.x, low.x, high.x), clamp(v.y, low.y, high.y));
                }
                // powN
                RTLIB_INLINE RTLIB_HOST_DEVICE int2 pow2(const int2& v)
                {
                    return make_int2(pow2(v.x), pow2(v.y));
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE int2 pow3(const int2& v)
                {
                    return make_int2(pow3(v.x), pow3(v.y));
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE int2 pow4(const int2& v)
                {
                    return make_int2(pow4(v.x), pow4(v.y));
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE int2 pow5(const int2& v)
                {
                    return make_int2(pow5(v.x), pow5(v.y));
                }
                // max
                RTLIB_INLINE RTLIB_HOST_DEVICE int3 max(const int3& v0, const int3& v1)
                {
#if defined(__cplusplus) && !defined(__CUDA_ARCH__)
                    using std::max;
#endif
                    return make_int3(max(v0.x, v1.x), max(v0.y, v1.y), max(v0.z, v1.z));
                }
                // min
                RTLIB_INLINE RTLIB_HOST_DEVICE int3 min(const int3& v0, const int3& v1)
                {
#if defined(__cplusplus) && !defined(__CUDA_ARCH__)
                    using std::min;
#endif
                    return make_int3(min(v0.x, v1.x), min(v0.y, v1.y), min(v0.z, v1.z));
                }
                // mix
                RTLIB_INLINE RTLIB_HOST_DEVICE int3 mix(const int3& v0, const int3& v1, const int3& a)
                {
                    return make_int3(mix(v0.x, v1.x, a.x), mix(v0.y, v1.y, a.y), mix(v0.z, v1.z, a.z));
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE int3 mix(const int3& v0, const int3& v1, const int a)
                {
                    return make_int3(mix(v0.x, v1.x, a), mix(v0.y, v1.y, a), mix(v0.z, v1.z, a));
                }
                // clamp
                RTLIB_INLINE RTLIB_HOST_DEVICE int3 clamp(const int3& v, const int3& low, const int3& high)
                {
                    return make_int3(clamp(v.x, low.x, high.x), clamp(v.y, low.y, high.y), clamp(v.z, low.z, high.z));
                }
                // powN
                RTLIB_INLINE RTLIB_HOST_DEVICE int3 pow2(const int3& v)
                {
                    return make_int3(pow2(v.x), pow2(v.y), pow2(v.z));
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE int3 pow3(const int3& v)
                {
                    return make_int3(pow3(v.x), pow3(v.y), pow3(v.z));
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE int3 pow4(const int3& v)
                {
                    return make_int3(pow4(v.x), pow4(v.y), pow4(v.z));
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE int3 pow5(const int3& v)
                {
                    return make_int3(pow5(v.x), pow5(v.y), pow5(v.z));
                }
                // max
                RTLIB_INLINE RTLIB_HOST_DEVICE int4 max(const int4& v0, const int4& v1)
                {
#if defined(__cplusplus) && !defined(__CUDA_ARCH__)
                    using std::max;
#endif
                    return make_int4(max(v0.x, v1.x), max(v0.y, v1.y), max(v0.z, v1.z), max(v0.w, v1.w));
                }
                // min
                RTLIB_INLINE RTLIB_HOST_DEVICE int4 min(const int4& v0, const int4& v1)
                {
#if defined(__cplusplus) && !defined(__CUDA_ARCH__)
                    using std::min;
#endif
                    return make_int4(min(v0.x, v1.x), min(v0.y, v1.y), min(v0.z, v1.z), min(v0.w, v1.w));
                }
                // mix
                RTLIB_INLINE RTLIB_HOST_DEVICE int4 mix(const int4& v0, const int4& v1, const int4& a)
                {
                    return make_int4(mix(v0.x, v1.x, a.x), mix(v0.y, v1.y, a.y), mix(v0.z, v1.z, a.z), mix(v0.w, v1.w, a.w));
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE int4 mix(const int4& v0, const int4& v1, const int a)
                {
                    return make_int4(mix(v0.x, v1.x, a), mix(v0.y, v1.y, a), mix(v0.z, v1.z, a), mix(v0.w, v1.w, a));
                }
                // clamp
                RTLIB_INLINE RTLIB_HOST_DEVICE int4 clamp(const int4& v, const int4& low, const int4& high)
                {
                    return make_int4(clamp(v.x, low.x, high.x), clamp(v.y, low.y, high.y), clamp(v.z, low.z, high.z), clamp(v.w, low.w, high.w));
                }
                // powN
                RTLIB_INLINE RTLIB_HOST_DEVICE int4 pow2(const int4& v)
                {
                    return make_int4(pow2(v.x), pow2(v.y), pow2(v.z), pow2(v.w));
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE int4 pow3(const int4& v)
                {
                    return make_int4(pow3(v.x), pow3(v.y), pow3(v.z), pow3(v.w));
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE int4 pow4(const int4& v)
                {
                    return make_int4(pow4(v.x), pow4(v.y), pow4(v.z), pow4(v.w));
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE int4 pow5(const int4& v)
                {
                    return make_int4(pow5(v.x), pow5(v.y), pow5(v.z), pow5(v.w));
                }
                // max
                RTLIB_INLINE RTLIB_HOST_DEVICE long2 max(const long2& v0, const long2& v1)
                {
#if defined(__cplusplus) && !defined(__CUDA_ARCH__)
                    using std::max;
#endif
                    return make_long2(max(v0.x, v1.x), max(v0.y, v1.y));
                }
                // min
                RTLIB_INLINE RTLIB_HOST_DEVICE long2 min(const long2& v0, const long2& v1)
                {
#if defined(__cplusplus) && !defined(__CUDA_ARCH__)
                    using std::min;
#endif
                    return make_long2(min(v0.x, v1.x), min(v0.y, v1.y));
                }
                // mix
                RTLIB_INLINE RTLIB_HOST_DEVICE long2 mix(const long2& v0, const long2& v1, const long2& a)
                {
                    return make_long2(mix(v0.x, v1.x, a.x), mix(v0.y, v1.y, a.y));
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE long2 mix(const long2& v0, const long2& v1, const long a)
                {
                    return make_long2(mix(v0.x, v1.x, a), mix(v0.y, v1.y, a));
                }
                // clamp
                RTLIB_INLINE RTLIB_HOST_DEVICE long2 clamp(const long2& v, const long2& low, const long2& high)
                {
                    return make_long2(clamp(v.x, low.x, high.x), clamp(v.y, low.y, high.y));
                }
                // powN
                RTLIB_INLINE RTLIB_HOST_DEVICE long2 pow2(const long2& v)
                {
                    return make_long2(pow2(v.x), pow2(v.y));
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE long2 pow3(const long2& v)
                {
                    return make_long2(pow3(v.x), pow3(v.y));
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE long2 pow4(const long2& v)
                {
                    return make_long2(pow4(v.x), pow4(v.y));
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE long2 pow5(const long2& v)
                {
                    return make_long2(pow5(v.x), pow5(v.y));
                }
                // max
                RTLIB_INLINE RTLIB_HOST_DEVICE long3 max(const long3& v0, const long3& v1)
                {
#if defined(__cplusplus) && !defined(__CUDA_ARCH__)
                    using std::max;
#endif
                    return make_long3(max(v0.x, v1.x), max(v0.y, v1.y), max(v0.z, v1.z));
                }
                // min
                RTLIB_INLINE RTLIB_HOST_DEVICE long3 min(const long3& v0, const long3& v1)
                {
#if defined(__cplusplus) && !defined(__CUDA_ARCH__)
                    using std::min;
#endif
                    return make_long3(min(v0.x, v1.x), min(v0.y, v1.y), min(v0.z, v1.z));
                }
                // mix
                RTLIB_INLINE RTLIB_HOST_DEVICE long3 mix(const long3& v0, const long3& v1, const long3& a)
                {
                    return make_long3(mix(v0.x, v1.x, a.x), mix(v0.y, v1.y, a.y), mix(v0.z, v1.z, a.z));
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE long3 mix(const long3& v0, const long3& v1, const long a)
                {
                    return make_long3(mix(v0.x, v1.x, a), mix(v0.y, v1.y, a), mix(v0.z, v1.z, a));
                }
                // clamp
                RTLIB_INLINE RTLIB_HOST_DEVICE long3 clamp(const long3& v, const long3& low, const long3& high)
                {
                    return make_long3(clamp(v.x, low.x, high.x), clamp(v.y, low.y, high.y), clamp(v.z, low.z, high.z));
                }
                // powN
                RTLIB_INLINE RTLIB_HOST_DEVICE long3 pow2(const long3& v)
                {
                    return make_long3(pow2(v.x), pow2(v.y), pow2(v.z));
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE long3 pow3(const long3& v)
                {
                    return make_long3(pow3(v.x), pow3(v.y), pow3(v.z));
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE long3 pow4(const long3& v)
                {
                    return make_long3(pow4(v.x), pow4(v.y), pow4(v.z));
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE long3 pow5(const long3& v)
                {
                    return make_long3(pow5(v.x), pow5(v.y), pow5(v.z));
                }
                // max
                RTLIB_INLINE RTLIB_HOST_DEVICE long4 max(const long4& v0, const long4& v1)
                {
#if defined(__cplusplus) && !defined(__CUDA_ARCH__)
                    using std::max;
#endif
                    return make_long4(max(v0.x, v1.x), max(v0.y, v1.y), max(v0.z, v1.z), max(v0.w, v1.w));
                }
                // min
                RTLIB_INLINE RTLIB_HOST_DEVICE long4 min(const long4& v0, const long4& v1)
                {
#if defined(__cplusplus) && !defined(__CUDA_ARCH__)
                    using std::min;
#endif
                    return make_long4(min(v0.x, v1.x), min(v0.y, v1.y), min(v0.z, v1.z), min(v0.w, v1.w));
                }
                // mix
                RTLIB_INLINE RTLIB_HOST_DEVICE long4 mix(const long4& v0, const long4& v1, const long4& a)
                {
                    return make_long4(mix(v0.x, v1.x, a.x), mix(v0.y, v1.y, a.y), mix(v0.z, v1.z, a.z), mix(v0.w, v1.w, a.w));
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE long4 mix(const long4& v0, const long4& v1, const long a)
                {
                    return make_long4(mix(v0.x, v1.x, a), mix(v0.y, v1.y, a), mix(v0.z, v1.z, a), mix(v0.w, v1.w, a));
                }
                // clamp
                RTLIB_INLINE RTLIB_HOST_DEVICE long4 clamp(const long4& v, const long4& low, const long4& high)
                {
                    return make_long4(clamp(v.x, low.x, high.x), clamp(v.y, low.y, high.y), clamp(v.z, low.z, high.z), clamp(v.w, low.w, high.w));
                }
                // powN
                RTLIB_INLINE RTLIB_HOST_DEVICE long4 pow2(const long4& v)
                {
                    return make_long4(pow2(v.x), pow2(v.y), pow2(v.z), pow2(v.w));
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE long4 pow3(const long4& v)
                {
                    return make_long4(pow3(v.x), pow3(v.y), pow3(v.z), pow3(v.w));
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE long4 pow4(const long4& v)
                {
                    return make_long4(pow4(v.x), pow4(v.y), pow4(v.z), pow4(v.w));
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE long4 pow5(const long4& v)
                {
                    return make_long4(pow5(v.x), pow5(v.y), pow5(v.z), pow5(v.w));
                }
                // max
                RTLIB_INLINE RTLIB_HOST_DEVICE longlong2 max(const longlong2& v0, const longlong2& v1)
                {
#if defined(__cplusplus) && !defined(__CUDA_ARCH__)
                    using std::max;
#endif
                    return make_longlong2(max(v0.x, v1.x), max(v0.y, v1.y));
                }
                // min
                RTLIB_INLINE RTLIB_HOST_DEVICE longlong2 min(const longlong2& v0, const longlong2& v1)
                {
#if defined(__cplusplus) && !defined(__CUDA_ARCH__)
                    using std::min;
#endif
                    return make_longlong2(min(v0.x, v1.x), min(v0.y, v1.y));
                }
                // mix
                RTLIB_INLINE RTLIB_HOST_DEVICE longlong2 mix(const longlong2& v0, const longlong2& v1, const longlong2& a)
                {
                    return make_longlong2(mix(v0.x, v1.x, a.x), mix(v0.y, v1.y, a.y));
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE longlong2 mix(const longlong2& v0, const longlong2& v1, const long long a)
                {
                    return make_longlong2(mix(v0.x, v1.x, a), mix(v0.y, v1.y, a));
                }
                // clamp
                RTLIB_INLINE RTLIB_HOST_DEVICE longlong2 clamp(const longlong2& v, const longlong2& low, const longlong2& high)
                {
                    return make_longlong2(clamp(v.x, low.x, high.x), clamp(v.y, low.y, high.y));
                }
                // powN
                RTLIB_INLINE RTLIB_HOST_DEVICE longlong2 pow2(const longlong2& v)
                {
                    return make_longlong2(pow2(v.x), pow2(v.y));
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE longlong2 pow3(const longlong2& v)
                {
                    return make_longlong2(pow3(v.x), pow3(v.y));
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE longlong2 pow4(const longlong2& v)
                {
                    return make_longlong2(pow4(v.x), pow4(v.y));
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE longlong2 pow5(const longlong2& v)
                {
                    return make_longlong2(pow5(v.x), pow5(v.y));
                }
                // max
                RTLIB_INLINE RTLIB_HOST_DEVICE longlong3 max(const longlong3& v0, const longlong3& v1)
                {
#if defined(__cplusplus) && !defined(__CUDA_ARCH__)
                    using std::max;
#endif
                    return make_longlong3(max(v0.x, v1.x), max(v0.y, v1.y), max(v0.z, v1.z));
                }
                // min
                RTLIB_INLINE RTLIB_HOST_DEVICE longlong3 min(const longlong3& v0, const longlong3& v1)
                {
#if defined(__cplusplus) && !defined(__CUDA_ARCH__)
                    using std::min;
#endif
                    return make_longlong3(min(v0.x, v1.x), min(v0.y, v1.y), min(v0.z, v1.z));
                }
                // mix
                RTLIB_INLINE RTLIB_HOST_DEVICE longlong3 mix(const longlong3& v0, const longlong3& v1, const longlong3& a)
                {
                    return make_longlong3(mix(v0.x, v1.x, a.x), mix(v0.y, v1.y, a.y), mix(v0.z, v1.z, a.z));
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE longlong3 mix(const longlong3& v0, const longlong3& v1, const long long a)
                {
                    return make_longlong3(mix(v0.x, v1.x, a), mix(v0.y, v1.y, a), mix(v0.z, v1.z, a));
                }
                // clamp
                RTLIB_INLINE RTLIB_HOST_DEVICE longlong3 clamp(const longlong3& v, const longlong3& low, const longlong3& high)
                {
                    return make_longlong3(clamp(v.x, low.x, high.x), clamp(v.y, low.y, high.y), clamp(v.z, low.z, high.z));
                }
                // powN
                RTLIB_INLINE RTLIB_HOST_DEVICE longlong3 pow2(const longlong3& v)
                {
                    return make_longlong3(pow2(v.x), pow2(v.y), pow2(v.z));
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE longlong3 pow3(const longlong3& v)
                {
                    return make_longlong3(pow3(v.x), pow3(v.y), pow3(v.z));
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE longlong3 pow4(const longlong3& v)
                {
                    return make_longlong3(pow4(v.x), pow4(v.y), pow4(v.z));
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE longlong3 pow5(const longlong3& v)
                {
                    return make_longlong3(pow5(v.x), pow5(v.y), pow5(v.z));
                }
                // max
                RTLIB_INLINE RTLIB_HOST_DEVICE longlong4 max(const longlong4& v0, const longlong4& v1)
                {
#if defined(__cplusplus) && !defined(__CUDA_ARCH__)
                    using std::max;
#endif
                    return make_longlong4(max(v0.x, v1.x), max(v0.y, v1.y), max(v0.z, v1.z), max(v0.w, v1.w));
                }
                // min
                RTLIB_INLINE RTLIB_HOST_DEVICE longlong4 min(const longlong4& v0, const longlong4& v1)
                {
#if defined(__cplusplus) && !defined(__CUDA_ARCH__)
                    using std::min;
#endif
                    return make_longlong4(min(v0.x, v1.x), min(v0.y, v1.y), min(v0.z, v1.z), min(v0.w, v1.w));
                }
                // mix
                RTLIB_INLINE RTLIB_HOST_DEVICE longlong4 mix(const longlong4& v0, const longlong4& v1, const longlong4& a)
                {
                    return make_longlong4(mix(v0.x, v1.x, a.x), mix(v0.y, v1.y, a.y), mix(v0.z, v1.z, a.z), mix(v0.w, v1.w, a.w));
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE longlong4 mix(const longlong4& v0, const longlong4& v1, const long long a)
                {
                    return make_longlong4(mix(v0.x, v1.x, a), mix(v0.y, v1.y, a), mix(v0.z, v1.z, a), mix(v0.w, v1.w, a));
                }
                // clamp
                RTLIB_INLINE RTLIB_HOST_DEVICE longlong4 clamp(const longlong4& v, const longlong4& low, const longlong4& high)
                {
                    return make_longlong4(clamp(v.x, low.x, high.x), clamp(v.y, low.y, high.y), clamp(v.z, low.z, high.z), clamp(v.w, low.w, high.w));
                }
                // powN
                RTLIB_INLINE RTLIB_HOST_DEVICE longlong4 pow2(const longlong4& v)
                {
                    return make_longlong4(pow2(v.x), pow2(v.y), pow2(v.z), pow2(v.w));
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE longlong4 pow3(const longlong4& v)
                {
                    return make_longlong4(pow3(v.x), pow3(v.y), pow3(v.z), pow3(v.w));
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE longlong4 pow4(const longlong4& v)
                {
                    return make_longlong4(pow4(v.x), pow4(v.y), pow4(v.z), pow4(v.w));
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE longlong4 pow5(const longlong4& v)
                {
                    return make_longlong4(pow5(v.x), pow5(v.y), pow5(v.z), pow5(v.w));
                }
                // max
                RTLIB_INLINE RTLIB_HOST_DEVICE uchar2 max(const uchar2& v0, const uchar2& v1)
                {
#if defined(__cplusplus) && !defined(__CUDA_ARCH__)
                    using std::max;
#endif
                    return make_uchar2(max(v0.x, v1.x), max(v0.y, v1.y));
                }
                // min
                RTLIB_INLINE RTLIB_HOST_DEVICE uchar2 min(const uchar2& v0, const uchar2& v1)
                {
#if defined(__cplusplus) && !defined(__CUDA_ARCH__)
                    using std::min;
#endif
                    return make_uchar2(min(v0.x, v1.x), min(v0.y, v1.y));
                }
                // mix
                RTLIB_INLINE RTLIB_HOST_DEVICE uchar2 mix(const uchar2& v0, const uchar2& v1, const uchar2& a)
                {
                    return make_uchar2(mix(v0.x, v1.x, a.x), mix(v0.y, v1.y, a.y));
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE uchar2 mix(const uchar2& v0, const uchar2& v1, const unsigned char a)
                {
                    return make_uchar2(mix(v0.x, v1.x, a), mix(v0.y, v1.y, a));
                }
                // clamp
                RTLIB_INLINE RTLIB_HOST_DEVICE uchar2 clamp(const uchar2& v, const uchar2& low, const uchar2& high)
                {
                    return make_uchar2(clamp(v.x, low.x, high.x), clamp(v.y, low.y, high.y));
                }
                // powN
                RTLIB_INLINE RTLIB_HOST_DEVICE uchar2 pow2(const uchar2& v)
                {
                    return make_uchar2(pow2(v.x), pow2(v.y));
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE uchar2 pow3(const uchar2& v)
                {
                    return make_uchar2(pow3(v.x), pow3(v.y));
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE uchar2 pow4(const uchar2& v)
                {
                    return make_uchar2(pow4(v.x), pow4(v.y));
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE uchar2 pow5(const uchar2& v)
                {
                    return make_uchar2(pow5(v.x), pow5(v.y));
                }
                // max
                RTLIB_INLINE RTLIB_HOST_DEVICE uchar3 max(const uchar3& v0, const uchar3& v1)
                {
#if defined(__cplusplus) && !defined(__CUDA_ARCH__)
                    using std::max;
#endif
                    return make_uchar3(max(v0.x, v1.x), max(v0.y, v1.y), max(v0.z, v1.z));
                }
                // min
                RTLIB_INLINE RTLIB_HOST_DEVICE uchar3 min(const uchar3& v0, const uchar3& v1)
                {
#if defined(__cplusplus) && !defined(__CUDA_ARCH__)
                    using std::min;
#endif
                    return make_uchar3(min(v0.x, v1.x), min(v0.y, v1.y), min(v0.z, v1.z));
                }
                // mix
                RTLIB_INLINE RTLIB_HOST_DEVICE uchar3 mix(const uchar3& v0, const uchar3& v1, const uchar3& a)
                {
                    return make_uchar3(mix(v0.x, v1.x, a.x), mix(v0.y, v1.y, a.y), mix(v0.z, v1.z, a.z));
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE uchar3 mix(const uchar3& v0, const uchar3& v1, const unsigned char a)
                {
                    return make_uchar3(mix(v0.x, v1.x, a), mix(v0.y, v1.y, a), mix(v0.z, v1.z, a));
                }
                // clamp
                RTLIB_INLINE RTLIB_HOST_DEVICE uchar3 clamp(const uchar3& v, const uchar3& low, const uchar3& high)
                {
                    return make_uchar3(clamp(v.x, low.x, high.x), clamp(v.y, low.y, high.y), clamp(v.z, low.z, high.z));
                }
                // powN
                RTLIB_INLINE RTLIB_HOST_DEVICE uchar3 pow2(const uchar3& v)
                {
                    return make_uchar3(pow2(v.x), pow2(v.y), pow2(v.z));
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE uchar3 pow3(const uchar3& v)
                {
                    return make_uchar3(pow3(v.x), pow3(v.y), pow3(v.z));
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE uchar3 pow4(const uchar3& v)
                {
                    return make_uchar3(pow4(v.x), pow4(v.y), pow4(v.z));
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE uchar3 pow5(const uchar3& v)
                {
                    return make_uchar3(pow5(v.x), pow5(v.y), pow5(v.z));
                }
                // max
                RTLIB_INLINE RTLIB_HOST_DEVICE uchar4 max(const uchar4& v0, const uchar4& v1)
                {
#if defined(__cplusplus) && !defined(__CUDA_ARCH__)
                    using std::max;
#endif
                    return make_uchar4(max(v0.x, v1.x), max(v0.y, v1.y), max(v0.z, v1.z), max(v0.w, v1.w));
                }
                // min
                RTLIB_INLINE RTLIB_HOST_DEVICE uchar4 min(const uchar4& v0, const uchar4& v1)
                {
#if defined(__cplusplus) && !defined(__CUDA_ARCH__)
                    using std::min;
#endif
                    return make_uchar4(min(v0.x, v1.x), min(v0.y, v1.y), min(v0.z, v1.z), min(v0.w, v1.w));
                }
                // mix
                RTLIB_INLINE RTLIB_HOST_DEVICE uchar4 mix(const uchar4& v0, const uchar4& v1, const uchar4& a)
                {
                    return make_uchar4(mix(v0.x, v1.x, a.x), mix(v0.y, v1.y, a.y), mix(v0.z, v1.z, a.z), mix(v0.w, v1.w, a.w));
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE uchar4 mix(const uchar4& v0, const uchar4& v1, const unsigned char a)
                {
                    return make_uchar4(mix(v0.x, v1.x, a), mix(v0.y, v1.y, a), mix(v0.z, v1.z, a), mix(v0.w, v1.w, a));
                }
                // clamp
                RTLIB_INLINE RTLIB_HOST_DEVICE uchar4 clamp(const uchar4& v, const uchar4& low, const uchar4& high)
                {
                    return make_uchar4(clamp(v.x, low.x, high.x), clamp(v.y, low.y, high.y), clamp(v.z, low.z, high.z), clamp(v.w, low.w, high.w));
                }
                // powN
                RTLIB_INLINE RTLIB_HOST_DEVICE uchar4 pow2(const uchar4& v)
                {
                    return make_uchar4(pow2(v.x), pow2(v.y), pow2(v.z), pow2(v.w));
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE uchar4 pow3(const uchar4& v)
                {
                    return make_uchar4(pow3(v.x), pow3(v.y), pow3(v.z), pow3(v.w));
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE uchar4 pow4(const uchar4& v)
                {
                    return make_uchar4(pow4(v.x), pow4(v.y), pow4(v.z), pow4(v.w));
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE uchar4 pow5(const uchar4& v)
                {
                    return make_uchar4(pow5(v.x), pow5(v.y), pow5(v.z), pow5(v.w));
                }
                // max
                RTLIB_INLINE RTLIB_HOST_DEVICE ushort2 max(const ushort2& v0, const ushort2& v1)
                {
#if defined(__cplusplus) && !defined(__CUDA_ARCH__)
                    using std::max;
#endif
                    return make_ushort2(max(v0.x, v1.x), max(v0.y, v1.y));
                }
                // min
                RTLIB_INLINE RTLIB_HOST_DEVICE ushort2 min(const ushort2& v0, const ushort2& v1)
                {
#if defined(__cplusplus) && !defined(__CUDA_ARCH__)
                    using std::min;
#endif
                    return make_ushort2(min(v0.x, v1.x), min(v0.y, v1.y));
                }
                // mix
                RTLIB_INLINE RTLIB_HOST_DEVICE ushort2 mix(const ushort2& v0, const ushort2& v1, const ushort2& a)
                {
                    return make_ushort2(mix(v0.x, v1.x, a.x), mix(v0.y, v1.y, a.y));
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE ushort2 mix(const ushort2& v0, const ushort2& v1, const unsigned short a)
                {
                    return make_ushort2(mix(v0.x, v1.x, a), mix(v0.y, v1.y, a));
                }
                // clamp
                RTLIB_INLINE RTLIB_HOST_DEVICE ushort2 clamp(const ushort2& v, const ushort2& low, const ushort2& high)
                {
                    return make_ushort2(clamp(v.x, low.x, high.x), clamp(v.y, low.y, high.y));
                }
                // powN
                RTLIB_INLINE RTLIB_HOST_DEVICE ushort2 pow2(const ushort2& v)
                {
                    return make_ushort2(pow2(v.x), pow2(v.y));
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE ushort2 pow3(const ushort2& v)
                {
                    return make_ushort2(pow3(v.x), pow3(v.y));
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE ushort2 pow4(const ushort2& v)
                {
                    return make_ushort2(pow4(v.x), pow4(v.y));
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE ushort2 pow5(const ushort2& v)
                {
                    return make_ushort2(pow5(v.x), pow5(v.y));
                }
                // max
                RTLIB_INLINE RTLIB_HOST_DEVICE ushort3 max(const ushort3& v0, const ushort3& v1)
                {
#if defined(__cplusplus) && !defined(__CUDA_ARCH__)
                    using std::max;
#endif
                    return make_ushort3(max(v0.x, v1.x), max(v0.y, v1.y), max(v0.z, v1.z));
                }
                // min
                RTLIB_INLINE RTLIB_HOST_DEVICE ushort3 min(const ushort3& v0, const ushort3& v1)
                {
#if defined(__cplusplus) && !defined(__CUDA_ARCH__)
                    using std::min;
#endif
                    return make_ushort3(min(v0.x, v1.x), min(v0.y, v1.y), min(v0.z, v1.z));
                }
                // mix
                RTLIB_INLINE RTLIB_HOST_DEVICE ushort3 mix(const ushort3& v0, const ushort3& v1, const ushort3& a)
                {
                    return make_ushort3(mix(v0.x, v1.x, a.x), mix(v0.y, v1.y, a.y), mix(v0.z, v1.z, a.z));
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE ushort3 mix(const ushort3& v0, const ushort3& v1, const unsigned short a)
                {
                    return make_ushort3(mix(v0.x, v1.x, a), mix(v0.y, v1.y, a), mix(v0.z, v1.z, a));
                }
                // clamp
                RTLIB_INLINE RTLIB_HOST_DEVICE ushort3 clamp(const ushort3& v, const ushort3& low, const ushort3& high)
                {
                    return make_ushort3(clamp(v.x, low.x, high.x), clamp(v.y, low.y, high.y), clamp(v.z, low.z, high.z));
                }
                // powN
                RTLIB_INLINE RTLIB_HOST_DEVICE ushort3 pow2(const ushort3& v)
                {
                    return make_ushort3(pow2(v.x), pow2(v.y), pow2(v.z));
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE ushort3 pow3(const ushort3& v)
                {
                    return make_ushort3(pow3(v.x), pow3(v.y), pow3(v.z));
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE ushort3 pow4(const ushort3& v)
                {
                    return make_ushort3(pow4(v.x), pow4(v.y), pow4(v.z));
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE ushort3 pow5(const ushort3& v)
                {
                    return make_ushort3(pow5(v.x), pow5(v.y), pow5(v.z));
                }
                // max
                RTLIB_INLINE RTLIB_HOST_DEVICE ushort4 max(const ushort4& v0, const ushort4& v1)
                {
#if defined(__cplusplus) && !defined(__CUDA_ARCH__)
                    using std::max;
#endif
                    return make_ushort4(max(v0.x, v1.x), max(v0.y, v1.y), max(v0.z, v1.z), max(v0.w, v1.w));
                }
                // min
                RTLIB_INLINE RTLIB_HOST_DEVICE ushort4 min(const ushort4& v0, const ushort4& v1)
                {
#if defined(__cplusplus) && !defined(__CUDA_ARCH__)
                    using std::min;
#endif
                    return make_ushort4(min(v0.x, v1.x), min(v0.y, v1.y), min(v0.z, v1.z), min(v0.w, v1.w));
                }
                // mix
                RTLIB_INLINE RTLIB_HOST_DEVICE ushort4 mix(const ushort4& v0, const ushort4& v1, const ushort4& a)
                {
                    return make_ushort4(mix(v0.x, v1.x, a.x), mix(v0.y, v1.y, a.y), mix(v0.z, v1.z, a.z), mix(v0.w, v1.w, a.w));
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE ushort4 mix(const ushort4& v0, const ushort4& v1, const unsigned short a)
                {
                    return make_ushort4(mix(v0.x, v1.x, a), mix(v0.y, v1.y, a), mix(v0.z, v1.z, a), mix(v0.w, v1.w, a));
                }
                // clamp
                RTLIB_INLINE RTLIB_HOST_DEVICE ushort4 clamp(const ushort4& v, const ushort4& low, const ushort4& high)
                {
                    return make_ushort4(clamp(v.x, low.x, high.x), clamp(v.y, low.y, high.y), clamp(v.z, low.z, high.z), clamp(v.w, low.w, high.w));
                }
                // powN
                RTLIB_INLINE RTLIB_HOST_DEVICE ushort4 pow2(const ushort4& v)
                {
                    return make_ushort4(pow2(v.x), pow2(v.y), pow2(v.z), pow2(v.w));
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE ushort4 pow3(const ushort4& v)
                {
                    return make_ushort4(pow3(v.x), pow3(v.y), pow3(v.z), pow3(v.w));
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE ushort4 pow4(const ushort4& v)
                {
                    return make_ushort4(pow4(v.x), pow4(v.y), pow4(v.z), pow4(v.w));
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE ushort4 pow5(const ushort4& v)
                {
                    return make_ushort4(pow5(v.x), pow5(v.y), pow5(v.z), pow5(v.w));
                }
                // max
                RTLIB_INLINE RTLIB_HOST_DEVICE uint2 max(const uint2& v0, const uint2& v1)
                {
#if defined(__cplusplus) && !defined(__CUDA_ARCH__)
                    using std::max;
#endif
                    return make_uint2(max(v0.x, v1.x), max(v0.y, v1.y));
                }
                // min
                RTLIB_INLINE RTLIB_HOST_DEVICE uint2 min(const uint2& v0, const uint2& v1)
                {
#if defined(__cplusplus) && !defined(__CUDA_ARCH__)
                    using std::min;
#endif
                    return make_uint2(min(v0.x, v1.x), min(v0.y, v1.y));
                }
                // mix
                RTLIB_INLINE RTLIB_HOST_DEVICE uint2 mix(const uint2& v0, const uint2& v1, const uint2& a)
                {
                    return make_uint2(mix(v0.x, v1.x, a.x), mix(v0.y, v1.y, a.y));
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE uint2 mix(const uint2& v0, const uint2& v1, const unsigned int a)
                {
                    return make_uint2(mix(v0.x, v1.x, a), mix(v0.y, v1.y, a));
                }
                // clamp
                RTLIB_INLINE RTLIB_HOST_DEVICE uint2 clamp(const uint2& v, const uint2& low, const uint2& high)
                {
                    return make_uint2(clamp(v.x, low.x, high.x), clamp(v.y, low.y, high.y));
                }
                // powN
                RTLIB_INLINE RTLIB_HOST_DEVICE uint2 pow2(const uint2& v)
                {
                    return make_uint2(pow2(v.x), pow2(v.y));
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE uint2 pow3(const uint2& v)
                {
                    return make_uint2(pow3(v.x), pow3(v.y));
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE uint2 pow4(const uint2& v)
                {
                    return make_uint2(pow4(v.x), pow4(v.y));
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE uint2 pow5(const uint2& v)
                {
                    return make_uint2(pow5(v.x), pow5(v.y));
                }
                // max
                RTLIB_INLINE RTLIB_HOST_DEVICE uint3 max(const uint3& v0, const uint3& v1)
                {
#if defined(__cplusplus) && !defined(__CUDA_ARCH__)
                    using std::max;
#endif
                    return make_uint3(max(v0.x, v1.x), max(v0.y, v1.y), max(v0.z, v1.z));
                }
                // min
                RTLIB_INLINE RTLIB_HOST_DEVICE uint3 min(const uint3& v0, const uint3& v1)
                {
#if defined(__cplusplus) && !defined(__CUDA_ARCH__)
                    using std::min;
#endif
                    return make_uint3(min(v0.x, v1.x), min(v0.y, v1.y), min(v0.z, v1.z));
                }
                // mix
                RTLIB_INLINE RTLIB_HOST_DEVICE uint3 mix(const uint3& v0, const uint3& v1, const uint3& a)
                {
                    return make_uint3(mix(v0.x, v1.x, a.x), mix(v0.y, v1.y, a.y), mix(v0.z, v1.z, a.z));
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE uint3 mix(const uint3& v0, const uint3& v1, const unsigned int a)
                {
                    return make_uint3(mix(v0.x, v1.x, a), mix(v0.y, v1.y, a), mix(v0.z, v1.z, a));
                }
                // clamp
                RTLIB_INLINE RTLIB_HOST_DEVICE uint3 clamp(const uint3& v, const uint3& low, const uint3& high)
                {
                    return make_uint3(clamp(v.x, low.x, high.x), clamp(v.y, low.y, high.y), clamp(v.z, low.z, high.z));
                }
                // powN
                RTLIB_INLINE RTLIB_HOST_DEVICE uint3 pow2(const uint3& v)
                {
                    return make_uint3(pow2(v.x), pow2(v.y), pow2(v.z));
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE uint3 pow3(const uint3& v)
                {
                    return make_uint3(pow3(v.x), pow3(v.y), pow3(v.z));
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE uint3 pow4(const uint3& v)
                {
                    return make_uint3(pow4(v.x), pow4(v.y), pow4(v.z));
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE uint3 pow5(const uint3& v)
                {
                    return make_uint3(pow5(v.x), pow5(v.y), pow5(v.z));
                }
                // max
                RTLIB_INLINE RTLIB_HOST_DEVICE uint4 max(const uint4& v0, const uint4& v1)
                {
#if defined(__cplusplus) && !defined(__CUDA_ARCH__)
                    using std::max;
#endif
                    return make_uint4(max(v0.x, v1.x), max(v0.y, v1.y), max(v0.z, v1.z), max(v0.w, v1.w));
                }
                // min
                RTLIB_INLINE RTLIB_HOST_DEVICE uint4 min(const uint4& v0, const uint4& v1)
                {
#if defined(__cplusplus) && !defined(__CUDA_ARCH__)
                    using std::min;
#endif
                    return make_uint4(min(v0.x, v1.x), min(v0.y, v1.y), min(v0.z, v1.z), min(v0.w, v1.w));
                }
                // mix
                RTLIB_INLINE RTLIB_HOST_DEVICE uint4 mix(const uint4& v0, const uint4& v1, const uint4& a)
                {
                    return make_uint4(mix(v0.x, v1.x, a.x), mix(v0.y, v1.y, a.y), mix(v0.z, v1.z, a.z), mix(v0.w, v1.w, a.w));
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE uint4 mix(const uint4& v0, const uint4& v1, const unsigned int a)
                {
                    return make_uint4(mix(v0.x, v1.x, a), mix(v0.y, v1.y, a), mix(v0.z, v1.z, a), mix(v0.w, v1.w, a));
                }
                // clamp
                RTLIB_INLINE RTLIB_HOST_DEVICE uint4 clamp(const uint4& v, const uint4& low, const uint4& high)
                {
                    return make_uint4(clamp(v.x, low.x, high.x), clamp(v.y, low.y, high.y), clamp(v.z, low.z, high.z), clamp(v.w, low.w, high.w));
                }
                // powN
                RTLIB_INLINE RTLIB_HOST_DEVICE uint4 pow2(const uint4& v)
                {
                    return make_uint4(pow2(v.x), pow2(v.y), pow2(v.z), pow2(v.w));
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE uint4 pow3(const uint4& v)
                {
                    return make_uint4(pow3(v.x), pow3(v.y), pow3(v.z), pow3(v.w));
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE uint4 pow4(const uint4& v)
                {
                    return make_uint4(pow4(v.x), pow4(v.y), pow4(v.z), pow4(v.w));
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE uint4 pow5(const uint4& v)
                {
                    return make_uint4(pow5(v.x), pow5(v.y), pow5(v.z), pow5(v.w));
                }
                // max
                RTLIB_INLINE RTLIB_HOST_DEVICE ulong2 max(const ulong2& v0, const ulong2& v1)
                {
#if defined(__cplusplus) && !defined(__CUDA_ARCH__)
                    using std::max;
#endif
                    return make_ulong2(max(v0.x, v1.x), max(v0.y, v1.y));
                }
                // min
                RTLIB_INLINE RTLIB_HOST_DEVICE ulong2 min(const ulong2& v0, const ulong2& v1)
                {
#if defined(__cplusplus) && !defined(__CUDA_ARCH__)
                    using std::min;
#endif
                    return make_ulong2(min(v0.x, v1.x), min(v0.y, v1.y));
                }
                // mix
                RTLIB_INLINE RTLIB_HOST_DEVICE ulong2 mix(const ulong2& v0, const ulong2& v1, const ulong2& a)
                {
                    return make_ulong2(mix(v0.x, v1.x, a.x), mix(v0.y, v1.y, a.y));
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE ulong2 mix(const ulong2& v0, const ulong2& v1, const unsigned long a)
                {
                    return make_ulong2(mix(v0.x, v1.x, a), mix(v0.y, v1.y, a));
                }
                // clamp
                RTLIB_INLINE RTLIB_HOST_DEVICE ulong2 clamp(const ulong2& v, const ulong2& low, const ulong2& high)
                {
                    return make_ulong2(clamp(v.x, low.x, high.x), clamp(v.y, low.y, high.y));
                }
                // powN
                RTLIB_INLINE RTLIB_HOST_DEVICE ulong2 pow2(const ulong2& v)
                {
                    return make_ulong2(pow2(v.x), pow2(v.y));
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE ulong2 pow3(const ulong2& v)
                {
                    return make_ulong2(pow3(v.x), pow3(v.y));
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE ulong2 pow4(const ulong2& v)
                {
                    return make_ulong2(pow4(v.x), pow4(v.y));
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE ulong2 pow5(const ulong2& v)
                {
                    return make_ulong2(pow5(v.x), pow5(v.y));
                }
                // max
                RTLIB_INLINE RTLIB_HOST_DEVICE ulong3 max(const ulong3& v0, const ulong3& v1)
                {
#if defined(__cplusplus) && !defined(__CUDA_ARCH__)
                    using std::max;
#endif
                    return make_ulong3(max(v0.x, v1.x), max(v0.y, v1.y), max(v0.z, v1.z));
                }
                // min
                RTLIB_INLINE RTLIB_HOST_DEVICE ulong3 min(const ulong3& v0, const ulong3& v1)
                {
#if defined(__cplusplus) && !defined(__CUDA_ARCH__)
                    using std::min;
#endif
                    return make_ulong3(min(v0.x, v1.x), min(v0.y, v1.y), min(v0.z, v1.z));
                }
                // mix
                RTLIB_INLINE RTLIB_HOST_DEVICE ulong3 mix(const ulong3& v0, const ulong3& v1, const ulong3& a)
                {
                    return make_ulong3(mix(v0.x, v1.x, a.x), mix(v0.y, v1.y, a.y), mix(v0.z, v1.z, a.z));
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE ulong3 mix(const ulong3& v0, const ulong3& v1, const unsigned long a)
                {
                    return make_ulong3(mix(v0.x, v1.x, a), mix(v0.y, v1.y, a), mix(v0.z, v1.z, a));
                }
                // clamp
                RTLIB_INLINE RTLIB_HOST_DEVICE ulong3 clamp(const ulong3& v, const ulong3& low, const ulong3& high)
                {
                    return make_ulong3(clamp(v.x, low.x, high.x), clamp(v.y, low.y, high.y), clamp(v.z, low.z, high.z));
                }
                // powN
                RTLIB_INLINE RTLIB_HOST_DEVICE ulong3 pow2(const ulong3& v)
                {
                    return make_ulong3(pow2(v.x), pow2(v.y), pow2(v.z));
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE ulong3 pow3(const ulong3& v)
                {
                    return make_ulong3(pow3(v.x), pow3(v.y), pow3(v.z));
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE ulong3 pow4(const ulong3& v)
                {
                    return make_ulong3(pow4(v.x), pow4(v.y), pow4(v.z));
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE ulong3 pow5(const ulong3& v)
                {
                    return make_ulong3(pow5(v.x), pow5(v.y), pow5(v.z));
                }
                // max
                RTLIB_INLINE RTLIB_HOST_DEVICE ulong4 max(const ulong4& v0, const ulong4& v1)
                {
#if defined(__cplusplus) && !defined(__CUDA_ARCH__)
                    using std::max;
#endif
                    return make_ulong4(max(v0.x, v1.x), max(v0.y, v1.y), max(v0.z, v1.z), max(v0.w, v1.w));
                }
                // min
                RTLIB_INLINE RTLIB_HOST_DEVICE ulong4 min(const ulong4& v0, const ulong4& v1)
                {
#if defined(__cplusplus) && !defined(__CUDA_ARCH__)
                    using std::min;
#endif
                    return make_ulong4(min(v0.x, v1.x), min(v0.y, v1.y), min(v0.z, v1.z), min(v0.w, v1.w));
                }
                // mix
                RTLIB_INLINE RTLIB_HOST_DEVICE ulong4 mix(const ulong4& v0, const ulong4& v1, const ulong4& a)
                {
                    return make_ulong4(mix(v0.x, v1.x, a.x), mix(v0.y, v1.y, a.y), mix(v0.z, v1.z, a.z), mix(v0.w, v1.w, a.w));
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE ulong4 mix(const ulong4& v0, const ulong4& v1, const unsigned long a)
                {
                    return make_ulong4(mix(v0.x, v1.x, a), mix(v0.y, v1.y, a), mix(v0.z, v1.z, a), mix(v0.w, v1.w, a));
                }
                // clamp
                RTLIB_INLINE RTLIB_HOST_DEVICE ulong4 clamp(const ulong4& v, const ulong4& low, const ulong4& high)
                {
                    return make_ulong4(clamp(v.x, low.x, high.x), clamp(v.y, low.y, high.y), clamp(v.z, low.z, high.z), clamp(v.w, low.w, high.w));
                }
                // powN
                RTLIB_INLINE RTLIB_HOST_DEVICE ulong4 pow2(const ulong4& v)
                {
                    return make_ulong4(pow2(v.x), pow2(v.y), pow2(v.z), pow2(v.w));
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE ulong4 pow3(const ulong4& v)
                {
                    return make_ulong4(pow3(v.x), pow3(v.y), pow3(v.z), pow3(v.w));
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE ulong4 pow4(const ulong4& v)
                {
                    return make_ulong4(pow4(v.x), pow4(v.y), pow4(v.z), pow4(v.w));
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE ulong4 pow5(const ulong4& v)
                {
                    return make_ulong4(pow5(v.x), pow5(v.y), pow5(v.z), pow5(v.w));
                }
                // max
                RTLIB_INLINE RTLIB_HOST_DEVICE ulonglong2 max(const ulonglong2& v0, const ulonglong2& v1)
                {
#if defined(__cplusplus) && !defined(__CUDA_ARCH__)
                    using std::max;
#endif
                    return make_ulonglong2(max(v0.x, v1.x), max(v0.y, v1.y));
                }
                // min
                RTLIB_INLINE RTLIB_HOST_DEVICE ulonglong2 min(const ulonglong2& v0, const ulonglong2& v1)
                {
#if defined(__cplusplus) && !defined(__CUDA_ARCH__)
                    using std::min;
#endif
                    return make_ulonglong2(min(v0.x, v1.x), min(v0.y, v1.y));
                }
                // mix
                RTLIB_INLINE RTLIB_HOST_DEVICE ulonglong2 mix(const ulonglong2& v0, const ulonglong2& v1, const ulonglong2& a)
                {
                    return make_ulonglong2(mix(v0.x, v1.x, a.x), mix(v0.y, v1.y, a.y));
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE ulonglong2 mix(const ulonglong2& v0, const ulonglong2& v1, const unsigned long long a)
                {
                    return make_ulonglong2(mix(v0.x, v1.x, a), mix(v0.y, v1.y, a));
                }
                // clamp
                RTLIB_INLINE RTLIB_HOST_DEVICE ulonglong2 clamp(const ulonglong2& v, const ulonglong2& low, const ulonglong2& high)
                {
                    return make_ulonglong2(clamp(v.x, low.x, high.x), clamp(v.y, low.y, high.y));
                }
                // powN
                RTLIB_INLINE RTLIB_HOST_DEVICE ulonglong2 pow2(const ulonglong2& v)
                {
                    return make_ulonglong2(pow2(v.x), pow2(v.y));
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE ulonglong2 pow3(const ulonglong2& v)
                {
                    return make_ulonglong2(pow3(v.x), pow3(v.y));
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE ulonglong2 pow4(const ulonglong2& v)
                {
                    return make_ulonglong2(pow4(v.x), pow4(v.y));
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE ulonglong2 pow5(const ulonglong2& v)
                {
                    return make_ulonglong2(pow5(v.x), pow5(v.y));
                }
                // max
                RTLIB_INLINE RTLIB_HOST_DEVICE ulonglong3 max(const ulonglong3& v0, const ulonglong3& v1)
                {
#if defined(__cplusplus) && !defined(__CUDA_ARCH__)
                    using std::max;
#endif
                    return make_ulonglong3(max(v0.x, v1.x), max(v0.y, v1.y), max(v0.z, v1.z));
                }
                // min
                RTLIB_INLINE RTLIB_HOST_DEVICE ulonglong3 min(const ulonglong3& v0, const ulonglong3& v1)
                {
#if defined(__cplusplus) && !defined(__CUDA_ARCH__)
                    using std::min;
#endif
                    return make_ulonglong3(min(v0.x, v1.x), min(v0.y, v1.y), min(v0.z, v1.z));
                }
                // mix
                RTLIB_INLINE RTLIB_HOST_DEVICE ulonglong3 mix(const ulonglong3& v0, const ulonglong3& v1, const ulonglong3& a)
                {
                    return make_ulonglong3(mix(v0.x, v1.x, a.x), mix(v0.y, v1.y, a.y), mix(v0.z, v1.z, a.z));
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE ulonglong3 mix(const ulonglong3& v0, const ulonglong3& v1, const unsigned long long a)
                {
                    return make_ulonglong3(mix(v0.x, v1.x, a), mix(v0.y, v1.y, a), mix(v0.z, v1.z, a));
                }
                // clamp
                RTLIB_INLINE RTLIB_HOST_DEVICE ulonglong3 clamp(const ulonglong3& v, const ulonglong3& low, const ulonglong3& high)
                {
                    return make_ulonglong3(clamp(v.x, low.x, high.x), clamp(v.y, low.y, high.y), clamp(v.z, low.z, high.z));
                }
                // powN
                RTLIB_INLINE RTLIB_HOST_DEVICE ulonglong3 pow2(const ulonglong3& v)
                {
                    return make_ulonglong3(pow2(v.x), pow2(v.y), pow2(v.z));
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE ulonglong3 pow3(const ulonglong3& v)
                {
                    return make_ulonglong3(pow3(v.x), pow3(v.y), pow3(v.z));
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE ulonglong3 pow4(const ulonglong3& v)
                {
                    return make_ulonglong3(pow4(v.x), pow4(v.y), pow4(v.z));
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE ulonglong3 pow5(const ulonglong3& v)
                {
                    return make_ulonglong3(pow5(v.x), pow5(v.y), pow5(v.z));
                }
                // max
                RTLIB_INLINE RTLIB_HOST_DEVICE ulonglong4 max(const ulonglong4& v0, const ulonglong4& v1)
                {
#if defined(__cplusplus) && !defined(__CUDA_ARCH__)
                    using std::max;
#endif
                    return make_ulonglong4(max(v0.x, v1.x), max(v0.y, v1.y), max(v0.z, v1.z), max(v0.w, v1.w));
                }
                // min
                RTLIB_INLINE RTLIB_HOST_DEVICE ulonglong4 min(const ulonglong4& v0, const ulonglong4& v1)
                {
#if defined(__cplusplus) && !defined(__CUDA_ARCH__)
                    using std::min;
#endif
                    return make_ulonglong4(min(v0.x, v1.x), min(v0.y, v1.y), min(v0.z, v1.z), min(v0.w, v1.w));
                }
                // mix
                RTLIB_INLINE RTLIB_HOST_DEVICE ulonglong4 mix(const ulonglong4& v0, const ulonglong4& v1, const ulonglong4& a)
                {
                    return make_ulonglong4(mix(v0.x, v1.x, a.x), mix(v0.y, v1.y, a.y), mix(v0.z, v1.z, a.z), mix(v0.w, v1.w, a.w));
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE ulonglong4 mix(const ulonglong4& v0, const ulonglong4& v1, const unsigned long long a)
                {
                    return make_ulonglong4(mix(v0.x, v1.x, a), mix(v0.y, v1.y, a), mix(v0.z, v1.z, a), mix(v0.w, v1.w, a));
                }
                // clamp
                RTLIB_INLINE RTLIB_HOST_DEVICE ulonglong4 clamp(const ulonglong4& v, const ulonglong4& low, const ulonglong4& high)
                {
                    return make_ulonglong4(clamp(v.x, low.x, high.x), clamp(v.y, low.y, high.y), clamp(v.z, low.z, high.z), clamp(v.w, low.w, high.w));
                }
                // powN
                RTLIB_INLINE RTLIB_HOST_DEVICE ulonglong4 pow2(const ulonglong4& v)
                {
                    return make_ulonglong4(pow2(v.x), pow2(v.y), pow2(v.z), pow2(v.w));
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE ulonglong4 pow3(const ulonglong4& v)
                {
                    return make_ulonglong4(pow3(v.x), pow3(v.y), pow3(v.z), pow3(v.w));
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE ulonglong4 pow4(const ulonglong4& v)
                {
                    return make_ulonglong4(pow4(v.x), pow4(v.y), pow4(v.z), pow4(v.w));
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE ulonglong4 pow5(const ulonglong4& v)
                {
                    return make_ulonglong4(pow5(v.x), pow5(v.y), pow5(v.z), pow5(v.w));
                }
                // max
                RTLIB_INLINE RTLIB_HOST_DEVICE float2 max(const float2& v0, const float2& v1)
                {
#if defined(__cplusplus) && !defined(__CUDA_ARCH__)
                    using std::max;
#endif
                    return make_float2(max(v0.x, v1.x), max(v0.y, v1.y));
                }
                // min
                RTLIB_INLINE RTLIB_HOST_DEVICE float2 min(const float2& v0, const float2& v1)
                {
#if defined(__cplusplus) && !defined(__CUDA_ARCH__)
                    using std::min;
#endif
                    return make_float2(min(v0.x, v1.x), min(v0.y, v1.y));
                }
                // mix
                RTLIB_INLINE RTLIB_HOST_DEVICE float2 mix(const float2& v0, const float2& v1, const float2& a)
                {
                    return make_float2(mix(v0.x, v1.x, a.x), mix(v0.y, v1.y, a.y));
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE float2 mix(const float2& v0, const float2& v1, const float a)
                {
                    return make_float2(mix(v0.x, v1.x, a), mix(v0.y, v1.y, a));
                }
                // clamp
                RTLIB_INLINE RTLIB_HOST_DEVICE float2 clamp(const float2& v, const float2& low, const float2& high)
                {
                    return make_float2(clamp(v.x, low.x, high.x), clamp(v.y, low.y, high.y));
                }
                // powN
                RTLIB_INLINE RTLIB_HOST_DEVICE float2 pow2(const float2& v)
                {
                    return make_float2(pow2(v.x), pow2(v.y));
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE float2 pow3(const float2& v)
                {
                    return make_float2(pow3(v.x), pow3(v.y));
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE float2 pow4(const float2& v)
                {
                    return make_float2(pow4(v.x), pow4(v.y));
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE float2 pow5(const float2& v)
                {
                    return make_float2(pow5(v.x), pow5(v.y));
                }
                // max
                RTLIB_INLINE RTLIB_HOST_DEVICE float3 max(const float3& v0, const float3& v1)
                {
#if defined(__cplusplus) && !defined(__CUDA_ARCH__)
                    using std::max;
#endif
                    return make_float3(max(v0.x, v1.x), max(v0.y, v1.y), max(v0.z, v1.z));
                }
                // min
                RTLIB_INLINE RTLIB_HOST_DEVICE float3 min(const float3& v0, const float3& v1)
                {
#if defined(__cplusplus) && !defined(__CUDA_ARCH__)
                    using std::min;
#endif
                    return make_float3(min(v0.x, v1.x), min(v0.y, v1.y), min(v0.z, v1.z));
                }
                // mix
                RTLIB_INLINE RTLIB_HOST_DEVICE float3 mix(const float3& v0, const float3& v1, const float3& a)
                {
                    return make_float3(mix(v0.x, v1.x, a.x), mix(v0.y, v1.y, a.y), mix(v0.z, v1.z, a.z));
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE float3 mix(const float3& v0, const float3& v1, const float a)
                {
                    return make_float3(mix(v0.x, v1.x, a), mix(v0.y, v1.y, a), mix(v0.z, v1.z, a));
                }
                // clamp
                RTLIB_INLINE RTLIB_HOST_DEVICE float3 clamp(const float3& v, const float3& low, const float3& high)
                {
                    return make_float3(clamp(v.x, low.x, high.x), clamp(v.y, low.y, high.y), clamp(v.z, low.z, high.z));
                }
                // powN
                RTLIB_INLINE RTLIB_HOST_DEVICE float3 pow2(const float3& v)
                {
                    return make_float3(pow2(v.x), pow2(v.y), pow2(v.z));
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE float3 pow3(const float3& v)
                {
                    return make_float3(pow3(v.x), pow3(v.y), pow3(v.z));
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE float3 pow4(const float3& v)
                {
                    return make_float3(pow4(v.x), pow4(v.y), pow4(v.z));
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE float3 pow5(const float3& v)
                {
                    return make_float3(pow5(v.x), pow5(v.y), pow5(v.z));
                }
                // max
                RTLIB_INLINE RTLIB_HOST_DEVICE float4 max(const float4& v0, const float4& v1)
                {
#if defined(__cplusplus) && !defined(__CUDA_ARCH__)
                    using std::max;
#endif
                    return make_float4(max(v0.x, v1.x), max(v0.y, v1.y), max(v0.z, v1.z), max(v0.w, v1.w));
                }
                // min
                RTLIB_INLINE RTLIB_HOST_DEVICE float4 min(const float4& v0, const float4& v1)
                {
#if defined(__cplusplus) && !defined(__CUDA_ARCH__)
                    using std::min;
#endif
                    return make_float4(min(v0.x, v1.x), min(v0.y, v1.y), min(v0.z, v1.z), min(v0.w, v1.w));
                }
                // mix
                RTLIB_INLINE RTLIB_HOST_DEVICE float4 mix(const float4& v0, const float4& v1, const float4& a)
                {
                    return make_float4(mix(v0.x, v1.x, a.x), mix(v0.y, v1.y, a.y), mix(v0.z, v1.z, a.z), mix(v0.w, v1.w, a.w));
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE float4 mix(const float4& v0, const float4& v1, const float a)
                {
                    return make_float4(mix(v0.x, v1.x, a), mix(v0.y, v1.y, a), mix(v0.z, v1.z, a), mix(v0.w, v1.w, a));
                }
                // clamp
                RTLIB_INLINE RTLIB_HOST_DEVICE float4 clamp(const float4& v, const float4& low, const float4& high)
                {
                    return make_float4(clamp(v.x, low.x, high.x), clamp(v.y, low.y, high.y), clamp(v.z, low.z, high.z), clamp(v.w, low.w, high.w));
                }
                // powN
                RTLIB_INLINE RTLIB_HOST_DEVICE float4 pow2(const float4& v)
                {
                    return make_float4(pow2(v.x), pow2(v.y), pow2(v.z), pow2(v.w));
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE float4 pow3(const float4& v)
                {
                    return make_float4(pow3(v.x), pow3(v.y), pow3(v.z), pow3(v.w));
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE float4 pow4(const float4& v)
                {
                    return make_float4(pow4(v.x), pow4(v.y), pow4(v.z), pow4(v.w));
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE float4 pow5(const float4& v)
                {
                    return make_float4(pow5(v.x), pow5(v.y), pow5(v.z), pow5(v.w));
                }
                // cross
                RTLIB_INLINE RTLIB_HOST_DEVICE float3 cross(const float3& v0, const float3& v1)
                {
                    return make_float3(v0.y * v1.z - v0.z * v1.y, v0.z * v1.x - v0.x * v1.z, v0.x * v1.y - v0.y * v1.x);
                }
                // dot
                RTLIB_INLINE RTLIB_HOST_DEVICE float dot(const float2& v0, const float2& v1)
                {
                    return (v0.x * v1.x) + (v0.y * v1.y);
                }
                // lengthSqr
                RTLIB_INLINE RTLIB_HOST_DEVICE float lengthSqr(const float2& v)
                {
                    return (v.x * v.x) + (v.y * v.y);
                }
                // length
                RTLIB_INLINE RTLIB_HOST_DEVICE float length(const float2& v)
                {
                    return sqrtf(lengthSqr(v));
                }
                // distanceSqr
                RTLIB_INLINE RTLIB_HOST_DEVICE float distanceSqr(const float2& v0, const float2& v1)
                {
                    return lengthSqr(v0 - v1);
                }
                // distance
                RTLIB_INLINE RTLIB_HOST_DEVICE float distance(const float2& v0, const float2& v1)
                {
                    return length(v0 - v1);
                }
                // normalize
                RTLIB_INLINE RTLIB_HOST_DEVICE float2 normalize(const float2& v)
                {
                    return v / length(v);
                }
                // expf
                RTLIB_INLINE RTLIB_HOST_DEVICE float2 expf(const float2& v)
                {
#if defined(__cplusplus) && !defined(__CUDA_ARCH__)
                    using std::expf;
#endif
                    return make_float2(::expf(v.x), ::expf(v.y));
                }
                // logf
                RTLIB_INLINE RTLIB_HOST_DEVICE float2 logf(const float2& v)
                {
#if defined(__cplusplus) && !defined(__CUDA_ARCH__)
                    using std::logf;
#endif
                    return make_float2(::logf(v.x), ::logf(v.y));
                }
                // sinf
                RTLIB_INLINE RTLIB_HOST_DEVICE float2 sinf(const float2& v)
                {
#if defined(__cplusplus) && !defined(__CUDA_ARCH__)
                    using std::sinf;
#endif
                    return make_float2(::sinf(v.x), ::sinf(v.y));
                }
                // cosf
                RTLIB_INLINE RTLIB_HOST_DEVICE float2 cosf(const float2& v)
                {
#if defined(__cplusplus) && !defined(__CUDA_ARCH__)
                    using std::cosf;
#endif
                    return make_float2(::cosf(v.x), ::cosf(v.y));
                }
                // tanf
                RTLIB_INLINE RTLIB_HOST_DEVICE float2 tanf(const float2& v)
                {
#if defined(__cplusplus) && !defined(__CUDA_ARCH__)
                    using std::tanf;
#endif
                    return make_float2(::tanf(v.x), ::tanf(v.y));
                }
                // powf
                RTLIB_INLINE RTLIB_HOST_DEVICE float2 powf(const float2& v0, const float v1)
                {
#if defined(__cplusplus) && !defined(__CUDA_ARCH__)
                    using std::powf;
#endif
                    return make_float2(::powf(v0.x, v1), ::powf(v0.y, v1));
                }
                // dot
                RTLIB_INLINE RTLIB_HOST_DEVICE float dot(const float3& v0, const float3& v1)
                {
                    return (v0.x * v1.x) + (v0.y * v1.y) + (v0.z * v1.z);
                }
                // lengthSqr
                RTLIB_INLINE RTLIB_HOST_DEVICE float lengthSqr(const float3& v)
                {
                    return (v.x * v.x) + (v.y * v.y) + (v.z * v.z);
                }
                // length
                RTLIB_INLINE RTLIB_HOST_DEVICE float length(const float3& v)
                {
                    return sqrtf(lengthSqr(v));
                }
                // distanceSqr
                RTLIB_INLINE RTLIB_HOST_DEVICE float distanceSqr(const float3& v0, const float3& v1)
                {
                    return lengthSqr(v0 - v1);
                }
                // distance
                RTLIB_INLINE RTLIB_HOST_DEVICE float distance(const float3& v0, const float3& v1)
                {
                    return length(v0 - v1);
                }
                // normalize
                RTLIB_INLINE RTLIB_HOST_DEVICE float3 normalize(const float3& v)
                {
                    return v / length(v);
                }
                // expf
                RTLIB_INLINE RTLIB_HOST_DEVICE float3 expf(const float3& v)
                {
#if defined(__cplusplus) && !defined(__CUDA_ARCH__)
                    using std::expf;
#endif
                    return make_float3(::expf(v.x), ::expf(v.y), ::expf(v.z));
                }
                // logf
                RTLIB_INLINE RTLIB_HOST_DEVICE float3 logf(const float3& v)
                {
#if defined(__cplusplus) && !defined(__CUDA_ARCH__)
                    using std::logf;
#endif
                    return make_float3(::logf(v.x), ::logf(v.y), ::logf(v.z));
                }
                // sinf
                RTLIB_INLINE RTLIB_HOST_DEVICE float3 sinf(const float3& v)
                {
#if defined(__cplusplus) && !defined(__CUDA_ARCH__)
                    using std::sinf;
#endif
                    return make_float3(::sinf(v.x), ::sinf(v.y), ::sinf(v.z));
                }
                // cosf
                RTLIB_INLINE RTLIB_HOST_DEVICE float3 cosf(const float3& v)
                {
#if defined(__cplusplus) && !defined(__CUDA_ARCH__)
                    using std::cosf;
#endif
                    return make_float3(::cosf(v.x), ::cosf(v.y), ::cosf(v.z));
                }
                // tanf
                RTLIB_INLINE RTLIB_HOST_DEVICE float3 tanf(const float3& v)
                {
#if defined(__cplusplus) && !defined(__CUDA_ARCH__)
                    using std::tanf;
#endif
                    return make_float3(::tanf(v.x), ::tanf(v.y), ::tanf(v.z));
                }
                // powf
                RTLIB_INLINE RTLIB_HOST_DEVICE float3 powf(const float3& v0, const float v1)
                {
#if defined(__cplusplus) && !defined(__CUDA_ARCH__)
                    using std::powf;
#endif
                    return make_float3(::powf(v0.x, v1), ::powf(v0.y, v1), ::powf(v0.z, v1));
                }
                // dot
                RTLIB_INLINE RTLIB_HOST_DEVICE float dot(const float4& v0, const float4& v1)
                {
                    return (v0.x * v1.x) + (v0.y * v1.y) + (v0.z * v1.z) + (v0.w * v1.w);
                }
                // lengthSqr
                RTLIB_INLINE RTLIB_HOST_DEVICE float lengthSqr(const float4& v)
                {
                    return (v.x * v.x) + (v.y * v.y) + (v.z * v.z) + (v.w * v.w);
                }
                // length
                RTLIB_INLINE RTLIB_HOST_DEVICE float length(const float4& v)
                {
                    return sqrtf(lengthSqr(v));
                }
                // distanceSqr
                RTLIB_INLINE RTLIB_HOST_DEVICE float distanceSqr(const float4& v0, const float4& v1)
                {
                    return lengthSqr(v0 - v1);
                }
                // distance
                RTLIB_INLINE RTLIB_HOST_DEVICE float distance(const float4& v0, const float4& v1)
                {
                    return length(v0 - v1);
                }
                // normalize
                RTLIB_INLINE RTLIB_HOST_DEVICE float4 normalize(const float4& v)
                {
                    return v / length(v);
                }
                // expf
                RTLIB_INLINE RTLIB_HOST_DEVICE float4 expf(const float4& v)
                {
#if defined(__cplusplus) && !defined(__CUDA_ARCH__)
                    using std::expf;
#endif
                    return make_float4(::expf(v.x), ::expf(v.y), ::expf(v.z), ::expf(v.w));
                }
                // logf
                RTLIB_INLINE RTLIB_HOST_DEVICE float4 logf(const float4& v)
                {
#if defined(__cplusplus) && !defined(__CUDA_ARCH__)
                    using std::logf;
#endif
                    return make_float4(::logf(v.x), ::logf(v.y), ::logf(v.z), ::logf(v.w));
                }
                // sinf
                RTLIB_INLINE RTLIB_HOST_DEVICE float4 sinf(const float4& v)
                {
#if defined(__cplusplus) && !defined(__CUDA_ARCH__)
                    using std::sinf;
#endif
                    return make_float4(::sinf(v.x), ::sinf(v.y), ::sinf(v.z), ::sinf(v.w));
                }
                // cosf
                RTLIB_INLINE RTLIB_HOST_DEVICE float4 cosf(const float4& v)
                {
#if defined(__cplusplus) && !defined(__CUDA_ARCH__)
                    using std::cosf;
#endif
                    return make_float4(::cosf(v.x), ::cosf(v.y), ::cosf(v.z), ::cosf(v.w));
                }
                // tanf
                RTLIB_INLINE RTLIB_HOST_DEVICE float4 tanf(const float4& v)
                {
#if defined(__cplusplus) && !defined(__CUDA_ARCH__)
                    using std::tanf;
#endif
                    return make_float4(::tanf(v.x), ::tanf(v.y), ::tanf(v.z), ::tanf(v.w));
                }
                // powf
                RTLIB_INLINE RTLIB_HOST_DEVICE float4 powf(const float4& v0, const float v1)
                {
#if defined(__cplusplus) && !defined(__CUDA_ARCH__)
                    using std::powf;
#endif
                    return make_float4(::powf(v0.x, v1), ::powf(v0.y, v1), ::powf(v0.z, v1), ::powf(v0.w, v1));
                }
                // srgb_to_rgb
                RTLIB_INLINE RTLIB_HOST_DEVICE uchar4 rgba_to_srgb(const uchar4& rgba)
                {
                    return make_uchar4(
                        static_cast<unsigned char>(linear_to_gamma(static_cast<float>(rgba.x / 255.99f)) * 255.99f),
                        static_cast<unsigned char>(linear_to_gamma(static_cast<float>(rgba.y / 255.99f)) * 255.99f),
                        static_cast<unsigned char>(linear_to_gamma(static_cast<float>(rgba.z / 255.99f)) * 255.99f),
                        rgba.w);
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE uchar4 srgb_to_rgba(const uchar4& srgb)
                {
                    return make_uchar4(
                        static_cast<unsigned char>(gamma_to_linear(static_cast<float>(srgb.x / 255.99f)) * 255.99f),
                        static_cast<unsigned char>(gamma_to_linear(static_cast<float>(srgb.y / 255.99f)) * 255.99f),
                        static_cast<unsigned char>(gamma_to_linear(static_cast<float>(srgb.z / 255.99f)) * 255.99f),
                        srgb.w);
                }

                RTLIB_INLINE RTLIB_HOST_DEVICE float2 reflect(const float2& direction, const float2& normal)
                {
                    return direction - 2.0f * RTLib::Ext::CUDA::Math::dot(direction, normal) * normal;
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE float3 reflect(const float3& direction, const float3& normal)
                {
                    return direction - 2.0f * RTLib::Ext::CUDA::Math::dot(direction, normal) * normal;
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE float schlick(float f0, float cosine_i)
                {
                    return f0 + (1.0f - f0) * RTLib::Ext::CUDA::Math::pow5(1.0f - cosine_i);
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE float fresnell(float cos_i, float ior)
                {
                    const float cos_i_2 = cos_i * cos_i;
                    const float sin_i_2 = RTLib::Ext::CUDA::Math::min(1.0f - cos_i_2, 1.0f);
                    const float sin_o_2 = ior * ior * sin_i_2;
                    const float cos_o_2 = 1.0f - sin_o_2;
                    if (cos_o_2 < 0.0f)
                    {
                        return 1.0f;
                    }
                    const float cos_i_cos_o = sqrtf(cos_i_2 * cos_o_2);
                    const float ref_p_2 = (cos_i_2 + ior * ior * cos_o_2 - 2.0f * ior * cos_i_cos_o) / (cos_i_2 + ior * ior * cos_o_2 + 2.0f * ior * cos_i_cos_o);
                    const float ref_s_2 = (ior * ior * cos_i_2 + cos_o_2 - 2.0f * ior * cos_i_cos_o) / (ior * ior * cos_i_2 + cos_o_2 + 2.0f * ior * cos_i_cos_o);
                    return (ref_p_2 + ref_s_2) / 2.0f;
                }
                struct ONB
                {
                    float3 m_Tangent;  // x�ɑ���
                    float3 m_Binormal; // y�ɑ���
                    float3 m_Normal;   // z�ɑ���
                    RTLIB_INLINE RTLIB_HOST_DEVICE ONB(const float3& normal)
                    {
                        m_Normal = normalize(normal); // w
                        if (fabsf(m_Normal.x) > fabsf(m_Normal.z))
                        {
                            m_Binormal.x = -m_Normal.y;
                            m_Binormal.y = m_Normal.x;
                            m_Binormal.z = 0.0f;
                        }
                        else
                        {
                            m_Binormal.x = 0.0f;
                            m_Binormal.y = -m_Normal.z;
                            m_Binormal.z = m_Normal.y;
                        }
                        m_Binormal = normalize(m_Binormal);
                        m_Tangent = normalize(cross(m_Normal, m_Binormal));
                    }
                    RTLIB_INLINE RTLIB_HOST_DEVICE float3 local(const float x, const float y, const float z) const
                    {
                        return x * m_Tangent + y * m_Binormal + z * m_Normal;
                    }
                    RTLIB_INLINE RTLIB_HOST_DEVICE float3 local(const float3& direction) const
                    {
                        return direction.x * m_Tangent + direction.y * m_Binormal + direction.z * m_Normal;
                    }
                };

            }

        }
    }
}
#endif