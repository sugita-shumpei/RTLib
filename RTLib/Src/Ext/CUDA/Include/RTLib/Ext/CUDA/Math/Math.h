#ifndef RTLIB_RTLIB_EXT_CUDA_MATH_RTLIB_MATH_H
#define RTLIB_RTLIB_EXT_CUDA_MATH_RTLIB_MATH_H
#include <cuda.h>
#if defined(__cplusplus) && !defined(__CUDACC__)
#include <bitset>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#endif

#include <RTLib/Ext/CUDA/Math/Preprocessors.h>
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
#ifndef RTLIB_M_INV_2PI
#define RTLIB_M_INV_2PI 0.15915494309189533576888376337251436203445964574046
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
                RTLIB_INLINE RTLIB_HOST_DEVICE RTLIB_CONSTEXPR signed char pow2(const signed char v)
                {
                    return v * v;
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE RTLIB_CONSTEXPR signed char pow3(const signed char v)
                {
                    return v * v * v;
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE RTLIB_CONSTEXPR signed char pow4(const signed char v)
                {
                    return v * v * v * v;
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE RTLIB_CONSTEXPR signed char pow5(const signed char v)
                {
                    return v * v * v * v * v;
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE RTLIB_CONSTEXPR short pow2(const short v)
                {
                    return v * v;
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE RTLIB_CONSTEXPR short pow3(const short v)
                {
                    return v * v * v;
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE RTLIB_CONSTEXPR short pow4(const short v)
                {
                    return v * v * v * v;
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE RTLIB_CONSTEXPR short pow5(const short v)
                {
                    return v * v * v * v * v;
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE RTLIB_CONSTEXPR int pow2(const int v)
                {
                    return v * v;
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE RTLIB_CONSTEXPR int pow3(const int v)
                {
                    return v * v * v;
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE RTLIB_CONSTEXPR int pow4(const int v)
                {
                    return v * v * v * v;
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE RTLIB_CONSTEXPR int pow5(const int v)
                {
                    return v * v * v * v * v;
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE RTLIB_CONSTEXPR long pow2(const long v)
                {
                    return v * v;
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE RTLIB_CONSTEXPR long pow3(const long v)
                {
                    return v * v * v;
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE RTLIB_CONSTEXPR long pow4(const long v)
                {
                    return v * v * v * v;
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE RTLIB_CONSTEXPR long pow5(const long v)
                {
                    return v * v * v * v * v;
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE RTLIB_CONSTEXPR long long pow2(const long long v)
                {
                    return v * v;
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE RTLIB_CONSTEXPR long long pow3(const long long v)
                {
                    return v * v * v;
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE RTLIB_CONSTEXPR long long pow4(const long long v)
                {
                    return v * v * v * v;
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE RTLIB_CONSTEXPR long long pow5(const long long v)
                {
                    return v * v * v * v * v;
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE RTLIB_CONSTEXPR unsigned char pow2(const unsigned char v)
                {
                    return v * v;
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE RTLIB_CONSTEXPR unsigned char pow3(const unsigned char v)
                {
                    return v * v * v;
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE RTLIB_CONSTEXPR unsigned char pow4(const unsigned char v)
                {
                    return v * v * v * v;
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE RTLIB_CONSTEXPR unsigned char pow5(const unsigned char v)
                {
                    return v * v * v * v * v;
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE RTLIB_CONSTEXPR unsigned short pow2(const unsigned short v)
                {
                    return v * v;
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE RTLIB_CONSTEXPR unsigned short pow3(const unsigned short v)
                {
                    return v * v * v;
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE RTLIB_CONSTEXPR unsigned short pow4(const unsigned short v)
                {
                    return v * v * v * v;
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE RTLIB_CONSTEXPR unsigned short pow5(const unsigned short v)
                {
                    return v * v * v * v * v;
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE RTLIB_CONSTEXPR unsigned int pow2(const unsigned int v)
                {
                    return v * v;
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE RTLIB_CONSTEXPR unsigned int pow3(const unsigned int v)
                {
                    return v * v * v;
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE RTLIB_CONSTEXPR unsigned int pow4(const unsigned int v)
                {
                    return v * v * v * v;
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE RTLIB_CONSTEXPR unsigned int pow5(const unsigned int v)
                {
                    return v * v * v * v * v;
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE RTLIB_CONSTEXPR unsigned long pow2(const unsigned long v)
                {
                    return v * v;
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE RTLIB_CONSTEXPR unsigned long pow3(const unsigned long v)
                {
                    return v * v * v;
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE RTLIB_CONSTEXPR unsigned long pow4(const unsigned long v)
                {
                    return v * v * v * v;
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE RTLIB_CONSTEXPR unsigned long pow5(const unsigned long v)
                {
                    return v * v * v * v * v;
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE RTLIB_CONSTEXPR unsigned long long pow2(const unsigned long long v)
                {
                    return v * v;
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE RTLIB_CONSTEXPR unsigned long long pow3(const unsigned long long v)
                {
                    return v * v * v;
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE RTLIB_CONSTEXPR unsigned long long pow4(const unsigned long long v)
                {
                    return v * v * v * v;
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE RTLIB_CONSTEXPR unsigned long long pow5(const unsigned long long v)
                {
                    return v * v * v * v * v;
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE RTLIB_CONSTEXPR float pow2(const float v)
                {
                    return v * v;
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE RTLIB_CONSTEXPR float pow3(const float v)
                {
                    return v * v * v;
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE RTLIB_CONSTEXPR float pow4(const float v)
                {
                    return v * v * v * v;
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE RTLIB_CONSTEXPR float pow5(const float v)
                {
                    return v * v * v * v * v;
                }
                RTLIB_INLINE RTLIB_DEVICE float powf(const float x, const float y)
                {
#if defined(__cplusplus) && !defined(__CUDA_ARCH__)
                    using std::powf;
#endif
                    return (y!=0.0f)?::powf(x,y) : static_cast<float>(x == 0.0f);
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
                RTLIB_INLINE RTLIB_DEVICE int   pop_count32(unsigned int ui32)
                {
#ifdef __CUDA_ARCH__
                    return __popc(ui32);
#else
                    return std::bitset<32>(ui32).count();
#endif
                }

            }

        }
    }
}
#endif