#ifndef RTLIB_VECTOR_FUNCTION_H
#define RTLIB_VECTOR_FUNCTION_H
#include <cuda_runtime.h>
#include <RTLib/Ext/CUDA/Math/Preprocessors.h>
#if defined(__cplusplus) && !defined(__CUDA_ARCH__)
#include <cmath>
#endif
RTLIB_INLINE RTLIB_HOST_DEVICE bool operator==(const char2 &v0, const char2 &v1)
{
  return (v0.x == v1.x) && (v0.y == v1.y);
}
RTLIB_INLINE RTLIB_HOST_DEVICE bool operator!=(const char2 &v0, const char2 &v1)
{
  return (v0.x != v1.x) || (v0.y != v1.y);
}
RTLIB_INLINE RTLIB_HOST_DEVICE char2 &operator+=(char2 &v0, const char2 &v1)
{
  v0.x += v1.x;
  v0.y += v1.y;
  return v0;
}
RTLIB_INLINE RTLIB_HOST_DEVICE char2 &operator-=(char2 &v0, const char2 &v1)
{
  v0.x -= v1.x;
  v0.y -= v1.y;
  return v0;
}
RTLIB_INLINE RTLIB_HOST_DEVICE char2 &operator*=(char2 &v0, const char2 &v1)
{
  v0.x *= v1.x;
  v0.y *= v1.y;
  return v0;
}
RTLIB_INLINE RTLIB_HOST_DEVICE char2 &operator*=(char2 &v0, const signed char v1)
{
  v0.x *= v1;
  v0.y *= v1;
  return v0;
}
RTLIB_INLINE RTLIB_HOST_DEVICE char2 &operator/=(char2 &v0, const char2 &v1)
{
  v0.x /= v1.x;
  v0.y /= v1.y;
  return v0;
}
RTLIB_INLINE RTLIB_HOST_DEVICE char2 &operator/=(char2 &v0, const signed char v1)
{
  v0.x /= v1;
  v0.y /= v1;
  return v0;
}
RTLIB_INLINE RTLIB_HOST_DEVICE char2 operator+(const char2 &v0, const char2 &v1)
{
  return make_char2(v0.x + v1.x, v0.y + v1.y);
}
RTLIB_INLINE RTLIB_HOST_DEVICE char2 operator-(const char2 &v0, const char2 &v1)
{
  return make_char2(v0.x - v1.x, v0.y - v1.y);
}
RTLIB_INLINE RTLIB_HOST_DEVICE char2 operator*(const char2 &v0, const char2 &v1)
{
  return make_char2(v0.x * v1.x, v0.y * v1.y);
}
RTLIB_INLINE RTLIB_HOST_DEVICE char2 operator*(const char2 &v0, const signed char v1)
{
  return make_char2(v0.x * v1, v0.y * v1);
}
RTLIB_INLINE RTLIB_HOST_DEVICE char2 operator*(const signed char v0, const char2 &v1)
{
  return make_char2(v0 * v1.x, v0 * v1.y);
}
RTLIB_INLINE RTLIB_HOST_DEVICE char2 operator/(const char2 &v0, const char2 &v1)
{
  return make_char2(v0.x / v1.x, v0.y / v1.y);
}
RTLIB_INLINE RTLIB_HOST_DEVICE char2 operator/(const char2 &v0, const signed char v1)
{
  return make_char2(v0.x / v1, v0.y / v1);
}
RTLIB_INLINE RTLIB_HOST_DEVICE char2 operator/(const signed char v0, const char2 &v1)
{
  return make_char2(v0 / v1.x, v0 / v1.y);
}
RTLIB_INLINE RTLIB_HOST_DEVICE bool operator==(const char3 &v0, const char3 &v1)
{
  return (v0.x == v1.x) && (v0.y == v1.y) && (v0.z == v1.z);
}
RTLIB_INLINE RTLIB_HOST_DEVICE bool operator!=(const char3 &v0, const char3 &v1)
{
  return (v0.x != v1.x) || (v0.y != v1.y) || (v0.z != v1.z);
}
RTLIB_INLINE RTLIB_HOST_DEVICE char3 &operator+=(char3 &v0, const char3 &v1)
{
  v0.x += v1.x;
  v0.y += v1.y;
  v0.z += v1.z;
  return v0;
}
RTLIB_INLINE RTLIB_HOST_DEVICE char3 &operator-=(char3 &v0, const char3 &v1)
{
  v0.x -= v1.x;
  v0.y -= v1.y;
  v0.z -= v1.z;
  return v0;
}
RTLIB_INLINE RTLIB_HOST_DEVICE char3 &operator*=(char3 &v0, const char3 &v1)
{
  v0.x *= v1.x;
  v0.y *= v1.y;
  v0.z *= v1.z;
  return v0;
}
RTLIB_INLINE RTLIB_HOST_DEVICE char3 &operator*=(char3 &v0, const signed char v1)
{
  v0.x *= v1;
  v0.y *= v1;
  v0.z *= v1;
  return v0;
}
RTLIB_INLINE RTLIB_HOST_DEVICE char3 &operator/=(char3 &v0, const char3 &v1)
{
  v0.x /= v1.x;
  v0.y /= v1.y;
  v0.z /= v1.z;
  return v0;
}
RTLIB_INLINE RTLIB_HOST_DEVICE char3 &operator/=(char3 &v0, const signed char v1)
{
  v0.x /= v1;
  v0.y /= v1;
  v0.z /= v1;
  return v0;
}
RTLIB_INLINE RTLIB_HOST_DEVICE char3 operator+(const char3 &v0, const char3 &v1)
{
  return make_char3(v0.x + v1.x, v0.y + v1.y, v0.z + v1.z);
}
RTLIB_INLINE RTLIB_HOST_DEVICE char3 operator-(const char3 &v0, const char3 &v1)
{
  return make_char3(v0.x - v1.x, v0.y - v1.y, v0.z - v1.z);
}
RTLIB_INLINE RTLIB_HOST_DEVICE char3 operator*(const char3 &v0, const char3 &v1)
{
  return make_char3(v0.x * v1.x, v0.y * v1.y, v0.z * v1.z);
}
RTLIB_INLINE RTLIB_HOST_DEVICE char3 operator*(const char3 &v0, const signed char v1)
{
  return make_char3(v0.x * v1, v0.y * v1, v0.z * v1);
}
RTLIB_INLINE RTLIB_HOST_DEVICE char3 operator*(const signed char v0, const char3 &v1)
{
  return make_char3(v0 * v1.x, v0 * v1.y, v0 * v1.z);
}
RTLIB_INLINE RTLIB_HOST_DEVICE char3 operator/(const char3 &v0, const char3 &v1)
{
  return make_char3(v0.x / v1.x, v0.y / v1.y, v0.z / v1.z);
}
RTLIB_INLINE RTLIB_HOST_DEVICE char3 operator/(const char3 &v0, const signed char v1)
{
  return make_char3(v0.x / v1, v0.y / v1, v0.z / v1);
}
RTLIB_INLINE RTLIB_HOST_DEVICE char3 operator/(const signed char v0, const char3 &v1)
{
  return make_char3(v0 / v1.x, v0 / v1.y, v0 / v1.z);
}
RTLIB_INLINE RTLIB_HOST_DEVICE bool operator==(const char4 &v0, const char4 &v1)
{
  return (v0.x == v1.x) && (v0.y == v1.y) && (v0.z == v1.z) && (v0.w == v1.w);
}
RTLIB_INLINE RTLIB_HOST_DEVICE bool operator!=(const char4 &v0, const char4 &v1)
{
  return (v0.x != v1.x) || (v0.y != v1.y) || (v0.z != v1.z) || (v0.w != v1.w);
}
RTLIB_INLINE RTLIB_HOST_DEVICE char4 &operator+=(char4 &v0, const char4 &v1)
{
  v0.x += v1.x;
  v0.y += v1.y;
  v0.z += v1.z;
  v0.w += v1.w;
  return v0;
}
RTLIB_INLINE RTLIB_HOST_DEVICE char4 &operator-=(char4 &v0, const char4 &v1)
{
  v0.x -= v1.x;
  v0.y -= v1.y;
  v0.z -= v1.z;
  v0.w -= v1.w;
  return v0;
}
RTLIB_INLINE RTLIB_HOST_DEVICE char4 &operator*=(char4 &v0, const char4 &v1)
{
  v0.x *= v1.x;
  v0.y *= v1.y;
  v0.z *= v1.z;
  v0.w *= v1.w;
  return v0;
}
RTLIB_INLINE RTLIB_HOST_DEVICE char4 &operator*=(char4 &v0, const signed char v1)
{
  v0.x *= v1;
  v0.y *= v1;
  v0.z *= v1;
  v0.w *= v1;
  return v0;
}
RTLIB_INLINE RTLIB_HOST_DEVICE char4 &operator/=(char4 &v0, const char4 &v1)
{
  v0.x /= v1.x;
  v0.y /= v1.y;
  v0.z /= v1.z;
  v0.w /= v1.w;
  return v0;
}
RTLIB_INLINE RTLIB_HOST_DEVICE char4 &operator/=(char4 &v0, const signed char v1)
{
  v0.x /= v1;
  v0.y /= v1;
  v0.z /= v1;
  v0.w /= v1;
  return v0;
}
RTLIB_INLINE RTLIB_HOST_DEVICE char4 operator+(const char4 &v0, const char4 &v1)
{
  return make_char4(v0.x + v1.x, v0.y + v1.y, v0.z + v1.z, v0.w + v1.w);
}
RTLIB_INLINE RTLIB_HOST_DEVICE char4 operator-(const char4 &v0, const char4 &v1)
{
  return make_char4(v0.x - v1.x, v0.y - v1.y, v0.z - v1.z, v0.w - v1.w);
}
RTLIB_INLINE RTLIB_HOST_DEVICE char4 operator*(const char4 &v0, const char4 &v1)
{
  return make_char4(v0.x * v1.x, v0.y * v1.y, v0.z * v1.z, v0.w * v1.w);
}
RTLIB_INLINE RTLIB_HOST_DEVICE char4 operator*(const char4 &v0, const signed char v1)
{
  return make_char4(v0.x * v1, v0.y * v1, v0.z * v1, v0.w * v1);
}
RTLIB_INLINE RTLIB_HOST_DEVICE char4 operator*(const signed char v0, const char4 &v1)
{
  return make_char4(v0 * v1.x, v0 * v1.y, v0 * v1.z, v0 * v1.w);
}
RTLIB_INLINE RTLIB_HOST_DEVICE char4 operator/(const char4 &v0, const char4 &v1)
{
  return make_char4(v0.x / v1.x, v0.y / v1.y, v0.z / v1.z, v0.w / v1.w);
}
RTLIB_INLINE RTLIB_HOST_DEVICE char4 operator/(const char4 &v0, const signed char v1)
{
  return make_char4(v0.x / v1, v0.y / v1, v0.z / v1, v0.w / v1);
}
RTLIB_INLINE RTLIB_HOST_DEVICE char4 operator/(const signed char v0, const char4 &v1)
{
  return make_char4(v0 / v1.x, v0 / v1.y, v0 / v1.z, v0 / v1.w);
}
RTLIB_INLINE RTLIB_HOST_DEVICE bool operator==(const short2 &v0, const short2 &v1)
{
  return (v0.x == v1.x) && (v0.y == v1.y);
}
RTLIB_INLINE RTLIB_HOST_DEVICE bool operator!=(const short2 &v0, const short2 &v1)
{
  return (v0.x != v1.x) || (v0.y != v1.y);
}
RTLIB_INLINE RTLIB_HOST_DEVICE short2 &operator+=(short2 &v0, const short2 &v1)
{
  v0.x += v1.x;
  v0.y += v1.y;
  return v0;
}
RTLIB_INLINE RTLIB_HOST_DEVICE short2 &operator-=(short2 &v0, const short2 &v1)
{
  v0.x -= v1.x;
  v0.y -= v1.y;
  return v0;
}
RTLIB_INLINE RTLIB_HOST_DEVICE short2 &operator*=(short2 &v0, const short2 &v1)
{
  v0.x *= v1.x;
  v0.y *= v1.y;
  return v0;
}
RTLIB_INLINE RTLIB_HOST_DEVICE short2 &operator*=(short2 &v0, const short v1)
{
  v0.x *= v1;
  v0.y *= v1;
  return v0;
}
RTLIB_INLINE RTLIB_HOST_DEVICE short2 &operator/=(short2 &v0, const short2 &v1)
{
  v0.x /= v1.x;
  v0.y /= v1.y;
  return v0;
}
RTLIB_INLINE RTLIB_HOST_DEVICE short2 &operator/=(short2 &v0, const short v1)
{
  v0.x /= v1;
  v0.y /= v1;
  return v0;
}
RTLIB_INLINE RTLIB_HOST_DEVICE short2 operator+(const short2 &v0, const short2 &v1)
{
  return make_short2(v0.x + v1.x, v0.y + v1.y);
}
RTLIB_INLINE RTLIB_HOST_DEVICE short2 operator-(const short2 &v0, const short2 &v1)
{
  return make_short2(v0.x - v1.x, v0.y - v1.y);
}
RTLIB_INLINE RTLIB_HOST_DEVICE short2 operator*(const short2 &v0, const short2 &v1)
{
  return make_short2(v0.x * v1.x, v0.y * v1.y);
}
RTLIB_INLINE RTLIB_HOST_DEVICE short2 operator*(const short2 &v0, const short v1)
{
  return make_short2(v0.x * v1, v0.y * v1);
}
RTLIB_INLINE RTLIB_HOST_DEVICE short2 operator*(const short v0, const short2 &v1)
{
  return make_short2(v0 * v1.x, v0 * v1.y);
}
RTLIB_INLINE RTLIB_HOST_DEVICE short2 operator/(const short2 &v0, const short2 &v1)
{
  return make_short2(v0.x / v1.x, v0.y / v1.y);
}
RTLIB_INLINE RTLIB_HOST_DEVICE short2 operator/(const short2 &v0, const short v1)
{
  return make_short2(v0.x / v1, v0.y / v1);
}
RTLIB_INLINE RTLIB_HOST_DEVICE short2 operator/(const short v0, const short2 &v1)
{
  return make_short2(v0 / v1.x, v0 / v1.y);
}
RTLIB_INLINE RTLIB_HOST_DEVICE bool operator==(const short3 &v0, const short3 &v1)
{
  return (v0.x == v1.x) && (v0.y == v1.y) && (v0.z == v1.z);
}
RTLIB_INLINE RTLIB_HOST_DEVICE bool operator!=(const short3 &v0, const short3 &v1)
{
  return (v0.x != v1.x) || (v0.y != v1.y) || (v0.z != v1.z);
}
RTLIB_INLINE RTLIB_HOST_DEVICE short3 &operator+=(short3 &v0, const short3 &v1)
{
  v0.x += v1.x;
  v0.y += v1.y;
  v0.z += v1.z;
  return v0;
}
RTLIB_INLINE RTLIB_HOST_DEVICE short3 &operator-=(short3 &v0, const short3 &v1)
{
  v0.x -= v1.x;
  v0.y -= v1.y;
  v0.z -= v1.z;
  return v0;
}
RTLIB_INLINE RTLIB_HOST_DEVICE short3 &operator*=(short3 &v0, const short3 &v1)
{
  v0.x *= v1.x;
  v0.y *= v1.y;
  v0.z *= v1.z;
  return v0;
}
RTLIB_INLINE RTLIB_HOST_DEVICE short3 &operator*=(short3 &v0, const short v1)
{
  v0.x *= v1;
  v0.y *= v1;
  v0.z *= v1;
  return v0;
}
RTLIB_INLINE RTLIB_HOST_DEVICE short3 &operator/=(short3 &v0, const short3 &v1)
{
  v0.x /= v1.x;
  v0.y /= v1.y;
  v0.z /= v1.z;
  return v0;
}
RTLIB_INLINE RTLIB_HOST_DEVICE short3 &operator/=(short3 &v0, const short v1)
{
  v0.x /= v1;
  v0.y /= v1;
  v0.z /= v1;
  return v0;
}
RTLIB_INLINE RTLIB_HOST_DEVICE short3 operator+(const short3 &v0, const short3 &v1)
{
  return make_short3(v0.x + v1.x, v0.y + v1.y, v0.z + v1.z);
}
RTLIB_INLINE RTLIB_HOST_DEVICE short3 operator-(const short3 &v0, const short3 &v1)
{
  return make_short3(v0.x - v1.x, v0.y - v1.y, v0.z - v1.z);
}
RTLIB_INLINE RTLIB_HOST_DEVICE short3 operator*(const short3 &v0, const short3 &v1)
{
  return make_short3(v0.x * v1.x, v0.y * v1.y, v0.z * v1.z);
}
RTLIB_INLINE RTLIB_HOST_DEVICE short3 operator*(const short3 &v0, const short v1)
{
  return make_short3(v0.x * v1, v0.y * v1, v0.z * v1);
}
RTLIB_INLINE RTLIB_HOST_DEVICE short3 operator*(const short v0, const short3 &v1)
{
  return make_short3(v0 * v1.x, v0 * v1.y, v0 * v1.z);
}
RTLIB_INLINE RTLIB_HOST_DEVICE short3 operator/(const short3 &v0, const short3 &v1)
{
  return make_short3(v0.x / v1.x, v0.y / v1.y, v0.z / v1.z);
}
RTLIB_INLINE RTLIB_HOST_DEVICE short3 operator/(const short3 &v0, const short v1)
{
  return make_short3(v0.x / v1, v0.y / v1, v0.z / v1);
}
RTLIB_INLINE RTLIB_HOST_DEVICE short3 operator/(const short v0, const short3 &v1)
{
  return make_short3(v0 / v1.x, v0 / v1.y, v0 / v1.z);
}
RTLIB_INLINE RTLIB_HOST_DEVICE bool operator==(const short4 &v0, const short4 &v1)
{
  return (v0.x == v1.x) && (v0.y == v1.y) && (v0.z == v1.z) && (v0.w == v1.w);
}
RTLIB_INLINE RTLIB_HOST_DEVICE bool operator!=(const short4 &v0, const short4 &v1)
{
  return (v0.x != v1.x) || (v0.y != v1.y) || (v0.z != v1.z) || (v0.w != v1.w);
}
RTLIB_INLINE RTLIB_HOST_DEVICE short4 &operator+=(short4 &v0, const short4 &v1)
{
  v0.x += v1.x;
  v0.y += v1.y;
  v0.z += v1.z;
  v0.w += v1.w;
  return v0;
}
RTLIB_INLINE RTLIB_HOST_DEVICE short4 &operator-=(short4 &v0, const short4 &v1)
{
  v0.x -= v1.x;
  v0.y -= v1.y;
  v0.z -= v1.z;
  v0.w -= v1.w;
  return v0;
}
RTLIB_INLINE RTLIB_HOST_DEVICE short4 &operator*=(short4 &v0, const short4 &v1)
{
  v0.x *= v1.x;
  v0.y *= v1.y;
  v0.z *= v1.z;
  v0.w *= v1.w;
  return v0;
}
RTLIB_INLINE RTLIB_HOST_DEVICE short4 &operator*=(short4 &v0, const short v1)
{
  v0.x *= v1;
  v0.y *= v1;
  v0.z *= v1;
  v0.w *= v1;
  return v0;
}
RTLIB_INLINE RTLIB_HOST_DEVICE short4 &operator/=(short4 &v0, const short4 &v1)
{
  v0.x /= v1.x;
  v0.y /= v1.y;
  v0.z /= v1.z;
  v0.w /= v1.w;
  return v0;
}
RTLIB_INLINE RTLIB_HOST_DEVICE short4 &operator/=(short4 &v0, const short v1)
{
  v0.x /= v1;
  v0.y /= v1;
  v0.z /= v1;
  v0.w /= v1;
  return v0;
}
RTLIB_INLINE RTLIB_HOST_DEVICE short4 operator+(const short4 &v0, const short4 &v1)
{
  return make_short4(v0.x + v1.x, v0.y + v1.y, v0.z + v1.z, v0.w + v1.w);
}
RTLIB_INLINE RTLIB_HOST_DEVICE short4 operator-(const short4 &v0, const short4 &v1)
{
  return make_short4(v0.x - v1.x, v0.y - v1.y, v0.z - v1.z, v0.w - v1.w);
}
RTLIB_INLINE RTLIB_HOST_DEVICE short4 operator*(const short4 &v0, const short4 &v1)
{
  return make_short4(v0.x * v1.x, v0.y * v1.y, v0.z * v1.z, v0.w * v1.w);
}
RTLIB_INLINE RTLIB_HOST_DEVICE short4 operator*(const short4 &v0, const short v1)
{
  return make_short4(v0.x * v1, v0.y * v1, v0.z * v1, v0.w * v1);
}
RTLIB_INLINE RTLIB_HOST_DEVICE short4 operator*(const short v0, const short4 &v1)
{
  return make_short4(v0 * v1.x, v0 * v1.y, v0 * v1.z, v0 * v1.w);
}
RTLIB_INLINE RTLIB_HOST_DEVICE short4 operator/(const short4 &v0, const short4 &v1)
{
  return make_short4(v0.x / v1.x, v0.y / v1.y, v0.z / v1.z, v0.w / v1.w);
}
RTLIB_INLINE RTLIB_HOST_DEVICE short4 operator/(const short4 &v0, const short v1)
{
  return make_short4(v0.x / v1, v0.y / v1, v0.z / v1, v0.w / v1);
}
RTLIB_INLINE RTLIB_HOST_DEVICE short4 operator/(const short v0, const short4 &v1)
{
  return make_short4(v0 / v1.x, v0 / v1.y, v0 / v1.z, v0 / v1.w);
}
RTLIB_INLINE RTLIB_HOST_DEVICE bool operator==(const int2 &v0, const int2 &v1)
{
  return (v0.x == v1.x) && (v0.y == v1.y);
}
RTLIB_INLINE RTLIB_HOST_DEVICE bool operator!=(const int2 &v0, const int2 &v1)
{
  return (v0.x != v1.x) || (v0.y != v1.y);
}
RTLIB_INLINE RTLIB_HOST_DEVICE int2 &operator+=(int2 &v0, const int2 &v1)
{
  v0.x += v1.x;
  v0.y += v1.y;
  return v0;
}
RTLIB_INLINE RTLIB_HOST_DEVICE int2 &operator-=(int2 &v0, const int2 &v1)
{
  v0.x -= v1.x;
  v0.y -= v1.y;
  return v0;
}
RTLIB_INLINE RTLIB_HOST_DEVICE int2 &operator*=(int2 &v0, const int2 &v1)
{
  v0.x *= v1.x;
  v0.y *= v1.y;
  return v0;
}
RTLIB_INLINE RTLIB_HOST_DEVICE int2 &operator*=(int2 &v0, const int v1)
{
  v0.x *= v1;
  v0.y *= v1;
  return v0;
}
RTLIB_INLINE RTLIB_HOST_DEVICE int2 &operator/=(int2 &v0, const int2 &v1)
{
  v0.x /= v1.x;
  v0.y /= v1.y;
  return v0;
}
RTLIB_INLINE RTLIB_HOST_DEVICE int2 &operator/=(int2 &v0, const int v1)
{
  v0.x /= v1;
  v0.y /= v1;
  return v0;
}
RTLIB_INLINE RTLIB_HOST_DEVICE int2 operator+(const int2 &v0, const int2 &v1)
{
  return make_int2(v0.x + v1.x, v0.y + v1.y);
}
RTLIB_INLINE RTLIB_HOST_DEVICE int2 operator-(const int2 &v0, const int2 &v1)
{
  return make_int2(v0.x - v1.x, v0.y - v1.y);
}
RTLIB_INLINE RTLIB_HOST_DEVICE int2 operator*(const int2 &v0, const int2 &v1)
{
  return make_int2(v0.x * v1.x, v0.y * v1.y);
}
RTLIB_INLINE RTLIB_HOST_DEVICE int2 operator*(const int2 &v0, const int v1)
{
  return make_int2(v0.x * v1, v0.y * v1);
}
RTLIB_INLINE RTLIB_HOST_DEVICE int2 operator*(const int v0, const int2 &v1)
{
  return make_int2(v0 * v1.x, v0 * v1.y);
}
RTLIB_INLINE RTLIB_HOST_DEVICE int2 operator/(const int2 &v0, const int2 &v1)
{
  return make_int2(v0.x / v1.x, v0.y / v1.y);
}
RTLIB_INLINE RTLIB_HOST_DEVICE int2 operator/(const int2 &v0, const int v1)
{
  return make_int2(v0.x / v1, v0.y / v1);
}
RTLIB_INLINE RTLIB_HOST_DEVICE int2 operator/(const int v0, const int2 &v1)
{
  return make_int2(v0 / v1.x, v0 / v1.y);
}
RTLIB_INLINE RTLIB_HOST_DEVICE bool operator==(const int3 &v0, const int3 &v1)
{
  return (v0.x == v1.x) && (v0.y == v1.y) && (v0.z == v1.z);
}
RTLIB_INLINE RTLIB_HOST_DEVICE bool operator!=(const int3 &v0, const int3 &v1)
{
  return (v0.x != v1.x) || (v0.y != v1.y) || (v0.z != v1.z);
}
RTLIB_INLINE RTLIB_HOST_DEVICE int3 &operator+=(int3 &v0, const int3 &v1)
{
  v0.x += v1.x;
  v0.y += v1.y;
  v0.z += v1.z;
  return v0;
}
RTLIB_INLINE RTLIB_HOST_DEVICE int3 &operator-=(int3 &v0, const int3 &v1)
{
  v0.x -= v1.x;
  v0.y -= v1.y;
  v0.z -= v1.z;
  return v0;
}
RTLIB_INLINE RTLIB_HOST_DEVICE int3 &operator*=(int3 &v0, const int3 &v1)
{
  v0.x *= v1.x;
  v0.y *= v1.y;
  v0.z *= v1.z;
  return v0;
}
RTLIB_INLINE RTLIB_HOST_DEVICE int3 &operator*=(int3 &v0, const int v1)
{
  v0.x *= v1;
  v0.y *= v1;
  v0.z *= v1;
  return v0;
}
RTLIB_INLINE RTLIB_HOST_DEVICE int3 &operator/=(int3 &v0, const int3 &v1)
{
  v0.x /= v1.x;
  v0.y /= v1.y;
  v0.z /= v1.z;
  return v0;
}
RTLIB_INLINE RTLIB_HOST_DEVICE int3 &operator/=(int3 &v0, const int v1)
{
  v0.x /= v1;
  v0.y /= v1;
  v0.z /= v1;
  return v0;
}
RTLIB_INLINE RTLIB_HOST_DEVICE int3 operator+(const int3 &v0, const int3 &v1)
{
  return make_int3(v0.x + v1.x, v0.y + v1.y, v0.z + v1.z);
}
RTLIB_INLINE RTLIB_HOST_DEVICE int3 operator-(const int3 &v0, const int3 &v1)
{
  return make_int3(v0.x - v1.x, v0.y - v1.y, v0.z - v1.z);
}
RTLIB_INLINE RTLIB_HOST_DEVICE int3 operator*(const int3 &v0, const int3 &v1)
{
  return make_int3(v0.x * v1.x, v0.y * v1.y, v0.z * v1.z);
}
RTLIB_INLINE RTLIB_HOST_DEVICE int3 operator*(const int3 &v0, const int v1)
{
  return make_int3(v0.x * v1, v0.y * v1, v0.z * v1);
}
RTLIB_INLINE RTLIB_HOST_DEVICE int3 operator*(const int v0, const int3 &v1)
{
  return make_int3(v0 * v1.x, v0 * v1.y, v0 * v1.z);
}
RTLIB_INLINE RTLIB_HOST_DEVICE int3 operator/(const int3 &v0, const int3 &v1)
{
  return make_int3(v0.x / v1.x, v0.y / v1.y, v0.z / v1.z);
}
RTLIB_INLINE RTLIB_HOST_DEVICE int3 operator/(const int3 &v0, const int v1)
{
  return make_int3(v0.x / v1, v0.y / v1, v0.z / v1);
}
RTLIB_INLINE RTLIB_HOST_DEVICE int3 operator/(const int v0, const int3 &v1)
{
  return make_int3(v0 / v1.x, v0 / v1.y, v0 / v1.z);
}
RTLIB_INLINE RTLIB_HOST_DEVICE bool operator==(const int4 &v0, const int4 &v1)
{
  return (v0.x == v1.x) && (v0.y == v1.y) && (v0.z == v1.z) && (v0.w == v1.w);
}
RTLIB_INLINE RTLIB_HOST_DEVICE bool operator!=(const int4 &v0, const int4 &v1)
{
  return (v0.x != v1.x) || (v0.y != v1.y) || (v0.z != v1.z) || (v0.w != v1.w);
}
RTLIB_INLINE RTLIB_HOST_DEVICE int4 &operator+=(int4 &v0, const int4 &v1)
{
  v0.x += v1.x;
  v0.y += v1.y;
  v0.z += v1.z;
  v0.w += v1.w;
  return v0;
}
RTLIB_INLINE RTLIB_HOST_DEVICE int4 &operator-=(int4 &v0, const int4 &v1)
{
  v0.x -= v1.x;
  v0.y -= v1.y;
  v0.z -= v1.z;
  v0.w -= v1.w;
  return v0;
}
RTLIB_INLINE RTLIB_HOST_DEVICE int4 &operator*=(int4 &v0, const int4 &v1)
{
  v0.x *= v1.x;
  v0.y *= v1.y;
  v0.z *= v1.z;
  v0.w *= v1.w;
  return v0;
}
RTLIB_INLINE RTLIB_HOST_DEVICE int4 &operator*=(int4 &v0, const int v1)
{
  v0.x *= v1;
  v0.y *= v1;
  v0.z *= v1;
  v0.w *= v1;
  return v0;
}
RTLIB_INLINE RTLIB_HOST_DEVICE int4 &operator/=(int4 &v0, const int4 &v1)
{
  v0.x /= v1.x;
  v0.y /= v1.y;
  v0.z /= v1.z;
  v0.w /= v1.w;
  return v0;
}
RTLIB_INLINE RTLIB_HOST_DEVICE int4 &operator/=(int4 &v0, const int v1)
{
  v0.x /= v1;
  v0.y /= v1;
  v0.z /= v1;
  v0.w /= v1;
  return v0;
}
RTLIB_INLINE RTLIB_HOST_DEVICE int4 operator+(const int4 &v0, const int4 &v1)
{
  return make_int4(v0.x + v1.x, v0.y + v1.y, v0.z + v1.z, v0.w + v1.w);
}
RTLIB_INLINE RTLIB_HOST_DEVICE int4 operator-(const int4 &v0, const int4 &v1)
{
  return make_int4(v0.x - v1.x, v0.y - v1.y, v0.z - v1.z, v0.w - v1.w);
}
RTLIB_INLINE RTLIB_HOST_DEVICE int4 operator*(const int4 &v0, const int4 &v1)
{
  return make_int4(v0.x * v1.x, v0.y * v1.y, v0.z * v1.z, v0.w * v1.w);
}
RTLIB_INLINE RTLIB_HOST_DEVICE int4 operator*(const int4 &v0, const int v1)
{
  return make_int4(v0.x * v1, v0.y * v1, v0.z * v1, v0.w * v1);
}
RTLIB_INLINE RTLIB_HOST_DEVICE int4 operator*(const int v0, const int4 &v1)
{
  return make_int4(v0 * v1.x, v0 * v1.y, v0 * v1.z, v0 * v1.w);
}
RTLIB_INLINE RTLIB_HOST_DEVICE int4 operator/(const int4 &v0, const int4 &v1)
{
  return make_int4(v0.x / v1.x, v0.y / v1.y, v0.z / v1.z, v0.w / v1.w);
}
RTLIB_INLINE RTLIB_HOST_DEVICE int4 operator/(const int4 &v0, const int v1)
{
  return make_int4(v0.x / v1, v0.y / v1, v0.z / v1, v0.w / v1);
}
RTLIB_INLINE RTLIB_HOST_DEVICE int4 operator/(const int v0, const int4 &v1)
{
  return make_int4(v0 / v1.x, v0 / v1.y, v0 / v1.z, v0 / v1.w);
}
RTLIB_INLINE RTLIB_HOST_DEVICE bool operator==(const long2 &v0, const long2 &v1)
{
  return (v0.x == v1.x) && (v0.y == v1.y);
}
RTLIB_INLINE RTLIB_HOST_DEVICE bool operator!=(const long2 &v0, const long2 &v1)
{
  return (v0.x != v1.x) || (v0.y != v1.y);
}
RTLIB_INLINE RTLIB_HOST_DEVICE long2 &operator+=(long2 &v0, const long2 &v1)
{
  v0.x += v1.x;
  v0.y += v1.y;
  return v0;
}
RTLIB_INLINE RTLIB_HOST_DEVICE long2 &operator-=(long2 &v0, const long2 &v1)
{
  v0.x -= v1.x;
  v0.y -= v1.y;
  return v0;
}
RTLIB_INLINE RTLIB_HOST_DEVICE long2 &operator*=(long2 &v0, const long2 &v1)
{
  v0.x *= v1.x;
  v0.y *= v1.y;
  return v0;
}
RTLIB_INLINE RTLIB_HOST_DEVICE long2 &operator*=(long2 &v0, const long v1)
{
  v0.x *= v1;
  v0.y *= v1;
  return v0;
}
RTLIB_INLINE RTLIB_HOST_DEVICE long2 &operator/=(long2 &v0, const long2 &v1)
{
  v0.x /= v1.x;
  v0.y /= v1.y;
  return v0;
}
RTLIB_INLINE RTLIB_HOST_DEVICE long2 &operator/=(long2 &v0, const long v1)
{
  v0.x /= v1;
  v0.y /= v1;
  return v0;
}
RTLIB_INLINE RTLIB_HOST_DEVICE long2 operator+(const long2 &v0, const long2 &v1)
{
  return make_long2(v0.x + v1.x, v0.y + v1.y);
}
RTLIB_INLINE RTLIB_HOST_DEVICE long2 operator-(const long2 &v0, const long2 &v1)
{
  return make_long2(v0.x - v1.x, v0.y - v1.y);
}
RTLIB_INLINE RTLIB_HOST_DEVICE long2 operator*(const long2 &v0, const long2 &v1)
{
  return make_long2(v0.x * v1.x, v0.y * v1.y);
}
RTLIB_INLINE RTLIB_HOST_DEVICE long2 operator*(const long2 &v0, const long v1)
{
  return make_long2(v0.x * v1, v0.y * v1);
}
RTLIB_INLINE RTLIB_HOST_DEVICE long2 operator*(const long v0, const long2 &v1)
{
  return make_long2(v0 * v1.x, v0 * v1.y);
}
RTLIB_INLINE RTLIB_HOST_DEVICE long2 operator/(const long2 &v0, const long2 &v1)
{
  return make_long2(v0.x / v1.x, v0.y / v1.y);
}
RTLIB_INLINE RTLIB_HOST_DEVICE long2 operator/(const long2 &v0, const long v1)
{
  return make_long2(v0.x / v1, v0.y / v1);
}
RTLIB_INLINE RTLIB_HOST_DEVICE long2 operator/(const long v0, const long2 &v1)
{
  return make_long2(v0 / v1.x, v0 / v1.y);
}
RTLIB_INLINE RTLIB_HOST_DEVICE bool operator==(const long3 &v0, const long3 &v1)
{
  return (v0.x == v1.x) && (v0.y == v1.y) && (v0.z == v1.z);
}
RTLIB_INLINE RTLIB_HOST_DEVICE bool operator!=(const long3 &v0, const long3 &v1)
{
  return (v0.x != v1.x) || (v0.y != v1.y) || (v0.z != v1.z);
}
RTLIB_INLINE RTLIB_HOST_DEVICE long3 &operator+=(long3 &v0, const long3 &v1)
{
  v0.x += v1.x;
  v0.y += v1.y;
  v0.z += v1.z;
  return v0;
}
RTLIB_INLINE RTLIB_HOST_DEVICE long3 &operator-=(long3 &v0, const long3 &v1)
{
  v0.x -= v1.x;
  v0.y -= v1.y;
  v0.z -= v1.z;
  return v0;
}
RTLIB_INLINE RTLIB_HOST_DEVICE long3 &operator*=(long3 &v0, const long3 &v1)
{
  v0.x *= v1.x;
  v0.y *= v1.y;
  v0.z *= v1.z;
  return v0;
}
RTLIB_INLINE RTLIB_HOST_DEVICE long3 &operator*=(long3 &v0, const long v1)
{
  v0.x *= v1;
  v0.y *= v1;
  v0.z *= v1;
  return v0;
}
RTLIB_INLINE RTLIB_HOST_DEVICE long3 &operator/=(long3 &v0, const long3 &v1)
{
  v0.x /= v1.x;
  v0.y /= v1.y;
  v0.z /= v1.z;
  return v0;
}
RTLIB_INLINE RTLIB_HOST_DEVICE long3 &operator/=(long3 &v0, const long v1)
{
  v0.x /= v1;
  v0.y /= v1;
  v0.z /= v1;
  return v0;
}
RTLIB_INLINE RTLIB_HOST_DEVICE long3 operator+(const long3 &v0, const long3 &v1)
{
  return make_long3(v0.x + v1.x, v0.y + v1.y, v0.z + v1.z);
}
RTLIB_INLINE RTLIB_HOST_DEVICE long3 operator-(const long3 &v0, const long3 &v1)
{
  return make_long3(v0.x - v1.x, v0.y - v1.y, v0.z - v1.z);
}
RTLIB_INLINE RTLIB_HOST_DEVICE long3 operator*(const long3 &v0, const long3 &v1)
{
  return make_long3(v0.x * v1.x, v0.y * v1.y, v0.z * v1.z);
}
RTLIB_INLINE RTLIB_HOST_DEVICE long3 operator*(const long3 &v0, const long v1)
{
  return make_long3(v0.x * v1, v0.y * v1, v0.z * v1);
}
RTLIB_INLINE RTLIB_HOST_DEVICE long3 operator*(const long v0, const long3 &v1)
{
  return make_long3(v0 * v1.x, v0 * v1.y, v0 * v1.z);
}
RTLIB_INLINE RTLIB_HOST_DEVICE long3 operator/(const long3 &v0, const long3 &v1)
{
  return make_long3(v0.x / v1.x, v0.y / v1.y, v0.z / v1.z);
}
RTLIB_INLINE RTLIB_HOST_DEVICE long3 operator/(const long3 &v0, const long v1)
{
  return make_long3(v0.x / v1, v0.y / v1, v0.z / v1);
}
RTLIB_INLINE RTLIB_HOST_DEVICE long3 operator/(const long v0, const long3 &v1)
{
  return make_long3(v0 / v1.x, v0 / v1.y, v0 / v1.z);
}
RTLIB_INLINE RTLIB_HOST_DEVICE bool operator==(const long4 &v0, const long4 &v1)
{
  return (v0.x == v1.x) && (v0.y == v1.y) && (v0.z == v1.z) && (v0.w == v1.w);
}
RTLIB_INLINE RTLIB_HOST_DEVICE bool operator!=(const long4 &v0, const long4 &v1)
{
  return (v0.x != v1.x) || (v0.y != v1.y) || (v0.z != v1.z) || (v0.w != v1.w);
}
RTLIB_INLINE RTLIB_HOST_DEVICE long4 &operator+=(long4 &v0, const long4 &v1)
{
  v0.x += v1.x;
  v0.y += v1.y;
  v0.z += v1.z;
  v0.w += v1.w;
  return v0;
}
RTLIB_INLINE RTLIB_HOST_DEVICE long4 &operator-=(long4 &v0, const long4 &v1)
{
  v0.x -= v1.x;
  v0.y -= v1.y;
  v0.z -= v1.z;
  v0.w -= v1.w;
  return v0;
}
RTLIB_INLINE RTLIB_HOST_DEVICE long4 &operator*=(long4 &v0, const long4 &v1)
{
  v0.x *= v1.x;
  v0.y *= v1.y;
  v0.z *= v1.z;
  v0.w *= v1.w;
  return v0;
}
RTLIB_INLINE RTLIB_HOST_DEVICE long4 &operator*=(long4 &v0, const long v1)
{
  v0.x *= v1;
  v0.y *= v1;
  v0.z *= v1;
  v0.w *= v1;
  return v0;
}
RTLIB_INLINE RTLIB_HOST_DEVICE long4 &operator/=(long4 &v0, const long4 &v1)
{
  v0.x /= v1.x;
  v0.y /= v1.y;
  v0.z /= v1.z;
  v0.w /= v1.w;
  return v0;
}
RTLIB_INLINE RTLIB_HOST_DEVICE long4 &operator/=(long4 &v0, const long v1)
{
  v0.x /= v1;
  v0.y /= v1;
  v0.z /= v1;
  v0.w /= v1;
  return v0;
}
RTLIB_INLINE RTLIB_HOST_DEVICE long4 operator+(const long4 &v0, const long4 &v1)
{
  return make_long4(v0.x + v1.x, v0.y + v1.y, v0.z + v1.z, v0.w + v1.w);
}
RTLIB_INLINE RTLIB_HOST_DEVICE long4 operator-(const long4 &v0, const long4 &v1)
{
  return make_long4(v0.x - v1.x, v0.y - v1.y, v0.z - v1.z, v0.w - v1.w);
}
RTLIB_INLINE RTLIB_HOST_DEVICE long4 operator*(const long4 &v0, const long4 &v1)
{
  return make_long4(v0.x * v1.x, v0.y * v1.y, v0.z * v1.z, v0.w * v1.w);
}
RTLIB_INLINE RTLIB_HOST_DEVICE long4 operator*(const long4 &v0, const long v1)
{
  return make_long4(v0.x * v1, v0.y * v1, v0.z * v1, v0.w * v1);
}
RTLIB_INLINE RTLIB_HOST_DEVICE long4 operator*(const long v0, const long4 &v1)
{
  return make_long4(v0 * v1.x, v0 * v1.y, v0 * v1.z, v0 * v1.w);
}
RTLIB_INLINE RTLIB_HOST_DEVICE long4 operator/(const long4 &v0, const long4 &v1)
{
  return make_long4(v0.x / v1.x, v0.y / v1.y, v0.z / v1.z, v0.w / v1.w);
}
RTLIB_INLINE RTLIB_HOST_DEVICE long4 operator/(const long4 &v0, const long v1)
{
  return make_long4(v0.x / v1, v0.y / v1, v0.z / v1, v0.w / v1);
}
RTLIB_INLINE RTLIB_HOST_DEVICE long4 operator/(const long v0, const long4 &v1)
{
  return make_long4(v0 / v1.x, v0 / v1.y, v0 / v1.z, v0 / v1.w);
}
RTLIB_INLINE RTLIB_HOST_DEVICE bool operator==(const longlong2 &v0, const longlong2 &v1)
{
  return (v0.x == v1.x) && (v0.y == v1.y);
}
RTLIB_INLINE RTLIB_HOST_DEVICE bool operator!=(const longlong2 &v0, const longlong2 &v1)
{
  return (v0.x != v1.x) || (v0.y != v1.y);
}
RTLIB_INLINE RTLIB_HOST_DEVICE longlong2 &operator+=(longlong2 &v0, const longlong2 &v1)
{
  v0.x += v1.x;
  v0.y += v1.y;
  return v0;
}
RTLIB_INLINE RTLIB_HOST_DEVICE longlong2 &operator-=(longlong2 &v0, const longlong2 &v1)
{
  v0.x -= v1.x;
  v0.y -= v1.y;
  return v0;
}
RTLIB_INLINE RTLIB_HOST_DEVICE longlong2 &operator*=(longlong2 &v0, const longlong2 &v1)
{
  v0.x *= v1.x;
  v0.y *= v1.y;
  return v0;
}
RTLIB_INLINE RTLIB_HOST_DEVICE longlong2 &operator*=(longlong2 &v0, const long long v1)
{
  v0.x *= v1;
  v0.y *= v1;
  return v0;
}
RTLIB_INLINE RTLIB_HOST_DEVICE longlong2 &operator/=(longlong2 &v0, const longlong2 &v1)
{
  v0.x /= v1.x;
  v0.y /= v1.y;
  return v0;
}
RTLIB_INLINE RTLIB_HOST_DEVICE longlong2 &operator/=(longlong2 &v0, const long long v1)
{
  v0.x /= v1;
  v0.y /= v1;
  return v0;
}
RTLIB_INLINE RTLIB_HOST_DEVICE longlong2 operator+(const longlong2 &v0, const longlong2 &v1)
{
  return make_longlong2(v0.x + v1.x, v0.y + v1.y);
}
RTLIB_INLINE RTLIB_HOST_DEVICE longlong2 operator-(const longlong2 &v0, const longlong2 &v1)
{
  return make_longlong2(v0.x - v1.x, v0.y - v1.y);
}
RTLIB_INLINE RTLIB_HOST_DEVICE longlong2 operator*(const longlong2 &v0, const longlong2 &v1)
{
  return make_longlong2(v0.x * v1.x, v0.y * v1.y);
}
RTLIB_INLINE RTLIB_HOST_DEVICE longlong2 operator*(const longlong2 &v0, const long long v1)
{
  return make_longlong2(v0.x * v1, v0.y * v1);
}
RTLIB_INLINE RTLIB_HOST_DEVICE longlong2 operator*(const long long v0, const longlong2 &v1)
{
  return make_longlong2(v0 * v1.x, v0 * v1.y);
}
RTLIB_INLINE RTLIB_HOST_DEVICE longlong2 operator/(const longlong2 &v0, const longlong2 &v1)
{
  return make_longlong2(v0.x / v1.x, v0.y / v1.y);
}
RTLIB_INLINE RTLIB_HOST_DEVICE longlong2 operator/(const longlong2 &v0, const long long v1)
{
  return make_longlong2(v0.x / v1, v0.y / v1);
}
RTLIB_INLINE RTLIB_HOST_DEVICE longlong2 operator/(const long long v0, const longlong2 &v1)
{
  return make_longlong2(v0 / v1.x, v0 / v1.y);
}
RTLIB_INLINE RTLIB_HOST_DEVICE bool operator==(const longlong3 &v0, const longlong3 &v1)
{
  return (v0.x == v1.x) && (v0.y == v1.y) && (v0.z == v1.z);
}
RTLIB_INLINE RTLIB_HOST_DEVICE bool operator!=(const longlong3 &v0, const longlong3 &v1)
{
  return (v0.x != v1.x) || (v0.y != v1.y) || (v0.z != v1.z);
}
RTLIB_INLINE RTLIB_HOST_DEVICE longlong3 &operator+=(longlong3 &v0, const longlong3 &v1)
{
  v0.x += v1.x;
  v0.y += v1.y;
  v0.z += v1.z;
  return v0;
}
RTLIB_INLINE RTLIB_HOST_DEVICE longlong3 &operator-=(longlong3 &v0, const longlong3 &v1)
{
  v0.x -= v1.x;
  v0.y -= v1.y;
  v0.z -= v1.z;
  return v0;
}
RTLIB_INLINE RTLIB_HOST_DEVICE longlong3 &operator*=(longlong3 &v0, const longlong3 &v1)
{
  v0.x *= v1.x;
  v0.y *= v1.y;
  v0.z *= v1.z;
  return v0;
}
RTLIB_INLINE RTLIB_HOST_DEVICE longlong3 &operator*=(longlong3 &v0, const long long v1)
{
  v0.x *= v1;
  v0.y *= v1;
  v0.z *= v1;
  return v0;
}
RTLIB_INLINE RTLIB_HOST_DEVICE longlong3 &operator/=(longlong3 &v0, const longlong3 &v1)
{
  v0.x /= v1.x;
  v0.y /= v1.y;
  v0.z /= v1.z;
  return v0;
}
RTLIB_INLINE RTLIB_HOST_DEVICE longlong3 &operator/=(longlong3 &v0, const long long v1)
{
  v0.x /= v1;
  v0.y /= v1;
  v0.z /= v1;
  return v0;
}
RTLIB_INLINE RTLIB_HOST_DEVICE longlong3 operator+(const longlong3 &v0, const longlong3 &v1)
{
  return make_longlong3(v0.x + v1.x, v0.y + v1.y, v0.z + v1.z);
}
RTLIB_INLINE RTLIB_HOST_DEVICE longlong3 operator-(const longlong3 &v0, const longlong3 &v1)
{
  return make_longlong3(v0.x - v1.x, v0.y - v1.y, v0.z - v1.z);
}
RTLIB_INLINE RTLIB_HOST_DEVICE longlong3 operator*(const longlong3 &v0, const longlong3 &v1)
{
  return make_longlong3(v0.x * v1.x, v0.y * v1.y, v0.z * v1.z);
}
RTLIB_INLINE RTLIB_HOST_DEVICE longlong3 operator*(const longlong3 &v0, const long long v1)
{
  return make_longlong3(v0.x * v1, v0.y * v1, v0.z * v1);
}
RTLIB_INLINE RTLIB_HOST_DEVICE longlong3 operator*(const long long v0, const longlong3 &v1)
{
  return make_longlong3(v0 * v1.x, v0 * v1.y, v0 * v1.z);
}
RTLIB_INLINE RTLIB_HOST_DEVICE longlong3 operator/(const longlong3 &v0, const longlong3 &v1)
{
  return make_longlong3(v0.x / v1.x, v0.y / v1.y, v0.z / v1.z);
}
RTLIB_INLINE RTLIB_HOST_DEVICE longlong3 operator/(const longlong3 &v0, const long long v1)
{
  return make_longlong3(v0.x / v1, v0.y / v1, v0.z / v1);
}
RTLIB_INLINE RTLIB_HOST_DEVICE longlong3 operator/(const long long v0, const longlong3 &v1)
{
  return make_longlong3(v0 / v1.x, v0 / v1.y, v0 / v1.z);
}
RTLIB_INLINE RTLIB_HOST_DEVICE bool operator==(const longlong4 &v0, const longlong4 &v1)
{
  return (v0.x == v1.x) && (v0.y == v1.y) && (v0.z == v1.z) && (v0.w == v1.w);
}
RTLIB_INLINE RTLIB_HOST_DEVICE bool operator!=(const longlong4 &v0, const longlong4 &v1)
{
  return (v0.x != v1.x) || (v0.y != v1.y) || (v0.z != v1.z) || (v0.w != v1.w);
}
RTLIB_INLINE RTLIB_HOST_DEVICE longlong4 &operator+=(longlong4 &v0, const longlong4 &v1)
{
  v0.x += v1.x;
  v0.y += v1.y;
  v0.z += v1.z;
  v0.w += v1.w;
  return v0;
}
RTLIB_INLINE RTLIB_HOST_DEVICE longlong4 &operator-=(longlong4 &v0, const longlong4 &v1)
{
  v0.x -= v1.x;
  v0.y -= v1.y;
  v0.z -= v1.z;
  v0.w -= v1.w;
  return v0;
}
RTLIB_INLINE RTLIB_HOST_DEVICE longlong4 &operator*=(longlong4 &v0, const longlong4 &v1)
{
  v0.x *= v1.x;
  v0.y *= v1.y;
  v0.z *= v1.z;
  v0.w *= v1.w;
  return v0;
}
RTLIB_INLINE RTLIB_HOST_DEVICE longlong4 &operator*=(longlong4 &v0, const long long v1)
{
  v0.x *= v1;
  v0.y *= v1;
  v0.z *= v1;
  v0.w *= v1;
  return v0;
}
RTLIB_INLINE RTLIB_HOST_DEVICE longlong4 &operator/=(longlong4 &v0, const longlong4 &v1)
{
  v0.x /= v1.x;
  v0.y /= v1.y;
  v0.z /= v1.z;
  v0.w /= v1.w;
  return v0;
}
RTLIB_INLINE RTLIB_HOST_DEVICE longlong4 &operator/=(longlong4 &v0, const long long v1)
{
  v0.x /= v1;
  v0.y /= v1;
  v0.z /= v1;
  v0.w /= v1;
  return v0;
}
RTLIB_INLINE RTLIB_HOST_DEVICE longlong4 operator+(const longlong4 &v0, const longlong4 &v1)
{
  return make_longlong4(v0.x + v1.x, v0.y + v1.y, v0.z + v1.z, v0.w + v1.w);
}
RTLIB_INLINE RTLIB_HOST_DEVICE longlong4 operator-(const longlong4 &v0, const longlong4 &v1)
{
  return make_longlong4(v0.x - v1.x, v0.y - v1.y, v0.z - v1.z, v0.w - v1.w);
}
RTLIB_INLINE RTLIB_HOST_DEVICE longlong4 operator*(const longlong4 &v0, const longlong4 &v1)
{
  return make_longlong4(v0.x * v1.x, v0.y * v1.y, v0.z * v1.z, v0.w * v1.w);
}
RTLIB_INLINE RTLIB_HOST_DEVICE longlong4 operator*(const longlong4 &v0, const long long v1)
{
  return make_longlong4(v0.x * v1, v0.y * v1, v0.z * v1, v0.w * v1);
}
RTLIB_INLINE RTLIB_HOST_DEVICE longlong4 operator*(const long long v0, const longlong4 &v1)
{
  return make_longlong4(v0 * v1.x, v0 * v1.y, v0 * v1.z, v0 * v1.w);
}
RTLIB_INLINE RTLIB_HOST_DEVICE longlong4 operator/(const longlong4 &v0, const longlong4 &v1)
{
  return make_longlong4(v0.x / v1.x, v0.y / v1.y, v0.z / v1.z, v0.w / v1.w);
}
RTLIB_INLINE RTLIB_HOST_DEVICE longlong4 operator/(const longlong4 &v0, const long long v1)
{
  return make_longlong4(v0.x / v1, v0.y / v1, v0.z / v1, v0.w / v1);
}
RTLIB_INLINE RTLIB_HOST_DEVICE longlong4 operator/(const long long v0, const longlong4 &v1)
{
  return make_longlong4(v0 / v1.x, v0 / v1.y, v0 / v1.z, v0 / v1.w);
}
RTLIB_INLINE RTLIB_HOST_DEVICE bool operator==(const uchar2 &v0, const uchar2 &v1)
{
  return (v0.x == v1.x) && (v0.y == v1.y);
}
RTLIB_INLINE RTLIB_HOST_DEVICE bool operator!=(const uchar2 &v0, const uchar2 &v1)
{
  return (v0.x != v1.x) || (v0.y != v1.y);
}
RTLIB_INLINE RTLIB_HOST_DEVICE uchar2 &operator+=(uchar2 &v0, const uchar2 &v1)
{
  v0.x += v1.x;
  v0.y += v1.y;
  return v0;
}
RTLIB_INLINE RTLIB_HOST_DEVICE uchar2 &operator-=(uchar2 &v0, const uchar2 &v1)
{
  v0.x -= v1.x;
  v0.y -= v1.y;
  return v0;
}
RTLIB_INLINE RTLIB_HOST_DEVICE uchar2 &operator*=(uchar2 &v0, const uchar2 &v1)
{
  v0.x *= v1.x;
  v0.y *= v1.y;
  return v0;
}
RTLIB_INLINE RTLIB_HOST_DEVICE uchar2 &operator*=(uchar2 &v0, const unsigned char v1)
{
  v0.x *= v1;
  v0.y *= v1;
  return v0;
}
RTLIB_INLINE RTLIB_HOST_DEVICE uchar2 &operator/=(uchar2 &v0, const uchar2 &v1)
{
  v0.x /= v1.x;
  v0.y /= v1.y;
  return v0;
}
RTLIB_INLINE RTLIB_HOST_DEVICE uchar2 &operator/=(uchar2 &v0, const unsigned char v1)
{
  v0.x /= v1;
  v0.y /= v1;
  return v0;
}
RTLIB_INLINE RTLIB_HOST_DEVICE uchar2 operator+(const uchar2 &v0, const uchar2 &v1)
{
  return make_uchar2(v0.x + v1.x, v0.y + v1.y);
}
RTLIB_INLINE RTLIB_HOST_DEVICE uchar2 operator-(const uchar2 &v0, const uchar2 &v1)
{
  return make_uchar2(v0.x - v1.x, v0.y - v1.y);
}
RTLIB_INLINE RTLIB_HOST_DEVICE uchar2 operator*(const uchar2 &v0, const uchar2 &v1)
{
  return make_uchar2(v0.x * v1.x, v0.y * v1.y);
}
RTLIB_INLINE RTLIB_HOST_DEVICE uchar2 operator*(const uchar2 &v0, const unsigned char v1)
{
  return make_uchar2(v0.x * v1, v0.y * v1);
}
RTLIB_INLINE RTLIB_HOST_DEVICE uchar2 operator*(const unsigned char v0, const uchar2 &v1)
{
  return make_uchar2(v0 * v1.x, v0 * v1.y);
}
RTLIB_INLINE RTLIB_HOST_DEVICE uchar2 operator/(const uchar2 &v0, const uchar2 &v1)
{
  return make_uchar2(v0.x / v1.x, v0.y / v1.y);
}
RTLIB_INLINE RTLIB_HOST_DEVICE uchar2 operator/(const uchar2 &v0, const unsigned char v1)
{
  return make_uchar2(v0.x / v1, v0.y / v1);
}
RTLIB_INLINE RTLIB_HOST_DEVICE uchar2 operator/(const unsigned char v0, const uchar2 &v1)
{
  return make_uchar2(v0 / v1.x, v0 / v1.y);
}
RTLIB_INLINE RTLIB_HOST_DEVICE bool operator==(const uchar3 &v0, const uchar3 &v1)
{
  return (v0.x == v1.x) && (v0.y == v1.y) && (v0.z == v1.z);
}
RTLIB_INLINE RTLIB_HOST_DEVICE bool operator!=(const uchar3 &v0, const uchar3 &v1)
{
  return (v0.x != v1.x) || (v0.y != v1.y) || (v0.z != v1.z);
}
RTLIB_INLINE RTLIB_HOST_DEVICE uchar3 &operator+=(uchar3 &v0, const uchar3 &v1)
{
  v0.x += v1.x;
  v0.y += v1.y;
  v0.z += v1.z;
  return v0;
}
RTLIB_INLINE RTLIB_HOST_DEVICE uchar3 &operator-=(uchar3 &v0, const uchar3 &v1)
{
  v0.x -= v1.x;
  v0.y -= v1.y;
  v0.z -= v1.z;
  return v0;
}
RTLIB_INLINE RTLIB_HOST_DEVICE uchar3 &operator*=(uchar3 &v0, const uchar3 &v1)
{
  v0.x *= v1.x;
  v0.y *= v1.y;
  v0.z *= v1.z;
  return v0;
}
RTLIB_INLINE RTLIB_HOST_DEVICE uchar3 &operator*=(uchar3 &v0, const unsigned char v1)
{
  v0.x *= v1;
  v0.y *= v1;
  v0.z *= v1;
  return v0;
}
RTLIB_INLINE RTLIB_HOST_DEVICE uchar3 &operator/=(uchar3 &v0, const uchar3 &v1)
{
  v0.x /= v1.x;
  v0.y /= v1.y;
  v0.z /= v1.z;
  return v0;
}
RTLIB_INLINE RTLIB_HOST_DEVICE uchar3 &operator/=(uchar3 &v0, const unsigned char v1)
{
  v0.x /= v1;
  v0.y /= v1;
  v0.z /= v1;
  return v0;
}
RTLIB_INLINE RTLIB_HOST_DEVICE uchar3 operator+(const uchar3 &v0, const uchar3 &v1)
{
  return make_uchar3(v0.x + v1.x, v0.y + v1.y, v0.z + v1.z);
}
RTLIB_INLINE RTLIB_HOST_DEVICE uchar3 operator-(const uchar3 &v0, const uchar3 &v1)
{
  return make_uchar3(v0.x - v1.x, v0.y - v1.y, v0.z - v1.z);
}
RTLIB_INLINE RTLIB_HOST_DEVICE uchar3 operator*(const uchar3 &v0, const uchar3 &v1)
{
  return make_uchar3(v0.x * v1.x, v0.y * v1.y, v0.z * v1.z);
}
RTLIB_INLINE RTLIB_HOST_DEVICE uchar3 operator*(const uchar3 &v0, const unsigned char v1)
{
  return make_uchar3(v0.x * v1, v0.y * v1, v0.z * v1);
}
RTLIB_INLINE RTLIB_HOST_DEVICE uchar3 operator*(const unsigned char v0, const uchar3 &v1)
{
  return make_uchar3(v0 * v1.x, v0 * v1.y, v0 * v1.z);
}
RTLIB_INLINE RTLIB_HOST_DEVICE uchar3 operator/(const uchar3 &v0, const uchar3 &v1)
{
  return make_uchar3(v0.x / v1.x, v0.y / v1.y, v0.z / v1.z);
}
RTLIB_INLINE RTLIB_HOST_DEVICE uchar3 operator/(const uchar3 &v0, const unsigned char v1)
{
  return make_uchar3(v0.x / v1, v0.y / v1, v0.z / v1);
}
RTLIB_INLINE RTLIB_HOST_DEVICE uchar3 operator/(const unsigned char v0, const uchar3 &v1)
{
  return make_uchar3(v0 / v1.x, v0 / v1.y, v0 / v1.z);
}
RTLIB_INLINE RTLIB_HOST_DEVICE bool operator==(const uchar4 &v0, const uchar4 &v1)
{
  return (v0.x == v1.x) && (v0.y == v1.y) && (v0.z == v1.z) && (v0.w == v1.w);
}
RTLIB_INLINE RTLIB_HOST_DEVICE bool operator!=(const uchar4 &v0, const uchar4 &v1)
{
  return (v0.x != v1.x) || (v0.y != v1.y) || (v0.z != v1.z) || (v0.w != v1.w);
}
RTLIB_INLINE RTLIB_HOST_DEVICE uchar4 &operator+=(uchar4 &v0, const uchar4 &v1)
{
  v0.x += v1.x;
  v0.y += v1.y;
  v0.z += v1.z;
  v0.w += v1.w;
  return v0;
}
RTLIB_INLINE RTLIB_HOST_DEVICE uchar4 &operator-=(uchar4 &v0, const uchar4 &v1)
{
  v0.x -= v1.x;
  v0.y -= v1.y;
  v0.z -= v1.z;
  v0.w -= v1.w;
  return v0;
}
RTLIB_INLINE RTLIB_HOST_DEVICE uchar4 &operator*=(uchar4 &v0, const uchar4 &v1)
{
  v0.x *= v1.x;
  v0.y *= v1.y;
  v0.z *= v1.z;
  v0.w *= v1.w;
  return v0;
}
RTLIB_INLINE RTLIB_HOST_DEVICE uchar4 &operator*=(uchar4 &v0, const unsigned char v1)
{
  v0.x *= v1;
  v0.y *= v1;
  v0.z *= v1;
  v0.w *= v1;
  return v0;
}
RTLIB_INLINE RTLIB_HOST_DEVICE uchar4 &operator/=(uchar4 &v0, const uchar4 &v1)
{
  v0.x /= v1.x;
  v0.y /= v1.y;
  v0.z /= v1.z;
  v0.w /= v1.w;
  return v0;
}
RTLIB_INLINE RTLIB_HOST_DEVICE uchar4 &operator/=(uchar4 &v0, const unsigned char v1)
{
  v0.x /= v1;
  v0.y /= v1;
  v0.z /= v1;
  v0.w /= v1;
  return v0;
}
RTLIB_INLINE RTLIB_HOST_DEVICE uchar4 operator+(const uchar4 &v0, const uchar4 &v1)
{
  return make_uchar4(v0.x + v1.x, v0.y + v1.y, v0.z + v1.z, v0.w + v1.w);
}
RTLIB_INLINE RTLIB_HOST_DEVICE uchar4 operator-(const uchar4 &v0, const uchar4 &v1)
{
  return make_uchar4(v0.x - v1.x, v0.y - v1.y, v0.z - v1.z, v0.w - v1.w);
}
RTLIB_INLINE RTLIB_HOST_DEVICE uchar4 operator*(const uchar4 &v0, const uchar4 &v1)
{
  return make_uchar4(v0.x * v1.x, v0.y * v1.y, v0.z * v1.z, v0.w * v1.w);
}
RTLIB_INLINE RTLIB_HOST_DEVICE uchar4 operator*(const uchar4 &v0, const unsigned char v1)
{
  return make_uchar4(v0.x * v1, v0.y * v1, v0.z * v1, v0.w * v1);
}
RTLIB_INLINE RTLIB_HOST_DEVICE uchar4 operator*(const unsigned char v0, const uchar4 &v1)
{
  return make_uchar4(v0 * v1.x, v0 * v1.y, v0 * v1.z, v0 * v1.w);
}
RTLIB_INLINE RTLIB_HOST_DEVICE uchar4 operator/(const uchar4 &v0, const uchar4 &v1)
{
  return make_uchar4(v0.x / v1.x, v0.y / v1.y, v0.z / v1.z, v0.w / v1.w);
}
RTLIB_INLINE RTLIB_HOST_DEVICE uchar4 operator/(const uchar4 &v0, const unsigned char v1)
{
  return make_uchar4(v0.x / v1, v0.y / v1, v0.z / v1, v0.w / v1);
}
RTLIB_INLINE RTLIB_HOST_DEVICE uchar4 operator/(const unsigned char v0, const uchar4 &v1)
{
  return make_uchar4(v0 / v1.x, v0 / v1.y, v0 / v1.z, v0 / v1.w);
}
RTLIB_INLINE RTLIB_HOST_DEVICE bool operator==(const ushort2 &v0, const ushort2 &v1)
{
  return (v0.x == v1.x) && (v0.y == v1.y);
}
RTLIB_INLINE RTLIB_HOST_DEVICE bool operator!=(const ushort2 &v0, const ushort2 &v1)
{
  return (v0.x != v1.x) || (v0.y != v1.y);
}
RTLIB_INLINE RTLIB_HOST_DEVICE ushort2 &operator+=(ushort2 &v0, const ushort2 &v1)
{
  v0.x += v1.x;
  v0.y += v1.y;
  return v0;
}
RTLIB_INLINE RTLIB_HOST_DEVICE ushort2 &operator-=(ushort2 &v0, const ushort2 &v1)
{
  v0.x -= v1.x;
  v0.y -= v1.y;
  return v0;
}
RTLIB_INLINE RTLIB_HOST_DEVICE ushort2 &operator*=(ushort2 &v0, const ushort2 &v1)
{
  v0.x *= v1.x;
  v0.y *= v1.y;
  return v0;
}
RTLIB_INLINE RTLIB_HOST_DEVICE ushort2 &operator*=(ushort2 &v0, const unsigned short v1)
{
  v0.x *= v1;
  v0.y *= v1;
  return v0;
}
RTLIB_INLINE RTLIB_HOST_DEVICE ushort2 &operator/=(ushort2 &v0, const ushort2 &v1)
{
  v0.x /= v1.x;
  v0.y /= v1.y;
  return v0;
}
RTLIB_INLINE RTLIB_HOST_DEVICE ushort2 &operator/=(ushort2 &v0, const unsigned short v1)
{
  v0.x /= v1;
  v0.y /= v1;
  return v0;
}
RTLIB_INLINE RTLIB_HOST_DEVICE ushort2 operator+(const ushort2 &v0, const ushort2 &v1)
{
  return make_ushort2(v0.x + v1.x, v0.y + v1.y);
}
RTLIB_INLINE RTLIB_HOST_DEVICE ushort2 operator-(const ushort2 &v0, const ushort2 &v1)
{
  return make_ushort2(v0.x - v1.x, v0.y - v1.y);
}
RTLIB_INLINE RTLIB_HOST_DEVICE ushort2 operator*(const ushort2 &v0, const ushort2 &v1)
{
  return make_ushort2(v0.x * v1.x, v0.y * v1.y);
}
RTLIB_INLINE RTLIB_HOST_DEVICE ushort2 operator*(const ushort2 &v0, const unsigned short v1)
{
  return make_ushort2(v0.x * v1, v0.y * v1);
}
RTLIB_INLINE RTLIB_HOST_DEVICE ushort2 operator*(const unsigned short v0, const ushort2 &v1)
{
  return make_ushort2(v0 * v1.x, v0 * v1.y);
}
RTLIB_INLINE RTLIB_HOST_DEVICE ushort2 operator/(const ushort2 &v0, const ushort2 &v1)
{
  return make_ushort2(v0.x / v1.x, v0.y / v1.y);
}
RTLIB_INLINE RTLIB_HOST_DEVICE ushort2 operator/(const ushort2 &v0, const unsigned short v1)
{
  return make_ushort2(v0.x / v1, v0.y / v1);
}
RTLIB_INLINE RTLIB_HOST_DEVICE ushort2 operator/(const unsigned short v0, const ushort2 &v1)
{
  return make_ushort2(v0 / v1.x, v0 / v1.y);
}
RTLIB_INLINE RTLIB_HOST_DEVICE bool operator==(const ushort3 &v0, const ushort3 &v1)
{
  return (v0.x == v1.x) && (v0.y == v1.y) && (v0.z == v1.z);
}
RTLIB_INLINE RTLIB_HOST_DEVICE bool operator!=(const ushort3 &v0, const ushort3 &v1)
{
  return (v0.x != v1.x) || (v0.y != v1.y) || (v0.z != v1.z);
}
RTLIB_INLINE RTLIB_HOST_DEVICE ushort3 &operator+=(ushort3 &v0, const ushort3 &v1)
{
  v0.x += v1.x;
  v0.y += v1.y;
  v0.z += v1.z;
  return v0;
}
RTLIB_INLINE RTLIB_HOST_DEVICE ushort3 &operator-=(ushort3 &v0, const ushort3 &v1)
{
  v0.x -= v1.x;
  v0.y -= v1.y;
  v0.z -= v1.z;
  return v0;
}
RTLIB_INLINE RTLIB_HOST_DEVICE ushort3 &operator*=(ushort3 &v0, const ushort3 &v1)
{
  v0.x *= v1.x;
  v0.y *= v1.y;
  v0.z *= v1.z;
  return v0;
}
RTLIB_INLINE RTLIB_HOST_DEVICE ushort3 &operator*=(ushort3 &v0, const unsigned short v1)
{
  v0.x *= v1;
  v0.y *= v1;
  v0.z *= v1;
  return v0;
}
RTLIB_INLINE RTLIB_HOST_DEVICE ushort3 &operator/=(ushort3 &v0, const ushort3 &v1)
{
  v0.x /= v1.x;
  v0.y /= v1.y;
  v0.z /= v1.z;
  return v0;
}
RTLIB_INLINE RTLIB_HOST_DEVICE ushort3 &operator/=(ushort3 &v0, const unsigned short v1)
{
  v0.x /= v1;
  v0.y /= v1;
  v0.z /= v1;
  return v0;
}
RTLIB_INLINE RTLIB_HOST_DEVICE ushort3 operator+(const ushort3 &v0, const ushort3 &v1)
{
  return make_ushort3(v0.x + v1.x, v0.y + v1.y, v0.z + v1.z);
}
RTLIB_INLINE RTLIB_HOST_DEVICE ushort3 operator-(const ushort3 &v0, const ushort3 &v1)
{
  return make_ushort3(v0.x - v1.x, v0.y - v1.y, v0.z - v1.z);
}
RTLIB_INLINE RTLIB_HOST_DEVICE ushort3 operator*(const ushort3 &v0, const ushort3 &v1)
{
  return make_ushort3(v0.x * v1.x, v0.y * v1.y, v0.z * v1.z);
}
RTLIB_INLINE RTLIB_HOST_DEVICE ushort3 operator*(const ushort3 &v0, const unsigned short v1)
{
  return make_ushort3(v0.x * v1, v0.y * v1, v0.z * v1);
}
RTLIB_INLINE RTLIB_HOST_DEVICE ushort3 operator*(const unsigned short v0, const ushort3 &v1)
{
  return make_ushort3(v0 * v1.x, v0 * v1.y, v0 * v1.z);
}
RTLIB_INLINE RTLIB_HOST_DEVICE ushort3 operator/(const ushort3 &v0, const ushort3 &v1)
{
  return make_ushort3(v0.x / v1.x, v0.y / v1.y, v0.z / v1.z);
}
RTLIB_INLINE RTLIB_HOST_DEVICE ushort3 operator/(const ushort3 &v0, const unsigned short v1)
{
  return make_ushort3(v0.x / v1, v0.y / v1, v0.z / v1);
}
RTLIB_INLINE RTLIB_HOST_DEVICE ushort3 operator/(const unsigned short v0, const ushort3 &v1)
{
  return make_ushort3(v0 / v1.x, v0 / v1.y, v0 / v1.z);
}
RTLIB_INLINE RTLIB_HOST_DEVICE bool operator==(const ushort4 &v0, const ushort4 &v1)
{
  return (v0.x == v1.x) && (v0.y == v1.y) && (v0.z == v1.z) && (v0.w == v1.w);
}
RTLIB_INLINE RTLIB_HOST_DEVICE bool operator!=(const ushort4 &v0, const ushort4 &v1)
{
  return (v0.x != v1.x) || (v0.y != v1.y) || (v0.z != v1.z) || (v0.w != v1.w);
}
RTLIB_INLINE RTLIB_HOST_DEVICE ushort4 &operator+=(ushort4 &v0, const ushort4 &v1)
{
  v0.x += v1.x;
  v0.y += v1.y;
  v0.z += v1.z;
  v0.w += v1.w;
  return v0;
}
RTLIB_INLINE RTLIB_HOST_DEVICE ushort4 &operator-=(ushort4 &v0, const ushort4 &v1)
{
  v0.x -= v1.x;
  v0.y -= v1.y;
  v0.z -= v1.z;
  v0.w -= v1.w;
  return v0;
}
RTLIB_INLINE RTLIB_HOST_DEVICE ushort4 &operator*=(ushort4 &v0, const ushort4 &v1)
{
  v0.x *= v1.x;
  v0.y *= v1.y;
  v0.z *= v1.z;
  v0.w *= v1.w;
  return v0;
}
RTLIB_INLINE RTLIB_HOST_DEVICE ushort4 &operator*=(ushort4 &v0, const unsigned short v1)
{
  v0.x *= v1;
  v0.y *= v1;
  v0.z *= v1;
  v0.w *= v1;
  return v0;
}
RTLIB_INLINE RTLIB_HOST_DEVICE ushort4 &operator/=(ushort4 &v0, const ushort4 &v1)
{
  v0.x /= v1.x;
  v0.y /= v1.y;
  v0.z /= v1.z;
  v0.w /= v1.w;
  return v0;
}
RTLIB_INLINE RTLIB_HOST_DEVICE ushort4 &operator/=(ushort4 &v0, const unsigned short v1)
{
  v0.x /= v1;
  v0.y /= v1;
  v0.z /= v1;
  v0.w /= v1;
  return v0;
}
RTLIB_INLINE RTLIB_HOST_DEVICE ushort4 operator+(const ushort4 &v0, const ushort4 &v1)
{
  return make_ushort4(v0.x + v1.x, v0.y + v1.y, v0.z + v1.z, v0.w + v1.w);
}
RTLIB_INLINE RTLIB_HOST_DEVICE ushort4 operator-(const ushort4 &v0, const ushort4 &v1)
{
  return make_ushort4(v0.x - v1.x, v0.y - v1.y, v0.z - v1.z, v0.w - v1.w);
}
RTLIB_INLINE RTLIB_HOST_DEVICE ushort4 operator*(const ushort4 &v0, const ushort4 &v1)
{
  return make_ushort4(v0.x * v1.x, v0.y * v1.y, v0.z * v1.z, v0.w * v1.w);
}
RTLIB_INLINE RTLIB_HOST_DEVICE ushort4 operator*(const ushort4 &v0, const unsigned short v1)
{
  return make_ushort4(v0.x * v1, v0.y * v1, v0.z * v1, v0.w * v1);
}
RTLIB_INLINE RTLIB_HOST_DEVICE ushort4 operator*(const unsigned short v0, const ushort4 &v1)
{
  return make_ushort4(v0 * v1.x, v0 * v1.y, v0 * v1.z, v0 * v1.w);
}
RTLIB_INLINE RTLIB_HOST_DEVICE ushort4 operator/(const ushort4 &v0, const ushort4 &v1)
{
  return make_ushort4(v0.x / v1.x, v0.y / v1.y, v0.z / v1.z, v0.w / v1.w);
}
RTLIB_INLINE RTLIB_HOST_DEVICE ushort4 operator/(const ushort4 &v0, const unsigned short v1)
{
  return make_ushort4(v0.x / v1, v0.y / v1, v0.z / v1, v0.w / v1);
}
RTLIB_INLINE RTLIB_HOST_DEVICE ushort4 operator/(const unsigned short v0, const ushort4 &v1)
{
  return make_ushort4(v0 / v1.x, v0 / v1.y, v0 / v1.z, v0 / v1.w);
}
RTLIB_INLINE RTLIB_HOST_DEVICE bool operator==(const uint2 &v0, const uint2 &v1)
{
  return (v0.x == v1.x) && (v0.y == v1.y);
}
RTLIB_INLINE RTLIB_HOST_DEVICE bool operator!=(const uint2 &v0, const uint2 &v1)
{
  return (v0.x != v1.x) || (v0.y != v1.y);
}
RTLIB_INLINE RTLIB_HOST_DEVICE uint2 &operator+=(uint2 &v0, const uint2 &v1)
{
  v0.x += v1.x;
  v0.y += v1.y;
  return v0;
}
RTLIB_INLINE RTLIB_HOST_DEVICE uint2 &operator-=(uint2 &v0, const uint2 &v1)
{
  v0.x -= v1.x;
  v0.y -= v1.y;
  return v0;
}
RTLIB_INLINE RTLIB_HOST_DEVICE uint2 &operator*=(uint2 &v0, const uint2 &v1)
{
  v0.x *= v1.x;
  v0.y *= v1.y;
  return v0;
}
RTLIB_INLINE RTLIB_HOST_DEVICE uint2 &operator*=(uint2 &v0, const unsigned int v1)
{
  v0.x *= v1;
  v0.y *= v1;
  return v0;
}
RTLIB_INLINE RTLIB_HOST_DEVICE uint2 &operator/=(uint2 &v0, const uint2 &v1)
{
  v0.x /= v1.x;
  v0.y /= v1.y;
  return v0;
}
RTLIB_INLINE RTLIB_HOST_DEVICE uint2 &operator/=(uint2 &v0, const unsigned int v1)
{
  v0.x /= v1;
  v0.y /= v1;
  return v0;
}
RTLIB_INLINE RTLIB_HOST_DEVICE uint2 operator+(const uint2 &v0, const uint2 &v1)
{
  return make_uint2(v0.x + v1.x, v0.y + v1.y);
}
RTLIB_INLINE RTLIB_HOST_DEVICE uint2 operator-(const uint2 &v0, const uint2 &v1)
{
  return make_uint2(v0.x - v1.x, v0.y - v1.y);
}
RTLIB_INLINE RTLIB_HOST_DEVICE uint2 operator*(const uint2 &v0, const uint2 &v1)
{
  return make_uint2(v0.x * v1.x, v0.y * v1.y);
}
RTLIB_INLINE RTLIB_HOST_DEVICE uint2 operator*(const uint2 &v0, const unsigned int v1)
{
  return make_uint2(v0.x * v1, v0.y * v1);
}
RTLIB_INLINE RTLIB_HOST_DEVICE uint2 operator*(const unsigned int v0, const uint2 &v1)
{
  return make_uint2(v0 * v1.x, v0 * v1.y);
}
RTLIB_INLINE RTLIB_HOST_DEVICE uint2 operator/(const uint2 &v0, const uint2 &v1)
{
  return make_uint2(v0.x / v1.x, v0.y / v1.y);
}
RTLIB_INLINE RTLIB_HOST_DEVICE uint2 operator/(const uint2 &v0, const unsigned int v1)
{
  return make_uint2(v0.x / v1, v0.y / v1);
}
RTLIB_INLINE RTLIB_HOST_DEVICE uint2 operator/(const unsigned int v0, const uint2 &v1)
{
  return make_uint2(v0 / v1.x, v0 / v1.y);
}
RTLIB_INLINE RTLIB_HOST_DEVICE bool operator==(const uint3 &v0, const uint3 &v1)
{
  return (v0.x == v1.x) && (v0.y == v1.y) && (v0.z == v1.z);
}
RTLIB_INLINE RTLIB_HOST_DEVICE bool operator!=(const uint3 &v0, const uint3 &v1)
{
  return (v0.x != v1.x) || (v0.y != v1.y) || (v0.z != v1.z);
}
RTLIB_INLINE RTLIB_HOST_DEVICE uint3 &operator+=(uint3 &v0, const uint3 &v1)
{
  v0.x += v1.x;
  v0.y += v1.y;
  v0.z += v1.z;
  return v0;
}
RTLIB_INLINE RTLIB_HOST_DEVICE uint3 &operator-=(uint3 &v0, const uint3 &v1)
{
  v0.x -= v1.x;
  v0.y -= v1.y;
  v0.z -= v1.z;
  return v0;
}
RTLIB_INLINE RTLIB_HOST_DEVICE uint3 &operator*=(uint3 &v0, const uint3 &v1)
{
  v0.x *= v1.x;
  v0.y *= v1.y;
  v0.z *= v1.z;
  return v0;
}
RTLIB_INLINE RTLIB_HOST_DEVICE uint3 &operator*=(uint3 &v0, const unsigned int v1)
{
  v0.x *= v1;
  v0.y *= v1;
  v0.z *= v1;
  return v0;
}
RTLIB_INLINE RTLIB_HOST_DEVICE uint3 &operator/=(uint3 &v0, const uint3 &v1)
{
  v0.x /= v1.x;
  v0.y /= v1.y;
  v0.z /= v1.z;
  return v0;
}
RTLIB_INLINE RTLIB_HOST_DEVICE uint3 &operator/=(uint3 &v0, const unsigned int v1)
{
  v0.x /= v1;
  v0.y /= v1;
  v0.z /= v1;
  return v0;
}
RTLIB_INLINE RTLIB_HOST_DEVICE uint3 operator+(const uint3 &v0, const uint3 &v1)
{
  return make_uint3(v0.x + v1.x, v0.y + v1.y, v0.z + v1.z);
}
RTLIB_INLINE RTLIB_HOST_DEVICE uint3 operator-(const uint3 &v0, const uint3 &v1)
{
  return make_uint3(v0.x - v1.x, v0.y - v1.y, v0.z - v1.z);
}
RTLIB_INLINE RTLIB_HOST_DEVICE uint3 operator*(const uint3 &v0, const uint3 &v1)
{
  return make_uint3(v0.x * v1.x, v0.y * v1.y, v0.z * v1.z);
}
RTLIB_INLINE RTLIB_HOST_DEVICE uint3 operator*(const uint3 &v0, const unsigned int v1)
{
  return make_uint3(v0.x * v1, v0.y * v1, v0.z * v1);
}
RTLIB_INLINE RTLIB_HOST_DEVICE uint3 operator*(const unsigned int v0, const uint3 &v1)
{
  return make_uint3(v0 * v1.x, v0 * v1.y, v0 * v1.z);
}
RTLIB_INLINE RTLIB_HOST_DEVICE uint3 operator/(const uint3 &v0, const uint3 &v1)
{
  return make_uint3(v0.x / v1.x, v0.y / v1.y, v0.z / v1.z);
}
RTLIB_INLINE RTLIB_HOST_DEVICE uint3 operator/(const uint3 &v0, const unsigned int v1)
{
  return make_uint3(v0.x / v1, v0.y / v1, v0.z / v1);
}
RTLIB_INLINE RTLIB_HOST_DEVICE uint3 operator/(const unsigned int v0, const uint3 &v1)
{
  return make_uint3(v0 / v1.x, v0 / v1.y, v0 / v1.z);
}
RTLIB_INLINE RTLIB_HOST_DEVICE bool operator==(const uint4 &v0, const uint4 &v1)
{
  return (v0.x == v1.x) && (v0.y == v1.y) && (v0.z == v1.z) && (v0.w == v1.w);
}
RTLIB_INLINE RTLIB_HOST_DEVICE bool operator!=(const uint4 &v0, const uint4 &v1)
{
  return (v0.x != v1.x) || (v0.y != v1.y) || (v0.z != v1.z) || (v0.w != v1.w);
}
RTLIB_INLINE RTLIB_HOST_DEVICE uint4 &operator+=(uint4 &v0, const uint4 &v1)
{
  v0.x += v1.x;
  v0.y += v1.y;
  v0.z += v1.z;
  v0.w += v1.w;
  return v0;
}
RTLIB_INLINE RTLIB_HOST_DEVICE uint4 &operator-=(uint4 &v0, const uint4 &v1)
{
  v0.x -= v1.x;
  v0.y -= v1.y;
  v0.z -= v1.z;
  v0.w -= v1.w;
  return v0;
}
RTLIB_INLINE RTLIB_HOST_DEVICE uint4 &operator*=(uint4 &v0, const uint4 &v1)
{
  v0.x *= v1.x;
  v0.y *= v1.y;
  v0.z *= v1.z;
  v0.w *= v1.w;
  return v0;
}
RTLIB_INLINE RTLIB_HOST_DEVICE uint4 &operator*=(uint4 &v0, const unsigned int v1)
{
  v0.x *= v1;
  v0.y *= v1;
  v0.z *= v1;
  v0.w *= v1;
  return v0;
}
RTLIB_INLINE RTLIB_HOST_DEVICE uint4 &operator/=(uint4 &v0, const uint4 &v1)
{
  v0.x /= v1.x;
  v0.y /= v1.y;
  v0.z /= v1.z;
  v0.w /= v1.w;
  return v0;
}
RTLIB_INLINE RTLIB_HOST_DEVICE uint4 &operator/=(uint4 &v0, const unsigned int v1)
{
  v0.x /= v1;
  v0.y /= v1;
  v0.z /= v1;
  v0.w /= v1;
  return v0;
}
RTLIB_INLINE RTLIB_HOST_DEVICE uint4 operator+(const uint4 &v0, const uint4 &v1)
{
  return make_uint4(v0.x + v1.x, v0.y + v1.y, v0.z + v1.z, v0.w + v1.w);
}
RTLIB_INLINE RTLIB_HOST_DEVICE uint4 operator-(const uint4 &v0, const uint4 &v1)
{
  return make_uint4(v0.x - v1.x, v0.y - v1.y, v0.z - v1.z, v0.w - v1.w);
}
RTLIB_INLINE RTLIB_HOST_DEVICE uint4 operator*(const uint4 &v0, const uint4 &v1)
{
  return make_uint4(v0.x * v1.x, v0.y * v1.y, v0.z * v1.z, v0.w * v1.w);
}
RTLIB_INLINE RTLIB_HOST_DEVICE uint4 operator*(const uint4 &v0, const unsigned int v1)
{
  return make_uint4(v0.x * v1, v0.y * v1, v0.z * v1, v0.w * v1);
}
RTLIB_INLINE RTLIB_HOST_DEVICE uint4 operator*(const unsigned int v0, const uint4 &v1)
{
  return make_uint4(v0 * v1.x, v0 * v1.y, v0 * v1.z, v0 * v1.w);
}
RTLIB_INLINE RTLIB_HOST_DEVICE uint4 operator/(const uint4 &v0, const uint4 &v1)
{
  return make_uint4(v0.x / v1.x, v0.y / v1.y, v0.z / v1.z, v0.w / v1.w);
}
RTLIB_INLINE RTLIB_HOST_DEVICE uint4 operator/(const uint4 &v0, const unsigned int v1)
{
  return make_uint4(v0.x / v1, v0.y / v1, v0.z / v1, v0.w / v1);
}
RTLIB_INLINE RTLIB_HOST_DEVICE uint4 operator/(const unsigned int v0, const uint4 &v1)
{
  return make_uint4(v0 / v1.x, v0 / v1.y, v0 / v1.z, v0 / v1.w);
}
RTLIB_INLINE RTLIB_HOST_DEVICE bool operator==(const ulong2 &v0, const ulong2 &v1)
{
  return (v0.x == v1.x) && (v0.y == v1.y);
}
RTLIB_INLINE RTLIB_HOST_DEVICE bool operator!=(const ulong2 &v0, const ulong2 &v1)
{
  return (v0.x != v1.x) || (v0.y != v1.y);
}
RTLIB_INLINE RTLIB_HOST_DEVICE ulong2 &operator+=(ulong2 &v0, const ulong2 &v1)
{
  v0.x += v1.x;
  v0.y += v1.y;
  return v0;
}
RTLIB_INLINE RTLIB_HOST_DEVICE ulong2 &operator-=(ulong2 &v0, const ulong2 &v1)
{
  v0.x -= v1.x;
  v0.y -= v1.y;
  return v0;
}
RTLIB_INLINE RTLIB_HOST_DEVICE ulong2 &operator*=(ulong2 &v0, const ulong2 &v1)
{
  v0.x *= v1.x;
  v0.y *= v1.y;
  return v0;
}
RTLIB_INLINE RTLIB_HOST_DEVICE ulong2 &operator*=(ulong2 &v0, const unsigned long v1)
{
  v0.x *= v1;
  v0.y *= v1;
  return v0;
}
RTLIB_INLINE RTLIB_HOST_DEVICE ulong2 &operator/=(ulong2 &v0, const ulong2 &v1)
{
  v0.x /= v1.x;
  v0.y /= v1.y;
  return v0;
}
RTLIB_INLINE RTLIB_HOST_DEVICE ulong2 &operator/=(ulong2 &v0, const unsigned long v1)
{
  v0.x /= v1;
  v0.y /= v1;
  return v0;
}
RTLIB_INLINE RTLIB_HOST_DEVICE ulong2 operator+(const ulong2 &v0, const ulong2 &v1)
{
  return make_ulong2(v0.x + v1.x, v0.y + v1.y);
}
RTLIB_INLINE RTLIB_HOST_DEVICE ulong2 operator-(const ulong2 &v0, const ulong2 &v1)
{
  return make_ulong2(v0.x - v1.x, v0.y - v1.y);
}
RTLIB_INLINE RTLIB_HOST_DEVICE ulong2 operator*(const ulong2 &v0, const ulong2 &v1)
{
  return make_ulong2(v0.x * v1.x, v0.y * v1.y);
}
RTLIB_INLINE RTLIB_HOST_DEVICE ulong2 operator*(const ulong2 &v0, const unsigned long v1)
{
  return make_ulong2(v0.x * v1, v0.y * v1);
}
RTLIB_INLINE RTLIB_HOST_DEVICE ulong2 operator*(const unsigned long v0, const ulong2 &v1)
{
  return make_ulong2(v0 * v1.x, v0 * v1.y);
}
RTLIB_INLINE RTLIB_HOST_DEVICE ulong2 operator/(const ulong2 &v0, const ulong2 &v1)
{
  return make_ulong2(v0.x / v1.x, v0.y / v1.y);
}
RTLIB_INLINE RTLIB_HOST_DEVICE ulong2 operator/(const ulong2 &v0, const unsigned long v1)
{
  return make_ulong2(v0.x / v1, v0.y / v1);
}
RTLIB_INLINE RTLIB_HOST_DEVICE ulong2 operator/(const unsigned long v0, const ulong2 &v1)
{
  return make_ulong2(v0 / v1.x, v0 / v1.y);
}
RTLIB_INLINE RTLIB_HOST_DEVICE bool operator==(const ulong3 &v0, const ulong3 &v1)
{
  return (v0.x == v1.x) && (v0.y == v1.y) && (v0.z == v1.z);
}
RTLIB_INLINE RTLIB_HOST_DEVICE bool operator!=(const ulong3 &v0, const ulong3 &v1)
{
  return (v0.x != v1.x) || (v0.y != v1.y) || (v0.z != v1.z);
}
RTLIB_INLINE RTLIB_HOST_DEVICE ulong3 &operator+=(ulong3 &v0, const ulong3 &v1)
{
  v0.x += v1.x;
  v0.y += v1.y;
  v0.z += v1.z;
  return v0;
}
RTLIB_INLINE RTLIB_HOST_DEVICE ulong3 &operator-=(ulong3 &v0, const ulong3 &v1)
{
  v0.x -= v1.x;
  v0.y -= v1.y;
  v0.z -= v1.z;
  return v0;
}
RTLIB_INLINE RTLIB_HOST_DEVICE ulong3 &operator*=(ulong3 &v0, const ulong3 &v1)
{
  v0.x *= v1.x;
  v0.y *= v1.y;
  v0.z *= v1.z;
  return v0;
}
RTLIB_INLINE RTLIB_HOST_DEVICE ulong3 &operator*=(ulong3 &v0, const unsigned long v1)
{
  v0.x *= v1;
  v0.y *= v1;
  v0.z *= v1;
  return v0;
}
RTLIB_INLINE RTLIB_HOST_DEVICE ulong3 &operator/=(ulong3 &v0, const ulong3 &v1)
{
  v0.x /= v1.x;
  v0.y /= v1.y;
  v0.z /= v1.z;
  return v0;
}
RTLIB_INLINE RTLIB_HOST_DEVICE ulong3 &operator/=(ulong3 &v0, const unsigned long v1)
{
  v0.x /= v1;
  v0.y /= v1;
  v0.z /= v1;
  return v0;
}
RTLIB_INLINE RTLIB_HOST_DEVICE ulong3 operator+(const ulong3 &v0, const ulong3 &v1)
{
  return make_ulong3(v0.x + v1.x, v0.y + v1.y, v0.z + v1.z);
}
RTLIB_INLINE RTLIB_HOST_DEVICE ulong3 operator-(const ulong3 &v0, const ulong3 &v1)
{
  return make_ulong3(v0.x - v1.x, v0.y - v1.y, v0.z - v1.z);
}
RTLIB_INLINE RTLIB_HOST_DEVICE ulong3 operator*(const ulong3 &v0, const ulong3 &v1)
{
  return make_ulong3(v0.x * v1.x, v0.y * v1.y, v0.z * v1.z);
}
RTLIB_INLINE RTLIB_HOST_DEVICE ulong3 operator*(const ulong3 &v0, const unsigned long v1)
{
  return make_ulong3(v0.x * v1, v0.y * v1, v0.z * v1);
}
RTLIB_INLINE RTLIB_HOST_DEVICE ulong3 operator*(const unsigned long v0, const ulong3 &v1)
{
  return make_ulong3(v0 * v1.x, v0 * v1.y, v0 * v1.z);
}
RTLIB_INLINE RTLIB_HOST_DEVICE ulong3 operator/(const ulong3 &v0, const ulong3 &v1)
{
  return make_ulong3(v0.x / v1.x, v0.y / v1.y, v0.z / v1.z);
}
RTLIB_INLINE RTLIB_HOST_DEVICE ulong3 operator/(const ulong3 &v0, const unsigned long v1)
{
  return make_ulong3(v0.x / v1, v0.y / v1, v0.z / v1);
}
RTLIB_INLINE RTLIB_HOST_DEVICE ulong3 operator/(const unsigned long v0, const ulong3 &v1)
{
  return make_ulong3(v0 / v1.x, v0 / v1.y, v0 / v1.z);
}
RTLIB_INLINE RTLIB_HOST_DEVICE bool operator==(const ulong4 &v0, const ulong4 &v1)
{
  return (v0.x == v1.x) && (v0.y == v1.y) && (v0.z == v1.z) && (v0.w == v1.w);
}
RTLIB_INLINE RTLIB_HOST_DEVICE bool operator!=(const ulong4 &v0, const ulong4 &v1)
{
  return (v0.x != v1.x) || (v0.y != v1.y) || (v0.z != v1.z) || (v0.w != v1.w);
}
RTLIB_INLINE RTLIB_HOST_DEVICE ulong4 &operator+=(ulong4 &v0, const ulong4 &v1)
{
  v0.x += v1.x;
  v0.y += v1.y;
  v0.z += v1.z;
  v0.w += v1.w;
  return v0;
}
RTLIB_INLINE RTLIB_HOST_DEVICE ulong4 &operator-=(ulong4 &v0, const ulong4 &v1)
{
  v0.x -= v1.x;
  v0.y -= v1.y;
  v0.z -= v1.z;
  v0.w -= v1.w;
  return v0;
}
RTLIB_INLINE RTLIB_HOST_DEVICE ulong4 &operator*=(ulong4 &v0, const ulong4 &v1)
{
  v0.x *= v1.x;
  v0.y *= v1.y;
  v0.z *= v1.z;
  v0.w *= v1.w;
  return v0;
}
RTLIB_INLINE RTLIB_HOST_DEVICE ulong4 &operator*=(ulong4 &v0, const unsigned long v1)
{
  v0.x *= v1;
  v0.y *= v1;
  v0.z *= v1;
  v0.w *= v1;
  return v0;
}
RTLIB_INLINE RTLIB_HOST_DEVICE ulong4 &operator/=(ulong4 &v0, const ulong4 &v1)
{
  v0.x /= v1.x;
  v0.y /= v1.y;
  v0.z /= v1.z;
  v0.w /= v1.w;
  return v0;
}
RTLIB_INLINE RTLIB_HOST_DEVICE ulong4 &operator/=(ulong4 &v0, const unsigned long v1)
{
  v0.x /= v1;
  v0.y /= v1;
  v0.z /= v1;
  v0.w /= v1;
  return v0;
}
RTLIB_INLINE RTLIB_HOST_DEVICE ulong4 operator+(const ulong4 &v0, const ulong4 &v1)
{
  return make_ulong4(v0.x + v1.x, v0.y + v1.y, v0.z + v1.z, v0.w + v1.w);
}
RTLIB_INLINE RTLIB_HOST_DEVICE ulong4 operator-(const ulong4 &v0, const ulong4 &v1)
{
  return make_ulong4(v0.x - v1.x, v0.y - v1.y, v0.z - v1.z, v0.w - v1.w);
}
RTLIB_INLINE RTLIB_HOST_DEVICE ulong4 operator*(const ulong4 &v0, const ulong4 &v1)
{
  return make_ulong4(v0.x * v1.x, v0.y * v1.y, v0.z * v1.z, v0.w * v1.w);
}
RTLIB_INLINE RTLIB_HOST_DEVICE ulong4 operator*(const ulong4 &v0, const unsigned long v1)
{
  return make_ulong4(v0.x * v1, v0.y * v1, v0.z * v1, v0.w * v1);
}
RTLIB_INLINE RTLIB_HOST_DEVICE ulong4 operator*(const unsigned long v0, const ulong4 &v1)
{
  return make_ulong4(v0 * v1.x, v0 * v1.y, v0 * v1.z, v0 * v1.w);
}
RTLIB_INLINE RTLIB_HOST_DEVICE ulong4 operator/(const ulong4 &v0, const ulong4 &v1)
{
  return make_ulong4(v0.x / v1.x, v0.y / v1.y, v0.z / v1.z, v0.w / v1.w);
}
RTLIB_INLINE RTLIB_HOST_DEVICE ulong4 operator/(const ulong4 &v0, const unsigned long v1)
{
  return make_ulong4(v0.x / v1, v0.y / v1, v0.z / v1, v0.w / v1);
}
RTLIB_INLINE RTLIB_HOST_DEVICE ulong4 operator/(const unsigned long v0, const ulong4 &v1)
{
  return make_ulong4(v0 / v1.x, v0 / v1.y, v0 / v1.z, v0 / v1.w);
}
RTLIB_INLINE RTLIB_HOST_DEVICE bool operator==(const ulonglong2 &v0, const ulonglong2 &v1)
{
  return (v0.x == v1.x) && (v0.y == v1.y);
}
RTLIB_INLINE RTLIB_HOST_DEVICE bool operator!=(const ulonglong2 &v0, const ulonglong2 &v1)
{
  return (v0.x != v1.x) || (v0.y != v1.y);
}
RTLIB_INLINE RTLIB_HOST_DEVICE ulonglong2 &operator+=(ulonglong2 &v0, const ulonglong2 &v1)
{
  v0.x += v1.x;
  v0.y += v1.y;
  return v0;
}
RTLIB_INLINE RTLIB_HOST_DEVICE ulonglong2 &operator-=(ulonglong2 &v0, const ulonglong2 &v1)
{
  v0.x -= v1.x;
  v0.y -= v1.y;
  return v0;
}
RTLIB_INLINE RTLIB_HOST_DEVICE ulonglong2 &operator*=(ulonglong2 &v0, const ulonglong2 &v1)
{
  v0.x *= v1.x;
  v0.y *= v1.y;
  return v0;
}
RTLIB_INLINE RTLIB_HOST_DEVICE ulonglong2 &operator*=(ulonglong2 &v0, const unsigned long long v1)
{
  v0.x *= v1;
  v0.y *= v1;
  return v0;
}
RTLIB_INLINE RTLIB_HOST_DEVICE ulonglong2 &operator/=(ulonglong2 &v0, const ulonglong2 &v1)
{
  v0.x /= v1.x;
  v0.y /= v1.y;
  return v0;
}
RTLIB_INLINE RTLIB_HOST_DEVICE ulonglong2 &operator/=(ulonglong2 &v0, const unsigned long long v1)
{
  v0.x /= v1;
  v0.y /= v1;
  return v0;
}
RTLIB_INLINE RTLIB_HOST_DEVICE ulonglong2 operator+(const ulonglong2 &v0, const ulonglong2 &v1)
{
  return make_ulonglong2(v0.x + v1.x, v0.y + v1.y);
}
RTLIB_INLINE RTLIB_HOST_DEVICE ulonglong2 operator-(const ulonglong2 &v0, const ulonglong2 &v1)
{
  return make_ulonglong2(v0.x - v1.x, v0.y - v1.y);
}
RTLIB_INLINE RTLIB_HOST_DEVICE ulonglong2 operator*(const ulonglong2 &v0, const ulonglong2 &v1)
{
  return make_ulonglong2(v0.x * v1.x, v0.y * v1.y);
}
RTLIB_INLINE RTLIB_HOST_DEVICE ulonglong2 operator*(const ulonglong2 &v0, const unsigned long long v1)
{
  return make_ulonglong2(v0.x * v1, v0.y * v1);
}
RTLIB_INLINE RTLIB_HOST_DEVICE ulonglong2 operator*(const unsigned long long v0, const ulonglong2 &v1)
{
  return make_ulonglong2(v0 * v1.x, v0 * v1.y);
}
RTLIB_INLINE RTLIB_HOST_DEVICE ulonglong2 operator/(const ulonglong2 &v0, const ulonglong2 &v1)
{
  return make_ulonglong2(v0.x / v1.x, v0.y / v1.y);
}
RTLIB_INLINE RTLIB_HOST_DEVICE ulonglong2 operator/(const ulonglong2 &v0, const unsigned long long v1)
{
  return make_ulonglong2(v0.x / v1, v0.y / v1);
}
RTLIB_INLINE RTLIB_HOST_DEVICE ulonglong2 operator/(const unsigned long long v0, const ulonglong2 &v1)
{
  return make_ulonglong2(v0 / v1.x, v0 / v1.y);
}
RTLIB_INLINE RTLIB_HOST_DEVICE bool operator==(const ulonglong3 &v0, const ulonglong3 &v1)
{
  return (v0.x == v1.x) && (v0.y == v1.y) && (v0.z == v1.z);
}
RTLIB_INLINE RTLIB_HOST_DEVICE bool operator!=(const ulonglong3 &v0, const ulonglong3 &v1)
{
  return (v0.x != v1.x) || (v0.y != v1.y) || (v0.z != v1.z);
}
RTLIB_INLINE RTLIB_HOST_DEVICE ulonglong3 &operator+=(ulonglong3 &v0, const ulonglong3 &v1)
{
  v0.x += v1.x;
  v0.y += v1.y;
  v0.z += v1.z;
  return v0;
}
RTLIB_INLINE RTLIB_HOST_DEVICE ulonglong3 &operator-=(ulonglong3 &v0, const ulonglong3 &v1)
{
  v0.x -= v1.x;
  v0.y -= v1.y;
  v0.z -= v1.z;
  return v0;
}
RTLIB_INLINE RTLIB_HOST_DEVICE ulonglong3 &operator*=(ulonglong3 &v0, const ulonglong3 &v1)
{
  v0.x *= v1.x;
  v0.y *= v1.y;
  v0.z *= v1.z;
  return v0;
}
RTLIB_INLINE RTLIB_HOST_DEVICE ulonglong3 &operator*=(ulonglong3 &v0, const unsigned long long v1)
{
  v0.x *= v1;
  v0.y *= v1;
  v0.z *= v1;
  return v0;
}
RTLIB_INLINE RTLIB_HOST_DEVICE ulonglong3 &operator/=(ulonglong3 &v0, const ulonglong3 &v1)
{
  v0.x /= v1.x;
  v0.y /= v1.y;
  v0.z /= v1.z;
  return v0;
}
RTLIB_INLINE RTLIB_HOST_DEVICE ulonglong3 &operator/=(ulonglong3 &v0, const unsigned long long v1)
{
  v0.x /= v1;
  v0.y /= v1;
  v0.z /= v1;
  return v0;
}
RTLIB_INLINE RTLIB_HOST_DEVICE ulonglong3 operator+(const ulonglong3 &v0, const ulonglong3 &v1)
{
  return make_ulonglong3(v0.x + v1.x, v0.y + v1.y, v0.z + v1.z);
}
RTLIB_INLINE RTLIB_HOST_DEVICE ulonglong3 operator-(const ulonglong3 &v0, const ulonglong3 &v1)
{
  return make_ulonglong3(v0.x - v1.x, v0.y - v1.y, v0.z - v1.z);
}
RTLIB_INLINE RTLIB_HOST_DEVICE ulonglong3 operator*(const ulonglong3 &v0, const ulonglong3 &v1)
{
  return make_ulonglong3(v0.x * v1.x, v0.y * v1.y, v0.z * v1.z);
}
RTLIB_INLINE RTLIB_HOST_DEVICE ulonglong3 operator*(const ulonglong3 &v0, const unsigned long long v1)
{
  return make_ulonglong3(v0.x * v1, v0.y * v1, v0.z * v1);
}
RTLIB_INLINE RTLIB_HOST_DEVICE ulonglong3 operator*(const unsigned long long v0, const ulonglong3 &v1)
{
  return make_ulonglong3(v0 * v1.x, v0 * v1.y, v0 * v1.z);
}
RTLIB_INLINE RTLIB_HOST_DEVICE ulonglong3 operator/(const ulonglong3 &v0, const ulonglong3 &v1)
{
  return make_ulonglong3(v0.x / v1.x, v0.y / v1.y, v0.z / v1.z);
}
RTLIB_INLINE RTLIB_HOST_DEVICE ulonglong3 operator/(const ulonglong3 &v0, const unsigned long long v1)
{
  return make_ulonglong3(v0.x / v1, v0.y / v1, v0.z / v1);
}
RTLIB_INLINE RTLIB_HOST_DEVICE ulonglong3 operator/(const unsigned long long v0, const ulonglong3 &v1)
{
  return make_ulonglong3(v0 / v1.x, v0 / v1.y, v0 / v1.z);
}
RTLIB_INLINE RTLIB_HOST_DEVICE bool operator==(const ulonglong4 &v0, const ulonglong4 &v1)
{
  return (v0.x == v1.x) && (v0.y == v1.y) && (v0.z == v1.z) && (v0.w == v1.w);
}
RTLIB_INLINE RTLIB_HOST_DEVICE bool operator!=(const ulonglong4 &v0, const ulonglong4 &v1)
{
  return (v0.x != v1.x) || (v0.y != v1.y) || (v0.z != v1.z) || (v0.w != v1.w);
}
RTLIB_INLINE RTLIB_HOST_DEVICE ulonglong4 &operator+=(ulonglong4 &v0, const ulonglong4 &v1)
{
  v0.x += v1.x;
  v0.y += v1.y;
  v0.z += v1.z;
  v0.w += v1.w;
  return v0;
}
RTLIB_INLINE RTLIB_HOST_DEVICE ulonglong4 &operator-=(ulonglong4 &v0, const ulonglong4 &v1)
{
  v0.x -= v1.x;
  v0.y -= v1.y;
  v0.z -= v1.z;
  v0.w -= v1.w;
  return v0;
}
RTLIB_INLINE RTLIB_HOST_DEVICE ulonglong4 &operator*=(ulonglong4 &v0, const ulonglong4 &v1)
{
  v0.x *= v1.x;
  v0.y *= v1.y;
  v0.z *= v1.z;
  v0.w *= v1.w;
  return v0;
}
RTLIB_INLINE RTLIB_HOST_DEVICE ulonglong4 &operator*=(ulonglong4 &v0, const unsigned long long v1)
{
  v0.x *= v1;
  v0.y *= v1;
  v0.z *= v1;
  v0.w *= v1;
  return v0;
}
RTLIB_INLINE RTLIB_HOST_DEVICE ulonglong4 &operator/=(ulonglong4 &v0, const ulonglong4 &v1)
{
  v0.x /= v1.x;
  v0.y /= v1.y;
  v0.z /= v1.z;
  v0.w /= v1.w;
  return v0;
}
RTLIB_INLINE RTLIB_HOST_DEVICE ulonglong4 &operator/=(ulonglong4 &v0, const unsigned long long v1)
{
  v0.x /= v1;
  v0.y /= v1;
  v0.z /= v1;
  v0.w /= v1;
  return v0;
}
RTLIB_INLINE RTLIB_HOST_DEVICE ulonglong4 operator+(const ulonglong4 &v0, const ulonglong4 &v1)
{
  return make_ulonglong4(v0.x + v1.x, v0.y + v1.y, v0.z + v1.z, v0.w + v1.w);
}
RTLIB_INLINE RTLIB_HOST_DEVICE ulonglong4 operator-(const ulonglong4 &v0, const ulonglong4 &v1)
{
  return make_ulonglong4(v0.x - v1.x, v0.y - v1.y, v0.z - v1.z, v0.w - v1.w);
}
RTLIB_INLINE RTLIB_HOST_DEVICE ulonglong4 operator*(const ulonglong4 &v0, const ulonglong4 &v1)
{
  return make_ulonglong4(v0.x * v1.x, v0.y * v1.y, v0.z * v1.z, v0.w * v1.w);
}
RTLIB_INLINE RTLIB_HOST_DEVICE ulonglong4 operator*(const ulonglong4 &v0, const unsigned long long v1)
{
  return make_ulonglong4(v0.x * v1, v0.y * v1, v0.z * v1, v0.w * v1);
}
RTLIB_INLINE RTLIB_HOST_DEVICE ulonglong4 operator*(const unsigned long long v0, const ulonglong4 &v1)
{
  return make_ulonglong4(v0 * v1.x, v0 * v1.y, v0 * v1.z, v0 * v1.w);
}
RTLIB_INLINE RTLIB_HOST_DEVICE ulonglong4 operator/(const ulonglong4 &v0, const ulonglong4 &v1)
{
  return make_ulonglong4(v0.x / v1.x, v0.y / v1.y, v0.z / v1.z, v0.w / v1.w);
}
RTLIB_INLINE RTLIB_HOST_DEVICE ulonglong4 operator/(const ulonglong4 &v0, const unsigned long long v1)
{
  return make_ulonglong4(v0.x / v1, v0.y / v1, v0.z / v1, v0.w / v1);
}
RTLIB_INLINE RTLIB_HOST_DEVICE ulonglong4 operator/(const unsigned long long v0, const ulonglong4 &v1)
{
  return make_ulonglong4(v0 / v1.x, v0 / v1.y, v0 / v1.z, v0 / v1.w);
}
RTLIB_INLINE RTLIB_HOST_DEVICE bool operator==(const float2 &v0, const float2 &v1)
{
  return (v0.x == v1.x) && (v0.y == v1.y);
}
RTLIB_INLINE RTLIB_HOST_DEVICE bool operator!=(const float2 &v0, const float2 &v1)
{
  return (v0.x != v1.x) || (v0.y != v1.y);
}
RTLIB_INLINE RTLIB_HOST_DEVICE float2 &operator+=(float2 &v0, const float2 &v1)
{
  v0.x += v1.x;
  v0.y += v1.y;
  return v0;
}
RTLIB_INLINE RTLIB_HOST_DEVICE float2 &operator-=(float2 &v0, const float2 &v1)
{
  v0.x -= v1.x;
  v0.y -= v1.y;
  return v0;
}
RTLIB_INLINE RTLIB_HOST_DEVICE float2 &operator*=(float2 &v0, const float2 &v1)
{
  v0.x *= v1.x;
  v0.y *= v1.y;
  return v0;
}
RTLIB_INLINE RTLIB_HOST_DEVICE float2 &operator*=(float2 &v0, const float v1)
{
  v0.x *= v1;
  v0.y *= v1;
  return v0;
}
RTLIB_INLINE RTLIB_HOST_DEVICE float2 &operator/=(float2 &v0, const float2 &v1)
{
  v0.x /= v1.x;
  v0.y /= v1.y;
  return v0;
}
RTLIB_INLINE RTLIB_HOST_DEVICE float2 &operator/=(float2 &v0, const float v1)
{
  v0.x /= v1;
  v0.y /= v1;
  return v0;
}
RTLIB_INLINE RTLIB_HOST_DEVICE float2 operator+(const float2 &v0, const float2 &v1)
{
  return make_float2(v0.x + v1.x, v0.y + v1.y);
}
RTLIB_INLINE RTLIB_HOST_DEVICE float2 operator-(const float2 &v0, const float2 &v1)
{
  return make_float2(v0.x - v1.x, v0.y - v1.y);
}
RTLIB_INLINE RTLIB_HOST_DEVICE float2 operator*(const float2 &v0, const float2 &v1)
{
  return make_float2(v0.x * v1.x, v0.y * v1.y);
}
RTLIB_INLINE RTLIB_HOST_DEVICE float2 operator*(const float2 &v0, const float v1)
{
  return make_float2(v0.x * v1, v0.y * v1);
}
RTLIB_INLINE RTLIB_HOST_DEVICE float2 operator*(const float v0, const float2 &v1)
{
  return make_float2(v0 * v1.x, v0 * v1.y);
}
RTLIB_INLINE RTLIB_HOST_DEVICE float2 operator/(const float2 &v0, const float2 &v1)
{
  return make_float2(v0.x / v1.x, v0.y / v1.y);
}
RTLIB_INLINE RTLIB_HOST_DEVICE float2 operator/(const float2 &v0, const float v1)
{
  return make_float2(v0.x / v1, v0.y / v1);
}
RTLIB_INLINE RTLIB_HOST_DEVICE float2 operator/(const float v0, const float2 &v1)
{
  return make_float2(v0 / v1.x, v0 / v1.y);
}
RTLIB_INLINE RTLIB_HOST_DEVICE bool operator==(const float3 &v0, const float3 &v1)
{
  return (v0.x == v1.x) && (v0.y == v1.y) && (v0.z == v1.z);
}
RTLIB_INLINE RTLIB_HOST_DEVICE bool operator!=(const float3 &v0, const float3 &v1)
{
  return (v0.x != v1.x) || (v0.y != v1.y) || (v0.z != v1.z);
}
RTLIB_INLINE RTLIB_HOST_DEVICE float3 &operator+=(float3 &v0, const float3 &v1)
{
  v0.x += v1.x;
  v0.y += v1.y;
  v0.z += v1.z;
  return v0;
}
RTLIB_INLINE RTLIB_HOST_DEVICE float3 &operator-=(float3 &v0, const float3 &v1)
{
  v0.x -= v1.x;
  v0.y -= v1.y;
  v0.z -= v1.z;
  return v0;
}
RTLIB_INLINE RTLIB_HOST_DEVICE float3 &operator*=(float3 &v0, const float3 &v1)
{
  v0.x *= v1.x;
  v0.y *= v1.y;
  v0.z *= v1.z;
  return v0;
}
RTLIB_INLINE RTLIB_HOST_DEVICE float3 &operator*=(float3 &v0, const float v1)
{
  v0.x *= v1;
  v0.y *= v1;
  v0.z *= v1;
  return v0;
}
RTLIB_INLINE RTLIB_HOST_DEVICE float3 &operator/=(float3 &v0, const float3 &v1)
{
  v0.x /= v1.x;
  v0.y /= v1.y;
  v0.z /= v1.z;
  return v0;
}
RTLIB_INLINE RTLIB_HOST_DEVICE float3 &operator/=(float3 &v0, const float v1)
{
  v0.x /= v1;
  v0.y /= v1;
  v0.z /= v1;
  return v0;
}
RTLIB_INLINE RTLIB_HOST_DEVICE float3 operator+(const float3 &v0, const float3 &v1)
{
  return make_float3(v0.x + v1.x, v0.y + v1.y, v0.z + v1.z);
}
RTLIB_INLINE RTLIB_HOST_DEVICE float3 operator-(const float3 &v0, const float3 &v1)
{
  return make_float3(v0.x - v1.x, v0.y - v1.y, v0.z - v1.z);
}
RTLIB_INLINE RTLIB_HOST_DEVICE float3 operator*(const float3 &v0, const float3 &v1)
{
  return make_float3(v0.x * v1.x, v0.y * v1.y, v0.z * v1.z);
}
RTLIB_INLINE RTLIB_HOST_DEVICE float3 operator*(const float3 &v0, const float v1)
{
  return make_float3(v0.x * v1, v0.y * v1, v0.z * v1);
}
RTLIB_INLINE RTLIB_HOST_DEVICE float3 operator*(const float v0, const float3 &v1)
{
  return make_float3(v0 * v1.x, v0 * v1.y, v0 * v1.z);
}
RTLIB_INLINE RTLIB_HOST_DEVICE float3 operator/(const float3 &v0, const float3 &v1)
{
  return make_float3(v0.x / v1.x, v0.y / v1.y, v0.z / v1.z);
}
RTLIB_INLINE RTLIB_HOST_DEVICE float3 operator/(const float3 &v0, const float v1)
{
  return make_float3(v0.x / v1, v0.y / v1, v0.z / v1);
}
RTLIB_INLINE RTLIB_HOST_DEVICE float3 operator/(const float v0, const float3 &v1)
{
  return make_float3(v0 / v1.x, v0 / v1.y, v0 / v1.z);
}
RTLIB_INLINE RTLIB_HOST_DEVICE bool operator==(const float4 &v0, const float4 &v1)
{
  return (v0.x == v1.x) && (v0.y == v1.y) && (v0.z == v1.z) && (v0.w == v1.w);
}
RTLIB_INLINE RTLIB_HOST_DEVICE bool operator!=(const float4 &v0, const float4 &v1)
{
  return (v0.x != v1.x) || (v0.y != v1.y) || (v0.z != v1.z) || (v0.w != v1.w);
}
RTLIB_INLINE RTLIB_HOST_DEVICE float4 &operator+=(float4 &v0, const float4 &v1)
{
  v0.x += v1.x;
  v0.y += v1.y;
  v0.z += v1.z;
  v0.w += v1.w;
  return v0;
}
RTLIB_INLINE RTLIB_HOST_DEVICE float4 &operator-=(float4 &v0, const float4 &v1)
{
  v0.x -= v1.x;
  v0.y -= v1.y;
  v0.z -= v1.z;
  v0.w -= v1.w;
  return v0;
}
RTLIB_INLINE RTLIB_HOST_DEVICE float4 &operator*=(float4 &v0, const float4 &v1)
{
  v0.x *= v1.x;
  v0.y *= v1.y;
  v0.z *= v1.z;
  v0.w *= v1.w;
  return v0;
}
RTLIB_INLINE RTLIB_HOST_DEVICE float4 &operator*=(float4 &v0, const float v1)
{
  v0.x *= v1;
  v0.y *= v1;
  v0.z *= v1;
  v0.w *= v1;
  return v0;
}
RTLIB_INLINE RTLIB_HOST_DEVICE float4 &operator/=(float4 &v0, const float4 &v1)
{
  v0.x /= v1.x;
  v0.y /= v1.y;
  v0.z /= v1.z;
  v0.w /= v1.w;
  return v0;
}
RTLIB_INLINE RTLIB_HOST_DEVICE float4 &operator/=(float4 &v0, const float v1)
{
  v0.x /= v1;
  v0.y /= v1;
  v0.z /= v1;
  v0.w /= v1;
  return v0;
}
RTLIB_INLINE RTLIB_HOST_DEVICE float4 operator+(const float4 &v0, const float4 &v1)
{
  return make_float4(v0.x + v1.x, v0.y + v1.y, v0.z + v1.z, v0.w + v1.w);
}
RTLIB_INLINE RTLIB_HOST_DEVICE float4 operator-(const float4 &v0, const float4 &v1)
{
  return make_float4(v0.x - v1.x, v0.y - v1.y, v0.z - v1.z, v0.w - v1.w);
}
RTLIB_INLINE RTLIB_HOST_DEVICE float4 operator*(const float4 &v0, const float4 &v1)
{
  return make_float4(v0.x * v1.x, v0.y * v1.y, v0.z * v1.z, v0.w * v1.w);
}
RTLIB_INLINE RTLIB_HOST_DEVICE float4 operator*(const float4 &v0, const float v1)
{
  return make_float4(v0.x * v1, v0.y * v1, v0.z * v1, v0.w * v1);
}
RTLIB_INLINE RTLIB_HOST_DEVICE float4 operator*(const float v0, const float4 &v1)
{
  return make_float4(v0 * v1.x, v0 * v1.y, v0 * v1.z, v0 * v1.w);
}
RTLIB_INLINE RTLIB_HOST_DEVICE float4 operator/(const float4 &v0, const float4 &v1)
{
  return make_float4(v0.x / v1.x, v0.y / v1.y, v0.z / v1.z, v0.w / v1.w);
}
RTLIB_INLINE RTLIB_HOST_DEVICE float4 operator/(const float4 &v0, const float v1)
{
  return make_float4(v0.x / v1, v0.y / v1, v0.z / v1, v0.w / v1);
}
RTLIB_INLINE RTLIB_HOST_DEVICE float4 operator/(const float v0, const float4 &v1)
{
  return make_float4(v0 / v1.x, v0 / v1.y, v0 / v1.z, v0 / v1.w);
}
RTLIB_INLINE RTLIB_HOST_DEVICE char2 make_char2(const signed char v)
{
  return make_char2(v, v);
}
RTLIB_INLINE RTLIB_HOST_DEVICE char2 make_char2(const char2 &v)
{
  return make_char2(v.x, v.y);
}
RTLIB_INLINE RTLIB_HOST_DEVICE char3 make_char3(const signed char v)
{
  return make_char3(v, v, v);
}
RTLIB_INLINE RTLIB_HOST_DEVICE char3 make_char3(const char2 &v)
{
  return make_char3(v.x, v.y, 0.0f);
}
RTLIB_INLINE RTLIB_HOST_DEVICE char3 make_char3(const char3 &v)
{
  return make_char3(v.x, v.y, v.z);
}
RTLIB_INLINE RTLIB_HOST_DEVICE char3 make_char3(const signed char v0, const char2 &v1)
{
  return make_char3(v0, v1.x, v1.y);
}
RTLIB_INLINE RTLIB_HOST_DEVICE char3 make_char3(const char2 &v0, const signed char v1)
{
  return make_char3(v0.x, v0.y, v1);
}
RTLIB_INLINE RTLIB_HOST_DEVICE char4 make_char4(const signed char v)
{
  return make_char4(v, v, v, v);
}
RTLIB_INLINE RTLIB_HOST_DEVICE char4 make_char4(const char2 &v)
{
  return make_char4(v.x, v.y, 0.0f, 0.0f);
}
RTLIB_INLINE RTLIB_HOST_DEVICE char4 make_char4(const char3 &v)
{
  return make_char4(v.x, v.y, v.z, 0.0f);
}
RTLIB_INLINE RTLIB_HOST_DEVICE char4 make_char4(const char4 &v)
{
  return make_char4(v.x, v.y, v.z, v.w);
}
RTLIB_INLINE RTLIB_HOST_DEVICE char4 make_char4(const signed char v0, const char3 &v1)
{
  return make_char4(v0, v1.x, v1.y, v1.z);
}
RTLIB_INLINE RTLIB_HOST_DEVICE char4 make_char4(const char2 &v0, const char2 &v1)
{
  return make_char4(v0.x, v0.y, v1.x, v1.y);
}
RTLIB_INLINE RTLIB_HOST_DEVICE char4 make_char4(const char3 &v0, const signed char v1)
{
  return make_char4(v0.x, v0.y, v0.z, v1);
}
RTLIB_INLINE RTLIB_HOST_DEVICE char4 make_char4(const signed char v0, const signed char v1, const char2 &v2)
{
  return make_char4(v0, v1, v2.x, v2.y);
}
RTLIB_INLINE RTLIB_HOST_DEVICE char4 make_char4(const signed char v0, const char2 &v1, const signed char v2)
{
  return make_char4(v0, v1.x, v1.y, v2);
}
RTLIB_INLINE RTLIB_HOST_DEVICE char4 make_char4(const char2 &v0, const signed char v1, const signed char v2)
{
  return make_char4(v0.x, v0.y, v1, v2);
}
RTLIB_INLINE RTLIB_HOST_DEVICE short2 make_short2(const short v)
{
  return make_short2(v, v);
}
RTLIB_INLINE RTLIB_HOST_DEVICE short2 make_short2(const short2 &v)
{
  return make_short2(v.x, v.y);
}
RTLIB_INLINE RTLIB_HOST_DEVICE short3 make_short3(const short v)
{
  return make_short3(v, v, v);
}
RTLIB_INLINE RTLIB_HOST_DEVICE short3 make_short3(const short2 &v)
{
  return make_short3(v.x, v.y, 0.0f);
}
RTLIB_INLINE RTLIB_HOST_DEVICE short3 make_short3(const short3 &v)
{
  return make_short3(v.x, v.y, v.z);
}
RTLIB_INLINE RTLIB_HOST_DEVICE short3 make_short3(const short v0, const short2 &v1)
{
  return make_short3(v0, v1.x, v1.y);
}
RTLIB_INLINE RTLIB_HOST_DEVICE short3 make_short3(const short2 &v0, const short v1)
{
  return make_short3(v0.x, v0.y, v1);
}
RTLIB_INLINE RTLIB_HOST_DEVICE short4 make_short4(const short v)
{
  return make_short4(v, v, v, v);
}
RTLIB_INLINE RTLIB_HOST_DEVICE short4 make_short4(const short2 &v)
{
  return make_short4(v.x, v.y, 0.0f, 0.0f);
}
RTLIB_INLINE RTLIB_HOST_DEVICE short4 make_short4(const short3 &v)
{
  return make_short4(v.x, v.y, v.z, 0.0f);
}
RTLIB_INLINE RTLIB_HOST_DEVICE short4 make_short4(const short4 &v)
{
  return make_short4(v.x, v.y, v.z, v.w);
}
RTLIB_INLINE RTLIB_HOST_DEVICE short4 make_short4(const short v0, const short3 &v1)
{
  return make_short4(v0, v1.x, v1.y, v1.z);
}
RTLIB_INLINE RTLIB_HOST_DEVICE short4 make_short4(const short2 &v0, const short2 &v1)
{
  return make_short4(v0.x, v0.y, v1.x, v1.y);
}
RTLIB_INLINE RTLIB_HOST_DEVICE short4 make_short4(const short3 &v0, const short v1)
{
  return make_short4(v0.x, v0.y, v0.z, v1);
}
RTLIB_INLINE RTLIB_HOST_DEVICE short4 make_short4(const short v0, const short v1, const short2 &v2)
{
  return make_short4(v0, v1, v2.x, v2.y);
}
RTLIB_INLINE RTLIB_HOST_DEVICE short4 make_short4(const short v0, const short2 &v1, const short v2)
{
  return make_short4(v0, v1.x, v1.y, v2);
}
RTLIB_INLINE RTLIB_HOST_DEVICE short4 make_short4(const short2 &v0, const short v1, const short v2)
{
  return make_short4(v0.x, v0.y, v1, v2);
}
RTLIB_INLINE RTLIB_HOST_DEVICE int2 make_int2(const int v)
{
  return make_int2(v, v);
}
RTLIB_INLINE RTLIB_HOST_DEVICE int2 make_int2(const int2 &v)
{
  return make_int2(v.x, v.y);
}
RTLIB_INLINE RTLIB_HOST_DEVICE int3 make_int3(const int v)
{
  return make_int3(v, v, v);
}
RTLIB_INLINE RTLIB_HOST_DEVICE int3 make_int3(const int2 &v)
{
  return make_int3(v.x, v.y, 0.0f);
}
RTLIB_INLINE RTLIB_HOST_DEVICE int3 make_int3(const int3 &v)
{
  return make_int3(v.x, v.y, v.z);
}
RTLIB_INLINE RTLIB_HOST_DEVICE int3 make_int3(const int v0, const int2 &v1)
{
  return make_int3(v0, v1.x, v1.y);
}
RTLIB_INLINE RTLIB_HOST_DEVICE int3 make_int3(const int2 &v0, const int v1)
{
  return make_int3(v0.x, v0.y, v1);
}
RTLIB_INLINE RTLIB_HOST_DEVICE int4 make_int4(const int v)
{
  return make_int4(v, v, v, v);
}
RTLIB_INLINE RTLIB_HOST_DEVICE int4 make_int4(const int2 &v)
{
  return make_int4(v.x, v.y, 0.0f, 0.0f);
}
RTLIB_INLINE RTLIB_HOST_DEVICE int4 make_int4(const int3 &v)
{
  return make_int4(v.x, v.y, v.z, 0.0f);
}
RTLIB_INLINE RTLIB_HOST_DEVICE int4 make_int4(const int4 &v)
{
  return make_int4(v.x, v.y, v.z, v.w);
}
RTLIB_INLINE RTLIB_HOST_DEVICE int4 make_int4(const int v0, const int3 &v1)
{
  return make_int4(v0, v1.x, v1.y, v1.z);
}
RTLIB_INLINE RTLIB_HOST_DEVICE int4 make_int4(const int2 &v0, const int2 &v1)
{
  return make_int4(v0.x, v0.y, v1.x, v1.y);
}
RTLIB_INLINE RTLIB_HOST_DEVICE int4 make_int4(const int3 &v0, const int v1)
{
  return make_int4(v0.x, v0.y, v0.z, v1);
}
RTLIB_INLINE RTLIB_HOST_DEVICE int4 make_int4(const int v0, const int v1, const int2 &v2)
{
  return make_int4(v0, v1, v2.x, v2.y);
}
RTLIB_INLINE RTLIB_HOST_DEVICE int4 make_int4(const int v0, const int2 &v1, const int v2)
{
  return make_int4(v0, v1.x, v1.y, v2);
}
RTLIB_INLINE RTLIB_HOST_DEVICE int4 make_int4(const int2 &v0, const int v1, const int v2)
{
  return make_int4(v0.x, v0.y, v1, v2);
}
RTLIB_INLINE RTLIB_HOST_DEVICE long2 make_long2(const long v)
{
  return make_long2(v, v);
}
RTLIB_INLINE RTLIB_HOST_DEVICE long2 make_long2(const long2 &v)
{
  return make_long2(v.x, v.y);
}
RTLIB_INLINE RTLIB_HOST_DEVICE long3 make_long3(const long v)
{
  return make_long3(v, v, v);
}
RTLIB_INLINE RTLIB_HOST_DEVICE long3 make_long3(const long2 &v)
{
  return make_long3(v.x, v.y, 0.0f);
}
RTLIB_INLINE RTLIB_HOST_DEVICE long3 make_long3(const long3 &v)
{
  return make_long3(v.x, v.y, v.z);
}
RTLIB_INLINE RTLIB_HOST_DEVICE long3 make_long3(const long v0, const long2 &v1)
{
  return make_long3(v0, v1.x, v1.y);
}
RTLIB_INLINE RTLIB_HOST_DEVICE long3 make_long3(const long2 &v0, const long v1)
{
  return make_long3(v0.x, v0.y, v1);
}
RTLIB_INLINE RTLIB_HOST_DEVICE long4 make_long4(const long v)
{
  return make_long4(v, v, v, v);
}
RTLIB_INLINE RTLIB_HOST_DEVICE long4 make_long4(const long2 &v)
{
  return make_long4(v.x, v.y, 0.0f, 0.0f);
}
RTLIB_INLINE RTLIB_HOST_DEVICE long4 make_long4(const long3 &v)
{
  return make_long4(v.x, v.y, v.z, 0.0f);
}
RTLIB_INLINE RTLIB_HOST_DEVICE long4 make_long4(const long4 &v)
{
  return make_long4(v.x, v.y, v.z, v.w);
}
RTLIB_INLINE RTLIB_HOST_DEVICE long4 make_long4(const long v0, const long3 &v1)
{
  return make_long4(v0, v1.x, v1.y, v1.z);
}
RTLIB_INLINE RTLIB_HOST_DEVICE long4 make_long4(const long2 &v0, const long2 &v1)
{
  return make_long4(v0.x, v0.y, v1.x, v1.y);
}
RTLIB_INLINE RTLIB_HOST_DEVICE long4 make_long4(const long3 &v0, const long v1)
{
  return make_long4(v0.x, v0.y, v0.z, v1);
}
RTLIB_INLINE RTLIB_HOST_DEVICE long4 make_long4(const long v0, const long v1, const long2 &v2)
{
  return make_long4(v0, v1, v2.x, v2.y);
}
RTLIB_INLINE RTLIB_HOST_DEVICE long4 make_long4(const long v0, const long2 &v1, const long v2)
{
  return make_long4(v0, v1.x, v1.y, v2);
}
RTLIB_INLINE RTLIB_HOST_DEVICE long4 make_long4(const long2 &v0, const long v1, const long v2)
{
  return make_long4(v0.x, v0.y, v1, v2);
}
RTLIB_INLINE RTLIB_HOST_DEVICE longlong2 make_longlong2(const long long v)
{
  return make_longlong2(v, v);
}
RTLIB_INLINE RTLIB_HOST_DEVICE longlong2 make_longlong2(const longlong2 &v)
{
  return make_longlong2(v.x, v.y);
}
RTLIB_INLINE RTLIB_HOST_DEVICE longlong3 make_longlong3(const long long v)
{
  return make_longlong3(v, v, v);
}
RTLIB_INLINE RTLIB_HOST_DEVICE longlong3 make_longlong3(const longlong2 &v)
{
  return make_longlong3(v.x, v.y, 0.0f);
}
RTLIB_INLINE RTLIB_HOST_DEVICE longlong3 make_longlong3(const longlong3 &v)
{
  return make_longlong3(v.x, v.y, v.z);
}
RTLIB_INLINE RTLIB_HOST_DEVICE longlong3 make_longlong3(const long long v0, const longlong2 &v1)
{
  return make_longlong3(v0, v1.x, v1.y);
}
RTLIB_INLINE RTLIB_HOST_DEVICE longlong3 make_longlong3(const longlong2 &v0, const long long v1)
{
  return make_longlong3(v0.x, v0.y, v1);
}
RTLIB_INLINE RTLIB_HOST_DEVICE longlong4 make_longlong4(const long long v)
{
  return make_longlong4(v, v, v, v);
}
RTLIB_INLINE RTLIB_HOST_DEVICE longlong4 make_longlong4(const longlong2 &v)
{
  return make_longlong4(v.x, v.y, 0.0f, 0.0f);
}
RTLIB_INLINE RTLIB_HOST_DEVICE longlong4 make_longlong4(const longlong3 &v)
{
  return make_longlong4(v.x, v.y, v.z, 0.0f);
}
RTLIB_INLINE RTLIB_HOST_DEVICE longlong4 make_longlong4(const longlong4 &v)
{
  return make_longlong4(v.x, v.y, v.z, v.w);
}
RTLIB_INLINE RTLIB_HOST_DEVICE longlong4 make_longlong4(const long long v0, const longlong3 &v1)
{
  return make_longlong4(v0, v1.x, v1.y, v1.z);
}
RTLIB_INLINE RTLIB_HOST_DEVICE longlong4 make_longlong4(const longlong2 &v0, const longlong2 &v1)
{
  return make_longlong4(v0.x, v0.y, v1.x, v1.y);
}
RTLIB_INLINE RTLIB_HOST_DEVICE longlong4 make_longlong4(const longlong3 &v0, const long long v1)
{
  return make_longlong4(v0.x, v0.y, v0.z, v1);
}
RTLIB_INLINE RTLIB_HOST_DEVICE longlong4 make_longlong4(const long long v0, const long long v1, const longlong2 &v2)
{
  return make_longlong4(v0, v1, v2.x, v2.y);
}
RTLIB_INLINE RTLIB_HOST_DEVICE longlong4 make_longlong4(const long long v0, const longlong2 &v1, const long long v2)
{
  return make_longlong4(v0, v1.x, v1.y, v2);
}
RTLIB_INLINE RTLIB_HOST_DEVICE longlong4 make_longlong4(const longlong2 &v0, const long long v1, const long long v2)
{
  return make_longlong4(v0.x, v0.y, v1, v2);
}
RTLIB_INLINE RTLIB_HOST_DEVICE uchar2 make_uchar2(const unsigned char v)
{
  return make_uchar2(v, v);
}
RTLIB_INLINE RTLIB_HOST_DEVICE uchar2 make_uchar2(const uchar2 &v)
{
  return make_uchar2(v.x, v.y);
}
RTLIB_INLINE RTLIB_HOST_DEVICE uchar3 make_uchar3(const unsigned char v)
{
  return make_uchar3(v, v, v);
}
RTLIB_INLINE RTLIB_HOST_DEVICE uchar3 make_uchar3(const uchar2 &v)
{
  return make_uchar3(v.x, v.y, 0.0f);
}
RTLIB_INLINE RTLIB_HOST_DEVICE uchar3 make_uchar3(const uchar3 &v)
{
  return make_uchar3(v.x, v.y, v.z);
}
RTLIB_INLINE RTLIB_HOST_DEVICE uchar3 make_uchar3(const unsigned char v0, const uchar2 &v1)
{
  return make_uchar3(v0, v1.x, v1.y);
}
RTLIB_INLINE RTLIB_HOST_DEVICE uchar3 make_uchar3(const uchar2 &v0, const unsigned char v1)
{
  return make_uchar3(v0.x, v0.y, v1);
}
RTLIB_INLINE RTLIB_HOST_DEVICE uchar4 make_uchar4(const unsigned char v)
{
  return make_uchar4(v, v, v, v);
}
RTLIB_INLINE RTLIB_HOST_DEVICE uchar4 make_uchar4(const uchar2 &v)
{
  return make_uchar4(v.x, v.y, 0.0f, 0.0f);
}
RTLIB_INLINE RTLIB_HOST_DEVICE uchar4 make_uchar4(const uchar3 &v)
{
  return make_uchar4(v.x, v.y, v.z, 0.0f);
}
RTLIB_INLINE RTLIB_HOST_DEVICE uchar4 make_uchar4(const uchar4 &v)
{
  return make_uchar4(v.x, v.y, v.z, v.w);
}
RTLIB_INLINE RTLIB_HOST_DEVICE uchar4 make_uchar4(const unsigned char v0, const uchar3 &v1)
{
  return make_uchar4(v0, v1.x, v1.y, v1.z);
}
RTLIB_INLINE RTLIB_HOST_DEVICE uchar4 make_uchar4(const uchar2 &v0, const uchar2 &v1)
{
  return make_uchar4(v0.x, v0.y, v1.x, v1.y);
}
RTLIB_INLINE RTLIB_HOST_DEVICE uchar4 make_uchar4(const uchar3 &v0, const unsigned char v1)
{
  return make_uchar4(v0.x, v0.y, v0.z, v1);
}
RTLIB_INLINE RTLIB_HOST_DEVICE uchar4 make_uchar4(const unsigned char v0, const unsigned char v1, const uchar2 &v2)
{
  return make_uchar4(v0, v1, v2.x, v2.y);
}
RTLIB_INLINE RTLIB_HOST_DEVICE uchar4 make_uchar4(const unsigned char v0, const uchar2 &v1, const unsigned char v2)
{
  return make_uchar4(v0, v1.x, v1.y, v2);
}
RTLIB_INLINE RTLIB_HOST_DEVICE uchar4 make_uchar4(const uchar2 &v0, const unsigned char v1, const unsigned char v2)
{
  return make_uchar4(v0.x, v0.y, v1, v2);
}
RTLIB_INLINE RTLIB_HOST_DEVICE ushort2 make_ushort2(const unsigned short v)
{
  return make_ushort2(v, v);
}
RTLIB_INLINE RTLIB_HOST_DEVICE ushort2 make_ushort2(const ushort2 &v)
{
  return make_ushort2(v.x, v.y);
}
RTLIB_INLINE RTLIB_HOST_DEVICE ushort3 make_ushort3(const unsigned short v)
{
  return make_ushort3(v, v, v);
}
RTLIB_INLINE RTLIB_HOST_DEVICE ushort3 make_ushort3(const ushort2 &v)
{
  return make_ushort3(v.x, v.y, 0.0f);
}
RTLIB_INLINE RTLIB_HOST_DEVICE ushort3 make_ushort3(const ushort3 &v)
{
  return make_ushort3(v.x, v.y, v.z);
}
RTLIB_INLINE RTLIB_HOST_DEVICE ushort3 make_ushort3(const unsigned short v0, const ushort2 &v1)
{
  return make_ushort3(v0, v1.x, v1.y);
}
RTLIB_INLINE RTLIB_HOST_DEVICE ushort3 make_ushort3(const ushort2 &v0, const unsigned short v1)
{
  return make_ushort3(v0.x, v0.y, v1);
}
RTLIB_INLINE RTLIB_HOST_DEVICE ushort4 make_ushort4(const unsigned short v)
{
  return make_ushort4(v, v, v, v);
}
RTLIB_INLINE RTLIB_HOST_DEVICE ushort4 make_ushort4(const ushort2 &v)
{
  return make_ushort4(v.x, v.y, 0.0f, 0.0f);
}
RTLIB_INLINE RTLIB_HOST_DEVICE ushort4 make_ushort4(const ushort3 &v)
{
  return make_ushort4(v.x, v.y, v.z, 0.0f);
}
RTLIB_INLINE RTLIB_HOST_DEVICE ushort4 make_ushort4(const ushort4 &v)
{
  return make_ushort4(v.x, v.y, v.z, v.w);
}
RTLIB_INLINE RTLIB_HOST_DEVICE ushort4 make_ushort4(const unsigned short v0, const ushort3 &v1)
{
  return make_ushort4(v0, v1.x, v1.y, v1.z);
}
RTLIB_INLINE RTLIB_HOST_DEVICE ushort4 make_ushort4(const ushort2 &v0, const ushort2 &v1)
{
  return make_ushort4(v0.x, v0.y, v1.x, v1.y);
}
RTLIB_INLINE RTLIB_HOST_DEVICE ushort4 make_ushort4(const ushort3 &v0, const unsigned short v1)
{
  return make_ushort4(v0.x, v0.y, v0.z, v1);
}
RTLIB_INLINE RTLIB_HOST_DEVICE ushort4 make_ushort4(const unsigned short v0, const unsigned short v1, const ushort2 &v2)
{
  return make_ushort4(v0, v1, v2.x, v2.y);
}
RTLIB_INLINE RTLIB_HOST_DEVICE ushort4 make_ushort4(const unsigned short v0, const ushort2 &v1, const unsigned short v2)
{
  return make_ushort4(v0, v1.x, v1.y, v2);
}
RTLIB_INLINE RTLIB_HOST_DEVICE ushort4 make_ushort4(const ushort2 &v0, const unsigned short v1, const unsigned short v2)
{
  return make_ushort4(v0.x, v0.y, v1, v2);
}
RTLIB_INLINE RTLIB_HOST_DEVICE uint2 make_uint2(const unsigned int v)
{
  return make_uint2(v, v);
}
RTLIB_INLINE RTLIB_HOST_DEVICE uint2 make_uint2(const uint2 &v)
{
  return make_uint2(v.x, v.y);
}
RTLIB_INLINE RTLIB_HOST_DEVICE uint3 make_uint3(const unsigned int v)
{
  return make_uint3(v, v, v);
}
RTLIB_INLINE RTLIB_HOST_DEVICE uint3 make_uint3(const uint2 &v)
{
  return make_uint3(v.x, v.y, 0.0f);
}
RTLIB_INLINE RTLIB_HOST_DEVICE uint3 make_uint3(const uint3 &v)
{
  return make_uint3(v.x, v.y, v.z);
}
RTLIB_INLINE RTLIB_HOST_DEVICE uint3 make_uint3(const unsigned int v0, const uint2 &v1)
{
  return make_uint3(v0, v1.x, v1.y);
}
RTLIB_INLINE RTLIB_HOST_DEVICE uint3 make_uint3(const uint2 &v0, const unsigned int v1)
{
  return make_uint3(v0.x, v0.y, v1);
}
RTLIB_INLINE RTLIB_HOST_DEVICE uint4 make_uint4(const unsigned int v)
{
  return make_uint4(v, v, v, v);
}
RTLIB_INLINE RTLIB_HOST_DEVICE uint4 make_uint4(const uint2 &v)
{
  return make_uint4(v.x, v.y, 0.0f, 0.0f);
}
RTLIB_INLINE RTLIB_HOST_DEVICE uint4 make_uint4(const uint3 &v)
{
  return make_uint4(v.x, v.y, v.z, 0.0f);
}
RTLIB_INLINE RTLIB_HOST_DEVICE uint4 make_uint4(const uint4 &v)
{
  return make_uint4(v.x, v.y, v.z, v.w);
}
RTLIB_INLINE RTLIB_HOST_DEVICE uint4 make_uint4(const unsigned int v0, const uint3 &v1)
{
  return make_uint4(v0, v1.x, v1.y, v1.z);
}
RTLIB_INLINE RTLIB_HOST_DEVICE uint4 make_uint4(const uint2 &v0, const uint2 &v1)
{
  return make_uint4(v0.x, v0.y, v1.x, v1.y);
}
RTLIB_INLINE RTLIB_HOST_DEVICE uint4 make_uint4(const uint3 &v0, const unsigned int v1)
{
  return make_uint4(v0.x, v0.y, v0.z, v1);
}
RTLIB_INLINE RTLIB_HOST_DEVICE uint4 make_uint4(const unsigned int v0, const unsigned int v1, const uint2 &v2)
{
  return make_uint4(v0, v1, v2.x, v2.y);
}
RTLIB_INLINE RTLIB_HOST_DEVICE uint4 make_uint4(const unsigned int v0, const uint2 &v1, const unsigned int v2)
{
  return make_uint4(v0, v1.x, v1.y, v2);
}
RTLIB_INLINE RTLIB_HOST_DEVICE uint4 make_uint4(const uint2 &v0, const unsigned int v1, const unsigned int v2)
{
  return make_uint4(v0.x, v0.y, v1, v2);
}
RTLIB_INLINE RTLIB_HOST_DEVICE ulong2 make_ulong2(const unsigned long v)
{
  return make_ulong2(v, v);
}
RTLIB_INLINE RTLIB_HOST_DEVICE ulong2 make_ulong2(const ulong2 &v)
{
  return make_ulong2(v.x, v.y);
}
RTLIB_INLINE RTLIB_HOST_DEVICE ulong3 make_ulong3(const unsigned long v)
{
  return make_ulong3(v, v, v);
}
RTLIB_INLINE RTLIB_HOST_DEVICE ulong3 make_ulong3(const ulong2 &v)
{
  return make_ulong3(v.x, v.y, 0.0f);
}
RTLIB_INLINE RTLIB_HOST_DEVICE ulong3 make_ulong3(const ulong3 &v)
{
  return make_ulong3(v.x, v.y, v.z);
}
RTLIB_INLINE RTLIB_HOST_DEVICE ulong3 make_ulong3(const unsigned long v0, const ulong2 &v1)
{
  return make_ulong3(v0, v1.x, v1.y);
}
RTLIB_INLINE RTLIB_HOST_DEVICE ulong3 make_ulong3(const ulong2 &v0, const unsigned long v1)
{
  return make_ulong3(v0.x, v0.y, v1);
}
RTLIB_INLINE RTLIB_HOST_DEVICE ulong4 make_ulong4(const unsigned long v)
{
  return make_ulong4(v, v, v, v);
}
RTLIB_INLINE RTLIB_HOST_DEVICE ulong4 make_ulong4(const ulong2 &v)
{
  return make_ulong4(v.x, v.y, 0.0f, 0.0f);
}
RTLIB_INLINE RTLIB_HOST_DEVICE ulong4 make_ulong4(const ulong3 &v)
{
  return make_ulong4(v.x, v.y, v.z, 0.0f);
}
RTLIB_INLINE RTLIB_HOST_DEVICE ulong4 make_ulong4(const ulong4 &v)
{
  return make_ulong4(v.x, v.y, v.z, v.w);
}
RTLIB_INLINE RTLIB_HOST_DEVICE ulong4 make_ulong4(const unsigned long v0, const ulong3 &v1)
{
  return make_ulong4(v0, v1.x, v1.y, v1.z);
}
RTLIB_INLINE RTLIB_HOST_DEVICE ulong4 make_ulong4(const ulong2 &v0, const ulong2 &v1)
{
  return make_ulong4(v0.x, v0.y, v1.x, v1.y);
}
RTLIB_INLINE RTLIB_HOST_DEVICE ulong4 make_ulong4(const ulong3 &v0, const unsigned long v1)
{
  return make_ulong4(v0.x, v0.y, v0.z, v1);
}
RTLIB_INLINE RTLIB_HOST_DEVICE ulong4 make_ulong4(const unsigned long v0, const unsigned long v1, const ulong2 &v2)
{
  return make_ulong4(v0, v1, v2.x, v2.y);
}
RTLIB_INLINE RTLIB_HOST_DEVICE ulong4 make_ulong4(const unsigned long v0, const ulong2 &v1, const unsigned long v2)
{
  return make_ulong4(v0, v1.x, v1.y, v2);
}
RTLIB_INLINE RTLIB_HOST_DEVICE ulong4 make_ulong4(const ulong2 &v0, const unsigned long v1, const unsigned long v2)
{
  return make_ulong4(v0.x, v0.y, v1, v2);
}
RTLIB_INLINE RTLIB_HOST_DEVICE ulonglong2 make_ulonglong2(const unsigned long long v)
{
  return make_ulonglong2(v, v);
}
RTLIB_INLINE RTLIB_HOST_DEVICE ulonglong2 make_ulonglong2(const ulonglong2 &v)
{
  return make_ulonglong2(v.x, v.y);
}
RTLIB_INLINE RTLIB_HOST_DEVICE ulonglong3 make_ulonglong3(const unsigned long long v)
{
  return make_ulonglong3(v, v, v);
}
RTLIB_INLINE RTLIB_HOST_DEVICE ulonglong3 make_ulonglong3(const ulonglong2 &v)
{
  return make_ulonglong3(v.x, v.y, 0.0f);
}
RTLIB_INLINE RTLIB_HOST_DEVICE ulonglong3 make_ulonglong3(const ulonglong3 &v)
{
  return make_ulonglong3(v.x, v.y, v.z);
}
RTLIB_INLINE RTLIB_HOST_DEVICE ulonglong3 make_ulonglong3(const unsigned long long v0, const ulonglong2 &v1)
{
  return make_ulonglong3(v0, v1.x, v1.y);
}
RTLIB_INLINE RTLIB_HOST_DEVICE ulonglong3 make_ulonglong3(const ulonglong2 &v0, const unsigned long long v1)
{
  return make_ulonglong3(v0.x, v0.y, v1);
}
RTLIB_INLINE RTLIB_HOST_DEVICE ulonglong4 make_ulonglong4(const unsigned long long v)
{
  return make_ulonglong4(v, v, v, v);
}
RTLIB_INLINE RTLIB_HOST_DEVICE ulonglong4 make_ulonglong4(const ulonglong2 &v)
{
  return make_ulonglong4(v.x, v.y, 0.0f, 0.0f);
}
RTLIB_INLINE RTLIB_HOST_DEVICE ulonglong4 make_ulonglong4(const ulonglong3 &v)
{
  return make_ulonglong4(v.x, v.y, v.z, 0.0f);
}
RTLIB_INLINE RTLIB_HOST_DEVICE ulonglong4 make_ulonglong4(const ulonglong4 &v)
{
  return make_ulonglong4(v.x, v.y, v.z, v.w);
}
RTLIB_INLINE RTLIB_HOST_DEVICE ulonglong4 make_ulonglong4(const unsigned long long v0, const ulonglong3 &v1)
{
  return make_ulonglong4(v0, v1.x, v1.y, v1.z);
}
RTLIB_INLINE RTLIB_HOST_DEVICE ulonglong4 make_ulonglong4(const ulonglong2 &v0, const ulonglong2 &v1)
{
  return make_ulonglong4(v0.x, v0.y, v1.x, v1.y);
}
RTLIB_INLINE RTLIB_HOST_DEVICE ulonglong4 make_ulonglong4(const ulonglong3 &v0, const unsigned long long v1)
{
  return make_ulonglong4(v0.x, v0.y, v0.z, v1);
}
RTLIB_INLINE RTLIB_HOST_DEVICE ulonglong4 make_ulonglong4(const unsigned long long v0, const unsigned long long v1, const ulonglong2 &v2)
{
  return make_ulonglong4(v0, v1, v2.x, v2.y);
}
RTLIB_INLINE RTLIB_HOST_DEVICE ulonglong4 make_ulonglong4(const unsigned long long v0, const ulonglong2 &v1, const unsigned long long v2)
{
  return make_ulonglong4(v0, v1.x, v1.y, v2);
}
RTLIB_INLINE RTLIB_HOST_DEVICE ulonglong4 make_ulonglong4(const ulonglong2 &v0, const unsigned long long v1, const unsigned long long v2)
{
  return make_ulonglong4(v0.x, v0.y, v1, v2);
}
RTLIB_INLINE RTLIB_HOST_DEVICE float2 make_float2(const float v)
{
  return make_float2(v, v);
}
RTLIB_INLINE RTLIB_HOST_DEVICE float2 make_float2(const float2 &v)
{
  return make_float2(v.x, v.y);
}
RTLIB_INLINE RTLIB_HOST_DEVICE float3 make_float3(const float v)
{
  return make_float3(v, v, v);
}
RTLIB_INLINE RTLIB_HOST_DEVICE float3 make_float3(const float2 &v)
{
  return make_float3(v.x, v.y, 0.0f);
}
RTLIB_INLINE RTLIB_HOST_DEVICE float3 make_float3(const float3 &v)
{
  return make_float3(v.x, v.y, v.z);
}
RTLIB_INLINE RTLIB_HOST_DEVICE float3 make_float3(const float v0, const float2 &v1)
{
  return make_float3(v0, v1.x, v1.y);
}
RTLIB_INLINE RTLIB_HOST_DEVICE float3 make_float3(const float2 &v0, const float v1)
{
  return make_float3(v0.x, v0.y, v1);
}
RTLIB_INLINE RTLIB_HOST_DEVICE float4 make_float4(const float v)
{
  return make_float4(v, v, v, v);
}
RTLIB_INLINE RTLIB_HOST_DEVICE float4 make_float4(const float2 &v)
{
  return make_float4(v.x, v.y, 0.0f, 0.0f);
}
RTLIB_INLINE RTLIB_HOST_DEVICE float4 make_float4(const float3 &v)
{
  return make_float4(v.x, v.y, v.z, 0.0f);
}
RTLIB_INLINE RTLIB_HOST_DEVICE float4 make_float4(const float4 &v)
{
  return make_float4(v.x, v.y, v.z, v.w);
}
RTLIB_INLINE RTLIB_HOST_DEVICE float4 make_float4(const float v0, const float3 &v1)
{
  return make_float4(v0, v1.x, v1.y, v1.z);
}
RTLIB_INLINE RTLIB_HOST_DEVICE float4 make_float4(const float2 &v0, const float2 &v1)
{
  return make_float4(v0.x, v0.y, v1.x, v1.y);
}
RTLIB_INLINE RTLIB_HOST_DEVICE float4 make_float4(const float3 &v0, const float v1)
{
  return make_float4(v0.x, v0.y, v0.z, v1);
}
RTLIB_INLINE RTLIB_HOST_DEVICE float4 make_float4(const float v0, const float v1, const float2 &v2)
{
  return make_float4(v0, v1, v2.x, v2.y);
}
RTLIB_INLINE RTLIB_HOST_DEVICE float4 make_float4(const float v0, const float2 &v1, const float v2)
{
  return make_float4(v0, v1.x, v1.y, v2);
}
RTLIB_INLINE RTLIB_HOST_DEVICE float4 make_float4(const float2 &v0, const float v1, const float v2)
{
  return make_float4(v0.x, v0.y, v1, v2);
}
RTLIB_INLINE RTLIB_HOST_DEVICE float2 make_float2(const signed char v)
{
  return make_float2(static_cast<float>(v), static_cast<float>(v));
}
RTLIB_INLINE RTLIB_HOST_DEVICE float2 make_float2(const char2 &v)
{
  return make_float2(static_cast<float>(v.x), static_cast<float>(v.y));
}
RTLIB_INLINE RTLIB_HOST_DEVICE float3 make_float3(const signed char v)
{
  return make_float3(static_cast<float>(v), static_cast<float>(v), static_cast<float>(v));
}
RTLIB_INLINE RTLIB_HOST_DEVICE float3 make_float3(const char2 &v)
{
  return make_float3(static_cast<float>(v.x), static_cast<float>(v.y), 0.0f);
}
RTLIB_INLINE RTLIB_HOST_DEVICE float3 make_float3(const char3 &v)
{
  return make_float3(static_cast<float>(v.x), static_cast<float>(v.y), static_cast<float>(v.z));
}
RTLIB_INLINE RTLIB_HOST_DEVICE float4 make_float4(const signed char v)
{
  return make_float4(static_cast<float>(v), static_cast<float>(v), static_cast<float>(v), static_cast<float>(v));
}
RTLIB_INLINE RTLIB_HOST_DEVICE float4 make_float4(const char2 &v)
{
  return make_float4(static_cast<float>(v.x), static_cast<float>(v.y), 0.0f, 0.0f);
}
RTLIB_INLINE RTLIB_HOST_DEVICE float4 make_float4(const char3 &v)
{
  return make_float4(static_cast<float>(v.x), static_cast<float>(v.y), static_cast<float>(v.z), 0.0f);
}
RTLIB_INLINE RTLIB_HOST_DEVICE float4 make_float4(const char4 &v)
{
  return make_float4(static_cast<float>(v.x), static_cast<float>(v.y), static_cast<float>(v.z), static_cast<float>(v.w));
}
RTLIB_INLINE RTLIB_HOST_DEVICE float2 make_float2(const short v)
{
  return make_float2(static_cast<float>(v), static_cast<float>(v));
}
RTLIB_INLINE RTLIB_HOST_DEVICE float2 make_float2(const short2 &v)
{
  return make_float2(static_cast<float>(v.x), static_cast<float>(v.y));
}
RTLIB_INLINE RTLIB_HOST_DEVICE float3 make_float3(const short v)
{
  return make_float3(static_cast<float>(v), static_cast<float>(v), static_cast<float>(v));
}
RTLIB_INLINE RTLIB_HOST_DEVICE float3 make_float3(const short2 &v)
{
  return make_float3(static_cast<float>(v.x), static_cast<float>(v.y), 0.0f);
}
RTLIB_INLINE RTLIB_HOST_DEVICE float3 make_float3(const short3 &v)
{
  return make_float3(static_cast<float>(v.x), static_cast<float>(v.y), static_cast<float>(v.z));
}
RTLIB_INLINE RTLIB_HOST_DEVICE float4 make_float4(const short v)
{
  return make_float4(static_cast<float>(v), static_cast<float>(v), static_cast<float>(v), static_cast<float>(v));
}
RTLIB_INLINE RTLIB_HOST_DEVICE float4 make_float4(const short2 &v)
{
  return make_float4(static_cast<float>(v.x), static_cast<float>(v.y), 0.0f, 0.0f);
}
RTLIB_INLINE RTLIB_HOST_DEVICE float4 make_float4(const short3 &v)
{
  return make_float4(static_cast<float>(v.x), static_cast<float>(v.y), static_cast<float>(v.z), 0.0f);
}
RTLIB_INLINE RTLIB_HOST_DEVICE float4 make_float4(const short4 &v)
{
  return make_float4(static_cast<float>(v.x), static_cast<float>(v.y), static_cast<float>(v.z), static_cast<float>(v.w));
}
RTLIB_INLINE RTLIB_HOST_DEVICE float2 make_float2(const int v)
{
  return make_float2(static_cast<float>(v), static_cast<float>(v));
}
RTLIB_INLINE RTLIB_HOST_DEVICE float2 make_float2(const int2 &v)
{
  return make_float2(static_cast<float>(v.x), static_cast<float>(v.y));
}
RTLIB_INLINE RTLIB_HOST_DEVICE float3 make_float3(const int v)
{
  return make_float3(static_cast<float>(v), static_cast<float>(v), static_cast<float>(v));
}
RTLIB_INLINE RTLIB_HOST_DEVICE float3 make_float3(const int2 &v)
{
  return make_float3(static_cast<float>(v.x), static_cast<float>(v.y), 0.0f);
}
RTLIB_INLINE RTLIB_HOST_DEVICE float3 make_float3(const int3 &v)
{
  return make_float3(static_cast<float>(v.x), static_cast<float>(v.y), static_cast<float>(v.z));
}
RTLIB_INLINE RTLIB_HOST_DEVICE float4 make_float4(const int v)
{
  return make_float4(static_cast<float>(v), static_cast<float>(v), static_cast<float>(v), static_cast<float>(v));
}
RTLIB_INLINE RTLIB_HOST_DEVICE float4 make_float4(const int2 &v)
{
  return make_float4(static_cast<float>(v.x), static_cast<float>(v.y), 0.0f, 0.0f);
}
RTLIB_INLINE RTLIB_HOST_DEVICE float4 make_float4(const int3 &v)
{
  return make_float4(static_cast<float>(v.x), static_cast<float>(v.y), static_cast<float>(v.z), 0.0f);
}
RTLIB_INLINE RTLIB_HOST_DEVICE float4 make_float4(const int4 &v)
{
  return make_float4(static_cast<float>(v.x), static_cast<float>(v.y), static_cast<float>(v.z), static_cast<float>(v.w));
}
RTLIB_INLINE RTLIB_HOST_DEVICE float2 make_float2(const long v)
{
  return make_float2(static_cast<float>(v), static_cast<float>(v));
}
RTLIB_INLINE RTLIB_HOST_DEVICE float2 make_float2(const long2 &v)
{
  return make_float2(static_cast<float>(v.x), static_cast<float>(v.y));
}
RTLIB_INLINE RTLIB_HOST_DEVICE float3 make_float3(const long v)
{
  return make_float3(static_cast<float>(v), static_cast<float>(v), static_cast<float>(v));
}
RTLIB_INLINE RTLIB_HOST_DEVICE float3 make_float3(const long2 &v)
{
  return make_float3(static_cast<float>(v.x), static_cast<float>(v.y), 0.0f);
}
RTLIB_INLINE RTLIB_HOST_DEVICE float3 make_float3(const long3 &v)
{
  return make_float3(static_cast<float>(v.x), static_cast<float>(v.y), static_cast<float>(v.z));
}
RTLIB_INLINE RTLIB_HOST_DEVICE float4 make_float4(const long v)
{
  return make_float4(static_cast<float>(v), static_cast<float>(v), static_cast<float>(v), static_cast<float>(v));
}
RTLIB_INLINE RTLIB_HOST_DEVICE float4 make_float4(const long2 &v)
{
  return make_float4(static_cast<float>(v.x), static_cast<float>(v.y), 0.0f, 0.0f);
}
RTLIB_INLINE RTLIB_HOST_DEVICE float4 make_float4(const long3 &v)
{
  return make_float4(static_cast<float>(v.x), static_cast<float>(v.y), static_cast<float>(v.z), 0.0f);
}
RTLIB_INLINE RTLIB_HOST_DEVICE float4 make_float4(const long4 &v)
{
  return make_float4(static_cast<float>(v.x), static_cast<float>(v.y), static_cast<float>(v.z), static_cast<float>(v.w));
}
RTLIB_INLINE RTLIB_HOST_DEVICE float2 make_float2(const long long v)
{
  return make_float2(static_cast<float>(v), static_cast<float>(v));
}
RTLIB_INLINE RTLIB_HOST_DEVICE float2 make_float2(const longlong2 &v)
{
  return make_float2(static_cast<float>(v.x), static_cast<float>(v.y));
}
RTLIB_INLINE RTLIB_HOST_DEVICE float3 make_float3(const long long v)
{
  return make_float3(static_cast<float>(v), static_cast<float>(v), static_cast<float>(v));
}
RTLIB_INLINE RTLIB_HOST_DEVICE float3 make_float3(const longlong2 &v)
{
  return make_float3(static_cast<float>(v.x), static_cast<float>(v.y), 0.0f);
}
RTLIB_INLINE RTLIB_HOST_DEVICE float3 make_float3(const longlong3 &v)
{
  return make_float3(static_cast<float>(v.x), static_cast<float>(v.y), static_cast<float>(v.z));
}
RTLIB_INLINE RTLIB_HOST_DEVICE float4 make_float4(const long long v)
{
  return make_float4(static_cast<float>(v), static_cast<float>(v), static_cast<float>(v), static_cast<float>(v));
}
RTLIB_INLINE RTLIB_HOST_DEVICE float4 make_float4(const longlong2 &v)
{
  return make_float4(static_cast<float>(v.x), static_cast<float>(v.y), 0.0f, 0.0f);
}
RTLIB_INLINE RTLIB_HOST_DEVICE float4 make_float4(const longlong3 &v)
{
  return make_float4(static_cast<float>(v.x), static_cast<float>(v.y), static_cast<float>(v.z), 0.0f);
}
RTLIB_INLINE RTLIB_HOST_DEVICE float4 make_float4(const longlong4 &v)
{
  return make_float4(static_cast<float>(v.x), static_cast<float>(v.y), static_cast<float>(v.z), static_cast<float>(v.w));
}
RTLIB_INLINE RTLIB_HOST_DEVICE float2 make_float2(const unsigned char v)
{
  return make_float2(static_cast<float>(v), static_cast<float>(v));
}
RTLIB_INLINE RTLIB_HOST_DEVICE float2 make_float2(const uchar2 &v)
{
  return make_float2(static_cast<float>(v.x), static_cast<float>(v.y));
}
RTLIB_INLINE RTLIB_HOST_DEVICE float3 make_float3(const unsigned char v)
{
  return make_float3(static_cast<float>(v), static_cast<float>(v), static_cast<float>(v));
}
RTLIB_INLINE RTLIB_HOST_DEVICE float3 make_float3(const uchar2 &v)
{
  return make_float3(static_cast<float>(v.x), static_cast<float>(v.y), 0.0f);
}
RTLIB_INLINE RTLIB_HOST_DEVICE float3 make_float3(const uchar3 &v)
{
  return make_float3(static_cast<float>(v.x), static_cast<float>(v.y), static_cast<float>(v.z));
}
RTLIB_INLINE RTLIB_HOST_DEVICE float4 make_float4(const unsigned char v)
{
  return make_float4(static_cast<float>(v), static_cast<float>(v), static_cast<float>(v), static_cast<float>(v));
}
RTLIB_INLINE RTLIB_HOST_DEVICE float4 make_float4(const uchar2 &v)
{
  return make_float4(static_cast<float>(v.x), static_cast<float>(v.y), 0.0f, 0.0f);
}
RTLIB_INLINE RTLIB_HOST_DEVICE float4 make_float4(const uchar3 &v)
{
  return make_float4(static_cast<float>(v.x), static_cast<float>(v.y), static_cast<float>(v.z), 0.0f);
}
RTLIB_INLINE RTLIB_HOST_DEVICE float4 make_float4(const uchar4 &v)
{
  return make_float4(static_cast<float>(v.x), static_cast<float>(v.y), static_cast<float>(v.z), static_cast<float>(v.w));
}
RTLIB_INLINE RTLIB_HOST_DEVICE float2 make_float2(const unsigned short v)
{
  return make_float2(static_cast<float>(v), static_cast<float>(v));
}
RTLIB_INLINE RTLIB_HOST_DEVICE float2 make_float2(const ushort2 &v)
{
  return make_float2(static_cast<float>(v.x), static_cast<float>(v.y));
}
RTLIB_INLINE RTLIB_HOST_DEVICE float3 make_float3(const unsigned short v)
{
  return make_float3(static_cast<float>(v), static_cast<float>(v), static_cast<float>(v));
}
RTLIB_INLINE RTLIB_HOST_DEVICE float3 make_float3(const ushort2 &v)
{
  return make_float3(static_cast<float>(v.x), static_cast<float>(v.y), 0.0f);
}
RTLIB_INLINE RTLIB_HOST_DEVICE float3 make_float3(const ushort3 &v)
{
  return make_float3(static_cast<float>(v.x), static_cast<float>(v.y), static_cast<float>(v.z));
}
RTLIB_INLINE RTLIB_HOST_DEVICE float4 make_float4(const unsigned short v)
{
  return make_float4(static_cast<float>(v), static_cast<float>(v), static_cast<float>(v), static_cast<float>(v));
}
RTLIB_INLINE RTLIB_HOST_DEVICE float4 make_float4(const ushort2 &v)
{
  return make_float4(static_cast<float>(v.x), static_cast<float>(v.y), 0.0f, 0.0f);
}
RTLIB_INLINE RTLIB_HOST_DEVICE float4 make_float4(const ushort3 &v)
{
  return make_float4(static_cast<float>(v.x), static_cast<float>(v.y), static_cast<float>(v.z), 0.0f);
}
RTLIB_INLINE RTLIB_HOST_DEVICE float4 make_float4(const ushort4 &v)
{
  return make_float4(static_cast<float>(v.x), static_cast<float>(v.y), static_cast<float>(v.z), static_cast<float>(v.w));
}
RTLIB_INLINE RTLIB_HOST_DEVICE float2 make_float2(const unsigned int v)
{
  return make_float2(static_cast<float>(v), static_cast<float>(v));
}
RTLIB_INLINE RTLIB_HOST_DEVICE float2 make_float2(const uint2 &v)
{
  return make_float2(static_cast<float>(v.x), static_cast<float>(v.y));
}
RTLIB_INLINE RTLIB_HOST_DEVICE float3 make_float3(const unsigned int v)
{
  return make_float3(static_cast<float>(v), static_cast<float>(v), static_cast<float>(v));
}
RTLIB_INLINE RTLIB_HOST_DEVICE float3 make_float3(const uint2 &v)
{
  return make_float3(static_cast<float>(v.x), static_cast<float>(v.y), 0.0f);
}
RTLIB_INLINE RTLIB_HOST_DEVICE float3 make_float3(const uint3 &v)
{
  return make_float3(static_cast<float>(v.x), static_cast<float>(v.y), static_cast<float>(v.z));
}
RTLIB_INLINE RTLIB_HOST_DEVICE float4 make_float4(const unsigned int v)
{
  return make_float4(static_cast<float>(v), static_cast<float>(v), static_cast<float>(v), static_cast<float>(v));
}
RTLIB_INLINE RTLIB_HOST_DEVICE float4 make_float4(const uint2 &v)
{
  return make_float4(static_cast<float>(v.x), static_cast<float>(v.y), 0.0f, 0.0f);
}
RTLIB_INLINE RTLIB_HOST_DEVICE float4 make_float4(const uint3 &v)
{
  return make_float4(static_cast<float>(v.x), static_cast<float>(v.y), static_cast<float>(v.z), 0.0f);
}
RTLIB_INLINE RTLIB_HOST_DEVICE float4 make_float4(const uint4 &v)
{
  return make_float4(static_cast<float>(v.x), static_cast<float>(v.y), static_cast<float>(v.z), static_cast<float>(v.w));
}
RTLIB_INLINE RTLIB_HOST_DEVICE float2 make_float2(const unsigned long v)
{
  return make_float2(static_cast<float>(v), static_cast<float>(v));
}
RTLIB_INLINE RTLIB_HOST_DEVICE float2 make_float2(const ulong2 &v)
{
  return make_float2(static_cast<float>(v.x), static_cast<float>(v.y));
}
RTLIB_INLINE RTLIB_HOST_DEVICE float3 make_float3(const unsigned long v)
{
  return make_float3(static_cast<float>(v), static_cast<float>(v), static_cast<float>(v));
}
RTLIB_INLINE RTLIB_HOST_DEVICE float3 make_float3(const ulong2 &v)
{
  return make_float3(static_cast<float>(v.x), static_cast<float>(v.y), 0.0f);
}
RTLIB_INLINE RTLIB_HOST_DEVICE float3 make_float3(const ulong3 &v)
{
  return make_float3(static_cast<float>(v.x), static_cast<float>(v.y), static_cast<float>(v.z));
}
RTLIB_INLINE RTLIB_HOST_DEVICE float4 make_float4(const unsigned long v)
{
  return make_float4(static_cast<float>(v), static_cast<float>(v), static_cast<float>(v), static_cast<float>(v));
}
RTLIB_INLINE RTLIB_HOST_DEVICE float4 make_float4(const ulong2 &v)
{
  return make_float4(static_cast<float>(v.x), static_cast<float>(v.y), 0.0f, 0.0f);
}
RTLIB_INLINE RTLIB_HOST_DEVICE float4 make_float4(const ulong3 &v)
{
  return make_float4(static_cast<float>(v.x), static_cast<float>(v.y), static_cast<float>(v.z), 0.0f);
}
RTLIB_INLINE RTLIB_HOST_DEVICE float4 make_float4(const ulong4 &v)
{
  return make_float4(static_cast<float>(v.x), static_cast<float>(v.y), static_cast<float>(v.z), static_cast<float>(v.w));
}
RTLIB_INLINE RTLIB_HOST_DEVICE float2 make_float2(const unsigned long long v)
{
  return make_float2(static_cast<float>(v), static_cast<float>(v));
}
RTLIB_INLINE RTLIB_HOST_DEVICE float2 make_float2(const ulonglong2 &v)
{
  return make_float2(static_cast<float>(v.x), static_cast<float>(v.y));
}
RTLIB_INLINE RTLIB_HOST_DEVICE float3 make_float3(const unsigned long long v)
{
  return make_float3(static_cast<float>(v), static_cast<float>(v), static_cast<float>(v));
}
RTLIB_INLINE RTLIB_HOST_DEVICE float3 make_float3(const ulonglong2 &v)
{
  return make_float3(static_cast<float>(v.x), static_cast<float>(v.y), 0.0f);
}
RTLIB_INLINE RTLIB_HOST_DEVICE float3 make_float3(const ulonglong3 &v)
{
  return make_float3(static_cast<float>(v.x), static_cast<float>(v.y), static_cast<float>(v.z));
}
RTLIB_INLINE RTLIB_HOST_DEVICE float4 make_float4(const unsigned long long v)
{
  return make_float4(static_cast<float>(v), static_cast<float>(v), static_cast<float>(v), static_cast<float>(v));
}
RTLIB_INLINE RTLIB_HOST_DEVICE float4 make_float4(const ulonglong2 &v)
{
  return make_float4(static_cast<float>(v.x), static_cast<float>(v.y), 0.0f, 0.0f);
}
RTLIB_INLINE RTLIB_HOST_DEVICE float4 make_float4(const ulonglong3 &v)
{
  return make_float4(static_cast<float>(v.x), static_cast<float>(v.y), static_cast<float>(v.z), 0.0f);
}
RTLIB_INLINE RTLIB_HOST_DEVICE float4 make_float4(const ulonglong4 &v)
{
  return make_float4(static_cast<float>(v.x), static_cast<float>(v.y), static_cast<float>(v.z), static_cast<float>(v.w));
}
#endif