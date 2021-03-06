#ifndef RTLIB_EXT_CUDA_MATH_HASH_H
#define RTLIB_EXT_CUDA_MATH_HASH_H
#include <RTLib/Ext/CUDA/math/VectorFunction.h>
#include <RTLib/Ext/CUDA/math/Math.h>
namespace RTLib
{
    namespace Ext
    {
        namespace CUDA
        {
            namespace Math
            {
                RTLIB_INLINE RTLIB_HOST_DEVICE unsigned int pcg1d(unsigned int  input)
                {
                    unsigned int state = input * 747796405u + 2891336453u;
                    unsigned int word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
                    return (word >> 22u) ^ word;
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE uint3 pcg3d(uint3 v)
                {
                    v = v * 1664525u + make_uint3(1013904223u, 1013904223u, 1013904223u);
                    v.x += v.y * v.z;
                    v.y += v.z * v.x;
                    v.z += v.x * v.y;
                    v.x = v.x ^ (v.x >> 16u);
                    v.y = v.y ^ (v.y >> 16u);
                    v.z = v.z ^ (v.z >> 16u);
                    v.x += v.y * v.z;
                    v.y += v.z * v.x;
                    v.z += v.x * v.y;
                    return v;
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE uint4 pcg4d(uint4 v)
                {
                    v = v * 1664525u + make_uint4(1013904223u, 1013904223u, 1013904223u, 1013904223u);
                    v.x += v.y * v.w;
                    v.y += v.z * v.x;
                    v.z += v.x * v.y;
                    v.w += v.y * v.z;
                    v.x = v.x ^ (v.x >> 16u);
                    v.y = v.y ^ (v.y >> 16u);
                    v.z = v.z ^ (v.z >> 16u);
                    v.w = v.w ^ (v.w >> 16u);
                    v.x += v.y * v.w;
                    v.y += v.z * v.x;
                    v.z += v.x * v.y;
                    v.w += v.y * v.z;
                    return v;
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE unsigned int xxhash(unsigned int data, unsigned int seed = 12345U)
                {
                    const unsigned int PRIME32_2 = 2246822519U;
                    const unsigned int PRIME32_3 = 3266489917U;
                    const unsigned int PRIME32_4 = 668265263U;
                    const unsigned int PRIME32_5 = 374761393U;
                    unsigned int h32 = (unsigned int)data * PRIME32_3;
                    h32 += seed + PRIME32_5 + 4U;
                    h32 = (h32 << 17) | (h32 >> 15);
                    h32 *= PRIME32_4;
                    h32 ^= h32 >> 15;
                    h32 *= PRIME32_2;
                    h32 ^= h32 >> 13;
                    h32 *= PRIME32_3;
                    h32 ^= h32 >> 16;
                    return (int)h32;
                }
                RTLIB_INLINE RTLIB_HOST_DEVICE unsigned int hash6432shift(unsigned long long key)
                {
                    key = (~key) + (key << 18);
                    key ^= key >> 31;
                    key *= 21;
                    key ^= key >> 11;
                    key += key << 6;
                    key ^= key >> 22;
                    return 0xFFFFFFFF & key;
                }

            }
        }
    }
}
#endif
