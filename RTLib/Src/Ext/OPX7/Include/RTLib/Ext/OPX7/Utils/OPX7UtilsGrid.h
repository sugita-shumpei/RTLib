#ifndef RTLIB_EXT_OPX7_UTILS_OPX7_UTILS_GRID_H
#define RTLIB_EXT_OPX7_UTILS_OPX7_UTILS_GRID_H
#include <RTLib/Ext/CUDA/Math/Hash.h>
#include <RTLib/Ext/CUDA/Math/Math.h>
#include <RTLib/Ext/CUDA/Math/Random.h>
#include <RTLib/Ext/CUDA/Math/VectorFunction.h>
#ifndef __CUDACC__
#include <RTLib/Ext/CUDA/CUDABuffer.h>
#include <fstream>
#include <stack>
#include <vector>
#include <iterator>
#include <fstream>
#include <stack>
#include <algorithm>
#include <functional>
#include <vector>
#include <iterator>
#include <fstream>
#include <stack>
#include <algorithm>
#include <functional>
#endif
namespace RTLib
{
    namespace Ext
    {
        namespace OPX7
        {
            namespace Utils
            {
                struct RegularGrid2
                {
                    float2 aabbOffset;
                    float2 aabbSize;
                    uint2  bounds;
                    RTLIB_INLINE RTLIB_HOST_DEVICE auto Find(const float2 p)const noexcept -> unsigned int
                    {
                        auto fLen = (p - aabbOffset) / aabbSize;
                        auto  iLen = RTLib::Ext::CUDA::Math::clamp(make_uint2(
                            static_cast<unsigned int>(fLen.x),
                            static_cast<unsigned int>(fLen.y)
                        ), make_uint2(0), bounds - make_uint2(1));
                        return bounds.x * iLen.y + iLen.x;
                    }
                };
                struct RegularGrid3
                {
                    float3 aabbOffset;
                    float3 aabbSize;
                    uint3  bounds;
                    RTLIB_INLINE RTLIB_HOST_DEVICE auto Find(const float3 p)const noexcept -> unsigned int
                    {
                        auto fLen = (p - aabbOffset) / aabbSize;
                        auto  iLen = RTLib::Ext::CUDA::Math::clamp(make_uint3(
                            static_cast<unsigned int>(fLen.x),
                            static_cast<unsigned int>(fLen.y),
                            static_cast<unsigned int>(fLen.z)
                        ), make_uint3(0), bounds - make_uint3(1));
                        return bounds.x * bounds.y * iLen.z + bounds.x * iLen.y + iLen.x;
                    }
                };
                struct HashGrid3
                {
                    static inline constexpr unsigned int kBlockSize = 32;
                    float3        aabbOffset;
                    float3        aabbSize;
                    uint3         bounds;
                    unsigned int  size;
                    unsigned int* checkSums;
                    RTLIB_INLINE RTLIB_HOST_DEVICE auto Find(const float3 p)const noexcept -> unsigned int
                    {
                        auto fLen = (p - aabbOffset) / aabbSize;
                        auto iLen = RTLib::Ext::CUDA::Math::clamp(make_uint3(
                            static_cast<unsigned int>(bounds.x * fLen.x),
                            static_cast<unsigned int>(bounds.y * fLen.y),
                            static_cast<unsigned int>(bounds.z * fLen.z)
                        ), make_uint3(0), bounds - make_uint3(1));
                        return GetCellIndex(iLen);

                    }
                private:
                    RTLIB_INLINE RTLIB_HOST_DEVICE auto GetCellIndex(const uint3 idx)const noexcept -> unsigned int
                    {
                        //unsigned long long baseIndex = bounds.x * bounds.y * iLen.z + bounds.x * iLen.y + iLen.x;
                        //return RTLib::Ext::CUDA::Math::hash6432shift(baseIndex) % size;
                        auto cellIndex = RTLib::Ext::CUDA::Math::pcg1d(idx.z + RTLib::Ext::CUDA::Math::pcg1d(idx.y + RTLib::Ext::CUDA::Math::pcg1d(idx.x))) % (size / kBlockSize);
                        auto checkSum = RTLib::Ext::CUDA::Math::max(RTLib::Ext::CUDA::Math::xxhash(idx.z + RTLib::Ext::CUDA::Math::xxhash(idx.y + RTLib::Ext::CUDA::Math::xxhash(idx.x))) % (size / kBlockSize), (unsigned int)1);
                        int i = 0;
                        for (i = 0; i < kBlockSize; ++i) {
                            auto hashIndex = i + cellIndex * kBlockSize;
#ifdef __CUDACC__
                            auto prvCheckSum = atomicCAS(&checkSums[kBlockSize * cellIndex + i], 0, checkSum);
#else
                            auto prvCheckSum = checkSums[kBlockSize * cellIndex + i];
                            if (prvCheckSum == 0)
                            {
                                checkSums[kBlockSize * cellIndex + i] = checkSum;
                            }
#endif
                            if ((prvCheckSum == 0) || (prvCheckSum == checkSum)) {
                                break;
                            }
                        }
                        if (i == kBlockSize) {
                            return UINT32_MAX;
                        }
                        return i + cellIndex * kBlockSize;
                    }
                };

            }
        }
    }
}
#endif
