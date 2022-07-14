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
                struct HashGridFindInfo
                {
                    unsigned int    checkSum;
                    unsigned int   cellIndex;
                    unsigned char blockIndex;
                    bool           isFounded;
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
                    RTLIB_INLINE RTLIB_HOST_DEVICE void Find(const float3 p, HashGridFindInfo& info)const noexcept
                    {
                        auto fLen = (p - aabbOffset) / aabbSize;
                        auto iLen = RTLib::Ext::CUDA::Math::clamp(make_uint3(
                            static_cast<unsigned int>(bounds.x * fLen.x),
                            static_cast<unsigned int>(bounds.y * fLen.y),
                            static_cast<unsigned int>(bounds.z * fLen.z)
                        ), make_uint3(0), bounds - make_uint3(1));
                        return GetCellInfo(iLen, info);
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
                    RTLIB_INLINE RTLIB_HOST_DEVICE void GetCellInfo(const uint3 idx, HashGridFindInfo& info)const noexcept
                    {
                        info.cellIndex = RTLib::Ext::CUDA::Math::pcg1d(idx.z + RTLib::Ext::CUDA::Math::pcg1d(idx.y + RTLib::Ext::CUDA::Math::pcg1d(idx.x))) % (size / kBlockSize);
                        info.checkSum  = RTLib::Ext::CUDA::Math::max(RTLib::Ext::CUDA::Math::xxhash(idx.z + RTLib::Ext::CUDA::Math::xxhash(idx.y + RTLib::Ext::CUDA::Math::xxhash(idx.x))) % (size / kBlockSize), (unsigned int)1);
                        info.isFounded = true;
                        int i = 0;
                        for (i = 0; i < kBlockSize; ++i) {
                            auto hashIndex = i + info.cellIndex * kBlockSize;
#ifdef __CUDACC__
                            auto prvCheckSum = atomicCAS(&checkSums[kBlockSize * info.cellIndex + i], 0, info.checkSum);
#else
                            auto prvCheckSum = checkSums[kBlockSize * info.cellIndex + i];
                            if (prvCheckSum == 0)
                            {
                                checkSums[kBlockSize * info.cellIndex + i] = info.checkSum;
                            }
#endif
                            if ((prvCheckSum == 0) || (prvCheckSum == info.checkSum)) {
                                break;
                            }
                        }
                        if (i == kBlockSize) {
                            info.isFounded  = false;
                            info.blockIndex = i;
                        }
                        info.blockIndex = i;
                    }
                };
                
                struct DoubleBufferedHashGrid3
                {
                    static inline constexpr unsigned int kBlockSize = 32;
                    float3        aabbOffset;
                    float3        aabbSize;
                    uint3         bounds;
                    unsigned int  size;
                    unsigned int* prvCheckSums;
                    unsigned int* curCheckSums;

                    RTLIB_INLINE RTLIB_HOST_DEVICE auto FindFromCur(const float3 p)const noexcept -> unsigned int
                    {
                        auto fLen = (p - aabbOffset) / aabbSize;
                        auto iLen = RTLib::Ext::CUDA::Math::clamp(make_uint3(
                            static_cast<unsigned int>(bounds.x * fLen.x),
                            static_cast<unsigned int>(bounds.y * fLen.y),
                            static_cast<unsigned int>(bounds.z * fLen.z)
                        ), make_uint3(0), bounds - make_uint3(1));
                        return GetCurCellIndex(iLen);

                    }
                    RTLIB_INLINE RTLIB_HOST_DEVICE void FindFromCur(const float3 p, HashGridFindInfo& info)const noexcept
                    {
                        auto fLen = (p - aabbOffset) / aabbSize;
                        auto iLen = RTLib::Ext::CUDA::Math::clamp(make_uint3(
                            static_cast<unsigned int>(bounds.x * fLen.x),
                            static_cast<unsigned int>(bounds.y * fLen.y),
                            static_cast<unsigned int>(bounds.z * fLen.z)
                        ), make_uint3(0), bounds - make_uint3(1));
                        return GetCurCellInfo(iLen, info);
                    }
                    RTLIB_INLINE RTLIB_HOST_DEVICE auto FindFromCur(const HashGridFindInfo& info)const noexcept -> unsigned int
                    {
                        if (info.isFounded) {
                            return info.blockIndex * kBlockSize + info.cellIndex;
                        }
                        unsigned int checkSum  = info.checkSum;
                        unsigned int cellIndex = info.cellIndex;
                        int i = 0;
                        for (i = 0; i < kBlockSize; ++i) {
                            auto hashIndex = i + cellIndex * kBlockSize;
#ifdef __CUDACC__
                            auto prvCheckSum = atomicCAS(&curCheckSums[kBlockSize * cellIndex + i], 0, checkSum);
#else
                            auto prvCheckSum = curCheckSums[kBlockSize * cellIndex + i];
                            if (prvCheckSum == 0)
                            {
                                curCheckSums[kBlockSize * cellIndex + i] = checkSum;
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
                    RTLIB_INLINE RTLIB_HOST_DEVICE auto FindFromPrv(const float3 p)const noexcept -> unsigned int
                    {
                        auto fLen = (p - aabbOffset) / aabbSize;
                        auto iLen = RTLib::Ext::CUDA::Math::clamp(make_uint3(
                            static_cast<unsigned int>(bounds.x * fLen.x),
                            static_cast<unsigned int>(bounds.y * fLen.y),
                            static_cast<unsigned int>(bounds.z * fLen.z)
                        ), make_uint3(0), bounds - make_uint3(1));
                        return GetPrvCellIndex(iLen);

                    }
                    RTLIB_INLINE RTLIB_HOST_DEVICE void FindFromPrv(const float3 p, HashGridFindInfo& info)const noexcept
                    {
                        auto fLen = (p - aabbOffset) / aabbSize;
                        auto iLen = RTLib::Ext::CUDA::Math::clamp(make_uint3(
                            static_cast<unsigned int>(bounds.x * fLen.x),
                            static_cast<unsigned int>(bounds.y * fLen.y),
                            static_cast<unsigned int>(bounds.z * fLen.z)
                        ), make_uint3(0), bounds - make_uint3(1));
                        return GetPrvCellInfo(iLen, info);
                    }
                    RTLIB_INLINE RTLIB_HOST_DEVICE auto FindFromPrv(const HashGridFindInfo& info)const noexcept -> unsigned int
                    {
                        if (info.isFounded) {
                            return info.blockIndex + info.cellIndex * kBlockSize;
                        }
                        return UINT32_MAX;
                    }

                private:
                    RTLIB_INLINE RTLIB_HOST_DEVICE auto GetCurCellIndex(const uint3 idx)const noexcept -> unsigned int
                    {
                        //unsigned long long baseIndex = bounds.x * bounds.y * iLen.z + bounds.x * iLen.y + iLen.x;
                        //return RTLib::Ext::CUDA::Math::hash6432shift(baseIndex) % size;
                        auto cellIndex = RTLib::Ext::CUDA::Math::pcg1d(idx.z + RTLib::Ext::CUDA::Math::pcg1d(idx.y + RTLib::Ext::CUDA::Math::pcg1d(idx.x))) % (size / kBlockSize);
                        //auto checkSum  = RTLib::Ext::CUDA::Math::max(RTLib::Ext::CUDA::Math::xxhash(idx.z + RTLib::Ext::CUDA::Math::xxhash(idx.y + RTLib::Ext::CUDA::Math::xxhash(idx.x))) % (size / kBlockSize), (unsigned int)1);
                        auto checkSum = 1 + bounds.x * bounds.y * idx.z + bounds.x * idx.y + idx.x;
                        int i = 0;
                        for (i = 0; i < kBlockSize; ++i) {
                            auto hashIndex = i + cellIndex * kBlockSize;
#ifdef __CUDACC__
                            auto prvCheckSum = atomicCAS(&curCheckSums[kBlockSize * cellIndex + i], 0, checkSum);
#else
                            auto prvCheckSum = curCheckSums[kBlockSize * cellIndex + i];
                            if (prvCheckSum == 0)
                            {
                                curCheckSums[kBlockSize * cellIndex + i] = checkSum;
                            }
#endif
                            if ((prvCheckSum == 0) || (prvCheckSum == checkSum)) {
                                break;
                            }
                        }
                        if (i == kBlockSize) {
                            printf("BUG IN GRID\n");
                            return UINT32_MAX;
                        }
                        return i + cellIndex * kBlockSize;
                    }
                    RTLIB_INLINE RTLIB_HOST_DEVICE auto GetPrvCellIndex(const uint3 idx)const noexcept -> unsigned int
                    {
                        //unsigned long long baseIndex = bounds.x * bounds.y * iLen.z + bounds.x * iLen.y + iLen.x;
                        //return RTLib::Ext::CUDA::Math::hash6432shift(baseIndex) % size;
                        auto cellIndex = RTLib::Ext::CUDA::Math::pcg1d(idx.z + RTLib::Ext::CUDA::Math::pcg1d(idx.y + RTLib::Ext::CUDA::Math::pcg1d(idx.x))) % (size / kBlockSize);
                        //auto checkSum = RTLib::Ext::CUDA::Math::max(RTLib::Ext::CUDA::Math::xxhash(idx.z + RTLib::Ext::CUDA::Math::xxhash(idx.y + RTLib::Ext::CUDA::Math::xxhash(idx.x))) % (size / kBlockSize), (unsigned int)1);
                        auto checkSum = 1 + bounds.x * bounds.y * idx.z + bounds.x * idx.y + idx.x;
                        int i = 0;
                        for (i = 0; i < kBlockSize; ++i) {
                            auto hashIndex = i + cellIndex * kBlockSize;
                            auto prvCheckSum = prvCheckSums[kBlockSize * cellIndex + i];
                            if ( prvCheckSum == 0)
                            {
                                return UINT32_MAX;
                            }
                            if (prvCheckSum == checkSum) {
                                return i + cellIndex * kBlockSize;
                            }
                        }
                        return UINT32_MAX;
                    }
                    RTLIB_INLINE RTLIB_HOST_DEVICE void GetCurCellInfo(const uint3 idx, HashGridFindInfo& info)const noexcept
                    {
                        info.cellIndex = RTLib::Ext::CUDA::Math::pcg1d(idx.z + RTLib::Ext::CUDA::Math::pcg1d(idx.y + RTLib::Ext::CUDA::Math::pcg1d(idx.x))) % (size / kBlockSize);
                        //info.checkSum  = RTLib::Ext::CUDA::Math::max(RTLib::Ext::CUDA::Math::xxhash(idx.z + RTLib::Ext::CUDA::Math::xxhash(idx.y + RTLib::Ext::CUDA::Math::xxhash(idx.x))) % (size / kBlockSize), (unsigned int)1);
                        info.checkSum = 1 + bounds.x * bounds.y * idx.z + bounds.x * idx.y + idx.x;
                        info.isFounded = true;
                        int i = 0;
                        for (i = 0; i < kBlockSize; ++i) {
                            auto hashIndex = i + info.cellIndex * kBlockSize;
#ifdef __CUDACC__
                            auto prvCheckSum = atomicCAS(&curCheckSums[kBlockSize * info.cellIndex + i], 0, info.checkSum);
#else
                            auto prvCheckSum = curCheckSums[kBlockSize * info.cellIndex + i];
                            if (prvCheckSum == 0)
                            {
                                curCheckSums[kBlockSize * info.cellIndex + i] = info.checkSum;
                            }
#endif
                            if ((prvCheckSum == 0) || (prvCheckSum == info.checkSum)) {
                                break;
                            }
                        }
                        if (i == kBlockSize) {
                            printf("BUG IN GRID\n");
                            info.isFounded = false;
                        }
                        info.blockIndex = i;
                    }
                    RTLIB_INLINE RTLIB_HOST_DEVICE void GetPrvCellInfo(const uint3 idx, HashGridFindInfo& info)const noexcept
                    {
                        info.cellIndex = RTLib::Ext::CUDA::Math::pcg1d(idx.z + RTLib::Ext::CUDA::Math::pcg1d(idx.y + RTLib::Ext::CUDA::Math::pcg1d(idx.x))) % (size / kBlockSize);
                        //info.checkSum  = RTLib::Ext::CUDA::Math::max(RTLib::Ext::CUDA::Math::xxhash(idx.z + RTLib::Ext::CUDA::Math::xxhash(idx.y + RTLib::Ext::CUDA::Math::xxhash(idx.x))) % (size / kBlockSize), (unsigned int)1);
                        info.checkSum = 1 + bounds.x * bounds.y * idx.z + bounds.x * idx.y + idx.x;
                        info.isFounded = true;
                        int i = 0;
                        for (i = 0; i < kBlockSize; ++i) {
                            auto prvCheckSum = prvCheckSums[kBlockSize * info.cellIndex + i];
                            if (prvCheckSum == 0)
                            {
                                info.blockIndex = 0;
                                info.isFounded = false;
                                return;
                            }
                            if (prvCheckSum == info.checkSum) {
                                info.blockIndex = i;
                                return;
                            }
                        }
                        info.blockIndex = 0;
                        info.isFounded = false;

                    }

                };
            }
        }
    }
}
#endif
