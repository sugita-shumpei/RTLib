#ifndef RTLIB_EXT_OPX7_UTILS_OPX7_UTILS_MORTON_H
#define RTLIB_EXT_OPX7_UTILS_OPX7_UTILS_MORTON_H
#include <RTLib/Ext/CUDA/Math/Math.h>
#include <RTLib/Ext/CUDA/Math/Random.h>
#include <RTLib/Ext/CUDA/Math/VectorFunction.h>
#ifndef __CUDACC__
#include <RTLib/Ext/CUDA/CUDAContext.h>
#include <RTLib/Ext/CUDA/CUDABuffer.h>
#include <RTLib/Ext/CUDA/CUDAModule.h>
#include <RTLib/Ext/CUDA/CUDAFunction.h>
#include <RTLib/Ext/CUDA/CUDANatives.h>
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
#ifndef RTLIB_EXT_OPX7_UTILS_OPX7_UTILS_MORTON_BUILD_BY_CUDA_KERNEL
#define RTLIB_EXT_OPX7_UTILS_OPX7_UTILS_MORTON_BUILD_BY_CUDA_KERNEL 1
#endif
namespace RTLib
{
    namespace Ext
    {
        namespace OPX7
        {
            namespace Utils
            {
                
                struct MortonUtils
                {
                    RTLIB_INLINE RTLIB_DEVICE static uint32_t Part1By1_16(uint32_t x) {
                        x &= 0x0000FFFF;
                        // 0000 0000 0000 0000 ABCD EFGH IJKL MNOP
                        //|0000 0000 ABCD EFGH IJKL MNOP 0000 0000
                        //&0000 0000 1111 1111 0000 0000 1111 1111
                        //=0000 0000 ABCD EFGH 0000 0000 IJKL MNOP
                        x = (x ^ (x << 8)) & 0x00FF00FF;
                        // 0000 0000 ABCD EFGH 0000 0000 IJKL MNOP
                        //|0000 ABCD EFGH 0000 0000 IJKL MNOP 0000
                        //&0000 1111 0000 1111 0000 1111 0000 1111
                        //=0000 ABCD 0000 EFGH 0000 IJKL 0000 MNOP
                        x = (x ^ (x << 4)) & 0x0F0F0F0F;
                        // 0000 ABCD 0000 EFGH 0000 IJKL 0000 MNOP
                        //|00AB CD00 00EF GH00 00IJ KL00 00MN OP00
                        //&0011 0011 0011 0011 0011 0011 0011 0011
                        //=00AB 00CD 00EF 00GH 00IJ 00KL 00MN 00OP
                        x = (x ^ (x << 2)) & 0x33333333;
                        // 00AB 00CD 00EF 00GH 00IJ 00KL 00MN 00OP
                        //|0AB0 0CD0 0EF0 0GH0 0IJ0 0KL0 0MN0 0OP0
                        //&0101 0101 0101 0101 0101 0101 0101 0101
                        //=0A0B 0C0D 0E0F 0G0H 0I0J 0K0L 0M0N 0O0P
                        x = (x ^ (x << 1)) & 0x55555555;
                        return x;
                    }
                    RTLIB_INLINE RTLIB_DEVICE static uint32_t Comp1By1_16(uint32_t x) {
                        // _A_B _C_D _E_F _G_H _I_J _K_L _M_N _O_P
                        //&0101 0101 0101 0101 0101 0101 0101 0101
                        //=0A0B 0C0D 0E0F 0G0H 0I0J 0K0L 0M0N OO0P
                        //|00A0 B0C0 D0E0 F0G0 H0I0 J0K0 L0M0 N0O0
                        //&0011 0011 0011 0011 0011 0011 0011 0011
                        //=00AB 00CD 00EF 00GH 00IJ 00KL 00MN 00OP
                        x &= 0x55555555;
                        x = (x ^ (x >> 1)) & 0x33333333;
                        // 00AB 00CD 00EF 00GH 00IJ 00KL 00MN 00OP
                        //|0000 AB00 CD00 EF00 GH00 IJ00 KL00 MN00
                        //&0000 1111 0000 1111 0000 1111 0000 1111
                        //=0000 ABCD 0000 EFGH 0000 IJKL 0000 MNOP
                        x = (x ^ (x >> 2)) & 0x0F0F0F0F;
                        // 0000 ABCD 0000 EFGH 0000 IJKL 0000 MNOP
                        //|0000 0000 ABCD 0000 EFGH 0000 IJKL 0000
                        //&0000 0000 1111 1111 0000 0000 1111 1111
                        //=0000 0000 ABCD EFGH 0000 0000 IJKL MNOP
                        x = (x ^ (x >> 4)) & 0x00FF00FF;
                        //&0000 0000 ABCD EFGH 0000 0000 IJKL MNOP
                        //|0000 0000 0000 0000 ABCD EFGH 0000 0000
                        //&0000 0000 0000 0000 1111 1111 1111 1111
                        x = (x ^ (x >> 8)) & 0x0000FFFF;
                        return x;
                    }

                    RTLIB_INLINE RTLIB_DEVICE static uint16_t Part1By1_8(uint16_t x) {
                        x &= 0x00FF;
                        // 0000 0000 ABCD EFGH
                        // 0000 ABCD EFGH 0000
                        x = (x ^ (x << 4)) & 0x0F0F;
                        // 0000 1111 0000 1111
                        // 0000 ABCD 0000 EFGH
                        // 00AB CD00 EF00 GH00
                        x = (x ^ (x << 2)) & 0x3333;
                        // 0011 0011 0011 0011
                        // 00AB 00CD 00EF 00GH
                        // 0AB0 0CD0 0EF0 0GH0
                        x = (x ^ (x << 1)) & 0x5555;
                        // 0101 0101 0101 0101
                        // 0A0B 0C0D 0E0F 0G0H
                        return x;
                    }
                    RTLIB_INLINE RTLIB_DEVICE static uint16_t Comp1By1_8(uint16_t x) {
                        x &= 0x5555;
                        // 0A0B 0C0D 0E0F 0G0H
                        // 00A0 B0C0 D0E0 F0G0
                        // 0011 0011 0011 0011
                        x = (x ^ (x >> 1)) & 0x3333;
                        // 00AB 00CD 00EF 00GH
                        // 0000 AB00 CD00 EF00
                        // 0000 1111 0000 1111
                        x = (x ^ (x >> 2)) & 0x0F0F;
                        // 0000 ABCD 0000 EFGH
                        x = (x ^ (x >> 4)) & 0x00FF;
                        return x;
                    }

                    RTLIB_INLINE RTLIB_DEVICE static uint8_t  Part1By1_4(uint8_t  x) {
                        x &= 0x0F;
                        x = (x ^ (x << 2)) & 0x33;
                        x = (x ^ (x << 1)) & 0x55;
                        return x;
                    }
                    RTLIB_INLINE RTLIB_DEVICE static uint8_t  Comp1By1_4(uint8_t  x) {
                        x &= 0x55;
                        // 0A0B 0C0D
                        // 00A0 B0C0
                        // 0011 0011
                        // 00AB 00CD
                        // 0000 AB00
                        x = (x ^ (x >> 1)) & 0x33;
                        // 00AB 00CD 00EF 00GH
                        // 0000 AB00 CD00 EF00
                        // 0000 1111 0000 1111
                        x = (x ^ (x >> 2)) & 0x0F;
                        return x;
                    }

                    RTLIB_INLINE RTLIB_DEVICE static uint8_t  Part1By1_2(uint8_t  x) {
                        x &= 0x3;
                        x = (x ^ (x << 1)) & 0x5;
                        return x;
                    }
                    RTLIB_INLINE RTLIB_DEVICE static uint8_t  Comp1By1_2(uint8_t  x) {
                        x &= 0x5;
                        // 00AB
                        // 0AB0
                        // 0A0B
                        // 00A0
                        x = (x ^ (x >> 1)) & 0x3;
                        return x;
                    }
                };

                template<unsigned int level>
                struct Morton2Utils;
                template<>
                struct Morton2Utils<4>
                {
                    RTLIB_INLINE RTLIB_DEVICE static auto GetCodeFromPosIdx(const unsigned short xIdx, const unsigned short yIdx) noexcept -> unsigned int {
                        return (MortonUtils::Part1By1_16(yIdx) << 1) + MortonUtils::Part1By1_16(xIdx);
                    }
                    RTLIB_INLINE RTLIB_DEVICE static void GetPosIdxFromCode(unsigned short& offX, unsigned short& offY, unsigned int code) noexcept {
                        offX = MortonUtils::Comp1By1_16(code);
                        offY = MortonUtils::Comp1By1_16(code >> 1);
                    }
                    RTLIB_INLINE RTLIB_DEVICE static auto GetPosIdxFromCode(unsigned int code)noexcept ->ushort2{
                        unsigned short offX, offY;
                        GetPosIdxFromCode(offX, offY, code);
                        return { offX,offY };
                    }
                };
                template<>
                struct Morton2Utils<3>
                {
                    RTLIB_INLINE RTLIB_DEVICE  static auto GetCodeFromPosIdx(const unsigned char xIdx, const unsigned char yIdx) noexcept -> unsigned short {
                        return (MortonUtils::Part1By1_8(yIdx) << 1) + MortonUtils::Part1By1_8(xIdx);
                    }
                    RTLIB_INLINE RTLIB_DEVICE static auto GetPosIdxFromCode(unsigned char& offX, unsigned char& offY, unsigned short code) noexcept {
                        offX = MortonUtils::Comp1By1_8(code);
                        offY = MortonUtils::Comp1By1_8(code >> 1);
                    }
                    RTLIB_INLINE RTLIB_DEVICE static auto GetPosIdxFromCode(unsigned short code)noexcept ->uchar2{
                        unsigned char offX, offY;
                        GetPosIdxFromCode(offX, offY, code);
                        return { offX,offY };
                    }
                };
                template<>
                struct Morton2Utils<2>
                {
                    RTLIB_INLINE RTLIB_DEVICE static auto GetCodeFromPosIdx(const unsigned char xIdx, const unsigned char yIdx) noexcept -> unsigned char {
                        return (MortonUtils::Part1By1_4(yIdx) << 1) + MortonUtils::Part1By1_4(xIdx);
                    }
                    RTLIB_INLINE RTLIB_DEVICE static auto GetPosIdxFromCode(unsigned char& offX, unsigned char& offY, unsigned char code) noexcept {
                        offX = MortonUtils::Comp1By1_4(code);
                        offY = MortonUtils::Comp1By1_4(code >> 1);
                    }
                    RTLIB_INLINE RTLIB_DEVICE static auto GetPosIdxFromCode(unsigned char code)noexcept ->uchar2{
                        unsigned char offX, offY;
                        GetPosIdxFromCode(offX, offY, code);
                        return { offX,offY };
                    }
                };
                template<>
                struct Morton2Utils<1>
                {
                    RTLIB_INLINE RTLIB_DEVICE static auto GetCodeFromPosIdx(const unsigned char xIdx, const unsigned char yIdx) noexcept -> unsigned char {
                        return (MortonUtils::Part1By1_2(yIdx) << 1) + MortonUtils::Part1By1_2(xIdx);
                    }
                    RTLIB_INLINE RTLIB_DEVICE static auto GetPosIdxFromCode(unsigned char& offX, unsigned char& offY, unsigned char code) noexcept {
                        offX = MortonUtils::Comp1By1_2(code);
                        offY = MortonUtils::Comp1By1_2(code >> 1);
                    }
                    RTLIB_INLINE RTLIB_DEVICE static auto GetPosIdxFromCode(unsigned char code)noexcept ->uchar2{
                        unsigned char offX, offY;
                        GetPosIdxFromCode(offX, offY, code);
                        return { offX,offY };
                    }
                };

                struct MortonQuadTreeUtils {
                    RTLIB_INLINE RTLIB_HOST_DEVICE static auto GetLevelFromArrayIndex(unsigned int arrayIndex) noexcept -> unsigned int
                    {
                        return log2((arrayIndex + 1) * 3) / 2;
                    }
                    RTLIB_INLINE RTLIB_HOST_DEVICE static auto GetMortonCodeGlobalFromArrayIndex(unsigned int arrayIndex) noexcept -> unsigned int
                    {
                        return arrayIndex - (pow(4, GetLevelFromArrayIndex(arrayIndex)) - 1) / 3;
                    }
                    RTLIB_INLINE RTLIB_HOST_DEVICE static auto GetPosIdxFromArrayIndex(unsigned int arrayIndex, unsigned short& x, unsigned short& y) noexcept {
                        Morton2Utils<4>::GetPosIdxFromCode(x, y, GetMortonCodeGlobalFromArrayIndex(arrayIndex));
                    }
                    RTLIB_INLINE RTLIB_HOST_DEVICE static auto GetArrayIndexFromMortonCodeLocal(unsigned int mortonCode, unsigned int level) noexcept -> unsigned int {
                        //1//4//16
                        //   4//16//64
                        mortonCode &= static_cast<unsigned int>(::exp2f(level) - 1);
                        return (::exp2f(2*level) - 1) / 3 + mortonCode;
                    }
                    RTLIB_INLINE RTLIB_HOST_DEVICE static auto GetArrayIndexFromMortonCodeGlobal(unsigned int mortonCode) noexcept -> unsigned int {
                        //1//4//16
                        //   4//16//64
                        unsigned int level = ::log2(mortonCode + 1) / 2;
                        return (::exp2f(2*level) - 1) / 3 + mortonCode;
                    }
                };

                template<unsigned int MaxLevel>
                struct MortonQuadTreeT {
                    //best grid idx
                    RTLIB_INLINE RTLIB_DEVICE MortonQuadTreeT(unsigned int level_, float* weights_)noexcept {
                        level   = RTLib::Ext::CUDA::Math::min(level_, MaxLevel);
                        weights = weights_;
                    }
                    RTLIB_INLINE RTLIB_DEVICE ~MortonQuadTreeT()noexcept {}

                    RTLIB_INLINE RTLIB_DEVICE void Record(const float2& w_in, float value)noexcept
                    {
                        //if (isinf(value)||isnan(value)) {
                        //    return;
                        //}
                        const float numNodes = ::exp2f(level);
                        unsigned int posX = w_in.x * numNodes;
                        unsigned int posY = w_in.y * numNodes;
                        unsigned int code = Morton2Utils<BestLevel()>::GetCodeFromPosIdx(posX, posY);
#if !RTLIB_EXT_OPX7_UTILS_OPX7_UTILS_MORTON_BUILD_BY_CUDA_KERNEL
                        for (unsigned int i = 1; i <= level; ++i)
                        {
                            //std::cout << "code: " << std::bitset<MaxLevel * 2>(code) << " offset: " << (::powf(4, level + 1 - i) - 1) / 3 << std::endl;
                            unsigned int arrIndex = (::powf(4, level + 1 - i) - 1) / 3 + code;
                            AtomicAdd(weights[arrIndex], value);
                            code >>= 2;
                        }
                        
                        AtomicAdd(weights[0], value);
#else
                        unsigned int arrIndex = (numNodes * numNodes - 1) / 3 + code;
                        AtomicAdd(weights[arrIndex], value);
#endif
                    }
                    template<typename RNG>
                    RTLIB_INLINE RTLIB_DEVICE auto SampleAndPdf(float& pdf, RNG& rng)const noexcept->float2
                    {
                        unsigned int size = 4;
                        unsigned int offset = 1;
                        unsigned int code = 0;

                        float total = weights[0];
                        if (total <= 0.0f)
                        {

                            float posX = CUDA::Math::random_float1(rng);
                            float posY = CUDA::Math::random_float1(rng);
                            pdf = 1.0f;
                            return  {posX, posY};
                        }
                        for (unsigned int i = 0; i < level; ++i)
                        {
                            float weiVals[4] = {
                                weights[offset + (code << 2) + 0b00],
                                weights[offset + (code << 2) + 0b01],
                                weights[offset + (code << 2) + 0b10],
                                weights[offset + (code << 2) + 0b11]
                            };
                            unsigned short codeLocal = 0;
                            //total > 0
                            float partial = weiVals[0b00] + weiVals[0b10];
                            float boundary = partial / total;
                            float sample = CUDA::Math::random_float1(rng);
                            if (sample < boundary)
                            {
                                sample /= boundary;
                                boundary = weiVals[0b00] / partial;
                            }
                            else {
                                partial = weiVals[0b01] + weiVals[0b11];
                                sample = (sample - boundary) / (1.0f - boundary);
                                boundary = weiVals[0b01] / partial;
                                codeLocal |= (1 << 0);
                            }
                            //boundary  = 0 -> 1
                            if (sample >= boundary) {
                                codeLocal |= (1 << 1);
                            }

                            total = weiVals[codeLocal];
                            code = (code << 2) + codeLocal;
                            offset += size;
                            size *= 4;
                        }
                        auto  indices = Morton2Utils<BestLevel()>::GetPosIdxFromCode(code);
                        auto  xPosIdx = indices.x;
                        auto  yPosIdx = indices.y;
                        float posX = (xPosIdx + CUDA::Math::random_float1(rng)) / ::exp2f(level);
                        float posY = (yPosIdx + CUDA::Math::random_float1(rng)) / ::exp2f(level);
                        pdf = total / weights[0] * ::exp2f(2.0f* level);
                        //printf("pdf=%lf\n", pdf);
                        return {posX, posY};
                    }
                    template<typename RNG>
                    RTLIB_INLINE RTLIB_DEVICE auto Sample(RNG& rng)const noexcept->float2
                    {
                        unsigned int size = 4;
                        unsigned int offset = 1;
                        unsigned int code = 0;

                        float total = weights[0];
                        if (total <= 0.0f)
                        {

                            float posX = CUDA::Math::random_float1(rng);
                            float posY = CUDA::Math::random_float1(rng);
                            return {posX, posY};
                        }
                        for (unsigned int i = 0; i < level; ++i)
                        {
                            float weiVals[4] = {
                                weights[offset + (code << 2) + 0b00],
                                weights[offset + (code << 2) + 0b01],
                                weights[offset + (code << 2) + 0b10],
                                weights[offset + (code << 2) + 0b11]
                            };
                            unsigned short codeLocal = 0;
                            float partial = weiVals[0b00] + weiVals[0b10];
                            float boundary = partial / total;
                            float sample = CUDA::Math::random_float1(0.0f, 1.0f)(rng);
                            if (sample < boundary)
                            {
                                sample /= boundary;
                                boundary = weiVals[0b00] / partial;
                            }
                            else {
                                partial = weiVals[0b01] + weiVals[0b11];
                                sample = (sample - boundary) / (1.0f - boundary);
                                boundary = weiVals[0b01] / partial;
                                codeLocal |= (1 << 0);
                            }

                            if (sample >= boundary) {
                                codeLocal |= (1 << 1);
                            }

                            total = weiVals[codeLocal];
                            code = (code << 2) + codeLocal;
                            offset += size;
                            size *= 4;
                        }
                        auto  indices = Morton2Utils<BestLevel()>::GetPosIdxFromCode(code);
                        auto  xPosIdx = indices[0];
                        auto  yPosIdx = indices[1];
                        const auto numNodes = ::exp2f(level);
                        float posX = (xPosIdx + CUDA::Math::random_float1(rng)) / numNodes;
                        float posY = (yPosIdx + CUDA::Math::random_float1(rng)) / numNodes;
                        return {posX, posY};
                    }
                    RTLIB_INLINE RTLIB_DEVICE auto Pdf(const  float2& w_in)const noexcept -> float
                    {
                        if (weights[0] <= 0.0f) {
                            return 1.0f;
                        }
                        const auto numNodes = ::exp2f(level);
                        unsigned int posX = w_in.x * numNodes;
                        unsigned int posY = w_in.y * numNodes;
                        unsigned int code = Morton2Utils<BestLevel()>::GetCodeFromPosIdx(posX, posY);
                        unsigned int arrIdx = (numNodes*numNodes - 1) / 3 + code;
                        auto pdf = weights[arrIdx] / weights[0] * numNodes*numNodes;
                        //printf("pdf=%lf\n", pdf);
                        return pdf;
                    }
                    RTLIB_INLINE RTLIB_DEVICE static constexpr auto BestLevel()noexcept -> unsigned int
                    {
                        if (MaxLevel == 0) { return 1; }//(001)00
                        if (MaxLevel == 1) { return 1; }//(005)AB00         -> A0 B0  
                        if (MaxLevel == 2) { return 1; }//(016)ABCD         -> AC BD  
                        if (MaxLevel == 3) { return 2; }//(085)ABCDEFG00    -> ACE0 BDF0  
                        if (MaxLevel == 4) { return 2; }//(149)ABCDEFGHI    -> ACEG BDFH 
                        if (MaxLevel == 5) { return 3; }//
                        if (MaxLevel == 6) { return 3; }//
                        if (MaxLevel == 7) { return 3; }//
                        if (MaxLevel == 8) { return 3; }//
                        if (MaxLevel == 9) { return 4; }
                        if (MaxLevel == 10) { return 4; }
                        if (MaxLevel == 11) { return 4; }
                        if (MaxLevel == 12) { return 4; }
                        if (MaxLevel == 13) { return 4; }
                        if (MaxLevel == 14) { return 4; }
                        if (MaxLevel == 15) { return 4; }
                        if (MaxLevel == 16) { return 4; }
                        return 4;
                    }
                    RTLIB_INLINE RTLIB_DEVICE static auto AtomicAdd(float& target, float value)->float
                    {
#ifdef __CUDACC__
                        return atomicAdd(&target, value);
#else
                        float prvTarget = target;
                        target += value;
                        return prvTarget;
#endif
                        
                    }
                    unsigned int  level;
                    float*        weights;
                };

                template<unsigned int MaxLevel>
                struct MortonQuadTreeWrapperT
                {
                    static inline constexpr auto kMaxTreeLevel = MaxLevel;
                    static inline constexpr auto kCountPerNodes= ((static_cast<size_t>(1) << (2 * (MaxLevel + 1))) - 1) / 3;
                    RTLIB_INLINE RTLIB_DEVICE MortonQuadTreeWrapperT()noexcept {}
                    RTLIB_INLINE RTLIB_DEVICE MortonQuadTreeWrapperT(unsigned int level_, float* weightsBuilding_, float* weightsSampling_)
                    {
                        level           = level_;
                        weightsBuilding = weightsBuilding_;
                        weightsSampling = weightsSampling_;
                        fraction        = 0.3f;
                    }

                    RTLIB_INLINE RTLIB_DEVICE auto GetBuildingTree(unsigned int spatialIndex)const ->MortonQuadTreeT<MaxLevel>
                    {
                        float* weightsBuildingForNode = weightsBuilding + kCountPerNodes * spatialIndex;
                        return MortonQuadTreeT<MaxLevel>(level, weightsBuildingForNode);
                    }

                    RTLIB_INLINE RTLIB_DEVICE auto GetSamplingTree(unsigned int spatialIndex)const ->MortonQuadTreeT<MaxLevel>
                    {
                        float* weightsSamplingForNode = weightsSampling + kCountPerNodes * spatialIndex;
                        return MortonQuadTreeT<MaxLevel>(level, weightsSamplingForNode);
                    }

                    RTLIB_INLINE RTLIB_DEVICE auto Record(unsigned int spatialIndex, const float3& direction, float value)const
                    {
                        if (spatialIndex == UINT_MAX) { return; }
                       // printf("spatialId: %d\n", spatialIndex);
                        auto tree = GetBuildingTree(spatialIndex);
                        auto dir2 = RTLib::Ext::CUDA::Math::dir_to_canonical(direction);
                        tree.Record(dir2, value);
                    }
                    
                    template<typename RNG>
                    RTLIB_INLINE RTLIB_DEVICE auto SampleAndPdf(unsigned int spatialIndex, float& pdf, RNG& rng)const -> float3
                    {
                        //if (spatialIndex == UINT_MAX)
                        //{
                        //    pdf = 0.25f * RTLIB_M_INV_PI;
                        //    float2 dir2 = RTLib::Ext::CUDA::Math::random_float2(rng);
                        //    return RTLib::Ext::CUDA::Math::canonical_to_dir(dir2);
                        //}
                        
                        auto tree = GetSamplingTree(spatialIndex);
                        auto dir2 = tree.SampleAndPdf(pdf,rng);
                        pdf *= (0.25f * RTLIB_M_INV_PI);
                        return RTLib::Ext::CUDA::Math::canonical_to_dir(dir2);
                    }
                    
                    template<typename RNG>
                    RTLIB_INLINE RTLIB_DEVICE auto Sample(unsigned int spatialIndex, RNG& rng)const -> float3
                    {
                        //if (spatialIndex == UINT_MAX) 
                        //{
                        //    float2 dir2 = RTLib::Ext::CUDA::Math::random_float2(rng);
                        //    return RTLib::Ext::CUDA::Math::dir_to_canonical(dir2);
                        //}
                        auto tree = GetSamplingTree(spatialIndex);
                        auto dir2 = tree.Sample(rng);
                        return RTLib::Ext::CUDA::Math::dir_to_canonical(dir2);
                    }

                    RTLIB_INLINE RTLIB_DEVICE auto Pdf(unsigned int spatialIndex, const float3& direction)const -> float
                    {
                        //if (spatialIndex == UINT_MAX) 
                        //{
                        //    return 0.25f * RTLIB_M_INV_PI;
                        //}
                        auto dir2 = RTLib::Ext::CUDA::Math::dir_to_canonical(direction);
                        auto tree = GetSamplingTree(spatialIndex);
                        return tree.Pdf(dir2) * 0.25f * RTLIB_M_INV_PI;
                    }

                    unsigned int level;
                    float        fraction;
                    float* weightsBuilding;
                    float* weightsSampling;
                };

                template<unsigned int MaxLevel>
                struct MortonTraceVertexT {
                    float3       rayOrigin;
                    float3    rayDirection;
                    float3      throughPut;
                    float3         bsdfVal;
                    float3        radiance;
                    float           cosine;
                    float            woPdf;
                    float          bsdfPdf;
                    float         dTreePdf;
                    unsigned int   gridIdx;
                    bool           isDelta;

                    RTLIB_INLINE RTLIB_HOST_DEVICE void Record(const float3& r) noexcept {
                        radiance += r;
                    }
                    RTLIB_INLINE RTLIB_HOST_DEVICE void Commit(MortonQuadTreeWrapperT<MaxLevel>& mortonQuadTree, float statisticalWeight)noexcept
                    {
                        if ((gridIdx == UINT32_MAX)||isDelta) {
                            return;
                        }

                        bool isValidRadiance =
                            (isfinite(radiance.x) && radiance.x >= 0.0f) &&
                            (isfinite(radiance.y) && radiance.y >= 0.0f) &&
                            (isfinite(radiance.z) && radiance.z >= 0.0f);
                        bool isValidBsdfVal =
                            (isfinite(bsdfVal.x) && bsdfVal.x >= 0.0f) &&
                            (isfinite(bsdfVal.y) && bsdfVal.y >= 0.0f) &&
                            (isfinite(bsdfVal.z) && bsdfVal.z >= 0.0f);
                        if (woPdf <= 0.0f || isnan(woPdf) || !isValidRadiance || !isValidBsdfVal)
                        {
                            return;
                        }
                        auto localRadiance = make_float3(0.0f);
                        if (throughPut.x * woPdf > 1e-4f) {
                            localRadiance.x = radiance.x / throughPut.x;
                        }
                        if (throughPut.y * woPdf > 1e-4f) {
                            localRadiance.y = radiance.y / throughPut.y;
                        }
                        if (throughPut.z * woPdf > 1e-4f) {
                            localRadiance.z = radiance.z / throughPut.z;
                        }

                        localRadiance *= fabsf(cosine);
                        //printf("localRadiance=(%f,%f,%f)\n",localRadiance.x,localRadiance.y,localRadiance.z);
                        float3 product = localRadiance * bsdfVal;
                        float localRadianceAvg = (localRadiance.x + localRadiance.y + localRadiance.z) / 3.0f;
                        //float productAvg = (product.x + product.y + product.z) / 3.0f;
                        mortonQuadTree.Record(gridIdx, rayDirection, localRadianceAvg / woPdf);
                    }
                };
#ifndef __CUDACC__
                auto GetPtxMortonQuadTreeBuildKernel() -> const unsigned char*;

                template<unsigned int MaxLevel>
                class  RTMortonQuadTreeWrapperT
                {
                public:
                    static inline constexpr auto kWeightBufferCountPerNodes = ((static_cast<size_t>(1) << (2 * (MaxLevel+1))) - 1) / 3;
                    RTMortonQuadTreeWrapperT(RTLib::Ext::CUDA::CUDAContext* context, unsigned int maxHashSize, unsigned int maxTreeLevel)
                    {
                        m_Context      = context;
                        m_MaxHashSize  = maxHashSize;
                        m_MaxTreeLevel = std::min(maxTreeLevel, MaxLevel);
                        m_WeightBufferIndexBuilding = 0;
                    }

                    void Allocate()
                    {
                        RTLib::Ext::CUDA::CUDABufferCreateDesc desc = {};

                        desc.flags = RTLib::Ext::CUDA::CUDAMemoryFlags::eDefault;
                        desc.sizeInBytes = sizeof(float) * m_MaxHashSize* kWeightBufferCountPerNodes;
                        if (!m_WeightBuffersCUDA[0]) {
                            m_WeightBuffersCUDA[0] = std::unique_ptr<RTLib::Ext::CUDA::CUDABuffer>(m_Context->CreateBuffer(desc));
                        }
                        if (!m_WeightBuffersCUDA[1]) {
                            m_WeightBuffersCUDA[1] = std::unique_ptr<RTLib::Ext::CUDA::CUDABuffer>(m_Context->CreateBuffer(desc));
                        }
                    }

                    void Update(RTLib::Ext::CUDA::CUDAStream* stream = nullptr)
                    {
                        m_WeightBufferIndexBuilding = 1 - m_WeightBufferIndexBuilding;
                        auto weightBufferGpuAddress = CUDA::CUDANatives::GetCUdeviceptr(GetWeightBufferBuilding());
                        cuMemsetD32Async(weightBufferGpuAddress, 0.0f, m_MaxHashSize * kWeightBufferCountPerNodes, CUDA::CUDANatives::GetCUstream(stream));
                    }

                    void Destroy() {
                        if (m_WeightBuffersCUDA[0]) {
                            m_WeightBuffersCUDA[0]->Destroy();
                            m_WeightBuffersCUDA[0].reset();
                        }
                        if (m_WeightBuffersCUDA[1]) {
                            m_WeightBuffersCUDA[1]->Destroy();
                            m_WeightBuffersCUDA[1].reset();
                        }
                    }

                    void Clear(RTLib::Ext::CUDA::CUDAStream* stream = nullptr) {

                        auto copy = RTLib::Ext::CUDA::CUDAMemoryBufferCopy();
                        copy.size = sizeof(float) * m_MaxHashSize * kWeightBufferCountPerNodes;
                        copy.dstOffset = 0;
                        auto data = std::vector<float>(copy.size / sizeof(float), 0.0f);
                        copy.srcData = data.data();
                        if (stream) {

                            RTLIB_CORE_ASSERT_IF_FAILED(
                                stream->CopyMemoryToBuffer(
                                    m_WeightBuffersCUDA[0].get(),
                                    { copy }
                                )
                            );
                            RTLIB_CORE_ASSERT_IF_FAILED(
                                stream->CopyMemoryToBuffer(
                                    m_WeightBuffersCUDA[1].get(),
                                    { copy }
                                )
                            );
                        }
                        else {

                            RTLIB_CORE_ASSERT_IF_FAILED(
                                m_Context->CopyMemoryToBuffer(
                                    m_WeightBuffersCUDA[0].get(),
                                    { copy }
                                )
                            );
                            RTLIB_CORE_ASSERT_IF_FAILED(
                                m_Context->CopyMemoryToBuffer(
                                    m_WeightBuffersCUDA[1].get(),
                                    { copy }
                                )
                            );
                        }
                    }

                    auto GetGpuHandle() noexcept -> MortonQuadTreeWrapperT<MaxLevel>
                    {
                        return MortonQuadTreeWrapperT<MaxLevel>(
                            m_MaxTreeLevel,
                            RTLib::Ext::CUDA::CUDANatives::GetGpuAddress<float>(GetWeightBufferBuilding()),
                            RTLib::Ext::CUDA::CUDANatives::GetGpuAddress<float>(GetWeightBufferSampling())
                        );
                    }

                    auto GetContext()noexcept -> CUDA::CUDAContext* { return m_Context; }

                    auto GetMaxTreeLevel()const noexcept -> unsigned int { return m_MaxTreeLevel; }

                    auto GetMaxHashSize()const noexcept  -> unsigned int { return m_MaxHashSize;  }

                    auto GetNodePerElement()const noexcept -> unsigned int {
                        return  ((static_cast<unsigned int>(1) << (2 * (m_MaxTreeLevel + 1))) - 1) / 3;
                    }

                    auto GetWeightBufferBuilding()      noexcept ->       RTLib::Ext::CUDA::CUDABuffer* {
                        return m_WeightBuffersCUDA[m_WeightBufferIndexBuilding].get();
                    }

                    auto GetWeightBufferSampling()      noexcept ->       RTLib::Ext::CUDA::CUDABuffer* {
                        return m_WeightBuffersCUDA[1-m_WeightBufferIndexBuilding].get();
                    }
                    RTLib::Ext::CUDA::CUDAContext*                m_Context;
                    unsigned int                                  m_MaxHashSize;
                    unsigned int                                  m_MaxTreeLevel;
                    std::unique_ptr<RTLib::Ext::CUDA::CUDABuffer> m_WeightBuffersCUDA[2];
                    uint8_t                                       m_WeightBufferIndexBuilding;
                };

                template<unsigned int MaxLevel>
                class  RTMortonQuadTreeControllerT
                {
                public:
                    static inline constexpr auto kWeightBufferCountPerNodes = RTMortonQuadTreeWrapperT<MaxLevel>::kWeightBufferCountPerNodes;

                    enum TraceState
                    {
                        TraceStateRecord = 0,
                        TraceStateRecordAndSample = 1,
                        TraceStateSample = 2,
                    };

                    RTMortonQuadTreeControllerT(
                        RTMortonQuadTreeWrapperT<MaxLevel>* tree,
                        unsigned int sampleForBudget  /*ALL SAMPLES FOR TRACE*/,
                        unsigned int iterationForBuilt/*ITERATION FOR BUILT*/ = 0,
                        float         ratioForBudget  /*RATIO FOR RECORDING TREE*/ = 0.5f,
                        unsigned int samplePerLaunch  /*SAMPLES PER LAUNCH*/ = 1
                    )noexcept
                        :m_Tree{ tree }, m_SampleForBudget{ sampleForBudget }, m_SamplePerLaunch{ samplePerLaunch }, m_IterationForBuilt{ iterationForBuilt }, m_RatioForBudget{ ratioForBudget }{
                        m_Module = std::unique_ptr<CUDA::CUDAModule>(RTLib::Ext::CUDA::CUDAModule::LoadFromData(m_Tree->GetContext(), GetPtxMortonQuadTreeBuildKernel()));
                        m_BuildKernel = std::unique_ptr<CUDA::CUDAFunction>(m_Module->LoadFunction("mortonBuildKernel"));
                        m_ClearKernel = std::unique_ptr<CUDA::CUDAFunction>(m_Module->LoadFunction("mortonClearKernel"));
                    }

                    void SetSampleForBudget(unsigned int sampleForBudget)noexcept { m_SampleForBudget = sampleForBudget; }

                    void Start() {
                        m_TraceStart = true;
                    }

                    void BegTrace() {
                        if (!m_Tree) {
                            return;
                        }
                        if (m_TraceStart) {

                            m_SamplePerAll = 0;
                            m_SamplePerTmp = 0;
                            m_CurIteration = 0;

                            m_TraceStart = false;
                            m_TraceExecuting = true;

                            m_SampleForRemain = ((m_SampleForBudget - 1 + m_SamplePerLaunch) / m_SamplePerLaunch) * m_SamplePerLaunch;
                            m_SampleForPass = 0;

                            m_TraceState = TraceStateRecord;

                            m_Tree->Destroy();
                            m_Tree->Allocate();
                        }
                        if (!m_TraceExecuting) {
                            return;
                        }

                        if (m_SamplePerTmp == 0)
                        {
                            m_SampleForRemain -= m_SampleForPass;
                            m_SampleForPass = std::min<uint32_t>(m_SampleForRemain, (1 << m_CurIteration) * m_SamplePerLaunch);
                            if ((m_SampleForRemain - m_SampleForPass < 2 * m_SampleForPass) ||
                                (m_SamplePerAll >= m_RatioForBudget * static_cast<float>(m_SampleForBudget))) {
#ifndef NDEBUG
                                std::cout << "Final: this->m_SamplePerAll=" << m_SamplePerAll << std::endl;
#endif
                                m_SampleForPass = m_SampleForRemain;
                            }

                            if (m_SampleForRemain > m_SampleForPass) {
                                //printf("UpdateHashTree\n");
                                //m_Tree->Update();
                            }
                        }
                        if (m_CurIteration > m_IterationForBuilt) {
                            if (m_SampleForRemain > m_SampleForPass) {
                                m_TraceState = TraceStateRecordAndSample;
                            }
                            else {
                                m_TraceState = TraceStateSample;
                            }
                        }
                        else {
                            m_TraceState = TraceStateRecord;
                        }
#ifndef NDEBUG
                        if (m_TraceState == TraceStateRecordAndSample) {
                            std::cout << "[Record And Sample] ";
                        }
                        if (m_TraceState == TraceStateSample) {
                            std::cout << "[Sample Only] ";
                        }
                        if (m_TraceState == TraceStateRecord) {
                            std::cout << "[Record Only] ";
                        }
                        std::cout << "CurIteration: " << m_CurIteration << " SamplePerTmp: " << m_SamplePerTmp << std::endl;
#endif

                    }

                    void EndTrace(CUDA::CUDAStream* stream = nullptr) {
                        if (!m_Tree || !m_TraceExecuting) {
                            return;
                        }
                        m_SamplePerAll += m_SamplePerLaunch;
                        m_SamplePerTmp += m_SamplePerLaunch;

                        if (m_SamplePerTmp >= m_SampleForPass)
                        {
#ifndef NDEBUG
                            printf("UpdateHashTree\n");
#endif
                            m_Tree->Update(stream);
                            //this->LaunchClearKernel(stream);
#if RTLIB_EXT_OPX7_UTILS_OPX7_UTILS_MORTON_BUILD_BY_CUDA_KERNEL
                            this->LaunchBuildKernel(stream);
#endif
                            m_CurIteration++;
                            m_SamplePerTmp = 0;
                        }
                        if (m_SamplePerAll > m_SampleForBudget) {
                            m_TraceExecuting = false;
                            m_SamplePerAll = 0;
                        }
                    }

                    auto GetGpuHandle()const noexcept -> MortonQuadTreeWrapperT<MaxLevel>
                    {
                        return m_Tree->GetGpuHandle();
                    }

                    auto GetState()const noexcept -> TraceState {
                        return m_TraceState;
                    }

                    bool LaunchBuildKernel(CUDA::CUDAStream* stream = nullptr)
                    {
                        CUDA::CUDAKernelLaunchDesc launchDesc = {};

                        launchDesc.blockDimX = 1024;
                        launchDesc.blockDimY = 1;
                        launchDesc.blockDimZ = 1;

                        launchDesc.gridDimX  = ((m_Tree->GetMaxHashSize() + 1024 - 1) / 1024)*1024;
                        launchDesc.gridDimY  = 1;
                        launchDesc.gridDimZ  = 1;
                        {
                            auto weightBuilding = CUDA::CUDANatives::GetCUdeviceptr(m_Tree->GetWeightBufferSampling());
                            unsigned int level            = m_Tree->GetMaxTreeLevel();
                            unsigned int nodesPerElement  = kWeightBufferCountPerNodes;
                            unsigned int numNodes         = m_Tree->GetMaxHashSize();
                            launchDesc.kernelParams.resize(4);
                            launchDesc.kernelParams[0] = &weightBuilding;
                            launchDesc.kernelParams[1] = &level;
                            launchDesc.kernelParams[2] = &nodesPerElement;
                            launchDesc.kernelParams[3] = &numNodes;
                        }
                        launchDesc.sharedMemBytes = 0;
                        launchDesc.stream = stream;
                        return m_BuildKernel->Launch(launchDesc);
                    }

                    bool LaunchClearKernel(CUDA::CUDAStream* stream = nullptr)
                    {
                        CUDA::CUDAKernelLaunchDesc launchDesc = {};

                        launchDesc.blockDimX = 1024;
                        launchDesc.blockDimY = 1;
                        launchDesc.blockDimZ = 1;

                        launchDesc.gridDimX = ((m_Tree->GetMaxHashSize() * kWeightBufferCountPerNodes + 1024 - 1) / 1024) * 1024;
                        launchDesc.gridDimY = 1;
                        launchDesc.gridDimZ = 1;
                        {
                            auto weightBuilding = CUDA::CUDANatives::GetCUdeviceptr(m_Tree->GetWeightBufferBuilding());
                            unsigned int numWeightBuilding = m_Tree->GetMaxHashSize() * kWeightBufferCountPerNodes;
                            launchDesc.kernelParams.resize(2);
                            launchDesc.kernelParams[0] = &weightBuilding;
                            launchDesc.kernelParams[1] = &numWeightBuilding;
                        }
                        launchDesc.sharedMemBytes = 0;
                        launchDesc.stream = stream;
                        return m_ClearKernel->Launch(launchDesc);
                    }

                    auto GetIteration()const noexcept -> unsigned int
                    {
                        return m_CurIteration;
                    }
                private:
                    RTMortonQuadTreeWrapperT<MaxLevel>* m_Tree   = nullptr;
                    std::unique_ptr<CUDA::CUDAModule>   m_Module = nullptr;
                    std::unique_ptr<CUDA::CUDAFunction> m_BuildKernel = nullptr;
                    std::unique_ptr<CUDA::CUDAFunction> m_ClearKernel = nullptr;
                    unsigned int    m_SampleForBudget = 0;
                    unsigned int    m_SamplePerLaunch = 0;
                    unsigned int    m_IterationForBuilt = 0;
                    float           m_RatioForBudget = 0.0f;

                    bool            m_TraceStart = false;
                    bool            m_TraceExecuting = false;
                    unsigned int    m_SamplePerAll = 0;
                    unsigned int    m_SampleForRemain = 0;
                    unsigned int    m_SampleForPass = 0;
                    TraceState      m_TraceState = TraceStateRecord;
                    unsigned int    m_CurIteration = 0;
                public:
                    unsigned int    m_SamplePerTmp = 0;
                };
#endif
            }
        }
    }
}
#endif