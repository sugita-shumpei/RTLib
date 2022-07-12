#ifndef RTLIB_EXT_OPX7_UTILS_OPX7_UTILS_MORTON_H
#define RTLIB_EXT_OPX7_UTILS_OPX7_UTILS_MORTON_H
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
                        mortonCode &= static_cast<unsigned int>(::powf(2.0f, level) - 1);
                        return (::powf(4, level) - 1) / 3 + mortonCode;
                    }
                    RTLIB_INLINE RTLIB_HOST_DEVICE static auto GetArrayIndexFromMortonCodeGlobal(unsigned int mortonCode) noexcept -> unsigned int {
                        //1//4//16
                        //   4//16//64
                        unsigned int level = ::log2(mortonCode + 1) / 2;
                        return (::powf(4, level) - 1) / 3 + mortonCode;
                    }
                };
                template<unsigned int MaxLevel>
                struct MortonQuadTree {
                    //best grid idx
                    MortonQuadTree(unsigned int level_, float* weights_)noexcept {
                        level   = std::min<size_t>(level_, MaxLevel);
                        weights = weights_;
                    }
                    ~MortonQuadTree()noexcept {}

                    RTLIB_INLINE RTLIB_DEVICE void Record(const float2& w_in, float value)noexcept
                    {
                        unsigned int posX = w_in.x * ::(2.0, level);
                        unsigned int posY = w_in.y * ::powf(2.0, level);
                        unsigned int code = Morton2Utils<BestLevel()>::GetCodeFromPosIdx(posX, posY);
                        for (unsigned int i = 1; i <= level; ++i)
                        {
                            //std::cout << "code: " << std::bitset<MaxLevel * 2>(code) << " offset: " << (::powf(4, level + 1 - i) - 1) / 3 << std::endl;
                            unsigned int arrIndex = (::powf(4, level + 1 - i) - 1) / 3 + code;
                            weights[arrIndex] += value;
                            code >>= 2;
                        }
                        weights[0] += value;
                    }
                    template<typename RNG>
                    RTLIB_INLINE RTLIB_DEVICE auto SampleAndPdf(RNG& rng, float& pdf)const noexcept->float2
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
                        float posX = (xPosIdx + 0.5f * CUDA::Math::random_float1(rng)) * ::powf(0.5, level);
                        float posY = (yPosIdx + 0.5f * CUDA::Math::random_float1(rng)) * ::powf(0.5, level);
                        pdf = total / weights[0] * ::powf(4.0f, level);

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
                        float posX = (xPosIdx + 0.5f * CUDA::Math::random_float1(rng)) * ::powf(0.5, level);
                        float posY = (yPosIdx + 0.5f * CUDA::Math::random_float1(rng)) * ::powf(0.5, level);
                        return {posX, posY};
                    }
                    RTLIB_INLINE RTLIB_DEVICE auto Pdf(const  float2& w_in)const noexcept -> float
                    {
                        if (weights[0] <= 0.0f) {
                            return 1.0f;
                        }
                        unsigned int posX = w_in.x * ::powf(2.0, level);
                        unsigned int posY = w_in.y * ::powf(2.0, level);
                        unsigned int code = Morton2Utils<BestLevel()>::GetCodeFromPosIdx(posX, posY);
                        unsigned int arrIdx = (::powf(4, level) - 1) / 3 + code;
                        return weights[arrIdx] / weights[0] * ::powf(4.0f, level);
                    }
                    RTLIB_INLINE RTLIB_DEVICE static constexpr auto BestLevel()noexcept -> unsigned int
                    {
                        if (MaxLevel == 0) { return 1; }
                        if (MaxLevel == 1) { return 1; }
                        if (MaxLevel == 2) { return 1; }
                        if (MaxLevel == 3) { return 2; }
                        if (MaxLevel == 4) { return 2; }
                        if (MaxLevel == 5) { return 3; }
                        if (MaxLevel == 6) { return 3; }
                        if (MaxLevel == 7) { return 3; }
                        if (MaxLevel == 8) { return 3; }
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
                        target += value;
#endif
                    }
                    unsigned int  level;
                    float*        weights;
                };

            }
        }
    }
}
#endif