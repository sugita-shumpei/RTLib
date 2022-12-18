#ifndef RTLIB_CORE_AABB_H
#define RTLIB_CORE_AABB_H
#include <array>
#include <vector>
namespace RTLib
{
	namespace Core {
        struct AABB {
            std::array<float, 3>  min = std::array<float, 3>{FLT_MAX, FLT_MAX, FLT_MAX};
            std::array<float, 3>  max = std::array<float, 3>{ -FLT_MAX, -FLT_MAX, -FLT_MAX};
        public:
            AABB()noexcept {}
            AABB(const AABB& aabb)noexcept = default;
            AABB& operator=(const AABB& aabb)noexcept = default;
            AABB(const std::array<float, 3>& min, const std::array<float, 3>& max)noexcept :min{ min }, max{ max } {}
            AABB(const std::vector<std::array<float, 3>>& vertices)noexcept :AABB() {
                for (auto& vertex : vertices) {
                    this->Update(vertex);
                }
            }
            auto GetArea()const noexcept -> float {
                std::array<float, 3> range = {
                    max[0] - min[0],
                    max[1] - min[1],
                    max[2] - min[2]
                };
                return 2.0f * (range[0] * range[1] + range[1] * range[2] + range[2] * range[0]);
            }
            void Update(const  std::array<float, 3>& vertex)noexcept {
                for (size_t i = 0; i < 3; ++i) {
                    min[i] = std::min(min[i], vertex[i]);
                    max[i] = std::max(max[i], vertex[i]);
                }
            }
            //Matrix: Column Major Order
            //           Row Major Representaion
            //make_float4(tmpInstanceData.transform[0], tmpInstanceData.transform[4], tmpInstanceData.transform[8], 0.0f),
            //make_float4(tmpInstanceData.transform[1], tmpInstanceData.transform[5], tmpInstanceData.transform[9], 0.0f),
            //make_float4(tmpInstanceData.transform[2], tmpInstanceData.transform[6], tmpInstanceData.transform[10], 0.0f),
            //make_float4(tmpInstanceData.transform[3], tmpInstanceData.transform[7], tmpInstanceData.transform[11], 1.0f)
            // 
            //  X0 X1 X2 X3 X
            //  Y0 Y1 Y2 Y3 Y
            //  Z0 Z1 Z2 Z3 Z
            //  00 00 00 01 W
            void Transform(const std::array<float, 16>& mat4x4) noexcept
            {
                
            }
        };
	}
}
#endif
