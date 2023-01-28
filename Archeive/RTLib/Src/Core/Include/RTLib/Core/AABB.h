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
        };
	}
}
#endif
