#ifndef RTLIB_CORE_ANIMATE_TRANSFORM__H
#define RTLIB_CORE_ANIMATE_TRANSFORM__H
#include <RTLib/Core/Transform.h>
#include <vector>
namespace RTLib
{
	namespace Core
	{
        enum class AnimateBehavior
        {
            eDefault,
            eConstant,
            eLinear,
            eRepeat
        };
        struct AnimateTransform
        {
            
        private:
            template<typename T>
            struct KeyValue {
                Float32 tick;
                T      value;
            };
        public:

            AnimateTransform() noexcept
                : 
                m_PreState{ AnimateBehavior::eDefault }, 
                m_PostState{ AnimateBehavior::eDefault },
                m_BasePosition{0.0f,0.0f,0.0f},
                m_BaseRotation{1.0f,0.0f,0.0f,0.0f},
                m_BaseScaling {1.0f,1.0f,1.0f},
                m_PositionKeys{}, 
                m_RotationKeys{}, 
                m_ScalingKeys{} 
            {}

            AnimateTransform(const AnimateTransform&) noexcept = default;
            AnimateTransform& operator=(const AnimateTransform&) noexcept = default;

            ~AnimateTransform() noexcept {}

            auto get_pre_state() const noexcept -> AnimateBehavior { return m_PreState; }
            void set_pre_state(AnimateBehavior preState) noexcept { m_PreState = preState; }
            
            auto get_post_state() const noexcept -> AnimateBehavior { return m_PostState; }
            void set_post_state(AnimateBehavior postState) noexcept { m_PostState = postState; }

            auto get_base_position() const noexcept -> Vector3 { return m_BasePosition; }
            void set_base_position(const Vector3& position) noexcept { m_BasePosition = position; }

            auto get_base_rotation() const noexcept -> Quat    { return m_BaseRotation; }
            void set_base_rotation(const Quat& rotation) noexcept { m_BaseRotation = rotation; }

            auto get_base_scaling() const noexcept -> Vector3 { return m_BaseScaling;  }
            void set_base_scaling(const Vector3& scaling) noexcept { m_BaseScaling = scaling; }

            void add_position(Float32 tick, const Vector3& position) noexcept
            {
                internal_add_value(m_PositionKeys, tick, position);
            }
            void add_rotation(Float32 tick, const Quat& rotation) noexcept
            {

                internal_add_value(m_RotationKeys, tick, rotation);
            }
            void add_scaling (Float32 tick, const Vector3& scaling) noexcept
            {
                internal_add_value(m_ScalingKeys, tick, scaling);
            }

            auto get_num_position_keys() const noexcept -> UInt32 { return m_PositionKeys.size(); }
            auto get_num_rotation_keys() const noexcept -> UInt32 { return m_RotationKeys.size(); }
            auto get_num_scaling_keys () const noexcept -> UInt32 { return m_ScalingKeys .size(); }

            auto get_position_keys() const noexcept -> const std::vector<KeyValue<Vector3>>& { return m_PositionKeys; }
            auto get_rotation_keys() const noexcept -> const std::vector<KeyValue<Quat>>&    { return m_RotationKeys; }
            auto get_scaling_keys () const noexcept -> const std::vector<KeyValue<Vector3>>& { return m_ScalingKeys;  }
            
            auto get_interpolated_position(Float32 tick) const noexcept -> Vector3 { return internal_get_value(tick, m_PreState, m_PostState, m_PositionKeys, m_BasePosition); }
            auto get_interpolated_rotation(Float32 tick) const noexcept -> Quat    { return internal_get_value(tick, m_PreState, m_PostState, m_RotationKeys, m_BaseRotation); }
            auto get_interpolated_scaling (Float32 tick) const noexcept -> Vector3 { return internal_get_value(tick, m_PreState, m_PostState, m_ScalingKeys , m_BaseScaling ); }

            auto get_interpolated_transform(Float32 tick) const noexcept -> Transform {
                Transform transform;
                transform.position = get_interpolated_position(tick);
                transform.rotation = get_interpolated_rotation(tick);
                transform.scaling  = get_interpolated_scaling (tick);
                return transform;
            }
        private:
            static inline constexpr UInt32 keyIndex_Pre = UINT32_MAX - 1;
            static inline constexpr UInt32 keyIndex_Post= UINT32_MAX;

            template<typename T>
            static void internal_add_value( std::vector<KeyValue<T>>& keys, Float32 tick, const T& value) noexcept
            {
                size_t insertTick = 0;
                for (auto& key : keys) {
                    if (key.tick == tick) { key.value = value; return; }
                    if (key.tick >  tick) {
                        break;
                    }
                    ++insertTick;
                }
                keys.insert(std::next(keys.begin(), insertTick), KeyValue<T>{tick,value});
            }
            template<typename T>
            static auto internal_get_interpolated_value(RTLib::Float32 tick, RTLib::UInt32 begIdx, const std::vector<KeyValue<T>>& keys) noexcept -> T
            {
                auto i0 = begIdx;
                auto i1 = begIdx + 1;
                const auto& k0 = keys.at(i0);
                const auto& k1 = keys.at(i1);
                const auto& t0 = k0.tick;
                const auto& t1 = k1.tick;
                const auto& v0 = k0.value;
                const auto& v1 = k1.value;
                auto alpha = (tick - t0) / (t1 - t0);
                if constexpr (std::is_same_v<T, RTLib::Quat>) {
                    return glm::normalize(glm::slerp(v0, v1, alpha));
                }
                else {
                    return glm::mix(v0, v1, alpha);
                }
            }
            template<typename T>
            static auto internal_get_nearest_key_index(RTLib::Float32 tick, const std::vector<KeyValue<T>>& keys) noexcept -> RTLib::UInt32
            {
                //最初の時刻以前なら
                if (keys.front().tick >= tick) { return keyIndex_Pre; }
                //最後の時刻以降なら
                if (keys.back().tick  <= tick) { return keyIndex_Post; }
                for (size_t i = 0; i < keys.size() - 1; ++i) {
                    if (keys[i + 1].tick > tick) {
                        return i;
                    }
                }
                assert(0);
            }
            template<typename T>
            static auto internal_get_value_without_key(RTLib::Float32 tick, AnimateBehavior state, const std::vector<KeyValue<T>>& keys, const T& baseVal, const T& constVal) noexcept -> T
            {
                if (state == AnimateBehavior::eDefault) { return baseVal; }
                if (state == AnimateBehavior::eConstant) { return constVal; }
                if (state == AnimateBehavior::eRepeat) {
                    auto t_beg = keys.front().tick;
                    auto t_end = keys.back().tick;
                    auto t_del = std::abs(t_end - t_beg);
                    auto t_near = 0.0f;
                    if (tick < t_beg)
                    {
                        t_near = std::fmod(tick - t_beg, t_del) + t_beg;
                    }
                    else {
                        t_near = std::fmod(tick - t_end, t_del) + t_beg;
                    }
                    auto i0 = internal_get_nearest_key_index(t_near, keys);
                    return internal_get_interpolated_value(tick, i0, keys);
                }
                if (state == AnimateBehavior::eLinear)
                {
                    auto t_beg = keys.front().tick;
                    auto t_end = keys.back().tick;
                    RTLib::UInt32 i0 = 0;
                    if (tick < t_beg)
                    {
                        i0 = 0;
                    }
                    else {
                        i0 = keys.size() - 2;
                    }
                    return internal_get_interpolated_value(tick, i0, keys);
                }
                return baseVal;
            }
            template<typename T>
            static auto internal_get_value(RTLib::Float32 tick, AnimateBehavior preState, AnimateBehavior postState, const std::vector<KeyValue<T>>& keys, const T& baseValue) noexcept -> T
            {
                if (keys.empty()) { return baseValue; }
                auto i0 = internal_get_nearest_key_index(tick, keys);
                if (i0 == keyIndex_Pre)  { return internal_get_value_without_key(tick, preState , keys, keys.front().value, keys.front().value); }
                if (i0 == keyIndex_Post) { return internal_get_value_without_key(tick, postState, keys, keys.back().value , keys.back().value); }
                return internal_get_interpolated_value(tick, i0, keys);
            }
        private:
            AnimateBehavior m_PreState  = AnimateBehavior::eDefault;
            AnimateBehavior m_PostState = AnimateBehavior::eDefault;

            std::vector<KeyValue<Vector3>> m_PositionKeys = {};
            std::vector<KeyValue<Quat>>    m_RotationKeys = {};
            std::vector<KeyValue<Vector3>> m_ScalingKeys  = {};

            Vector3 m_BasePosition = Vector3(0.0f,0.0f,0.0f);
            Quat    m_BaseRotation = Quat(1.0f,0.0f,0.0f,0.0f);
            Vector3 m_BaseScaling  = Vector3(1.0f,1.0f,1.0f);

        };
	}
}
#endif
