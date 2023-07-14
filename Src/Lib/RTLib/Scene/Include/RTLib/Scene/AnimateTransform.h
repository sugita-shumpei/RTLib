#ifndef RTLIB_SCENE_ANIMATE_TRANSFORM__H
#define RTLIB_SCENE_ANIMATE_TRANSFORM__H
#include <RTLib/Scene/ObjectTypeID.h>
#include <RTLib/Scene/Component.h>
#include <RTLib/Scene/Transform.h>
#include <RTLib/Core/AnimateTransform.h>
namespace RTLib
{
    RTLIB_SCENE_DEFINE_OBJECT_TYPE_ID(AnimateTransform, "A66EF2F2-343A-4A40-B483-141E465D1081");
    namespace Scene
    {
        struct Animator;
        struct Object;
        struct AnimateTransform : public RTLib::Scene::Component
        {
            friend class Animator;

            static auto New(std::shared_ptr<RTLib::Scene::Object> object) -> std::shared_ptr<RTLib::Scene::AnimateTransform>;
            virtual ~AnimateTransform() noexcept;
            // Derive From RTLib::Core::Object
            virtual auto query_object(const TypeID& typeID) -> std::shared_ptr<RTLib::Core::Object> override;
            virtual auto get_type_id() const noexcept -> TypeID override;
            virtual auto get_name() const noexcept -> String override;
            // Derive From RTLib::Scene::Component
            virtual auto get_transform() -> std::shared_ptr<RTLib::Scene::Transform> override;
            virtual auto get_object() -> std::shared_ptr<RTLib::Scene::Object> override;

            auto get_pre_state() const noexcept -> AnimateBehavior { return m_Local.get_pre_state(); }
            void set_pre_state(AnimateBehavior preState) noexcept { return m_Local.set_pre_state(preState); }

            auto get_post_state() const noexcept -> AnimateBehavior { return m_Local.get_post_state(); }
            void set_post_state(AnimateBehavior postState) noexcept { return m_Local.set_post_state(postState); }

            auto get_base_position() const noexcept -> Vector3 { return m_Local.get_base_position(); }
            void set_base_position(const Vector3& position) noexcept { m_Local.set_base_position(position); }
            
            auto get_base_rotation() const noexcept -> Quat    { return m_Local.get_base_rotation(); }
            void set_base_rotation(const Quat& rotation) noexcept { m_Local.set_base_rotation(rotation); }

            auto get_base_scaling () const noexcept -> Vector3 { return m_Local.get_base_scaling (); }
            void set_base_scaling (const Vector3& scaling) noexcept { m_Local.set_base_scaling(scaling); }

            void add_local_position(Float32 tick, const Vector3& position) noexcept { m_Local.add_position(tick, position); }
            void add_local_rotation(Float32 tick, const Quat&    rotation) noexcept { m_Local.add_rotation(tick, rotation); }
            void add_local_scaling( Float32 tick, const Vector3& scaling ) noexcept { m_Local.add_scaling (tick, scaling ); }

            auto get_num_local_position_keys() const noexcept -> UInt32 { return m_Local.get_num_position_keys(); }
            auto get_num_local_rotation_keys() const noexcept -> UInt32 { return m_Local.get_num_rotation_keys(); }
            auto get_num_local_scaling_keys()  const noexcept -> UInt32 { return m_Local.get_num_scaling_keys(); }

            void set_animator(std::shared_ptr<Animator> animator);
            auto get_animator() const ->std::shared_ptr<Animator>;

            void update_tick(Float32 tick);
        private:
            AnimateTransform(std::shared_ptr<RTLib::Scene::Object> object);
            auto internal_get_transform() const noexcept -> std::shared_ptr<RTLib::Scene::Transform>;
            auto internal_get_object() const noexcept -> std::shared_ptr<RTLib::Scene::Object>;
            bool internal_update_tick(Float32 tick);
        private:
            RTLib::Core::AnimateTransform m_Local;
            std::weak_ptr<RTLib::Scene::Object> m_Object;
            std::weak_ptr<RTLib::Scene::Animator> m_Animator;
        };
    }
}
#endif
