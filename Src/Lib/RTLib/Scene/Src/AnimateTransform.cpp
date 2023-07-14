#include <RTLib/Scene/AnimateTransform.h>
#include <RTLib/Scene/Animator.h>
#include <RTLib/Scene/Object.h>
auto RTLib::Scene::AnimateTransform::New(std::shared_ptr<Scene::Object> object) -> std::shared_ptr<Scene::AnimateTransform>
{
    return std::shared_ptr<Scene::AnimateTransform>(new Scene::AnimateTransform(object));
}

void RTLib::Scene::AnimateTransform::set_animator(std::shared_ptr<Animator> animator)
{
}

auto RTLib::Scene::AnimateTransform::get_animator() const -> std::shared_ptr<Animator>
{
    return std::shared_ptr<Animator>();
}

void RTLib::Scene::AnimateTransform::update_tick(Float32 tick)
{
    if (internal_update_tick(tick)) {
        auto transform = get_transform();
        if (!transform) { return; }
        transform->internal_update_cache();
    }
}


RTLib::Scene::AnimateTransform::~AnimateTransform() noexcept {}

auto RTLib::Scene::AnimateTransform::get_transform() -> std::shared_ptr<Transform>
{
    return internal_get_transform();
}

auto RTLib::Scene::AnimateTransform::get_object() -> std::shared_ptr<RTLib::Scene::Object> 
{
    return internal_get_object();
}

auto RTLib::Scene::AnimateTransform::query_object(const ObjectTypeID& typeID) -> std::shared_ptr<Core::Object>
{
    if (typeID == ObjectTypeID_Unknown || typeID == ObjectTypeID_SceneComponent || typeID == ObjectTypeID_SceneAnimateTransform) {
        return shared_from_this();
    }
    else {
        return {};
    }
}
auto RTLib::Scene::AnimateTransform::get_type_id() const noexcept -> ObjectTypeID
{
    return ObjectTypeID_SceneAnimateTransform;
}
auto RTLib::Scene::AnimateTransform::get_name() const noexcept -> String 
{
    auto object = m_Object.lock();
    if (object) {
        return object->get_name();
    }
    else {
        return "";
    }
}

bool RTLib::Scene::AnimateTransform::internal_update_tick(Float32 tick) {
    auto transform = get_transform();
    if (!transform) { return false; }

    auto curLocalPosi = transform->get_local_position();
    auto curLocalRota = transform->get_local_rotation();
    auto curLocalScal = transform->get_local_scaling();

    auto nxtLocalPosi = m_Local.get_interpolated_position(tick);
    auto nxtLocalRota = m_Local.get_interpolated_rotation(tick);
    auto nxtLocalScal = m_Local.get_interpolated_scaling (tick);

    bool nxtDirty = false;
    if (curLocalPosi != nxtLocalPosi) {
        transform->internal_set_local_position_without_update(nxtLocalPosi);
        nxtDirty = true;
    }
    if (curLocalRota != nxtLocalRota) {
        transform->internal_set_local_rotation_without_update(nxtLocalRota);
        nxtDirty = true;
    }
    if (curLocalScal != nxtLocalScal) {
        transform->internal_set_local_scaling_without_update(nxtLocalScal);
        nxtDirty = true;
    }
    return nxtDirty;
}

RTLib::Scene::AnimateTransform::AnimateTransform(std::shared_ptr<Scene::Object> object)
    :m_Object{ object }, m_Animator{}, m_Local {}
{
    m_Local.set_base_position(object->get_transform()->get_local_position());
    m_Local.set_base_rotation(object->get_transform()->get_local_rotation());
    m_Local.set_base_scaling (object->get_transform()->get_local_scaling ());
}

auto RTLib::Scene::AnimateTransform::internal_get_transform() const noexcept -> std::shared_ptr<Transform>
{
    auto object = internal_get_object();
    if (object) {
        return object->get_transform();
    }
    else {
        return nullptr;
    }
}

auto RTLib::Scene::AnimateTransform::internal_get_object() const noexcept -> std::shared_ptr<Scene::Object>
{
    return m_Object.lock();
}
