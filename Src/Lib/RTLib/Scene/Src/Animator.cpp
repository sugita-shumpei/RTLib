#include <RTLib/Scene/Animator.h>
#include <RTLib/Scene/AnimateTransform.h>
#include <RTLib/Scene/Object.h>
auto RTLib::Scene::Animator::New(std::shared_ptr<RTLib::Scene::Object> object) -> std::shared_ptr<Animator>
{
    return std::shared_ptr<Animator>(new Animator(object));
}

void RTLib::Scene::Animator::set_duration(Float32 duration) noexcept
{
    m_Duration = duration;
}

auto RTLib::Scene::Animator::get_duration() const noexcept -> Float32
{
    return m_Duration;
}

void RTLib::Scene::Animator::set_ticks_per_second(Float32 ticks) noexcept
{
    m_TicksPerSecond = ticks;
}

auto RTLib::Scene::Animator::get_ticks_per_second() const noexcept -> Float32
{
    return m_TicksPerSecond;
}

void RTLib::Scene::Animator::internal_add_animate_transform(const std::shared_ptr<RTLib::Scene::AnimateTransform>& transform)
{
    if (!transform) { return; }
    if (m_HandleMap.count(transform.get()) > 0) { return; }
    m_HandleMap.insert({ transform.get(),transform });
    // 依存関係を構築する
    // もし, 今回追加するtransformに対応するnodeが既存のノードに含まれている場合
    // →追加せず, 通常通りの更新に任せる
    // そうではない場合
    // →追加
    {
        RTLib::Scene::AnimateTransform* keyUpdate = nullptr;
        std::vector<RTLib::Scene::AnimateTransform*> shouldErase = {};
        for (auto [transformKeyUpdate,transformKeysDepend] : m_UpdateMap)
        {
            auto transformUpdateW = m_HandleMap.at(transformKeyUpdate);
            if (!transformUpdateW.expired()) {
                auto transformUpdate = transformUpdateW.lock();
                if (transformUpdate->get_transform()->contain_in_graph(transform->get_transform())) {
                    keyUpdate = transformKeyUpdate;
                    break;
                }
            }
        }
        if (keyUpdate) {
            m_UpdateMap.at(keyUpdate).push_back(transform.get());
            return;
        }
    }
    //　逆にほかの依存ノードのうち新たに追加するノードに含まれるものは削除
    {
        std::vector<RTLib::Scene::AnimateTransform*> keyDependsRoot = {};
        for (auto [transformKeyUpdate, transformKeysDepend] : m_UpdateMap)
        {
            auto transformUpdateW = m_HandleMap.at(transformKeyUpdate);
            if (!transformUpdateW.expired()) {
                auto transformUpdate = transformUpdateW.lock();
                if (transform->get_transform()->contain_in_graph(transformUpdate->get_transform())) {
                    keyDependsRoot.push_back(transformKeyUpdate);
                }
            }
        }
        std::vector<RTLib::Scene::AnimateTransform*> keyDependsNode = {};
        for (auto keyDependRoot : keyDependsRoot) {
            keyDependsNode.push_back(keyDependRoot);
            for (auto& transformKeyDepend : m_UpdateMap.at(keyDependRoot)) {
                auto transformDependW = m_HandleMap.at(transformKeyDepend);
                if (!transformDependW.expired()) {
                    keyDependsNode.push_back(transformKeyDepend);
                }
            }
        }
        for (auto keyDependRoot : keyDependsRoot) {
            m_UpdateMap.erase(keyDependRoot);
        }

        m_UpdateMap.insert({ transform.get(),keyDependsNode });
    }
}

auto RTLib::Scene::Animator::get_transforms() const noexcept -> std::vector<std::shared_ptr<RTLib::Scene::AnimateTransform>>
{
    std::vector<std::shared_ptr<RTLib::Scene::AnimateTransform>> transforms = {};
    for (auto& [key,valueW] : m_HandleMap) {
        auto value = valueW.lock();
        if (value) {
            transforms.push_back(value);
        }
    }
    return transforms;
}

void RTLib::Scene::Animator::update_time(Float32 time)
{
    auto tick = std::fmod(time * m_TicksPerSecond, m_Duration);
    // Local Update
    for (auto& [key, valueW] : m_HandleMap) {
        auto updateW = m_HandleMap.at(key);
        if (!updateW.expired()) {
            auto updateTransform = updateW.lock();
            updateTransform->internal_update_tick(tick);
        }
    }
    // Global Update
    for (auto& [updateKey,dependKeys]:m_UpdateMap) {
        auto updateW = m_HandleMap.at(updateKey);
        if (!updateW.expired()) {
            auto updateTransform = updateW.lock();
            updateTransform->get_transform()->internal_update_cache();
        }
    }
}

RTLib::Scene::Animator::Animator(std::shared_ptr<RTLib::Scene::Object> object) noexcept
    :m_Object{object}
{
}

auto RTLib::Scene::Animator::internal_get_transform() const noexcept -> std::shared_ptr<RTLib::Scene::Transform>
{
    return internal_get_transform();
}

auto RTLib::Scene::Animator::internal_get_object() const noexcept -> std::shared_ptr<RTLib::Scene::Object>
{
    return internal_get_object();
}

auto RTLib::Scene::Animator::query_object(const TypeID& typeID) -> std::shared_ptr<RTLib::Core::Object>
{
    if (typeID == ObjectTypeID_Unknown || typeID == ObjectTypeID_SceneComponent || typeID == ObjectTypeID_SceneAnimator)
    {
        return shared_from_this();
    }
    else {
        return nullptr;
    }
}

auto RTLib::Scene::Animator::get_type_id() const noexcept -> TypeID
{
    return ObjectTypeID_SceneAnimator;
}

auto RTLib::Scene::Animator::get_name() const noexcept -> String
{
    auto object = internal_get_object();
    if (!object) { return ""; }
    return object->get_name();
}

auto RTLib::Scene::Animator::get_transform() -> std::shared_ptr<Scene::Transform>
{
    auto object = internal_get_object();
    if (!object) { return nullptr;  }
    return object->get_transform();
}

auto RTLib::Scene::Animator::get_object() -> std::shared_ptr<Scene::Object>
{
    return m_Object.lock();
}
