#include <RTLib/Scene/Animator.h>
#include <RTLib/Scene/AnimateTransform.h>
#include <RTLib/Scene/Object.h>
#include "Internals/Container.h"
#include <stack>
#include <queue>

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

void RTLib::Scene::Animator::internal_add_animate_transform(const std::shared_ptr<RTLib::Scene::AnimateTransform>& newAnim)
{
    if (!newAnim) { return; }
    if (m_HandleMap.count(newAnim.get())) { return; }
    if (!newAnim->m_Animator.expired()) {
        auto oldAnimator = newAnim->m_Animator.lock();
        if (oldAnimator) {
            oldAnimator->m_HandleMap.erase(newAnim.get());
        }
    }
    m_HandleMap.insert({ newAnim.get(), newAnim });
    newAnim->m_Animator = std::static_pointer_cast<Animator>(shared_from_this());
}

void RTLib::Scene::Animator::internal_pop_animate_transform(const std::shared_ptr<RTLib::Scene::AnimateTransform>& newAnim)
{
    if (!newAnim) { return; }
    if (!m_HandleMap.count(newAnim.get())) { return; }
    m_HandleMap.erase(newAnim.get());
    newAnim->m_Animator = {};
}

auto RTLib::Scene::Animator::get_transforms() const noexcept -> std::vector<std::shared_ptr<RTLib::Scene::AnimateTransform>>
{
    std::vector<std::shared_ptr<RTLib::Scene::AnimateTransform>> transforms = {};
    for (auto& [key, valueW] : m_HandleMap) {
        auto value = valueW.lock();
        if (!value) { transforms.push_back(value); }
    }
    return transforms;
}

void RTLib::Scene::Animator::update_time(Float32 time)
{
    auto tick = std::fmod(time * m_TicksPerSecond, m_Duration);
    auto candiQueue = RTLib::Scene::Internals::VectorQueue<std::shared_ptr<Transform>>();
    // Local Update
    for (auto& [key, valueW] : m_HandleMap) {
        auto value = valueW.lock();
        // もしいずれかの
        if (value) {
            if (value->internal_update_tick(tick)) {
                candiQueue.push(value->get_transform());
            }
        }
    }
    auto updateTransforms = std::vector<std::shared_ptr<Transform>>();
    updateTransforms.reserve(candiQueue.get_deque().size());
    {
        //選択ノードの各ノードのうち
        //  自分自身がいずれかのgraph上に存在
        //→最終ノードに追加しない
        //  いずれかが自分自身のgraph上に存在
        //→選択ノードから削除
        while (!candiQueue.empty()) {
            //一つ取り出す
            auto candidate = candiQueue.front();
            candiQueue.pop();

            bool insertFinal = true;
            // 候補を調べる
            auto candiQueueSize = candiQueue.get_deque().size();
            {
                auto i = 0;
                while (!candiQueue.empty() && (i < candiQueueSize)) {
                    auto other = candiQueue.front();
                    candiQueue.pop();
                    // もし取り出したノードがgraphに含まれていたら
                    if (other->contain_in_graph(candidate)) {
                        // 最終的なupdateは行わない
                        insertFinal = false;
                    }
                    // もし取り出したノードのgraphに含まれていなかったら,
                    if (!candidate->contain_in_graph(other)) {
                        // 再度候補に追加
                        candiQueue.push(other);
                    }
                    ++i;
                }
            }
            // もし他のノードのgraphに含まれていなかったら
            if (insertFinal) {
                // 最終的なノードに追加
                updateTransforms.push_back(candidate);
            }
        }
    }
    for (auto& value : updateTransforms) {
        value->internal_update_cache();
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
