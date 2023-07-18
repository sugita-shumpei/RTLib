#ifndef RTLIB_SCENE_ANIMATOR__H
#define RTLIB_SCENE_ANIMATOR__H
#include <RTLib/Scene/ObjectTypeID.h>
#include <RTLib/Scene/Component.h>
#include <RTLib/Scene/Transform.h>
#include <RTLib/Core/AnimateTransform.h>
#include <unordered_map>
#include <unordered_set>
#include <utility>
namespace RTLib
{
	RTLIB_SCENE_DEFINE_OBJECT_TYPE_ID(Animator,"5E03C33D-5D77-4247-B052-A1C724EE11AD");
	namespace Scene
	{
		struct Animator : public RTLib::Scene::Component
		{
			friend class AnimateTransform;
			static auto New(std::shared_ptr<RTLib::Scene::Object> object) -> std::shared_ptr<Animator>;
			virtual ~Animator() noexcept {}
			// Derive From RTLib::Core::Object
			virtual auto query_object(const TypeID& typeID) -> std::shared_ptr<RTLib::Core::Object> override;
			virtual auto get_type_id() const noexcept -> TypeID override;
			virtual auto get_name() const noexcept -> String override;
			// Derive From RTLib::Scene::Component
			virtual auto get_transform() -> std::shared_ptr<RTLib::Scene::Transform> override;
			virtual auto get_object() -> std::shared_ptr<RTLib::Scene::Object> override;

			void set_duration(Float32 duration) noexcept;
			auto get_duration() const noexcept -> Float32;

			void set_ticks_per_second(Float32 ticks) noexcept;
			auto get_ticks_per_second() const noexcept -> Float32;

			auto get_transforms() const noexcept -> std::vector<std::shared_ptr<RTLib::Scene::AnimateTransform>>;
			// 依存関係にある複数のAnimateTransformを更新
			void update_time(Float32 time);
		private:
			Animator(std::shared_ptr<RTLib::Scene::Object> object) noexcept;
			auto internal_get_transform() const noexcept -> std::shared_ptr<RTLib::Scene::Transform>;
			auto internal_get_object() const noexcept -> std::shared_ptr<RTLib::Scene::Object>;
			void internal_add_animate_transform(const std::shared_ptr<RTLib::Scene::AnimateTransform>& transform);
			void internal_pop_animate_transform(const std::shared_ptr<RTLib::Scene::AnimateTransform>& transform);
		private:
			RTLib::Float32 m_Duration = 1.0f;
			RTLib::Float32 m_TicksPerSecond = 60.0f;
			std::weak_ptr<RTLib::Scene::Object> m_Object = {};
			std::unordered_map<RTLib::Scene::AnimateTransform*,std::weak_ptr<RTLib::Scene::AnimateTransform>> m_HandleMap = {};
		};
	}
}
#endif
