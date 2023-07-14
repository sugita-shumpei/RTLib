#ifndef RTLIB_SCENE_OBJECT__H
#define RTLIB_SCENE_OBJECT__H
#include <RTLib/Core/Object.h>
#include <RTLib/Scene/Component.h>
#include <RTLib/Core/Transform.h>
#include <vector>
#include <variant>
namespace RTLib
{
	RTLIB_SCENE_DEFINE_OBJECT_TYPE_ID(Object, "1BA5E933-7911-464F-88CA-9ED19E40DD84");
	namespace Scene
	{
		struct Transform;
		struct TransformGraphNode;

		struct Object : public Core::Object
		{

			friend class Transform;

			static auto New(String name = "", const Core::Transform& transform = {}) -> std::shared_ptr<RTLib::Scene::Object>;
			virtual ~Object() noexcept;
			// Derive From RTLib::Core::Object
			virtual auto query_object(const TypeID& typeID) -> std::shared_ptr<Core::Object> override;
			virtual auto get_type_id() const noexcept -> TypeID override;
			virtual auto get_name() const noexcept -> String override;

			auto get_transform() const noexcept -> std::shared_ptr<RTLib::Scene::Transform>;

			template<typename ComponentType>
			auto get_component() const noexcept ->  std::shared_ptr<ComponentType>
			{
				return std::static_pointer_cast<ComponentType>(internal_get_component(ObjectTraits<ComponentType>::typeID));
			}

			template<typename ComponentType>
			auto get_components() const noexcept ->  std::vector<std::shared_ptr<ComponentType>>
			{
				auto components = internal_get_components(ObjectTraits<ComponentType>::typeID);
				auto res = std::vector<std::shared_ptr<ComponentType>>();
				res.reserve(components.size());
				for (auto& comp : components) {
					res.push_back(std::static_pointer_cast<ComponentType>(comp));
				}
				return res;
			}

			template<typename ComponentType>
			auto get_components_in_children() const noexcept ->  std::vector<std::shared_ptr<ComponentType>>
			{
				auto components = internal_get_components_in_children(ObjectTraits<ComponentType>::typeID);
				auto res = std::vector<std::shared_ptr<ComponentType>>();
				res.reserve(components.size());
				for (auto& comp : components) {
					res.push_back(std::static_pointer_cast<ComponentType>(comp));
				}
				return res;
			}

			template<typename ComponentType>
			auto get_components_in_parent() const noexcept ->  std::vector<std::shared_ptr<ComponentType>>
			{
				auto components = internal_get_components_in_parent(ObjectTraits<ComponentType>::typeID);
				auto res = std::vector<std::shared_ptr<ComponentType>>();
				res.reserve(components.size());
				for (auto& comp : components) {
					res.push_back(std::static_pointer_cast<ComponentType>(comp));
				}
				return res;
			}

			template<typename ComponentType>
			auto add_component() noexcept -> std::shared_ptr<ComponentType>
			{
				if constexpr (std::is_same_v<ComponentType, RTLib::Scene::Transform>)
				{
					return get_transform();
				}
				else {
					auto component = ComponentType::New(internal_get_object());
					m_Components.push_back(component);
					return component;
				}
			}

			template<typename ComponentType>
			void remove_component() noexcept
			{
				internal_remove_component(ObjectTraits<ComponentType>::typeID);
			}

		private:
			Object(String name = "") noexcept;
			void internal_set_transform(std::shared_ptr<RTLib::Scene::Transform> transform);
			auto internal_get_object() noexcept -> std::shared_ptr<RTLib::Scene::Object>;
			auto internal_get_component(ObjectTypeID typeID) const noexcept ->  std::shared_ptr<RTLib::Scene::Component>;
			auto internal_get_components(ObjectTypeID typeID) const noexcept -> std::vector<std::shared_ptr<RTLib::Scene::Component>>;
			auto internal_get_components_in_parent(ObjectTypeID typeID) const noexcept -> std::vector<std::shared_ptr<RTLib::Scene::Component>>;
			auto internal_get_components_in_children(ObjectTypeID typeID) const noexcept -> std::vector<std::shared_ptr<RTLib::Scene::Component>>;
			auto internal_get_transforms_in_children() const noexcept -> std::vector<std::shared_ptr<RTLib::Scene::Transform>>;
			void internal_remove_component(ObjectTypeID typeID) noexcept;
		private:
			// Transformの扱い
			// Sceneに紐づいていないときのみ, 強参照,　
			// そうではない時は弱参照で保持
			std::vector<std::shared_ptr<RTLib::Scene::Component>> m_Components = {};
			std::shared_ptr<RTLib::Scene::Transform> m_Transform = {};
			String m_Name = "";
		};
	}
}
#endif
