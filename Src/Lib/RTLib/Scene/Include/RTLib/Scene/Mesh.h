#ifndef RTLIB_SCENE_MESH__H
#define RTLIB_SCENE_MESH__H
#include <RTLib/Scene/Component.h>
#include <RTLib/Core/Mesh.h>
#include <memory>
namespace RTLib
{
	RTLIB_SCENE_DEFINE_OBJECT_TYPE_ID(Mesh, "E9E16008-1371-448E-B5A9-D6D3B26BC5FA");
	namespace Scene
	{
		struct Object;
		struct Mesh : public Scene::Component
		{
			static auto New(std::shared_ptr<Scene::Object> object) -> std::shared_ptr<Scene::Mesh>;
			virtual~Mesh() noexcept;

			virtual auto query_object(const TypeID& typeID) -> std::shared_ptr<Core::Object> override;
			virtual auto get_transform() -> std::shared_ptr<Scene::Transform> override;
			virtual auto get_object() -> std::shared_ptr<Scene::Object> override;
			virtual auto get_type_id() const noexcept -> TypeID override;
			virtual auto get_name() const noexcept -> String override;

			std::shared_ptr<Core::Mesh> mesh = nullptr;
		private:
			Mesh(std::shared_ptr<Scene::Object> object);
		private:
			std::weak_ptr<Scene::Object> m_SceneObject;
		};
	}
}
#endif
