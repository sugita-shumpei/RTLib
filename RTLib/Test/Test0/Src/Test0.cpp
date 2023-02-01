#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <cuda.h>
#include <optix.h>
#include <optix_stubs.h>
#include <optix_function_table_definition.h>
#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>
#include <nlohmann/json.hpp>
#include <uuid.h>
#include <iostream>
#include <array>
#include <vector>
#include <fstream>
#include <utility>
#include <unordered_map>
#include <unordered_set>

using  UniqueId = int;
template<typename ObjectType>
class  UniqueIdHandler
{
public:
	UniqueIdHandler() noexcept {}
	UniqueIdHandler(const UniqueIdHandler&) = delete;
	UniqueIdHandler& operator=(const UniqueIdHandler&) = delete;

	UniqueIdHandler(UniqueIdHandler&& rhs)noexcept
		:m_ID{std::exchange(rhs.m_ID,0)} {
	}
	UniqueIdHandler& operator=(UniqueIdHandler&& rhs)noexcept
	{
		if (this != &rhs) {
			m_ID = std::exchange(rhs.m_ID, 0);
		}
		return *this;
	}
	UniqueId GetID()const noexcept
	{
		if (m_ID==0) {
			static std::atomic<UniqueId> globalUniqueIdCounter{ 0 };
			m_ID = globalUniqueIdCounter.fetch_add(1) + 1;
		}
		return m_ID;
	}
private:
	mutable UniqueId m_ID = 0;
};
struct Scene {
	static auto New() -> std::shared_ptr<Scene> {
		return std::shared_ptr<Scene>(new Scene());
	}
	auto RootActors()const noexcept -> const std::vector<std::shared_ptr<class Actor>>
	{
		return m_Actors;
	}

	static void AddActor(const std::shared_ptr<Scene>& scene, const std::shared_ptr<class Actor>& actor)noexcept;

	void Serialize(nlohmann::json& json);
private:
	std::vector<std::shared_ptr<class Actor>> m_Actors = {};
};
//  Object に UUID と UUIDのセットを記録
//  Object の 生成時にUUIDのセットを保存
//->GetComponentに利用
template<typename T, size_t N>
inline constexpr auto AddArray(const std::array<T, N>& arr, T val) -> std::array<T, N+1>{
	std::array<T, N + 1> res;
	size_t i = 0;
	for (auto& elm : arr) {
		res[i] = elm;
		++i;
	}
	res[N] = val;
	return res;
}
inline constexpr auto NewUuid(std::uint32_t v1, std::uint16_t v2, std::uint16_t v3, const std::array<std::uint8_t, 8>& v4) -> uuids::uuid {
	std::array<std::uint8_t, 16> data = {};
	//0x00FF
	data[0] = std::uint8_t( v1 & std::uint32_t(0xFF));
	data[1] = std::uint8_t((v1 & std::uint32_t(0xFF00))>>8);
	data[2] = std::uint8_t((v1 & std::uint32_t(0xFF0000))>>16);
	data[3] = std::uint8_t((v1 & std::uint32_t(0xFF000000)) >> 24);

	data[4] = std::uint8_t( v2 & std::uint32_t(0xFF));
	data[5] = std::uint8_t((v2 & std::uint32_t(0xFF00)) >> 8);

	data[6] = std::uint8_t( v3 & std::uint32_t(0xFF));
	data[7] = std::uint8_t((v3 & std::uint32_t(0xFF00)) >> 8);

	for (int i = 0; i < 8; ++i) {
		data[8 + i] = v4[i];
	}
	return uuids::uuid(data);
}

class  Object {
public:
	Object() noexcept :m_InstanceID(UniqueIdHandler<Object>().GetID()), name{} {}
	virtual ~Object() noexcept {}

	std::string name;

	auto GetInstanceID()const noexcept -> UniqueId {
		return m_InstanceID;
	}
	auto ToString()const noexcept -> std::string {
		return name;
	}

	template<typename ObjectDerived, typename ...Args,bool cond = std::is_base_of_v<Object, ObjectDerived>>
	static auto New(Args&& ...args)->std::shared_ptr<ObjectDerived>{
		auto object = ObjectDerived::NewObject(std::forward<Args&&>(args)...);
		object.m_TypeInfo = std::unique_ptr<ITypeInfo>(new ObjectDerived::TypeInfo())
	}
private:
	UniqueId m_InstanceID;
};
class  Actor: public Object {
	friend class Scene;
	friend class Transform;
protected:
	Actor() noexcept : Object(), m_ActiveLocal{ true }, tag{}, m_Scene{}, m_Components{} {}
	Actor(std::string name)noexcept:Object(), m_ActiveLocal{ true }, tag{ name }, m_Scene{}, m_Components{} {}
public:
	static auto New()noexcept -> std::shared_ptr<Actor>;
	static auto New(std::string name)noexcept -> std::shared_ptr<Actor>;
	virtual ~Actor() noexcept {}
	
	template<typename ComponentType,bool Cond = std::is_base_of_v<class Component, ComponentType>>
	static auto AddComponent(const std::shared_ptr<Actor>& actor)noexcept->std::shared_ptr< ComponentType>;
	//After Add Transform
	template<typename ComponentType, bool Cond = std::is_base_of_v<class Component, ComponentType>>
	auto AddComponent() noexcept -> std::shared_ptr< ComponentType>;
	template<typename ComponentType, bool Cond = std::is_base_of_v<class Component, ComponentType>>
	auto GetComponent()const noexcept -> std::shared_ptr< ComponentType>;
	template<typename ComponentType, bool Cond = std::is_base_of_v<class Component, ComponentType>>
	auto GetComponentInChildren()const noexcept -> std::shared_ptr< ComponentType>;
	template<typename ComponentType, bool Cond = std::is_base_of_v<class Component, ComponentType>>
	auto GetComponents()const noexcept -> std::vector<std::shared_ptr< ComponentType>>;
	template<typename ComponentType, bool Cond = std::is_base_of_v<class Component, ComponentType>>
	auto GetComponentsInChildren()const noexcept -> std::vector<std::shared_ptr< ComponentType>>;

	void Serialize(nlohmann::json& json) {}

	auto  IsActive()const noexcept -> bool;
	void SetActive(bool active)noexcept  ;

	auto Scene()const noexcept -> std::shared_ptr<class Scene> {
		return m_Scene.lock();
	}
	auto Transform()const noexcept -> std::shared_ptr<class Transform>;

	std::string tag;
private:
	void Scene(const std::shared_ptr<class Scene>& scene)noexcept {
		m_Scene = scene;
	}

	void OnEnabled();
	void OnDisable();

	bool m_ActiveLocal;
	std::weak_ptr<class Scene> m_Scene;
	std::vector<std::shared_ptr<class Component>> m_Components;
};
class  Component :public Object
{
private:
	friend class Actor;
protected:
	Component() noexcept : Object(), tag{}, m_Actor{}, m_pID{ nullptr }, m_cID{ 0 } {}
public:
	static inline constexpr uuids::uuid typeID  = uuids::uuid();

	template<typename ComponentDerived,bool Cond = std::is_base_of_v< Component, ComponentDerived>>
	static auto New() -> std::shared_ptr<ComponentDerived>;

	virtual ~Component() noexcept {}
	
	virtual void Awake() = 0;
	virtual void Start() = 0;
	virtual void Update() = 0;
	virtual void FixedUpdate() {};
	virtual void LastUpdate()  {};
	virtual void OnEnabled() {};
	virtual void OnDisable() {};

	 
	auto Actor()const noexcept -> std::shared_ptr<class Actor> {
		return m_Actor.lock();
	}

	std::string tag;

protected:
	bool IsSafeDownCast(const uuids::uuid& targetType)const noexcept {
		for (int i = 0; i < m_cID; ++i) {
			if (m_pID[i] == targetType) {
				return true;
			}
		}
		return false;
	}
private:
	void Actor(const std::shared_ptr<class Actor>& actor) noexcept {
		m_Actor = actor;
	}
	std::weak_ptr<class Actor> m_Actor;
	const uuids::uuid* m_pID;
	size_t m_cID;
};
template <typename ComponentType>
struct ComponentTraits {
	using type = ComponentType;
	using base_type = typename ComponentType::base_type;
	static inline constexpr auto typeID = ComponentType::typeID;
	static inline constexpr auto typeIDs = AddArray(ComponentTraits<typename ComponentTraits::base_type>::typeIDs, typeID);
};
template<>
struct ComponentTraits<Component>
{
	using type = Component;
	using base_type = void;
	static inline constexpr uuids::uuid typeID = Component::typeID;
	static inline constexpr std::array<uuids::uuid, 1> typeIDs = { typeID };
};

class  Transform :public Component
{
private:
	friend class Component;
	Transform()noexcept :Component(), m_Parent{}, m_DependTransforms{},
		m_LocalPosition{ }, m_LocalScale{ 1.0f }, m_LocalRotation { glm::identity<glm::quat>() },
		m_Position{ }, m_LossyScale{ 1.0f }, m_Rotation{ glm::identity<glm::quat>() }
	{}

	static auto NewComponent() -> std::shared_ptr<Transform>;
public:
	// {BC4475B5-74A9-4CC4-A48C-6C3939409747}
	static inline constexpr uuids::uuid typeID = NewUuid(0xbc4475b5, 0x74a9, 0x4cc4, { 0xa4, 0x8c, 0x6c, 0x39, 0x39, 0x40, 0x97, 0x47 });
	using base_type = Component;

	virtual ~Transform()noexcept {}

	auto Parent()const noexcept -> std::shared_ptr<class Transform>;
	void Parent(const std::shared_ptr<Transform>& parentTransform) noexcept;

	auto Position()const noexcept -> glm::vec3 { return m_Position; }
	void Position(const glm::vec3& position);

	auto Rotation()const noexcept -> glm::quat { return m_Rotation; }
	void Rotation(const glm::quat& rotation);

	auto LossyScale()const noexcept -> glm::vec3 { return m_LossyScale; }

	auto LocalPosition()const noexcept -> glm::vec3 { return m_LocalPosition; }
	void LocalPosition(const glm::vec3& localPosition);

	auto LocalRotation()const noexcept -> glm::quat { return m_LocalRotation; }
	void LocalRotation(const glm::quat& localRotation);

	auto LocalScale()const noexcept -> glm::vec3 { return m_LocalScale; }
	void LocalScale(const glm::vec3& localScale);

	auto GetChild(size_t idx)const->std::shared_ptr<class Transform>;
	auto GetChildCount()const noexcept -> size_t;
	//World -> to -> Local
	auto InverseTransformPoint(const glm::vec3& point)const noexcept -> glm::vec3;
	//Local -> to -> World
	auto TransformPoint(const glm::vec3& point)const noexcept -> glm::vec3;

	auto GetLocalToWorldMatrix()const noexcept -> glm::mat4x4;

	virtual void Awake() override
	{
	}
	virtual void Start() override
	{
	}
	virtual void Update() override
	{
	}

	virtual void OnEnabled() override;
	virtual void OnDisable() override;
private:
	void RemoveChild(size_t i)
	{
		GetChild(i)->m_Parent = {};
		m_DependTransforms.erase(std::next(m_DependTransforms.begin(), i + 1));
	}

	std::weak_ptr<Transform>                    m_Parent;
	std::vector<std::weak_ptr<class Transform>> m_DependTransforms;

	glm::vec3 m_Position;
	glm::quat m_Rotation;
	glm::vec3 m_LossyScale;

	glm::vec3 m_LocalPosition;
	glm::quat m_LocalRotation;
	glm::vec3 m_LocalScale;
};

template<typename ComponentDerived, bool Cond>
auto Component::New() -> std::shared_ptr<ComponentDerived> {
	std::shared_ptr<ComponentDerived> component = ComponentDerived::NewComponent();
	Component* pComponent = component.get();
	pComponent->m_pID = ComponentTraits<ComponentDerived>::typeIDs.data();
	pComponent->m_cID = std::size(ComponentTraits<ComponentDerived>::typeIDs);
	return component;
}
void Scene::AddActor(const std::shared_ptr<Scene>& scene, const std::shared_ptr<class Actor>& actor)noexcept
{
	scene->m_Actors.push_back(actor);
	scene->m_Actors.back()->Scene(scene);
}
auto Actor::New()noexcept -> std::shared_ptr<Actor> {
	auto actor = std::shared_ptr<Actor>(new Actor());
	AddComponent<class Transform>(actor);
	return actor;
}
auto Actor::New(std::string name)noexcept -> std::shared_ptr<Actor> {
	auto actor = std::shared_ptr<Actor>(new Actor(name));
	AddComponent<class Transform>(actor);
	return actor;
}
template <typename ComponentType, bool Cond>
auto Actor::AddComponent(const std::shared_ptr<Actor>& actor)noexcept->std::shared_ptr<ComponentType>
{
	auto component = Component::New<ComponentType>();
	actor->m_Components.push_back(component);
	actor->m_Components.back()->Actor(actor);
	if (actor->IsActive()) {
		component->OnEnabled();
	}
	return component;
}
template <typename ComponentType, bool Cond>
auto Actor::AddComponent()noexcept->std::shared_ptr<ComponentType>
{
	auto actor = Transform()->Actor();
	return AddComponent<ComponentType>(actor);
}
template <typename ComponentType, bool Cond>
auto Actor::GetComponent()const noexcept -> std::shared_ptr< ComponentType> {
	for (auto& component : m_Components) {
		if (component->IsSafeDownCast(ComponentType::typeID))
		{
			return std::static_pointer_cast<ComponentType>(component);
		}
	}
	return nullptr;
}
template <typename ComponentType, bool Cond>
auto Actor::GetComponents()const noexcept -> std::vector<std::shared_ptr< ComponentType>>
{
	std::vector<std::shared_ptr<ComponentType>> res;
	for (auto& component : m_Components) {
		if (component->IsSafeDownCast(ComponentType::typeID))
		{
			res.push_back(std::static_pointer_cast<ComponentType>(component));
		}
	}
	return res;
}
auto Actor::Transform()const noexcept -> std::shared_ptr<class Transform>
{
	return std::static_pointer_cast<class Transform>(m_Components.front());
}
auto Actor::IsActive()const noexcept -> bool {
	auto parent = Transform()->Parent();
	if ( parent) {
		auto parentActor = parent->Actor();
		if (!parentActor->IsActive()) { return false; }
	}
	return m_ActiveLocal;
}
void Actor::SetActive(bool isActive)noexcept
{
	bool prvActiveLocal  = m_ActiveLocal;
	auto parentTransform = Transform()->Parent();
	bool isParentActive  = true;
	if  (parentTransform) {
		isParentActive = parentTransform->Actor()->IsActive();
	}
	if (isParentActive &&  prvActiveLocal && !isActive)
	{
		OnDisable();
	}
	if (isParentActive && !prvActiveLocal && isActive)
	{
		OnEnabled();
	}
}
void Actor::OnEnabled() {
	auto transform = Transform();
	for (auto& component : m_Components)
	{
		component->OnEnabled();
	}
}
void Actor::OnDisable() {
	auto transform = Transform();
	for (auto& component : m_Components)
	{
		component->OnDisable();
	}
}

auto Transform::NewComponent() -> std::shared_ptr<Transform> {
	auto transform = std::shared_ptr<Transform>(new Transform());
	transform->m_DependTransforms.push_back(transform);
	return transform;
}
auto Transform::GetChild(size_t idx)const -> std::shared_ptr<class Transform> { return m_DependTransforms.at(idx + 1).lock(); }
auto Transform::GetChildCount()const noexcept -> size_t { return m_DependTransforms.size() - 1; }

void Transform::Parent(const std::shared_ptr<class Transform>& parentTransform) noexcept
{
	auto childTransform = this->Actor()->Transform();
	if (!childTransform->m_Parent.expired()) {

		auto curParentTransform = childTransform->m_Parent.lock();
		size_t i = 0;
		for (i = 0; i < curParentTransform->GetChildCount(); ++i) {
			if (childTransform == curParentTransform->GetChild(i)) {

				break;
			}
		}
		curParentTransform->RemoveChild(i);
	}
	childTransform->m_Parent = parentTransform;
	parentTransform->m_DependTransforms.push_back(childTransform);
}
auto Transform::Parent()const noexcept -> std::shared_ptr<class Transform>
{
	return m_Parent.lock();
}

void Transform::OnEnabled() {
	if (name.empty()) {
		name = Actor()->name;
	}
	for (auto i = 0; i < GetChildCount(); ++i) {
		auto childActor = GetChild(i)->Actor();
		if (childActor->m_ActiveLocal) {
			childActor->OnEnabled();
		}
	}
}
void Transform::OnDisable()
{
	for (auto i = 0;i<GetChildCount();++i){
		auto childActor = GetChild(i)->Actor();
		if (childActor->m_ActiveLocal) {
			childActor->OnDisable();
		}
	}
}

template <typename ComponentType, bool Cond>
auto Actor::GetComponentInChildren()const noexcept -> std::shared_ptr< ComponentType> {
	auto transform = Transform();
	{
		auto childComponent = GetComponent<ComponentType>();
		if (childComponent) {
			return childComponent;
		}
	}
	for (auto i = 0; i < transform->GetChildCount(); ++i) {
		auto childActor     = transform->GetChild(i)->Actor();
		auto childComponentInChildren = childActor->GetComponentInChildren<ComponentType>();
		if (childComponentInChildren) {
			return childComponentInChildren;
		}
	}
	return nullptr;
}
template <typename ComponentType, bool Cond>
auto Actor::GetComponentsInChildren()const noexcept -> std::vector<std::shared_ptr< ComponentType>>
{
	std::vector<std::shared_ptr<ComponentType>> res;
	auto transform = Transform();
	{
		auto childComponents = GetComponents<ComponentType>();
		if (!childComponents.empty()) {
			res.insert(std::end(res), std::begin(childComponents), std::end(childComponents));
		}
	}
	for (auto i = 0; i < transform->GetChildCount(); ++i) {
		auto childActor = transform->GetChild(i)->Actor();
		auto childComponentsInChildren = childActor->GetComponentsInChildren<ComponentType>();
		if (!childComponentsInChildren.empty()) {
			res.insert(std::end(res), std::begin(childComponentsInChildren), std::end(childComponentsInChildren));
		}
	}
	return res;
}
void Scene::Serialize(nlohmann::json& json) {
	for (auto& actor : m_Actors) {
		if (actor->IsActive()) {
			actor->Serialize(json);
		}
	}
}

void to_json(nlohmann::json& json, std::shared_ptr<Scene>& scene)
{
	scene->Serialize(json);
}

class SampleComponent :public Component
{
private:
	friend class Component;
	SampleComponent()noexcept :Component(), x{0} {}

	static auto NewComponent() -> std::shared_ptr<SampleComponent> {
		return std::shared_ptr<SampleComponent>(new SampleComponent());
	}
public:
	// {4C7F3F25-A1EE-4101-A425-63BDDC6AC8B8}
	static inline constexpr uuids::uuid typeID = NewUuid(0x4c7f3f25, 0xa1ee, 0x4101, { 0xa4, 0x25, 0x63, 0xbd, 0xdc, 0x6a, 0xc8, 0xb8 });
	using base_type = Component;

	virtual ~SampleComponent()noexcept {}

	virtual void Awake()  override
	{
	}
	virtual void Start()  override
	{
	}
	virtual void Update() override
	{
	}

	int x;
};

int main() {
	auto scene = Scene::New();
	Scene::AddActor(scene, Actor::New("Actor1"));
	Scene::AddActor(scene, Actor::New("Actor2"));
	Scene::AddActor(scene, Actor::New("Actor3"));
	{
		auto actor1 = scene->RootActors()[0];
		auto scene1 = actor1->Scene();

		actor1->AddComponent<SampleComponent>();
		actor1->AddComponent<SampleComponent>();
		actor1->AddComponent<SampleComponent>();

		actor1->name = "actor1";
		actor1->tag  = "tag";
		auto actor2  = scene->RootActors()[1];

		actor1->Transform()->Parent(actor2->Transform());

		auto actor3 = scene->RootActors()[2];
		actor1->Transform()->Parent(actor3->Transform());
		{
			auto transform = actor1->Transform();
			//transform->LocalPosition(glm::vec3(1.0f, 1.0f, 1.0f));
			//transform->LocalRotation(glm::quat());
			//transform->LocalScale(glm::vec3(2.0f));
		}
	}
	return 0;
}
