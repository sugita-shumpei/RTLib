#ifndef RTLIB_SAMPLES_SAMPLE1__H
#define RTLIB_SAMPLES_SAMPLE1__H
#include <RTLib/Core/Vector.h>
#include <assimp/material.h>
#include <assimp/mesh.h>
#include <vector>
#include <optional>

struct MetaData
{
	MetaData(aiMetadata* metadata = nullptr) noexcept
	{
		if (!metadata) {
			return;
		}
		for (auto i = 0; i < metadata->mNumProperties; ++i) {
			auto key = metadata->mKeys[i].C_Str();
			auto val = metadata->mValues[i];
			switch (val.mType)
			{
			case AI_BOOL:
			{
				auto val = false; metadata->Get<bool>(key, val);
				m_KeyMap.insert({ key, m_TypeIndices.size() });
				m_TypeIndices.push_back({ Type::eBool, m_Bools.size() });
				m_Bools.push_back(val);
			}
			break;
			case AI_INT32:
			{
				auto val = RTLib::Int32(0); metadata->Get<RTLib::Int32>(key, val);
				m_KeyMap.insert({ key, m_TypeIndices.size() });
				m_TypeIndices.push_back({ Type::eInt32, m_Int32s.size() });
				m_Int32s.push_back(val);
			}
			break;
			case AI_UINT64:
			{
				auto val = RTLib::UInt64(0); metadata->Get<RTLib::UInt64>(key, val);
				m_KeyMap.insert({ key, m_TypeIndices.size() });
				m_TypeIndices.push_back({ Type::eUInt64, m_UInt64s.size() });
				m_UInt64s.push_back(val);
			}
			break;
			case AI_FLOAT:
			{
				auto val = RTLib::Float32(0); metadata->Get<RTLib::Float32>(key, val);
				m_KeyMap.insert({ key, m_TypeIndices.size() });
				m_TypeIndices.push_back({ Type::eFloat32, m_Float32s.size() });
				m_Float32s.push_back(val);
			}
			break;
			case AI_DOUBLE:
			{
				auto val = RTLib::Float64(0); metadata->Get<RTLib::Float64>(key, val);
				m_KeyMap.insert({ key, m_TypeIndices.size() });
				m_TypeIndices.push_back({ Type::eFloat64, m_Float64s.size() });
				m_Float64s.push_back(val);
			}
			break;
			case AI_AISTRING:
			{
				auto val = aiString(""); metadata->Get<aiString>(key, val);
				m_KeyMap.insert({ key, m_TypeIndices.size() });
				m_TypeIndices.push_back({ Type::eString, m_Strings.size() });
				m_Strings.push_back(val.C_Str());
			}
			break;
			case AI_AIVECTOR3D:
			{
				auto val = aiVector3D(); metadata->Get<aiVector3D>(key, val);
				m_KeyMap.insert({ key, m_TypeIndices.size() });
				m_TypeIndices.push_back({ Type::eVector3, m_Vector3s.size() });
				m_Vector3s.push_back(RTLib::Vector3(val.x, val.y, val.z));
			}
			break;
			case AI_AIMETADATA:
			{
				aiMetadata val; metadata->Get<aiMetadata>(key, val);
				m_KeyMap.insert({ key, m_TypeIndices.size() });
				m_TypeIndices.push_back({ Type::eMetaData, m_Metadatas.size() });
				auto metadata = MetaData(&val);
				m_Metadatas.push_back(std::move(metadata));
			}
			break;
			case AI_META_MAX:
				break;
			case FORCE_32BIT:
				break;
			default:
				break;
			}
		}
	}

	bool has_key(std::string name) const noexcept { return m_KeyMap.count(name) > 0; }

	template<typename T>
	bool get_val(std::string name, T& val) const noexcept {
		if (!has_key(name)) { return false; }
		auto [type, index] = m_TypeIndices[m_KeyMap.at(name)];
		if (type == Type::eBool) {
			if constexpr (std::is_same_v<T, RTLib::Bool>) { val = m_Bools[index]; return true; }
		}
		if (type == Type::eInt32) {
			if constexpr (std::is_same_v<T, RTLib::Int32>) { val = m_Int32s[index]; return true; }
		}
		if (type == Type::eUInt64) {
			if constexpr (std::is_same_v<T, RTLib::UInt64>) { val = m_UInt64s[index]; return true; }
		}
		if (type == Type::eFloat32) {
			if constexpr (std::is_same_v<T, RTLib::Float32>) { val = m_Float32s[index]; return true; }
		}
		if (type == Type::eFloat64) {
			if constexpr (std::is_same_v<T, RTLib::Float64>) { val = m_Float64s[index]; return true; }
		}
		if (type == Type::eString) {
			if constexpr (std::is_same_v<T, RTLib::String>) { val = m_Strings[index]; return true; }
		}
		if (type == Type::eVector3) {
			if constexpr (std::is_same_v<T, RTLib::Vector3>) { val = m_Vector3s[index]; return true; }
		}
		if (type == Type::eMetaData) {
			if constexpr (std::is_same_v<T, MetaData>) { val = m_Metadatas[index]; return true; }
		}
		return false;
	}

	auto get_keys() const noexcept -> std::vector<std::string> {
		std::vector<std::string> keys; keys.reserve(m_KeyMap.size());
		for (auto& [key, idx] : m_KeyMap) { keys.push_back(key); }
		return keys;
	}

	template<typename T>
	auto get_keys_and_vals() const noexcept -> std::vector<std::pair<std::string, T>>
	{
		std::vector<std::pair<std::string, T>> res = {};
		for (auto& [key, idx] : m_KeyMap) {
			T val = {};
			if (get_val<T>(key, val)) { res.push_back({ key,val }); }
		}
		return res;
	}


	enum class Type
	{
		eBool,
		eInt32,
		eUInt64,
		eFloat32,
		eFloat64,
		eString,
		eVector3,
		eMetaData,
	};

	std::unordered_map<std::string, std::size_t> m_KeyMap = {};
	std::vector<std::pair<Type, std::size_t>>    m_TypeIndices = {};
	std::vector<RTLib::UInt8>                   m_Bools = {};
	std::vector<RTLib::Int32>                   m_Int32s = {};
	std::vector<RTLib::UInt64>                  m_UInt64s = {};
	std::vector<RTLib::Float32>                 m_Float32s = {};
	std::vector<RTLib::Float64>                 m_Float64s = {};
	std::vector<RTLib::String>                  m_Strings = {};
	std::vector<RTLib::Vector3>                 m_Vector3s = {};
	std::vector<MetaData>                       m_Metadatas = {};
};
template <typename T>
struct BoneKey
{
	T                   value;
	RTLib::Core::Float32 time;
};
struct Node : public std::enable_shared_from_this<Node>
{
	static auto New(aiNode* rootNode) noexcept -> std::shared_ptr<Node>
	{
		auto root = std::shared_ptr<Node>(new Node(rootNode));
		for (auto i = 0; i < rootNode->mNumChildren; ++i) {
			root->m_Children[i] = New(root, i);
		}
		return root;
	}
	static auto New(std::shared_ptr<Node> parent, RTLib::UInt32 childIdx) noexcept -> std::shared_ptr<Node>
	{

		auto node = std::shared_ptr<Node>(new Node(parent, childIdx));
		for (auto i = 0; i < node->m_Node->mNumChildren; ++i) {
			node->m_Children[i] = New(node, i);
		}
		return node;
	}
	virtual ~Node()noexcept {}

	auto get_local_to_parent_matrix() const noexcept -> RTLib::Matrix4x4
	{
		auto scalMat = glm::scale(RTLib::Matrix4x4(1.0f), m_LocalScaling);
		auto rotaMat = glm::toMat4(m_LocalRotation);
		auto tranMat = glm::translate(RTLib::Matrix4x4(1.0f), m_LocalPosition);
		return tranMat * rotaMat * scalMat;
	}
	auto get_name() const noexcept -> const char* { return m_Node->mName.C_Str(); }

	aiNode* m_Node;
	RTLib::UInt32                      m_Idx;
	std::weak_ptr<Node>                m_Parent;
	std::vector<std::shared_ptr<Node>> m_Children;
	RTLib::Vector3                     m_LocalPosition;
	RTLib::Quat                        m_LocalRotation;
	RTLib::Vector3                     m_LocalScaling;
	RTLib::Bool                        m_Dirty = true;
private:
	Node(aiNode* rootNode) noexcept
	{
		m_Parent = {};
		m_Idx = 0;
		m_Node = rootNode;
		m_Children.resize(rootNode->mNumChildren);

		auto mTransformation = m_Node->mTransformation;
		aiVector3D mPosi; aiQuaternion mRota; aiVector3D mScal;
		mTransformation.Decompose(mScal, mRota, mPosi);

		m_LocalPosition.x = mPosi.x;
		m_LocalPosition.y = mPosi.y;
		m_LocalPosition.z = mPosi.z;

		m_LocalRotation.x = mRota.x;
		m_LocalRotation.y = mRota.y;
		m_LocalRotation.z = mRota.z;
		m_LocalRotation.w = mRota.w;

		m_LocalScaling.x = mScal.x;
		m_LocalScaling.y = mScal.y;
		m_LocalScaling.z = mScal.z;
	}
	Node(std::shared_ptr<Node> parent, RTLib::UInt32 childIdx) noexcept
	{
		m_Parent = parent;
		m_Idx = childIdx;
		m_Node = parent->m_Node->mChildren[childIdx];
		m_Children.resize(m_Node->mNumChildren);

		auto mTransformation = m_Node->mTransformation;
		aiVector3D mPosi; aiQuaternion mRota; aiVector3D mScal;
		mTransformation.Decompose(mScal, mRota, mPosi);

		m_LocalPosition.x = mPosi.x;
		m_LocalPosition.y = mPosi.y;
		m_LocalPosition.z = mPosi.z;

		m_LocalRotation.x = mRota.x;
		m_LocalRotation.y = mRota.y;
		m_LocalRotation.z = mRota.z;
		m_LocalRotation.w = mRota.w;

		m_LocalScaling.x = mScal.x;
		m_LocalScaling.y = mScal.y;
		m_LocalScaling.z = mScal.z;
	}
};
struct NodeTransform
{
	std::weak_ptr<Node>   node = {};
	RTLib::Matrix4x4 transform = RTLib::Matrix4x4(1.0f);
};
struct RootNode
{
	RootNode(aiNode* root_)noexcept {
		rootNode = Node::New(root_);
		nodeMap = { };

		std::stack<aiNode*> keyStack;

		NodeTransform rootTran = { rootNode,rootNode->get_local_to_parent_matrix() };
		nodeMap.insert({ root_,{ rootNode,rootNode->get_local_to_parent_matrix() } });
		keyStack.push(root_);

		while (!keyStack.empty())
		{
			auto key = keyStack.top();
			keyStack.pop();

			auto& parentNodeTran = nodeMap.at(key);
			auto  parentNode = parentNodeTran.node.lock();
			auto& parentTran = parentNodeTran.transform;
			parentNode->m_Dirty = false;

			for (auto& child : parentNode->m_Children)
			{
				auto childNode = child->m_Node;
				auto childLocal = child->get_local_to_parent_matrix();
				nodeMap.insert({ childNode, { child, parentTran * childLocal } });
				keyStack.push(child->m_Node);
			}
		}

	}
	auto find_node_transform(const char* key) -> NodeTransform
	{
		auto node = rootNode->m_Node->FindNode(key);
		if (!node)
		{
			return NodeTransform{};
		}
		else {
			return nodeMap.at(node);
		}
	}
	auto find_node(const char* key) -> std::weak_ptr<Node>
	{
		return find_node_transform(key).node;
	}
	void update_dirty()
	{
		std::stack<aiNode*> keyStack;
		keyStack.push(rootNode->m_Node);

		while (!keyStack.empty())
		{
			auto key = keyStack.top();
			keyStack.pop();
			auto& nodeTran = nodeMap.at(key);
			auto  node = nodeTran.node.lock();
			//親がdirtyなら, 子もdirty
			if (node->m_Dirty)
			{
				for (auto& child : node->m_Children)
				{
					child->m_Dirty = true;
				}
			}
			//伝搬する
			for (auto& child : node->m_Children)
			{
				keyStack.push(child->m_Node);
			}
		}
	}
	void update_transform()
	{
		std::stack<aiNode*> keyStack;

		keyStack.push(rootNode->m_Node);

		while (!keyStack.empty())
		{
			auto key = keyStack.top();
			keyStack.pop();
			auto& nodeTran = nodeMap.at(key);
			auto  node = nodeTran.node.lock();

			auto parent = node->m_Parent.lock();
			auto parentTransform = RTLib::Matrix4x4(1.0f);
			if (parent)
			{
				parentTransform = nodeMap.at(parent->m_Node).transform;
			}
			bool isDirty = node->m_Dirty;
			nodeTran.transform = parentTransform * node->get_local_to_parent_matrix();
			node->m_Dirty = false;

			for (auto& child : node->m_Children)
			{
				keyStack.push(child->m_Node);
			}
		}
	}

	std::shared_ptr<Node>                     rootNode = {};
	std::unordered_map<aiNode*, NodeTransform> nodeMap = {};
};
struct Bone
{
	static inline constexpr RTLib::UInt32 keyIndex_Pre = UINT32_MAX - 1;
	static inline constexpr RTLib::UInt32 keyIndex_Post = UINT32_MAX;

	Bone(aiNodeAnim* nodeAnim, std::shared_ptr<Node> node) noexcept :m_Node{ node }
	{
		m_PreState = nodeAnim->mPreState;
		m_PostState = nodeAnim->mPostState;

		m_KeyPositions.resize(nodeAnim->mNumPositionKeys);
		m_KeyRotations.resize(nodeAnim->mNumRotationKeys);
		m_KeyScalings.resize(nodeAnim->mNumScalingKeys);

		for (auto i = 0; i < nodeAnim->mNumPositionKeys; ++i)
		{
			m_KeyPositions[i].time = nodeAnim->mPositionKeys[i].mTime;
			m_KeyPositions[i].value.x = nodeAnim->mPositionKeys[i].mValue.x;
			m_KeyPositions[i].value.y = nodeAnim->mPositionKeys[i].mValue.y;
			m_KeyPositions[i].value.z = nodeAnim->mPositionKeys[i].mValue.z;

		}
		for (auto i = 0; i < nodeAnim->mNumRotationKeys; ++i)
		{
			m_KeyRotations[i].time = nodeAnim->mRotationKeys[i].mTime;
			m_KeyRotations[i].value.x = nodeAnim->mRotationKeys[i].mValue.x;
			m_KeyRotations[i].value.y = nodeAnim->mRotationKeys[i].mValue.y;
			m_KeyRotations[i].value.z = nodeAnim->mRotationKeys[i].mValue.z;
			m_KeyRotations[i].value.w = nodeAnim->mRotationKeys[i].mValue.w;
		}
		for (auto i = 0; i < nodeAnim->mNumScalingKeys; ++i)
		{
			m_KeyScalings[i].time = nodeAnim->mScalingKeys[i].mTime;
			m_KeyScalings[i].value.x = nodeAnim->mScalingKeys[i].mValue.x;
			m_KeyScalings[i].value.y = nodeAnim->mScalingKeys[i].mValue.y;
			m_KeyScalings[i].value.z = nodeAnim->mScalingKeys[i].mValue.z;

		}

		m_BasePosition = node->m_LocalPosition;
		m_BaseRotation = node->m_LocalRotation;
		m_BaseScaling = node->m_LocalScaling;
	}
	~Bone() noexcept {}

	auto get_position(RTLib::Float32 time) const noexcept -> RTLib::Vector3 { return get_value(time, m_PreState, m_PostState, m_KeyPositions, m_BasePosition); }
	auto get_rotation(RTLib::Float32 time) const noexcept -> RTLib::Quat { return get_value(time, m_PreState, m_PostState, m_KeyRotations, m_BaseRotation); }
	auto get_scaling(RTLib::Float32 time) const noexcept -> RTLib::Vector3 { return get_value(time, m_PreState, m_PostState, m_KeyScalings, m_BaseScaling); }

	void update(RTLib::Float32 time) noexcept
	{
		auto node = m_Node.lock();
		assert(node);
		auto curLocalPosi = node->m_LocalPosition;
		auto curLocalRota = node->m_LocalRotation;
		auto curLocalScal = node->m_LocalScaling;

		auto nxtLocalPosi = get_position(time);
		auto nxtLocalRota = get_rotation(time);
		auto nxtLocalScal = get_scaling(time);

		bool nxtDirty = false;
		if (curLocalPosi != nxtLocalPosi) {
			node->m_LocalPosition = nxtLocalPosi;
			nxtDirty = true;
		}
		if (curLocalRota != nxtLocalRota) {
			node->m_LocalRotation = nxtLocalRota;
			nxtDirty = true;
		}
		if (curLocalScal != nxtLocalScal) {
			node->m_LocalScaling = nxtLocalScal;
			nxtDirty = true;
		}

		node->m_Dirty = nxtDirty;
	}

	auto get_node() const noexcept-> std::shared_ptr<Node> { return m_Node.lock(); }
private:

	template<typename T>
	static auto get_interpolated_value(RTLib::Float32 time, RTLib::UInt32 begIdx, const std::vector<BoneKey<T>>& keys) noexcept -> T
	{
		auto i0 = begIdx;
		auto i1 = begIdx + 1;
		const auto& k0 = keys.at(i0);
		const auto& k1 = keys.at(i1);
		const auto& t0 = k0.time;
		const auto& t1 = k1.time;
		const auto& v0 = k0.value;
		const auto& v1 = k1.value;
		auto alpha = (time - t0) / (t1 - t0);
		if constexpr (std::is_same_v<T, RTLib::Quat>) {
			return glm::normalize(glm::slerp(v0, v1, alpha));
		}
		else {
			return glm::lerp(v0, v1, alpha);
		}
	}
	template<typename T>
	static auto get_nearest_key_index(RTLib::Float32 time, const std::vector<BoneKey<T>>& keys) noexcept -> RTLib::UInt32
	{
		//最初の時刻以前なら
		if (keys.front().time >= time) { return keyIndex_Pre; }
		//最後の時刻以降なら
		if (keys.back().time <= time) { return keyIndex_Post; }
		for (size_t i = 0; i < keys.size() - 1; ++i) {
			if (keys[i + 1].time > time) {
				return i;
			}
		}
		assert(0);
	}
	template<typename T>
	static auto get_value_without_key(RTLib::Float32 time, aiAnimBehaviour state, const std::vector<BoneKey<T>>& keys, const T& baseVal, const T& constVal) noexcept -> T
	{
		if (state == aiAnimBehaviour_DEFAULT) { return baseVal; }
		if (state == aiAnimBehaviour_CONSTANT) { return constVal; }
		if (state == aiAnimBehaviour_REPEAT) {
			auto t_beg = keys.front().time;
			auto t_end = keys.back().time;
			auto t_del = std::abs(t_end - t_beg);
			auto t_near = 0.0f;
			if (time < t_beg)
			{
				t_near = std::fmod(time - t_beg, t_del) + t_beg;
			}
			else {
				t_near = std::fmod(time - t_end, t_del) + t_beg;
			}
			auto i0 = get_nearest_key_index(t_near, keys);
			return get_interpolated_value(time, i0, keys);
		}
		if (state == aiAnimBehaviour_LINEAR)
		{
			auto t_beg = keys.front().time;
			auto t_end = keys.back().time;
			RTLib::UInt32 i0 = 0;
			if (time < t_beg)
			{
				i0 = 0;
			}
			else {
				i0 = keys.size() - 2;
			}
			return get_interpolated_value(time, i0, keys);
		}
		return baseVal;
	}
	template<typename T>
	static auto get_value(RTLib::Float32 time, aiAnimBehaviour preState, aiAnimBehaviour postState, const std::vector<BoneKey<T>>& keys, const T& baseVal) noexcept -> T
	{
		auto i0 = get_nearest_key_index(time, keys);
		if (i0 == keyIndex_Pre) { return get_value_without_key(time, preState, keys, keys.front().value, keys.front().value); }
		if (i0 == keyIndex_Post) { return get_value_without_key(time, postState, keys, keys.back().value, keys.back().value); }
		return get_interpolated_value(time, i0, keys);
	}
private:
	std::weak_ptr<Node> m_Node;

	aiAnimBehaviour m_PreState;
	aiAnimBehaviour m_PostState;

	std::vector<BoneKey<RTLib::Core::Vector3>> m_KeyPositions;
	std::vector<BoneKey<RTLib::Core::Quat>>    m_KeyRotations;
	std::vector<BoneKey<RTLib::Core::Vector3>> m_KeyScalings;

	RTLib::Core::Vector3 m_BasePosition;
	RTLib::Core::Quat    m_BaseRotation;
	RTLib::Core::Vector3 m_BaseScaling;

};
struct Animation
{
	Animation(aiAnimation* anim, const std::shared_ptr<RootNode> rootNode)
		:m_RootNode{ rootNode }
	{
		m_Duration = anim->mDuration;
		m_TickPerSecond = anim->mTicksPerSecond;
		m_Bones.reserve(anim->mNumChannels);
		for (auto i = 0; i < anim->mNumChannels; ++i)
		{
			auto name = anim->mChannels[i]->mNodeName;
			auto node = m_RootNode->find_node(name.C_Str()).lock();
			assert(node);
			m_Bones.push_back(std::unique_ptr<Bone>(new Bone(anim->mChannels[i], node)));
		}
	}

	void update_frames(RTLib::Float32 timeInSecond) {
		update_bones(timeInSecond);
		update_nodes();
	}
	void update_bones(RTLib::Float32 timeInSecond) {
		auto timeInTick = timeInSecond * m_TickPerSecond;
		auto timeInAnim = std::fmod(timeInTick, m_Duration);
		for (auto& bone : m_Bones)
		{
			bone->update(timeInAnim);
		}
	}
	void update_nodes() {
		m_RootNode->update_dirty();
		m_RootNode->update_transform();
	}

	std::shared_ptr<RootNode>  m_RootNode;
	RTLib::Float32 m_Duration;
	RTLib::Float32 m_TickPerSecond;
	std::vector<std::unique_ptr<Bone>> m_Bones;
};
struct Texture
{
	aiReturn Load(aiMaterial* pMaterial, aiTextureType type, unsigned int index)
	{
		aiString         path_ = aiString();
		aiTextureMapping mapping_ = _aiTextureMapping_Force32Bit;
		unsigned int     uvIndex_ = UINT_MAX;
		ai_real          blend_ = FLT_MAX;
		aiTextureOp      op_ = _aiTextureOp_Force32Bit;
		aiTextureMapMode mapmode_ = _aiTextureMapMode_Force32Bit;
		auto res = pMaterial->GetTexture(type, index, &path_, &mapping_, &uvIndex_, &blend_, &op_, &mapmode_);
		if (path_.C_Str() != std::string("")) {
			path = std::string(path_.C_Str());
		}
		else {
			path = "";
		}
		if (mapping_ != _aiTextureMapping_Force32Bit)
		{
			mapping = mapping_;
		}
		else {
			mapping = std::nullopt;
		}
		if (uvIndex_ != UINT32_MAX) {
			uvIndex = uvIndex_;
		}
		else {
			uvIndex = std::nullopt;
		}
		if (blend_ != FLT_MAX) {
			blend = blend_;
		}
		else {
			blend = std::nullopt;
		}
		if (op_ != _aiTextureOp_Force32Bit) {
			op = op_;
		}
		else {
			op = std::nullopt;
		}
		if (mapmode_ != _aiTextureMapMode_Force32Bit) {
			mapMode = mapmode_;
		}
		else {
			mapMode = std::nullopt;
		}
		return res;
	}

	std::string path = "";
	std::optional<aiTextureMapping> mapping = std::nullopt;
	std::optional<unsigned int>     uvIndex = std::nullopt;
	std::optional<float>            blend = std::nullopt;
	std::optional<aiTextureOp>      op = std::nullopt;
	std::optional<aiTextureMapMode> mapMode = std::nullopt;

};
struct Material
{

};

#endif
