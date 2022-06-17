
#include <RTLib/Core/BinaryReader.h>
#include <filesystem>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <iostream>
#include <fstream>
void RTLib::Core::MeshGroup::SetSharedResource(const MeshSharedResourcePtr& res) noexcept
{
    this->m_SharedResource = res;
}

auto RTLib::Core::MeshGroup::GetSharedResource() const noexcept -> MeshSharedResourcePtr
{
    return this->m_SharedResource;
}

void RTLib::Core::MeshGroup::SetUniqueResource(const std::string& name, const MeshUniqueResourcePtr& res) noexcept
{
    this->m_UniqueResources[name] = res;
}

auto RTLib::Core::MeshGroup::GetUniqueResource(const std::string& name) const -> MeshUniqueResourcePtr
{
    return this->m_UniqueResources.at(name);
}

auto RTLib::Core::MeshGroup::GetUniqueResources() const noexcept -> const std::unordered_map<std::string, MeshUniqueResourcePtr>&
{
    return m_UniqueResources;
}

auto RTLib::Core::MeshGroup::GetUniqueNames() const noexcept -> std::vector<std::string>
{
    auto uniqueNames = std::vector<std::string>();
    uniqueNames.reserve(m_UniqueResources.size());
    for (auto& [name, res] : m_UniqueResources) {
        uniqueNames.push_back(name);
    }
    return uniqueNames;
}

auto RTLib::Core::MeshGroup::LoadMesh(const std::string& name) const -> MeshPtr
{
    auto mesh = std::make_shared<Mesh>();
    mesh->m_Name = name;
    mesh->m_SharedResource = this->GetSharedResource();
    mesh->m_UniqueResource = this->GetUniqueResource(name);
    return mesh;
}

bool RTLib::Core::MeshGroup::RemoveMesh(const std::string& name)
{
    if (m_UniqueResources.count(name) > 0) {
        m_UniqueResources.erase(name);
        return true;
    }
    return false;
}

void RTLib::Core::Mesh::SetSharedResource(const MeshSharedResourcePtr& res) noexcept
{
    m_SharedResource = res;
}

auto RTLib::Core::Mesh::GetSharedResource() const noexcept -> MeshSharedResourcePtr
{
    return m_SharedResource;
}

void RTLib::Core::Mesh::SetUniqueResource(const MeshUniqueResourcePtr& res) noexcept
{
    m_UniqueResource = res;
}

void RTLib::Core::Mesh::SetUniqueResource(const std::string& name, const MeshUniqueResourcePtr& res) noexcept
{
    m_Name = name;
    m_UniqueResource = res;
}

auto RTLib::Core::Mesh::GetUniqueResource() const -> MeshUniqueResourcePtr
{
    return m_UniqueResource;
}
void RTLib::Core::ObjModel::SplitLight()
{
    {
        auto splitMeshGroup = RTLib::Core::MeshGroup::New();
        splitMeshGroup->SetSharedResource(meshGroup->GetSharedResource());
        for (auto& [name, uniqueResource] : meshGroup->GetUniqueResources())
        {
            std::unordered_set<bool> materialEmitSet = {};
            for (auto& matIdx : uniqueResource->materials) {
                auto& material = materials[matIdx];
                auto emitCol = material.GetFloat3As<std::array<float,3>>("emitCol");
                if (emitCol[0] + emitCol[1] + emitCol[2] > 0.0f) {
                    materialEmitSet.insert(true);
                }
                else {
                    materialEmitSet.insert(false);
                }
            }
            if (materialEmitSet.size() == 2) {
                splitMeshGroup->SetUniqueResource(name, uniqueResource);
            }
            else if (materialEmitSet.count(true) > 0) {
                uniqueResource->variables.SetBool("hasLight", true);
            }
            else
            {
                uniqueResource->variables.SetBool("hasLight", false);
            }
        }
        //split mesh
        for (auto& [name, uniqueResource] : splitMeshGroup->GetUniqueResources())
        {
            meshGroup->RemoveMesh(name);
            auto newSurfMatIndMap = std::unordered_map<uint32_t, uint32_t>();
            auto newEmitMatIndMap = std::unordered_map<uint32_t, uint32_t>();
            auto newSurfUniqueResource = RTLib::Core::MeshUniqueResource::New();
            auto newEmitUniqueResource = RTLib::Core::MeshUniqueResource::New();
            newSurfUniqueResource->name = uniqueResource->name + ".Surface";
            newSurfUniqueResource->variables.SetBool("hasLight", false);
            newEmitUniqueResource->name = uniqueResource->name + ".Emission";
            newEmitUniqueResource->variables.SetBool("hasLight", true);
            for (auto i = 0; i < uniqueResource->matIndBuffer.size(); ++i) {
                auto  matIndex = uniqueResource->matIndBuffer[i];
                auto& material = materials[uniqueResource->materials[matIndex]];
                auto emitCol = material.GetFloat3As<std::array<float,3>> ("emitCol");
                //emitCol = 0.0f -> Surface
                if (emitCol[0] + emitCol[1] + emitCol[2] > 0.0f) {
                    if (newEmitMatIndMap.count(matIndex) == 0)
                    {
                        uint32_t newMatIndex = newEmitUniqueResource->materials.size();
                        newEmitUniqueResource->materials.push_back(uniqueResource->materials[matIndex]);
                        newEmitMatIndMap[matIndex] = newMatIndex;
                    }
                    newEmitUniqueResource->matIndBuffer.push_back(newEmitMatIndMap[matIndex]);
                    newEmitUniqueResource->triIndBuffer.push_back(uniqueResource->triIndBuffer[i]);
                }
                else {
                    if (newSurfMatIndMap.count(matIndex) == 0)
                    {
                        uint32_t newMatIndex = newSurfUniqueResource->materials.size();
                        newSurfUniqueResource->materials.push_back(uniqueResource->materials[matIndex]);
                        newSurfMatIndMap[matIndex] = newMatIndex;
                    }
                    newSurfUniqueResource->matIndBuffer.push_back(newSurfMatIndMap[matIndex]);
                    newSurfUniqueResource->triIndBuffer.push_back(uniqueResource->triIndBuffer[i]);
                }
            }
            meshGroup->SetUniqueResource(newSurfUniqueResource->name, newSurfUniqueResource);
            meshGroup->SetUniqueResource(newEmitUniqueResource->name, newEmitUniqueResource);
        }
    }
    {
        auto splitMeshGroup = RTLib::Core::MeshGroup::New();
        splitMeshGroup->SetSharedResource(meshGroup->GetSharedResource());
        for (auto& [name, uniqueResource] : meshGroup->GetUniqueResources())
        {
            if (uniqueResource->variables.GetBool("hasLight") && uniqueResource->materials.size() > 1) {
                splitMeshGroup->SetUniqueResource(name, uniqueResource);
            }
        }
        //split mesh
        for (auto& [name, uniqueResource] : splitMeshGroup->GetUniqueResources())
        {
            meshGroup->RemoveMesh(name);
            auto newEmitUniqueResources = std::vector< RTLib::Core::MeshUniqueResourcePtr>(uniqueResource->materials.size());
            for (auto i = 0; i < uniqueResource->materials.size(); ++i)
            {
                newEmitUniqueResources[i] = RTLib::Core::MeshUniqueResource::New();
                newEmitUniqueResources[i]->name = uniqueResource->name + "." + std::to_string(i);
                newEmitUniqueResources[i]->materials.push_back(uniqueResource->materials[i]);
                newEmitUniqueResources[i]->variables.SetBool("hasLight", true);
            }
            for (auto i = 0; i < uniqueResource->matIndBuffer.size(); ++i) {
                auto  matIndex = uniqueResource->matIndBuffer[i];
                newEmitUniqueResources[matIndex]->matIndBuffer.push_back(0);
                newEmitUniqueResources[matIndex]->triIndBuffer.push_back(uniqueResource->triIndBuffer[i]);
            }
            for (auto i = 0; i < uniqueResource->materials.size(); ++i)
            {
                meshGroup->SetUniqueResource(newEmitUniqueResources[i]->name, newEmitUniqueResources[i]);
            }
        }
    }
}

void RTLib::Core::ObjModel::InitAABB()
{
    struct    AABB {
        std::array<float, 3>  min = std::array<float, 3>{FLT_MAX, FLT_MAX, FLT_MAX};
        std::array<float, 3>  max = std::array<float, 3>{ -FLT_MAX, -FLT_MAX, -FLT_MAX};
    public:
        AABB()noexcept {}
        AABB(const AABB& aabb)noexcept = default;
        AABB& operator=(const AABB& aabb)noexcept = default;
        AABB(const std::array<float, 3>& min, const std::array<float, 3>& max)noexcept :min{ min }, max{ max }{}
        AABB(const std::vector<std::array<float, 3>>& vertices)noexcept :AABB() {
            for (auto& vertex : vertices) {
                this->Update(vertex);
            }
        }
        auto GetArea()const noexcept -> float {
            std::array<float, 3> range = {
                max[0] - min[0],
                max[1] - min[1],
                max[2] - min[2]
            };
            return 2.0f * (range[0]*range[1]+range[1]*range[2]+range[2]*range[0]);
        }
        void Update(const  std::array<float, 3>& vertex)noexcept {
            for (size_t i = 0; i < 3; ++i) {
                min[i] = std::min(min[i], vertex[i]);
                max[i] = std::max(max[i], vertex[i]);
            }
        }
    };
    for (auto& [name, uniqueResource] : meshGroup->GetUniqueResources())
    {
        AABB aabb;
        for (auto& triIdx : uniqueResource->triIndBuffer) {
            aabb.Update(meshGroup->GetSharedResource()->vertexBuffer[triIdx[0]]);
            aabb.Update(meshGroup->GetSharedResource()->vertexBuffer[triIdx[1]]);
            aabb.Update(meshGroup->GetSharedResource()->vertexBuffer[triIdx[2]]);
        }
        uniqueResource->variables.SetFloat3From("aabb.min", aabb.min);
        uniqueResource->variables.SetFloat3From("aabb.max", aabb.max);
    }
    //AABB
    AABB aabb;
    for (auto& vertex : meshGroup->GetSharedResource()->vertexBuffer) {
        aabb.Update(vertex);
    }
    meshGroup->GetSharedResource()->variables.SetFloat3From("aabb.min", aabb.min);
    meshGroup->GetSharedResource()->variables.SetFloat3From("aabb.max", aabb.max);
}

bool RTLib::Core::ObjModelAssetManager::LoadAsset(const std::string& keyName, const std::string& objPath, bool useCache) 
{
    if (useCache) {
        if (LoadAssetCache(keyName)) {
            return true;
        }
    }
    auto mtlBaseDir = std::filesystem::canonical(std::filesystem::path(objPath).parent_path());
    tinyobj::ObjReaderConfig readerConfig = {};
    readerConfig.mtl_search_path = mtlBaseDir.string() + "\\";

    tinyobj::ObjReader reader = {};
    if (!reader.ParseFromFile(objPath, readerConfig)) {
        if (!reader.Error().empty()) {
            std::cerr << "TinyObjReader: " << reader.Error();
        }
        return false;
    }
    auto meshGroup = std::make_shared<MeshGroup>();
    auto phongMaterials = std::vector<VariableMap>();
    if (!reader.Warning().empty()) {
        std::cout << "TinyObjReader: " << reader.Warning();
    }
    auto& attrib = reader.GetAttrib();
    auto& shapes = reader.GetShapes();
    auto& materials = reader.GetMaterials();
    {
        meshGroup->SetSharedResource(std::make_shared<RTLib::Core::MeshSharedResource>());
        auto& vertexBuffer = meshGroup->GetSharedResource()->vertexBuffer;
        auto& texCrdBuffer = meshGroup->GetSharedResource()->texCrdBuffer;
        auto& normalBuffer = meshGroup->GetSharedResource()->normalBuffer;

        struct MyHash
        {
            MyHash()noexcept {}
            MyHash(const MyHash&)noexcept = default;
            MyHash(MyHash&&)noexcept = default;
            ~MyHash()noexcept {}
            MyHash& operator=(const MyHash&)noexcept = default;
            MyHash& operator=(MyHash&&)noexcept = default;
            size_t operator()(tinyobj::index_t key)const
            {
                size_t vertexHash = std::hash<int>()(key.vertex_index)   & 0x3FFFFF;
                size_t normalHash = std::hash<int>()(key.normal_index)   & 0x1FFFFF;
                size_t texCrdHash = std::hash<int>()(key.texcoord_index) & 0x1FFFFF;
                return vertexHash + (normalHash << 22) + (texCrdHash << 43);
            }
        };
        struct MyEqualTo
        {
            using first_argument_type = tinyobj::index_t;
            using second_argument_type = tinyobj::index_t;
            using result_type = bool;
            constexpr bool operator()(const tinyobj::index_t& x, const tinyobj::index_t& y)const
            {
                return (x.vertex_index == y.vertex_index) && (x.texcoord_index == y.texcoord_index) && (x.normal_index == y.normal_index);
            }
        };

        std::vector< tinyobj::index_t> indices = {};
        std::unordered_map<tinyobj::index_t, size_t, MyHash, MyEqualTo> indicesMap = {};
        for (size_t i = 0; i < shapes.size(); ++i) {
            for (size_t j = 0; j < shapes[i].mesh.num_face_vertices.size(); ++j) {
                for (size_t k = 0; k < 3; ++k) {
                    //tinyobj::idx
                    tinyobj::index_t idx = shapes[i].mesh.indices[3 * j + k];
                    if (indicesMap.count(idx) == 0) {
                        size_t indicesCount = std::size(indices);
                        indicesMap[idx] = indicesCount;
                        indices.push_back(idx);
                    }
                }
            }
        }
        std::cout << "VertexBuffer: " << attrib.vertices.size() / 3 << "->" << indices.size() << std::endl;
        std::cout << "NormalBuffer: " << attrib.normals.size() / 3 << "->" << indices.size() << std::endl;
        std::cout << "TexCrdBuffer: " << attrib.texcoords.size() / 2 << "->" << indices.size() << std::endl;
        vertexBuffer.resize(indices.size());
        texCrdBuffer.resize(indices.size());
        normalBuffer.resize(indices.size());

        for (size_t i = 0; i < indices.size(); ++i) {
            tinyobj::index_t idx = indices[i];
            vertexBuffer[i] = std::array<float,3>{
                attrib.vertices[3 * idx.vertex_index + 0],
                attrib.vertices[3 * idx.vertex_index + 1],
                attrib.vertices[3 * idx.vertex_index + 2] };
            if (idx.normal_index >= 0) {
                normalBuffer[i] = std::array<float, 3>{
                    attrib.normals[3 * idx.normal_index + 0],
                    attrib.normals[3 * idx.normal_index + 1],
                    attrib.normals[3 * idx.normal_index + 2]};
            }
            else {
                normalBuffer[i] = std::array<float, 3>{0.0f, 1.0f, 0.0f};
            }
            if (idx.texcoord_index >= 0) {
                texCrdBuffer[i] = std::array<float, 2>{
                    attrib.texcoords[2 * idx.texcoord_index + 0],
                    attrib.texcoords[2 * idx.texcoord_index + 1]
                };
            }
            else {
                texCrdBuffer[i] = std::array<float, 2>{0.5f, 0.5f};
            }
        }

        std::unordered_map<std::size_t, std::size_t> texCrdMap = {};
        for (size_t i = 0; i < shapes.size(); ++i) {
            std::unordered_map<uint32_t, uint32_t> tmpMaterials = {};
            auto uniqueResource = std::make_shared<RTLib::Core::MeshUniqueResource>();
            uniqueResource->name = shapes[i].name;
            uniqueResource->triIndBuffer.resize(shapes[i].mesh.num_face_vertices.size());
            for (size_t j = 0; j < shapes[i].mesh.num_face_vertices.size(); ++j) {
                uint32_t idx0 = indicesMap.at(shapes[i].mesh.indices[3 * j + 0]);
                uint32_t idx1 = indicesMap.at(shapes[i].mesh.indices[3 * j + 1]);
                uint32_t idx2 = indicesMap.at(shapes[i].mesh.indices[3 * j + 2]);
                uniqueResource->triIndBuffer[j] = std::array<uint32_t, 3>{idx0, idx1, idx2};
            }
            uniqueResource->matIndBuffer.resize(shapes[i].mesh.material_ids.size());
            for (size_t j = 0; j < shapes[i].mesh.material_ids.size(); ++j) {
                if (tmpMaterials.count(shapes[i].mesh.material_ids[j]) != 0) {
                    uniqueResource->matIndBuffer[j] = tmpMaterials.at(shapes[i].mesh.material_ids[j]);
                }
                else {
                    int newValue = tmpMaterials.size();
                    tmpMaterials[shapes[i].mesh.material_ids[j]] = newValue;
                    uniqueResource->matIndBuffer[j] = newValue;
                }
            }
            uniqueResource->materials.resize(tmpMaterials.size());
            for (auto& [Ind, RelInd] : tmpMaterials) {
                uniqueResource->materials[RelInd] = Ind;
            }
            meshGroup->SetUniqueResource(shapes[i].name, uniqueResource);
        }
    }
    {
        phongMaterials.resize(materials.size());
        for (size_t i = 0; i < phongMaterials.size(); ++i) {
            phongMaterials[i].SetString("name", materials[i].name);
            phongMaterials[i].SetUInt32("illum", materials[i].illum);
            phongMaterials[i].SetFloat3("diffCol",
                { materials[i].diffuse[0],
                  materials[i].diffuse[1],
                  materials[i].diffuse[2]
                });
            phongMaterials[i].SetFloat3("specCol",
                { materials[i].specular[0],
                  materials[i].specular[1],
                  materials[i].specular[2]
                });
            phongMaterials[i].SetFloat3("tranCol",
                { materials[i].transmittance[0],
                  materials[i].transmittance[1] ,
                  materials[i].transmittance[2]
                });
            phongMaterials[i].SetFloat3("emitCol",
                { materials[i].emission[0],
                  materials[i].emission[1] ,
                  materials[i].emission[2]
                });

            if (!materials[i].diffuse_texname.empty()) {
                phongMaterials[i].SetString("diffTex", mtlBaseDir.string() + "\\" + materials[i].diffuse_texname);
            }
            else {
                phongMaterials[i].SetString("diffTex", "");
            }
            if (!materials[i].specular_texname.empty()) {
                phongMaterials[i].SetString("specTex", mtlBaseDir.string() + "\\" + materials[i].specular_texname);
            }
            else {
                phongMaterials[i].SetString("specTex", "");
            }
            if (!materials[i].emissive_texname.empty()) {
                phongMaterials[i].SetString("emitTex", mtlBaseDir.string() + "\\" + materials[i].emissive_texname);
            }
            else {
                phongMaterials[i].SetString("emitTex", "");
            }
            if (!materials[i].specular_highlight_texname.empty()) {
                phongMaterials[i].SetString("shinTex", mtlBaseDir.string() + "\\" + materials[i].specular_highlight_texname);
            }
            else {
                phongMaterials[i].SetString("shinTex", "");
            }
            phongMaterials[i].SetFloat1("shinness", materials[i].shininess);
            phongMaterials[i].SetFloat1("refrIndx", materials[i].ior);
        }
    }
    {
        auto splitMeshGroup = RTLib::Core::MeshGroup::New();
        splitMeshGroup->SetSharedResource(meshGroup->GetSharedResource());
        for (auto& [name, uniqueResource] : meshGroup->GetUniqueResources())
        {
            std::unordered_set<bool> materialEmitSet = {};
            for (auto& matIdx : uniqueResource->materials) {
                auto& material = phongMaterials[matIdx];
                auto emitCol = material.GetFloat3As<std::array<float,3>>("emitCol");
                if (emitCol[0] + emitCol[1] + emitCol[2] > 0.0f) {
                    materialEmitSet.insert(true);
                }
                else {
                    materialEmitSet.insert(false);
                }
            }
            if (materialEmitSet.size() == 2) {
                splitMeshGroup->SetUniqueResource(name, uniqueResource);
            }
            else if (materialEmitSet.count(true) > 0) {
                uniqueResource->variables.SetBool("hasLight", true);
            }
            else
            {
                uniqueResource->variables.SetBool("hasLight", false);
            }
        }
        //split mesh
        for (auto& [name, uniqueResource] : splitMeshGroup->GetUniqueResources())
        {
            meshGroup->RemoveMesh(name);
            auto newSurfMatIndMap = std::unordered_map<uint32_t, uint32_t>();
            auto newEmitMatIndMap = std::unordered_map<uint32_t, uint32_t>();
            auto newSurfUniqueResource = RTLib::Core::MeshUniqueResource::New();
            auto newEmitUniqueResource = RTLib::Core::MeshUniqueResource::New();
            newSurfUniqueResource->name = uniqueResource->name + ".Surface";
            newSurfUniqueResource->variables.SetBool("hasLight", false);
            newEmitUniqueResource->name = uniqueResource->name + ".Emission";
            newEmitUniqueResource->variables.SetBool("hasLight", true);
            for (auto i = 0; i < uniqueResource->matIndBuffer.size(); ++i) {
                auto  matIndex = uniqueResource->matIndBuffer[i];
                auto& material = phongMaterials[uniqueResource->materials[matIndex]];
                auto emitCol = material.GetFloat3As<std::array<float,3>>("emitCol");
                //emitCol = 0.0f -> Surface
                if (emitCol[0]+ emitCol[1] + emitCol[2] > 0.0f) {
                    if (newEmitMatIndMap.count(matIndex) == 0)
                    {
                        uint32_t newMatIndex = newEmitUniqueResource->materials.size();
                        newEmitUniqueResource->materials.push_back(uniqueResource->materials[matIndex]);
                        newEmitMatIndMap[matIndex] = newMatIndex;
                    }
                    newEmitUniqueResource->matIndBuffer.push_back(newEmitMatIndMap[matIndex]);
                    newEmitUniqueResource->triIndBuffer.push_back(uniqueResource->triIndBuffer[i]);
                }
                else {
                    if (newSurfMatIndMap.count(matIndex) == 0)
                    {
                        uint32_t newMatIndex = newSurfUniqueResource->materials.size();
                        newSurfUniqueResource->materials.push_back(uniqueResource->materials[matIndex]);
                        newSurfMatIndMap[matIndex] = newMatIndex;
                    }
                    newSurfUniqueResource->matIndBuffer.push_back(newSurfMatIndMap[matIndex]);
                    newSurfUniqueResource->triIndBuffer.push_back(uniqueResource->triIndBuffer[i]);
                }
            }
            meshGroup->SetUniqueResource(newSurfUniqueResource->name, newSurfUniqueResource);
            meshGroup->SetUniqueResource(newEmitUniqueResource->name, newEmitUniqueResource);
        }
    }
    {
        auto splitMeshGroup = RTLib::Core::MeshGroup::New();
        splitMeshGroup->SetSharedResource(meshGroup->GetSharedResource());
        for (auto& [name, uniqueResource] : meshGroup->GetUniqueResources())
        {
            if (uniqueResource->variables.GetBool("hasLight") && uniqueResource->materials.size() > 1) {
                splitMeshGroup->SetUniqueResource(name, uniqueResource);
            }
        }
        //split mesh
        for (auto& [name, uniqueResource] : splitMeshGroup->GetUniqueResources())
        {
            meshGroup->RemoveMesh(name);
            auto newEmitUniqueResources = std::vector< RTLib::Core::MeshUniqueResourcePtr>(uniqueResource->materials.size());
            for (auto i = 0; i < uniqueResource->materials.size(); ++i)
            {
                newEmitUniqueResources[i] = RTLib::Core::MeshUniqueResource::New();
                newEmitUniqueResources[i]->name = uniqueResource->name + "." + std::to_string(i);
                newEmitUniqueResources[i]->materials.push_back(uniqueResource->materials[i]);
                newEmitUniqueResources[i]->variables.SetBool("hasLight", true);
            }
            for (auto i = 0; i < uniqueResource->matIndBuffer.size(); ++i) {
                auto  matIndex = uniqueResource->matIndBuffer[i];
                newEmitUniqueResources[matIndex]->matIndBuffer.push_back(0);
                newEmitUniqueResources[matIndex]->triIndBuffer.push_back(uniqueResource->triIndBuffer[i]);
            }
            for (auto i = 0; i < uniqueResource->materials.size(); ++i)
            {
                meshGroup->SetUniqueResource(newEmitUniqueResources[i]->name, newEmitUniqueResources[i]);
            }
        }
    }
    meshGroup->GetSharedResource()->variables.SetString("path", objPath);
    m_ObjModels[keyName] = { meshGroup,std::move(phongMaterials) };
    m_ObjModels[keyName].SplitLight();
    m_ObjModels[keyName].InitAABB();
    SaveAssetCache(keyName);
    return true;
}


void RTLib::Core::ObjModelAssetManager::FreeAsset(const std::string& keyName)
{
    m_ObjModels.erase(keyName);
}

auto RTLib::Core::ObjModelAssetManager::GetAsset(const std::string& keyName) const -> const ObjModel&
{
    // TODO: return �X�e�[�g�����g�������ɑ}�����܂�
    return m_ObjModels.at(keyName);
}

auto RTLib::Core::ObjModelAssetManager::GetAsset(const std::string& keyName) -> ObjModel&
{
    // TODO: return �X�e�[�g�����g�������ɑ}�����܂�
    return m_ObjModels.at(keyName);
}

auto RTLib::Core::ObjModelAssetManager::PopAsset(const std::string& keyName) -> ObjModel
{
    if (HasAsset(keyName)) {
        auto objModel = m_ObjModels.at(keyName);
        m_ObjModels.erase(keyName);
        return std::move(objModel);
    }
    return {};
}

bool RTLib::Core::ObjModelAssetManager::HasAsset(const std::string& keyName) const noexcept
{
    return m_ObjModels.count(keyName) != 0;
}

auto RTLib::Core::ObjModelAssetManager::GetAssets() const -> const std::unordered_map<std::string, ObjModel>&
{
    // TODO: return �X�e�[�g�����g�������ɑ}�����܂�
    return m_ObjModels;
}

auto RTLib::Core::ObjModelAssetManager::GetAssets()       ->       std::unordered_map<std::string, ObjModel>&
{
    // TODO: return �X�e�[�g�����g�������ɑ}�����܂�
    return m_ObjModels;
}

void RTLib::Core::ObjModelAssetManager::Reset()
{
    m_ObjModels.clear();
}

RTLib::Core::ObjModelAssetManager::~ObjModelAssetManager()
{
}
bool RTLib::Core::ObjModelAssetManager::LoadAssetCache(const std::string& keyName) noexcept
{
    if (m_CacheDir.empty()) { return false; }
    auto cacheRootDir  = m_CacheDir   + "\\" + keyName;
    auto cacheJsonPath = cacheRootDir + "\\" + keyName + ".json";
    auto cacheJsonFile = std::ifstream(cacheJsonPath, std::ios::binary);
    if (!cacheJsonFile.is_open()) {
        return false;
    }
    std::cout << cacheJsonPath << std::endl;
    auto cacheJsonData = nlohmann::json();
    try
    {
        cacheJsonData = nlohmann::json::parse(cacheJsonFile);
    }
    catch (nlohmann::json::parse_error& ex)
    {
        std::cerr << "parse error at byte " << ex.byte << std::endl;
    }
    for (auto& elem : cacheJsonData.items()) {
        std::cout << elem.key() << std::endl;
    }
    auto& meshGroupJson = cacheJsonData.at("MeshGroup");
    auto& materialsJson = cacheJsonData.at("Materials");
    auto materialIdxMap = std::unordered_map<std::string, uint32_t>();
    std::vector<VariableMap> materials;
    materials.reserve(materialsJson.size());
    for (auto& materialElem : materialsJson.items()) {
        materialIdxMap[materialElem.key()]  = materials.size();
        materials.push_back(materialElem.value());
    }
    auto    meshGroup = RTLib::Core::MeshGroup::New();

    auto LoadBuffer = [](const nlohmann::json& json,auto& buffer)->void {
        auto strideInBytes = sizeof(buffer[0]);
        auto sizeInBytes   = json.at("SizeInBytes").get<uint32_t>();
        if (json.count("Path")==0) {
            if (std::is_same_v<std::array<float, 3>, std::remove_const_t<std::remove_reference_t<decltype(buffer[0])>>>) {
                buffer.resize(sizeInBytes / strideInBytes);
                auto bufferData = json.at("Data").get<std::vector<float>>();
                std::memcpy(buffer.data(), bufferData.data(), sizeInBytes);
            }
            else if (std::is_same_v<std::array<float, 2>, std::remove_const_t<std::remove_reference_t<decltype(buffer[0])>>>) {
                buffer.resize(sizeInBytes / strideInBytes);
                auto bufferData = json.at("Data").get<std::vector<float>>();
                std::memcpy(buffer.data(), bufferData.data(), sizeInBytes);
            }
            else if (std::is_same_v<std::array<uint32_t, 3>, std::remove_const_t<std::remove_reference_t<decltype(buffer[0])>>>) {
                buffer.resize(sizeInBytes / strideInBytes);
                auto bufferData = json.at("Data").get<std::vector<uint32_t>>();
                std::memcpy(buffer.data(), bufferData.data(), sizeInBytes);
            }
            else {
                buffer = json.at("Data").get<std::vector<std::remove_const_t<std::remove_reference_t<decltype(buffer[0])>>>>();
            }
        }
        else {
            auto loadPath = json.at("Path").get<std::string>();
            auto file = std::ifstream(loadPath, std::ios::binary);
            if (file.is_open()) {
                buffer.resize(sizeInBytes / strideInBytes);
                file.read((char*)buffer.data(), sizeInBytes);
            }
            file.close();
        }
    };
    {
        auto meshSharedResource = RTLib::Core::MeshSharedResource::New();
        auto& sharedResourceJson = meshGroupJson.at("SharedResource");
        LoadBuffer(sharedResourceJson.at("VertexBuffer"), meshSharedResource->vertexBuffer);
        LoadBuffer(sharedResourceJson.at("NormalBuffer"), meshSharedResource->normalBuffer);
        LoadBuffer(sharedResourceJson.at("TexCrdBuffer"), meshSharedResource->texCrdBuffer);
        meshSharedResource->variables = sharedResourceJson.at("Variables").get<VariableMap>();
        meshGroup->SetSharedResource(meshSharedResource);
    }
    {
        auto& uniqueResourcesJson = meshGroupJson.at("UniqueResources");
        for (auto& elem : uniqueResourcesJson.items()) {
            auto  uniqueResourceName = elem.key();
            auto& uniqueResourceJson = elem.value();
            auto meshUniqueResource = RTLib::Core::MeshUniqueResource::New();
            LoadBuffer(uniqueResourceJson.at("MatIndBuffer"), meshUniqueResource->matIndBuffer);
            LoadBuffer(uniqueResourceJson.at("TriIndBuffer"), meshUniqueResource->triIndBuffer);
            meshUniqueResource->variables = uniqueResourceJson.at("Variables").get<VariableMap>();
            auto uniqueMaterialNames = uniqueResourceJson.at("Materials").get<std::vector<std::string>>();
            meshUniqueResource->materials.reserve(uniqueMaterialNames.size());
            for (auto& uniqueMaterialName : uniqueMaterialNames) {
                meshUniqueResource->materials.push_back(materialIdxMap[uniqueMaterialName]);
            }
            meshGroup->SetUniqueResource(uniqueResourceName, meshUniqueResource);
        }
    }
    m_ObjModels[keyName] = { std::move(meshGroup),std::move(materials) };

    cacheJsonFile.close();
    return true;
}
void RTLib::Core::ObjModelAssetManager::SaveAssetCache(const std::string& keyName) const noexcept
{
    if (m_CacheDir.empty() || m_ObjModels.count(keyName)==0) { return; }
    auto cacheRootDir   = m_CacheDir + "\\" + keyName;
    if (!std::filesystem::exists(cacheRootDir)) {
        std::filesystem::create_directory(cacheRootDir);
    }
    auto cacheJsonPath = cacheRootDir + "\\" + keyName + ".json";
    auto cacheJsonFile = std::ofstream(cacheJsonPath, std::ios::binary);
    auto cacheJsonData = nlohmann::json();
    cacheJsonData["MeshGroup"] = {};
    auto& [meshGroup, materials] = m_ObjModels.at(keyName);
    auto sharedResource = meshGroup->GetSharedResource();
    auto SaveBuffer = [](const auto& buffer, const std::string& savePath) {
        auto strideInBytes = sizeof(buffer[0]);
        auto sizeInBytes = buffer.size() * strideInBytes;
        auto json = nlohmann::json();
        if (sizeInBytes < 10*1024) {
            if (std::is_same_v<std::array<float, 3>, std::remove_const_t<std::remove_reference_t<decltype(buffer[0])>>>) {
                auto bufferData = std::vector<float>(sizeInBytes / sizeof(float));
                std::memcpy(bufferData.data(), buffer.data(), sizeInBytes);
               json["Data"] = bufferData;
            }else if (std::is_same_v<std::array<float, 2>, std::remove_const_t<std::remove_reference_t<decltype(buffer[0])>>>) {
                auto bufferData = std::vector<float>(sizeInBytes / sizeof(float));
                std::memcpy(bufferData.data(), buffer.data(), sizeInBytes);
                json["Data"] = bufferData;
            }
            else if (std::is_same_v<std::array<uint32_t, 3>, std::remove_const_t<std::remove_reference_t<decltype(buffer[0])>>>) {
                auto bufferData = std::vector<uint32_t>(sizeInBytes / sizeof(uint32_t));
                std::memcpy(bufferData.data(), buffer.data(), sizeInBytes);
                json["Data"] = bufferData;
            }
            else {
                json["Data"] = buffer;
            }
        }
        else {
            auto file = std::ofstream(savePath, std::ios::binary);
            if (file.is_open()) {
                file.write((const char*)buffer.data(), sizeInBytes);
            }
            file.close();
            json["Path"] = savePath;
        }
        json["SizeInBytes"] = sizeInBytes;
        json["StrideInBytes"] = strideInBytes;

        return json;
    };
    cacheJsonData["MeshGroup"]["SharedResource"]["VertexBuffer"] = SaveBuffer(sharedResource->vertexBuffer, cacheRootDir + "\\VertexBuffer.bin");
    cacheJsonData["MeshGroup"]["SharedResource"]["NormalBuffer"] = SaveBuffer(sharedResource->normalBuffer, cacheRootDir + "\\NormalBuffer.bin");
    cacheJsonData["MeshGroup"]["SharedResource"]["TexCrdBuffer"] = SaveBuffer(sharedResource->normalBuffer, cacheRootDir + "\\TexCrdBuffer.bin");
    cacheJsonData["MeshGroup"]["SharedResource"][   "Variables"] = sharedResource->variables;
    for (auto& [uniqueName, uniqueResource] : meshGroup->GetUniqueResources()) {
        cacheJsonData["MeshGroup"]["UniqueResources"][uniqueName]["TriIndBuffer"] = SaveBuffer(uniqueResource->triIndBuffer, cacheRootDir + "\\"+ uniqueName  + "-TriIndBuffer.bin");
        cacheJsonData["MeshGroup"]["UniqueResources"][uniqueName]["MatIndBuffer"] = SaveBuffer(uniqueResource->matIndBuffer, cacheRootDir + "\\" + uniqueName + "-MatIndBuffer.bin");
        cacheJsonData["MeshGroup"]["UniqueResources"][uniqueName]["Variables"]    = uniqueResource->variables;
        for (auto& matIdx : uniqueResource->materials) {
            auto matName = materials[matIdx].GetString("name");
            cacheJsonData["MeshGroup"]["UniqueResources"][uniqueName]["Materials"].push_back(matName);
        }
    }
    cacheJsonData["Materials"] = {};
    for (auto& material: m_ObjModels.at(keyName).materials) {
        auto matName = material.GetString("name");
        cacheJsonData["Materials"][matName] = material;
    }
    cacheJsonFile << cacheJsonData;
    cacheJsonFile.close();
    return;
}