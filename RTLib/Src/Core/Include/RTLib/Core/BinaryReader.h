#ifndef RTLIB_CORE_BINARY_READER_H
#define RTLIB_CORE_BINARY_READER_H
#include <RTLib/Core/Exceptions.h>
#include <tiny_obj_loader.h>
#include <unordered_map>
#include <nlohmann/json.hpp>
#include <vector>
#include <array>
#include <string>
#include <memory>
#include <cstdint>
#include <type_traits>
#define RTLIB_CORE_VARIABLE_MAP_METHOD_DECLARE_SET(Name) void Set##Name(const std::string& keyName, const Internal##Name& value)noexcept { m_##Name##Data[keyName] = value; }
#define RTLIB_CORE_VARIABLE_MAP_METHOD_DECLARE_SET_FLOAT_N_FROM(CNT) template<typename T> void SetFloat##CNT##From(const std::string& keyName, const T& value)noexcept { \
    static_assert(sizeof(T)==sizeof(float)*CNT);\
    Internal##Float##CNT middle; \
    std::memcpy(&middle, &value, sizeof(float)*CNT);\
    return SetFloat##CNT(keyName,middle); \
}
#define RTLIB_CORE_VARIABLE_MAP_METHOD_DECLARE_GET(Name) auto Get##Name(const std::string& keyName)const -> Internal##Name { return m_##Name##Data.at(keyName); }
#define RTLIB_CORE_VARIABLE_MAP_METHOD_DECLARE_GET_FLOAT_N_AS(CNT) template<typename T> auto GetFloat##CNT##As(const std::string& keyName)const noexcept -> T{ \
    static_assert(sizeof(T)==sizeof(float)*CNT);\
    Internal##Float##CNT middle = GetFloat##CNT(keyName); \
    T value {}; \
    std::memcpy(&value, &middle, sizeof(float)*CNT);\
    return value; \
}
#define RTLIB_CORE_VARIABLE_MAP_METHOD_DECLARE_POP(Name) auto Pop##Name(const std::string& keyName)noexcept -> Internal##Name { \
    auto val = Get##Name(keyName); \
    m_##Name##Data.erase(keyName); \
    return val;\
}
#define RTLIB_CORE_VARIABLE_MAP_METHOD_DECLARE_HAS(Name) bool Has##Name(const std::string& keyName)const noexcept{ return m_##Name##Data.count(keyName) > 0; }

namespace RTLib {
	namespace Core {
        class  VariableMap
        {
        private:
            using InternalUInt32 = uint32_t;
            using InternalBool = bool;
            using InternalFloat1 = float;
            using InternalFloat2 = std::array<float, 2>;
            using InternalFloat3 = std::array<float, 3>;
            using InternalFloat4 = std::array<float, 4>;
            //For String
            using InternalString = std::string;
        public:

            //Set
            RTLIB_CORE_VARIABLE_MAP_METHOD_DECLARE_SET(UInt32);
            RTLIB_CORE_VARIABLE_MAP_METHOD_DECLARE_SET(Bool);
            RTLIB_CORE_VARIABLE_MAP_METHOD_DECLARE_SET(Float1);
            RTLIB_CORE_VARIABLE_MAP_METHOD_DECLARE_SET(Float2);
            RTLIB_CORE_VARIABLE_MAP_METHOD_DECLARE_SET(Float3);
            RTLIB_CORE_VARIABLE_MAP_METHOD_DECLARE_SET(Float4);
            RTLIB_CORE_VARIABLE_MAP_METHOD_DECLARE_SET(String);
            RTLIB_CORE_VARIABLE_MAP_METHOD_DECLARE_SET_FLOAT_N_FROM(1);
            RTLIB_CORE_VARIABLE_MAP_METHOD_DECLARE_SET_FLOAT_N_FROM(2);
            RTLIB_CORE_VARIABLE_MAP_METHOD_DECLARE_SET_FLOAT_N_FROM(3);
            RTLIB_CORE_VARIABLE_MAP_METHOD_DECLARE_SET_FLOAT_N_FROM(4);
            //Get
            RTLIB_CORE_VARIABLE_MAP_METHOD_DECLARE_GET(UInt32);
            RTLIB_CORE_VARIABLE_MAP_METHOD_DECLARE_GET(Bool);
            RTLIB_CORE_VARIABLE_MAP_METHOD_DECLARE_GET(Float1);
            RTLIB_CORE_VARIABLE_MAP_METHOD_DECLARE_GET(Float2);
            RTLIB_CORE_VARIABLE_MAP_METHOD_DECLARE_GET(Float3);
            RTLIB_CORE_VARIABLE_MAP_METHOD_DECLARE_GET(Float4);
            RTLIB_CORE_VARIABLE_MAP_METHOD_DECLARE_GET(String);
            RTLIB_CORE_VARIABLE_MAP_METHOD_DECLARE_GET_FLOAT_N_AS(1);
            RTLIB_CORE_VARIABLE_MAP_METHOD_DECLARE_GET_FLOAT_N_AS(2);
            RTLIB_CORE_VARIABLE_MAP_METHOD_DECLARE_GET_FLOAT_N_AS(3);
            RTLIB_CORE_VARIABLE_MAP_METHOD_DECLARE_GET_FLOAT_N_AS(4);
            //Pop
            RTLIB_CORE_VARIABLE_MAP_METHOD_DECLARE_POP(UInt32);
            RTLIB_CORE_VARIABLE_MAP_METHOD_DECLARE_POP(Bool);
            RTLIB_CORE_VARIABLE_MAP_METHOD_DECLARE_POP(Float1);
            RTLIB_CORE_VARIABLE_MAP_METHOD_DECLARE_POP(Float2);
            RTLIB_CORE_VARIABLE_MAP_METHOD_DECLARE_POP(Float3);
            RTLIB_CORE_VARIABLE_MAP_METHOD_DECLARE_POP(Float4);
            RTLIB_CORE_VARIABLE_MAP_METHOD_DECLARE_POP(String);
            //Has
            RTLIB_CORE_VARIABLE_MAP_METHOD_DECLARE_HAS(UInt32);
            RTLIB_CORE_VARIABLE_MAP_METHOD_DECLARE_HAS(Bool);
            RTLIB_CORE_VARIABLE_MAP_METHOD_DECLARE_HAS(Float1);
            RTLIB_CORE_VARIABLE_MAP_METHOD_DECLARE_HAS(Float2);
            RTLIB_CORE_VARIABLE_MAP_METHOD_DECLARE_HAS(Float3);
            RTLIB_CORE_VARIABLE_MAP_METHOD_DECLARE_HAS(Float4);
            RTLIB_CORE_VARIABLE_MAP_METHOD_DECLARE_HAS(String);

            auto EnumerateUInt32Data()const noexcept -> const std::unordered_map<std::string, uint32_t>& { return m_UInt32Data; }
            auto EnumerateBoolData()  const noexcept -> const std::unordered_map<std::string, bool    >& { return m_BoolData; }
            auto EnumerateFloat1Data()const noexcept -> const std::unordered_map<std::string, float               >& { return m_Float1Data; }
            auto EnumerateFloat2Data()const noexcept -> const std::unordered_map<std::string, std::array<float, 2>>& { return m_Float2Data; }
            auto EnumerateFloat3Data()const noexcept -> const std::unordered_map<std::string, std::array<float, 3>>& { return m_Float3Data; }
            auto EnumerateFloat4Data()const noexcept -> const std::unordered_map<std::string, std::array<float, 4>>& { return m_Float4Data; }
            auto EnumerateStringData()const noexcept -> const std::unordered_map<std::string, std::string>& { return m_StringData; }
        private:
            std::unordered_map<std::string, uint32_t>              m_UInt32Data;
            std::unordered_map<std::string, bool>                  m_BoolData;
            std::unordered_map<std::string, float>                 m_Float1Data;
            std::unordered_map<std::string, std::array<float, 2>>  m_Float2Data;
            std::unordered_map<std::string, std::array<float, 3>>  m_Float3Data;
            std::unordered_map<std::string, std::array<float, 4>>  m_Float4Data;
            std::unordered_map<std::string, std::string>           m_StringData;
        };
        using  VariableMapList = std::vector<VariableMap>;
        using  VariableMapListPtr = std::shared_ptr<VariableMapList>;

        class  MeshSharedResource;
        struct MeshSharedResourceExtData {
        public:
            MeshSharedResourceExtData(MeshSharedResource* pMeshSharedResource)noexcept {
                m_MeshSharedResource = pMeshSharedResource;
            }
            virtual ~MeshSharedResourceExtData()noexcept {}
        protected:
            auto GetParent()const noexcept -> const MeshSharedResource* { return m_MeshSharedResource; }
            auto GetParent()      noexcept ->       MeshSharedResource* { return m_MeshSharedResource; }
        private:
            MeshSharedResource* m_MeshSharedResource = nullptr;
        };

        struct MeshSharedResource {
            using Float2 = std::array<float, 2>;
            using Float3 = std::array<float, 3>;
            std::string          name         = {};
            std::vector<Float3>  vertexBuffer = {};
            std::vector<Float3>  normalBuffer = {};
            std::vector<Float2>  texCrdBuffer = {};
            VariableMap          variables    = {};
            std::unique_ptr<MeshSharedResourceExtData> extData = nullptr;
            static auto New() ->std::shared_ptr<MeshSharedResource> {
                return std::shared_ptr<MeshSharedResource>(new MeshSharedResource());
            }
            template<typename MeshSharedResourceExtT, typename ...Args, bool Cond = std::is_base_of_v<MeshSharedResourceExtData, MeshSharedResourceExtT>>
            bool AddExtData(Args&&... args) {
                extData = std::unique_ptr<MeshSharedResourceExtData>(MeshSharedResourceExtT::New(this, std::forward<Args>(args)...));
                return extData != nullptr;
            }
            template<typename MeshSharedResourceExtT, typename ...Args, bool Cond = std::is_base_of_v<MeshSharedResourceExtData, MeshSharedResourceExtT>>
            auto GetExtData()const ->const MeshSharedResourceExtT* {
                return dynamic_cast<const MeshSharedResourceExtT*>(extData.get());
            }
            template<typename MeshSharedResourceExtT, typename ...Args, bool Cond = std::is_base_of_v<MeshSharedResourceExtData, MeshSharedResourceExtT>>
            auto GetExtData()->MeshSharedResourceExtT* {
                return dynamic_cast<MeshSharedResourceExtT*>(extData.get());
            }
        };
        using  MeshSharedResourcePtr = std::shared_ptr<MeshSharedResource>;
        class  MeshUniqueResource;
        struct MeshUniqueResourceExtData {
        public:
            MeshUniqueResourceExtData(MeshUniqueResource* pMeshUniqueResource)noexcept {
                m_MeshUniqueResource = pMeshUniqueResource;
            }
            virtual ~MeshUniqueResourceExtData()noexcept {}
        protected:
            auto GetParent()const noexcept -> const MeshUniqueResource* { return m_MeshUniqueResource; }
            auto GetParent()      noexcept ->       MeshUniqueResource* { return m_MeshUniqueResource; }
        private:
            MeshUniqueResource* m_MeshUniqueResource = nullptr;
        };
        struct MeshUniqueResource {
            using UInt3 = std::array<uint32_t, 3>;
            std::string            name = {};
            std::vector<uint32_t>  materials = {};
            std::vector<UInt3>     triIndBuffer = {};
            std::vector<uint32_t>  matIndBuffer = {};
            VariableMap            variables = {};
            std::unique_ptr<MeshUniqueResourceExtData> extData = nullptr;
            static auto New() ->std::shared_ptr<MeshUniqueResource> {
                return std::shared_ptr<MeshUniqueResource>(new MeshUniqueResource());
            }
            template<typename MeshUniqueResourceExtT, typename ...Args, bool Cond = std::is_base_of_v<MeshUniqueResourceExtData, MeshUniqueResourceExtT>>
            bool AddExtData(Args&&... args) {
                extData = std::unique_ptr<MeshUniqueResourceExtData>(MeshUniqueResourceExtT::New(this, std::forward<Args>(args)...));
                return extData != nullptr;
            }
            template<typename MeshUniqueResourceExtT, typename ...Args, bool Cond = std::is_base_of_v<MeshUniqueResourceExtData, MeshUniqueResourceExtT>>
            auto GetExtData()const ->const MeshUniqueResourceExtT* {
                return dynamic_cast<const MeshUniqueResourceExtT*>(extData.get());
            }
            template<typename MeshUniqueResourceExtT, typename ...Args, bool Cond = std::is_base_of_v<MeshUniqueResourceExtData, MeshUniqueResourceExtT>>
            auto GetExtData()->MeshUniqueResourceExtT* {
                return dynamic_cast<MeshUniqueResourceExtT*>(extData.get());
            }
        };
        using  MeshUniqueResourcePtr = std::shared_ptr<MeshUniqueResource>;
        class  MeshGroup;
        class  Mesh {
        public:
            Mesh()noexcept {}
            void SetSharedResource(const  MeshSharedResourcePtr& res)noexcept;
            auto GetSharedResource()const noexcept -> MeshSharedResourcePtr;
            void SetUniqueResource(const  MeshUniqueResourcePtr& res)noexcept;
            void SetUniqueResource(const std::string& name, const MeshUniqueResourcePtr& res)noexcept;
            auto GetUniqueResource()const->MeshUniqueResourcePtr;
            static auto New()->std::shared_ptr<Mesh> {
                return std::shared_ptr<Mesh>(new Mesh());
            }
        private:
            friend class MeshGroup;
            std::string              m_Name = {};
            MeshSharedResourcePtr    m_SharedResource = {};
            MeshUniqueResourcePtr    m_UniqueResource = {};
        };
        using  MeshPtr = std::shared_ptr<Mesh>;
        class  MeshGroup {
        public:
            MeshGroup()noexcept {}
            void SetSharedResource(const MeshSharedResourcePtr& res)noexcept;
            auto GetSharedResource()const noexcept -> MeshSharedResourcePtr;
            void SetUniqueResource(const std::string& name, const MeshUniqueResourcePtr& res)noexcept;
            auto GetUniqueResource(const std::string& name) const->MeshUniqueResourcePtr;
            auto GetUniqueResources()const noexcept -> const std::unordered_map<std::string, MeshUniqueResourcePtr>&;
            auto GetUniqueNames()const noexcept -> std::vector<std::string>;
            auto LoadMesh(const std::string& name)const->MeshPtr;
            bool RemoveMesh(const std::string& name);
            static auto New()->std::shared_ptr<MeshGroup> {
                return std::shared_ptr<MeshGroup>(new MeshGroup());
            }
        private:
            using MeshUniqueResourcePtrMap = std::unordered_map<std::string, MeshUniqueResourcePtr>;
            std::string              m_Name = {};
            MeshSharedResourcePtr    m_SharedResource = {};
            MeshUniqueResourcePtrMap m_UniqueResources = {};
        };
        using  MeshGroupPtr = std::shared_ptr<MeshGroup>;
        struct ObjModel
        {
            MeshGroupPtr             meshGroup;
            std::vector<VariableMap> materials;
            void SplitLight();
            void   InitAABB();
        };
        class  ObjModelAssetManager
        {
        public:
            ObjModelAssetManager(const std::string& cacheDir = "")noexcept :m_CacheDir{ cacheDir } {}
            auto GetCacheDir()const noexcept -> std::string { return m_CacheDir; }
            void SetCacheDir(const std::string& cacheDir)noexcept { m_CacheDir = cacheDir; }
            bool LoadAsset(const std::string& keyName, const std::string& objPath, bool useCache = true);
            void FreeAsset(const std::string& keyName);
            auto  GetAsset(const std::string& keyName)const -> const ObjModel&;
            auto  GetAsset(const std::string& keyName)->ObjModel&;
            auto  PopAsset(const std::string& keyName)->ObjModel;
            auto  GetAssets()const -> const std::unordered_map<std::string, ObjModel>&;
            auto  GetAssets()->std::unordered_map<std::string, ObjModel>&;
            bool  HasAsset(const std::string& keyName)const noexcept;
            void  Reset();
            ~ObjModelAssetManager();
        private:
            bool LoadAssetCache(const std::string& keyName)noexcept;
            void SaveAssetCache(const std::string& keyName)const noexcept;
        private:
            std::string                               m_CacheDir  = {};
            std::unordered_map<std::string, ObjModel> m_ObjModels = {};
        };
        //ObjModelAssetManager
        template<typename JsonType>
        inline void   to_json(JsonType& j, const RTLib::Core::ObjModelAssetManager& obj)
        {
            j["CacheDir"] = obj.GetCacheDir();
            for (auto& [name, objAsset] : obj.GetAssets()) {
                j["Assets"][name] = objAsset.meshGroup->GetSharedResource()->variables.GetString("path");
            }
        }
        template<typename JsonType>
        inline void from_json(const JsonType& j, RTLib::Core::ObjModelAssetManager& obj)
        {
            obj = ObjModelAssetManager(j.at("CacheDir").get<std::string>());
            for (auto& assetJson : j.at("Assets").items())
            {
                RTLIB_CORE_ASSERT_IF_FAILED(obj.LoadAsset(assetJson.key(), assetJson.value().get<std::string>()));
            }

        }
        //Variable
        template<typename JsonType>
        inline void   to_json(      JsonType& j, const RTLib::Core::VariableMap& v) {

            for (auto& [key, value] : v.EnumerateBoolData()) {
                j[key] = value;
            }
            for (auto& [key, value] : v.EnumerateFloat1Data()) {
                j[key] = value;
            }
            for (auto& [key, value] : v.EnumerateFloat2Data()) {
                j[key] = value;
            }
            for (auto& [key, value] : v.EnumerateFloat3Data()) {
                j[key] = value;
            }
            for (auto& [key, value] : v.EnumerateFloat4Data()) {
                j[key] = value;
            }
            for (auto& [key, value] : v.EnumerateUInt32Data()) {
                j[key] = value;
            }
            for (auto& [key, value] : v.EnumerateStringData()) {
                j[key] = value;
            }


        }
        template<typename JsonType>
        inline void from_json(const JsonType& j, RTLib::Core::VariableMap&       v) {
            for (auto& elem : j.items()) {
                if (elem.value().is_string()) {
                    v.SetString(elem.key(), elem.value().get<std::string>());
                }
                if (elem.value().is_boolean()) {
                    v.SetBool(elem.key(), elem.value().get<bool>());
                }
                if (elem.value().is_number_integer()) {
                    v.SetUInt32(elem.key(), elem.value().get<unsigned int>());
                }
                if (elem.value().is_number_float()) {
                    v.SetFloat1(elem.key(), elem.value().get<float>());
                }
                if (elem.value().is_array()) {
                    auto size = elem.value().size();
                    switch (size) {
                    case 2:
                        v.SetFloat2(elem.key(), elem.value().get < std::array<float, 2>>());
                        break;
                    case 3:
                        v.SetFloat3(elem.key(), elem.value().get < std::array<float, 3>>());
                        break;
                    case 4:
                        v.SetFloat4(elem.key(), elem.value().get < std::array<float, 4>>());
                        break;
                    default:
                        break;
                    }
                }
            }
        }
        //VariableArray
        template<typename JsonType>
        inline void   to_json(JsonType& j, const std::vector<RTLib::Core::VariableMap>& v) {
            for (auto& elem : v) {
                auto elemJson = JsonType(v);
                j.push_back(elemJson);
            }
        }
        template<typename JsonType>
        inline void from_json(const JsonType& j, std::vector<RTLib::Core::VariableMap>& v) {
            v.clear();
            v.reserve(j.size());
            for (auto& elem : j) {
                v.push_back(elem.get<RTLib::Core::VariableMap>());
            }
        }
        
        struct WorldElementGeometryObjModelMesh
        {
            std::string base = {};
            std::vector<RTLib::Core::VariableMap> materials = {};
            std::array<float, 12> transform = {};
            bool useTransform = false;
            bool useMaterials = false;
        };
        //VariableArray
        template<typename JsonType>
        inline void   to_json(JsonType& j, const WorldElementGeometryObjModelMesh& v) {
            j["Base"] = v.base;
            if (v.useTransform) {
                j["Transform"] = v.transform;
            }
            if (v.useMaterials) {
                j["Materials"] = v.materials;
            }
        }
        template<typename JsonType>
        inline void from_json(const JsonType& j, WorldElementGeometryObjModelMesh& v) {
            v.base = j.at("Base").get<std::string>();
            v.useMaterials = j.count("Materials") > 0;
            v.useTransform = j.count("Transform") > 0;
            if (v.useMaterials) {
                v.materials = j.at("Materials").get<std::vector<RTLib::Core::VariableMap>>();
            }
            if (v.useTransform) {
                v.transform = j.at("Transform").get<std::array<float, 12>>();
            }
        }
        struct WorldElementGeometryObjModel
        {
            std::string base = "";
            std::unordered_map<std::string, WorldElementGeometryObjModelMesh> meshes = {};
            bool useMeshes = false;
        };
        template<typename JsonType>
        inline void   to_json(JsonType& j, const WorldElementGeometryObjModel& v) {
            j["Type"] = "ObjModel";
            j["Base"] = v.base;
            if (!v.meshes.empty() && v.useMeshes) {
                auto meshJsons = std::vector<JsonType>();
                meshJsons.reserve(v.meshes.size());
                for (auto& mesh : v.meshes) {
                    meshJsons.push_back(nlohmann::json(mesh));
                }
                j["Meshes"] = meshJsons;
            }
        }
        template<typename JsonType>
        inline void from_json(const JsonType& j, WorldElementGeometryObjModel& v) {
            v.base   = j.at("Base").get<std::string>();
            v.meshes.clear();
            if ((j.count("Meshes") > 0)) {
                for (auto& meshElem : j.at("Meshes").items()) {
                    v.meshes[meshElem.key()] = meshElem.value().get<WorldElementGeometryObjModelMesh>();
                   
                } 
                v.useMeshes = true;
            }
            else {
                v.useMeshes = false;
            }
        }

        struct WorldElementGeometryAS
        {
            std::vector<std::string> geometries = {};
        };
        template<typename JsonType>
        inline void   to_json(JsonType& j, const WorldElementGeometryAS& v) {
            j["Type"] = "GeometryAS";
            j["Geometries"] = v.geometries;
        }
        template<typename JsonType>
        inline void from_json(const JsonType& j, WorldElementGeometryAS& v) {
            v.geometries = j.at("Geometries").get<std::vector<std::string>>();
        }
        struct WorldElementInstance
        {
            std::string base;
            std::string asType;
            std::array<float, 12> transform;
            
        };
        template<typename JsonType>
        inline void   to_json(JsonType& j, const WorldElementInstance& v) {
            j["Type"]      ="Instance";
            j["Base"]      = v.base;
            j["ASType"]    = v.asType;
            j["Transform"] = v.transform;
        }
        template<typename JsonType>
        inline void from_json(const JsonType& j, WorldElementInstance& v) {
            v.base         = j.at("Base").get<std::string>();
            v.asType       = j.at("ASType").get<std::string>();
            v.transform    = j.at("Transform").get<std::array<float, 12>>();
        }
        struct WorldElementInstanceAS
        {
            std::vector<std::string> instances = {};
        };
        template<typename JsonType>
        inline void   to_json(JsonType& j, const WorldElementInstanceAS& v) {
            j["Type"] = "InstanceAS";
            j["Instances"] = v.instances;
        }
        template<typename JsonType>
        inline void from_json(const JsonType& j, WorldElementInstanceAS& v) {
            v.instances = j.at("Instances").get<std::vector<std::string>>();
        }
        struct WorldData
        {
            std::unordered_map<std::string, WorldElementGeometryObjModel>     geometryObjModels;
            std::unordered_map<std::string, WorldElementGeometryAS>           geometryASs;
            std::unordered_map<std::string, WorldElementInstance>             instances;
            std::unordered_map<std::string, WorldElementInstanceAS>           instanceASs;
        };

        template<typename JsonType>
        inline void   to_json(JsonType& j, const WorldData& v) {
            for (auto& [name,geometry] : v.geometryObjModels) {
                j["Geometries"][name]  = geometry;
            }
            for (auto& [name, geometryAS] : v.geometryASs) {
                j["GeometryASs"][name] = geometryAS;
            }
            for (auto& [name, instance] : v.instances) {
                j["Instances"][name]   = instance;
            }
            for (auto& [name, instanceAS] : v.instanceASs) {
                j["InstanceASs"][name] = instanceAS;
            }
        }
        template<typename JsonType>
        inline void from_json(const JsonType& j, WorldData& v) {
            for (auto& elem : j.at("Geometries" ).items()) {
                if (elem.value().at("Type").get<std::string>() == "ObjModel") {
                    v.geometryObjModels[elem.key()] = elem.value().get<WorldElementGeometryObjModel>();
                }
            }
            for (auto& elem : j.at("GeometryASs").items()) {
                v.geometryASs[elem.key()] = elem.value().get<WorldElementGeometryAS>();
            }
            for (auto& elem : j.at("Instances"  ).items()) {
                v.instances[elem.key()] = elem.value().get<WorldElementInstance>();
            }
            for (auto& elem : j.at("InstanceASs").items()) {
                v.instanceASs[elem.key()] = elem.value().get<WorldElementInstanceAS>();
            }
        }
        

    }
}
#endif
