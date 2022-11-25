#ifndef RTLIB_CORE_BINARY_READER_H
#define RTLIB_CORE_BINARY_READER_H
#include <RTLib/Core/VariableMap.h>
#include <RTLib/Core/Exceptions.h>
#include <unordered_map>
#include <nlohmann/json.hpp>
#include <vector>
#include <array>
#include <string>
#include <memory>
#include <cstdint>
#include <type_traits>

namespace RTLib {
	namespace Core {

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

        class  SphereResource;
        class  SphereResourceExtData {
        public:
            SphereResourceExtData(SphereResource* pSphereSharedResource)noexcept {
                m_SphereSharedResource = pSphereSharedResource;
            }
            virtual ~SphereResourceExtData()noexcept {}
        protected:
            auto GetParent()const noexcept -> const SphereResource* { return m_SphereSharedResource; }
            auto GetParent()      noexcept ->       SphereResource* { return m_SphereSharedResource; }
        private:
            SphereResource* m_SphereSharedResource = nullptr;
        };

        struct SphereResource
        {
            using Float3                                  = std::array<float, 3>;
            std::string                              name = {};
            std::vector<Float3>              centerBuffer = {};
            std::vector<float >              radiusBuffer = {};
            std::vector<unsigned int>        matIndBuffer = {};
            std::vector<VariableMap>            materials = {};
            VariableMap                         variables = {};
            std::unique_ptr<SphereResourceExtData> extData = nullptr;

            static auto New() ->std::shared_ptr<SphereResource> {
                return std::shared_ptr<SphereResource>(new SphereResource());
            }
            template<typename SphereResourceExtT, typename ...Args, bool Cond = std::is_base_of_v<SphereResourceExtData, SphereResourceExtT>>
            bool AddExtData(Args&&... args) {
                extData = std::unique_ptr<SphereResourceExtData>(SphereResourceExtT::New(this, std::forward<Args>(args)...));
                return extData != nullptr;
            }
            template<typename SphereResourceExtT, typename ...Args, bool Cond = std::is_base_of_v<SphereResourceExtData, SphereResourceExtT>>
            auto GetExtData()const ->const SphereResourceExtT* {
                return dynamic_cast<const SphereResourceExtT*>(extData.get());
            }
            template<typename SphereResourceExtT, typename ...Args, bool Cond = std::is_base_of_v<SphereResourceExtData, SphereResourceExtT>>
            auto GetExtData()->SphereResourceExtT* {
                return dynamic_cast<SphereResourceExtT*>(extData.get());
            }
        };
        using  SphereResourcePtr = std::shared_ptr<SphereResource>;

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
        
        
        

    }
}
#endif
