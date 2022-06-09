#ifndef RTLIB_CORE_BINARY_READER_H
#define RTLIB_CORE_BINARY_READER_H
#include <tiny_obj_loader.h>
#include <unordered_map>
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
        private:
            std::unordered_map<std::string, uint32_t>             m_UInt32Data;
            std::unordered_map<std::string, bool>                 m_BoolData;
            std::unordered_map<std::string, float>                m_Float1Data;
            std::unordered_map<std::string, std::array<float, 2>>  m_Float2Data;
            std::unordered_map<std::string, std::array<float, 3>>  m_Float3Data;
            std::unordered_map<std::string, std::array<float, 4>>  m_Float4Data;
            std::unordered_map<std::string, std::string>          m_StringData;
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
        class ObjModelAssetManager
        {
        public:
            bool LoadAsset(const std::string& keyName, const std::string& objPath);
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
            std::unordered_map<std::string, ObjModel> m_ObjModels = {};
        };
		class  ObjModelReader
		{
		public:
			bool Load(const std::string& filename)noexcept;

		};
		
	}
}
#endif
