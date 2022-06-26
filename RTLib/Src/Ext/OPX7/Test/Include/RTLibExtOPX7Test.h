#ifndef RTLIB_EXT_OPX7_TEST_H
#define RTLIB_EXT_OPX7_TEST_H
#include <RTLib/Core/Exceptions.h>
#include <RTLib/Core/BinaryReader.h>
#include <RTLib/Utils/Camera.h>
#include <RTLib/Ext/OPX7/OPX7Context.h>
#include <RTLib/Ext/OPX7/OPX7Module.h>
#include <RTLib/Ext/OPX7/OPX7ProgramGroup.h>
#include <RTLib/Ext/OPX7/OPX7ShaderTable.h>
#include <RTLib/Ext/OPX7/OPX7ShaderTableLayout.h>
#include <RTLib/Ext/OPX7/OPX7AccelerationStructure.h>
#include <RTLib/Ext/OPX7/OPX7Pipeline.h>
#include <RTLib/Ext/OPX7/OPX7Natives.h>
#include <RTLib/Ext/OPX7/OPX7Exceptions.h>
#include <RTLib/Ext/CUDA/CUDAExceptions.h>
#include <RTLib/Ext/CUDA/CUDAImage.h>
#include <RTLib/Ext/CUDA/CUDATexture.h>
#include <RTLib/Ext/CUDA/CUDAStream.h>
#include <RTLib/Ext/CUDA/CUDANatives.h>
#include <RTLib/Ext/CUDA/Math/VectorFunction.h>
#include <RTLib/Ext/CUGL/CUGLBuffer.h>
#include <RTLib/Ext/GLFW/GLFWContext.h>
#include <RTLib/Ext/GLFW/GL/GLFWOpenGLWindow.h>
#include <RTLib/Ext/GLFW/GL/GLFWOpenGLContext.h>
#include <RTLib/Ext/GL/GLRectRenderer.h>
#include <RTLib/Ext/GL/GLTexture.h>
#include <RTLib/Ext/GL/GLImage.h>
#include <optix_stubs.h>
#include <GLFW/glfw3.h>
#include <stb_image.h>
#include <stb_image_write.h>
#include <cuda/SimpleKernel.h>
#include <RTLibExtOPX7TestConfig.h>
#include <memory>
#include <iostream>
#include <filesystem>
#include <random>
#include <fstream>
#include <unordered_map>
#include <unordered_set>
#include <array>
#include <string>
#include <string_view>
#define RTLIB_DECLARE_GET_BY_REFERENCE(class_name,type_name,func_name_base,member_name) \
const type_name& get##func_name_base()const noexcept { return member_name; }
#define RTLIB_DECLARE_GET_BY_VALUE(class_name,type_name,func_name_base,member_name) \
type_name get##func_name_base()const noexcept{ return member_name; }
#define RTLIB_DECLARE_SET_BY_REFERENCE(class_name,type_name,func_name_base,member_name) \
void  set##func_name_base(const type_name& v)noexcept{ member_name = v; } 
#define RTLIB_DECLARE_SET_BY_VALUE(class_name,type_name,func_name_base,member_name) \
void  set##func_name_base(const type_name v)noexcept{ member_name = v; }
#define RTLIB_DECLARE_GET_AND_SET_BY_REFERENCE(class_name,type_name,func_name_base,member_name) \
RTLIB_DECLARE_GET_BY_REFERENCE(class_name,type_name,func_name_base,member_name); \
RTLIB_DECLARE_SET_BY_REFERENCE(class_name,type_name,func_name_base,member_name)
#define RTLIB_DECLARE_GET_AND_SET_BY_VALUE(class_name,type_name,func_name_base,member_name) \
RTLIB_DECLARE_GET_BY_VALUE(class_name,type_name,func_name_base,member_name); \
RTLIB_DECLARE_SET_BY_VALUE(class_name,type_name,func_name_base,member_name)
namespace rtlib
{
    namespace test {
        using namespace RTLib::Core;
        using namespace RTLib::Utils;
        using namespace RTLib::Ext;
        using namespace RTLib::Ext::CUDA::Math;


        struct TraceConfigData
        {
            unsigned int width;
            unsigned int height;
            unsigned int samples;
            unsigned int maxSamples;
        };

        template<typename JsonType>
        inline void   to_json(JsonType& j, const TraceConfigData& v) {
            j["Width"     ] = v.width;
            j["Height"    ] = v.height;
            j["Samples"   ] = v.samples;
            j["MaxSamples"] = v.maxSamples;
        }
        template<typename JsonType>
        inline void from_json(const JsonType& j, TraceConfigData& v) {
            v.width      = j.at("Width"     ).get<unsigned int>();
            v.height     = j.at("Height"    ).get<unsigned int>();
            v.samples    = j.at("Samples"   ).get<unsigned int>();
            v.maxSamples = j.at("MaxSamples").get<unsigned int>();
        }

        struct SceneData
        {
            ObjModelAssetManager objAssetManager;
            WorldData            world ;
            TraceConfigData      config;
            CameraController     cameraController;
            auto GetFrameBufferSizeInBytes()const noexcept -> size_t
            {
                return static_cast<size_t>(config.width * config.height);
            }
            auto GetCamera()const noexcept -> Camera
            {
                return cameraController.GetCamera(static_cast<float>(config.width) / static_cast<float>(config.height));
            }
        };
        template<typename JsonType>
        inline void   to_json(JsonType& j, const SceneData& v) {
            j["CameraController"] = v.cameraController;
            j["ObjModels"]        = v.objAssetManager;
            j["World" ]           = v.world;
            j["Config"]           = v.config;
        }
        template<typename JsonType>
        inline void from_json(const JsonType& j, SceneData& v) {
            v.cameraController = j.at("CameraController").get<RTLib::Utils::CameraController>();
            v.objAssetManager  = j.at("ObjModels"       ).get<RTLib::Core::ObjModelAssetManager>();
            v.world            = j.at("World"           ).get<RTLib::Core::WorldData>();
            v.config           = j.at("Config"          ).get<TraceConfigData>();
            for (auto& [geometryASName, geometryAS] : v.world.geometryASs)
            {
                for (auto& geometryName : geometryAS.geometries) {
                    if (v.world.geometryObjModels.count(geometryName) > 0) {
                        auto& geometryObjModel = v.world.geometryObjModels.at(geometryName);
                        auto& objAsset         = v.objAssetManager.GetAsset(geometryObjModel.base);
                        if (geometryObjModel.meshes.empty()) {
                            auto meshNames = objAsset.meshGroup->GetUniqueNames();
                            for (auto& meshName : meshNames)
                            {
                                auto mesh = objAsset.meshGroup->LoadMesh(meshName);
                                geometryObjModel.meshes[meshName].base = meshName;
                                geometryObjModel.meshes[meshName].transform = {};
                                geometryObjModel.meshes[meshName].materials.reserve(mesh->GetUniqueResource()->materials.size());
                                for (auto& matIdx : mesh->GetUniqueResource()->materials) {
                                    geometryObjModel.meshes[meshName].materials.push_back(objAsset.materials[matIdx]);
                                }
                            }
                        }
                        else {
                            for (auto& [meshName, meshData] : geometryObjModel.meshes)
                            {
                                auto mesh = objAsset.meshGroup->LoadMesh(meshData.base);
                                if (meshData.materials.empty()) {
                                    meshData.materials.reserve(mesh->GetUniqueResource()->materials.size());
                                    for (auto& matIdx : mesh->GetUniqueResource()->materials) {
                                        meshData.materials.push_back(objAsset.materials[matIdx]);
                                    }
                                }

                            }
                        }
                    }
                }
            }
        }


        inline auto CreateGLFWWindow(RTLib::Ext::GLFW::GLFWContext* glfwContext,int width, int height, const char* title)->RTLib::Ext::GLFW::GL::GLFWOpenGLWindow* {
            auto desc          = RTLib::Ext::GLFW::GL::GLFWOpenGLWindowCreateDesc();
            desc.width         = width;
            desc.height        = height;
            desc.title         = title;
            desc.isCoreProfile = true;
            desc.isVisible     = false;
            desc.isResizable   = false;
            std::vector<std::pair<int, int>> glVersions = {
                {4,6},{4,5},{4,4},{4,3},{4,2},{4,1},{4,0},
                {3,3},{3,2},{3,1},{3,0},
                {2,1},{2,0}
            };
            for (auto& [majorVersion, minorVersion] : glVersions) {
                desc.versionMajor = majorVersion;
                desc.versionMinor = minorVersion;
                auto window = RTLib::Ext::GLFW::GL::GLFWOpenGLWindow::New(glfwContext, desc);

                if (window) { return window; }
            }
            return nullptr;
        }
        
        inline auto LoadShaderSource(const char* filename)->std::vector<GLchar>
        {
            auto shaderSource = std::vector<GLchar>();
            auto sourceFile = std::ifstream(filename, std::ios::binary);
            if (sourceFile.is_open()) {
                sourceFile.seekg(0, std::ios::end);
                auto size = static_cast<size_t>(sourceFile.tellg());
                shaderSource.resize(size / sizeof(shaderSource[0]));
                sourceFile.seekg(0, std::ios::beg);
                sourceFile.read((char*)shaderSource.data(), size);
                sourceFile.close();
            }
            return shaderSource;
        }

        inline auto LoadBinary(const char* filename)->std::vector<uint32_t>
        {
            auto shaderBinary = std::vector<uint32_t>();
            auto sourceFile = std::ifstream(filename, std::ios::binary);
            if (sourceFile.is_open()) {
                sourceFile.seekg(0, std::ios::end);
                auto size = static_cast<size_t>(sourceFile.tellg());
                shaderBinary.resize(size / sizeof(shaderBinary[0]));
                sourceFile.seekg(0, std::ios::beg);
                sourceFile.read((char*)shaderBinary.data(), size);
                sourceFile.close();
            }
            return shaderBinary;
        }

        inline auto GetDefaultSceneJson()->nlohmann::json {
            auto cameraController = RTLib::Utils::CameraController({ 0.0f, 1.0f, 5.0f });
            cameraController.SetMovementSpeed(10.0f);
            return {
                {"ObjModels",
                    {
                        {"CacheDir", std::filesystem::absolute(".").string()},
                        {"Assets",
                            {{"CornellBox-Original", RTLIB_EXT_OPX7_TEST_DATA_PATH "/Models/CornellBox/CornellBox-Original.obj"}}
                        }
                    }
                },
                {"World",
                 {
                     {"Geometries",
                        {{"CornellBox-Original", {{"Type", "ObjModel"},{"Base", "CornellBox-Original"}}}}
                     },
                     {"GeometryASs",
                        {{"CornellBox-Original", {{"Type", "GeometryAS"}, {"Geometries", std::vector<std::string>{"CornellBox-Original"}}}}}
                      },
                     {"Instances",
                        {{"Instance0",{{"Type", "Instance"}, {"ASType", "Geometry"},{"Base", "CornellBox-Original"},{"Transform",std::vector<float>{
                            1.0f,0.0f,0.0f,0.0f,
                            0.0f,1.0f,0.0f,0.0f,
                            0.0f,0.0f,1.0f,0.0f
                        }}}}}
                     },
                     {"InstanceASs",
                      {{"Root",{{"Type", "InstanceAS"},{"Instances", std::vector<std::string>{"Instance0"}}}}}
                     },
                 }},
                {"CameraController", cameraController},
                {"Config",{
                    {"Width" ,1024},
                    {"Height",1024},
                    {"Samples",1},
                    {"MaxSamples",10000},
                }}
            };
        }
        
        inline void RenderFrameGL(
            RTLib::Ext::GL::GLContext*      context,
            RTLib::Ext::GL::GLRectRenderer* rectRenderer,
            RTLib::Ext::GL::GLBuffer *      frameBuffer,
            RTLib::Ext::GL::GLTexture*      frameTexture
        ) {
            auto extent = frameTexture->GetImage()->GetExtent();
            context->SetClearBuffer(RTLib::Ext::GL::GLClearBufferFlagsColor);
            context->SetClearColor(0.0f, 0.0f, 0.0f, 0.0f);
            context->CopyBufferToImage(frameBuffer, frameTexture->GetImage(),
                { RTLib::Ext::GL::GLBufferImageCopy{
                    ((size_t)0),
                    ((size_t)0),
                    ((size_t)0),
                    {((uint32_t)0), ((uint32_t)0), ((uint32_t)1)},
                    {0, 0, 0},
                    {((uint32_t)extent.width), ((uint32_t)extent.height), ((uint32_t)1)},
                } });
            rectRenderer->DrawTexture(frameTexture);
        }

        inline auto UpdateCameraMovement(
            RTLib::Utils::CameraController& cameraController,
            RTLib::Ext::GLFW::GLFWWindow* window,
            float delTime,
            float delCurPosX, 
            float delCurPosY){
            bool isMovedCamera = false;
            const bool pressKeyLeft     = (glfwGetKey(window->GetGLFWwindow(), GLFW_KEY_A) == GLFW_PRESS) || (glfwGetKey(window->GetGLFWwindow(), GLFW_KEY_LEFT) == GLFW_PRESS);
            const bool pressKeyRight    = (glfwGetKey(window->GetGLFWwindow(), GLFW_KEY_D) == GLFW_PRESS) || (glfwGetKey(window->GetGLFWwindow(), GLFW_KEY_RIGHT) == GLFW_PRESS);
            const bool pressKeyForward  = (glfwGetKey(window->GetGLFWwindow(), GLFW_KEY_W) == GLFW_PRESS);
            const bool pressKeyBackward = (glfwGetKey(window->GetGLFWwindow(), GLFW_KEY_S) == GLFW_PRESS);
            const bool pressKeyUp       = (glfwGetKey(window->GetGLFWwindow(), GLFW_KEY_UP) == GLFW_PRESS);
            const bool pressKeyDown     = (glfwGetKey(window->GetGLFWwindow(), GLFW_KEY_DOWN) == GLFW_PRESS);
            if (pressKeyLeft    )
            {
                cameraController.ProcessKeyboard(RTLib::Utils::CameraMovement::eLeft, delTime);
                isMovedCamera = true;
            }
            if (pressKeyRight   )
            {
                cameraController.ProcessKeyboard(RTLib::Utils::CameraMovement::eRight, delTime);
                isMovedCamera = true;
            }
            if (pressKeyForward )
            {
                cameraController.ProcessKeyboard(RTLib::Utils::CameraMovement::eForward, delTime);
                isMovedCamera = true;
            }
            if (pressKeyBackward)
            {
                cameraController.ProcessKeyboard(RTLib::Utils::CameraMovement::eBackward,  delTime);
                isMovedCamera = true;
            }
            if (pressKeyUp      )
            {
                cameraController.ProcessKeyboard(RTLib::Utils::CameraMovement::eUp,  delTime);
                isMovedCamera = true;
            }
            if (pressKeyDown    )
            {
                cameraController.ProcessKeyboard(RTLib::Utils::CameraMovement::eDown,  delTime);
                isMovedCamera = true;
            }
            if (glfwGetMouseButton(window->GetGLFWwindow(), GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS)
            {
                cameraController.ProcessMouseMovement(-delCurPosX, delCurPosY);
                isMovedCamera = true;
            }
            return isMovedCamera;
        }

        class OPX7MeshSharedResourceExtData:public RTLib::Core::MeshSharedResourceExtData {
        public:
            static auto New(MeshSharedResource* pMeshSharedResource, OPX7::OPX7Context* context)noexcept->OPX7MeshSharedResourceExtData* {
                auto extData = new OPX7MeshSharedResourceExtData(pMeshSharedResource);
                auto parent  = extData->GetParent();
                extData->m_VertexBuffer = std::unique_ptr<RTLib::Ext::CUDA::CUDABuffer>(context->CreateBuffer(
                    {RTLib::Ext::CUDA::CUDAMemoryFlags::eDefault, sizeof(float3) * std::size(parent->vertexBuffer), std::data(parent->vertexBuffer)}
                ));
                extData->m_NormalBuffer = std::unique_ptr<RTLib::Ext::CUDA::CUDABuffer>(context->CreateBuffer(
                    { RTLib::Ext::CUDA::CUDAMemoryFlags::eDefault, sizeof(float3) * std::size(parent->normalBuffer), std::data(parent->normalBuffer) }
                ));
                extData->m_TexCrdBuffer = std::unique_ptr<RTLib::Ext::CUDA::CUDABuffer>(context->CreateBuffer(
                    { RTLib::Ext::CUDA::CUDAMemoryFlags::eDefault, sizeof(float2) * std::size(parent->texCrdBuffer), std::data(parent->texCrdBuffer) }
                ));
                return extData;
            }
            virtual ~OPX7MeshSharedResourceExtData()noexcept {}
            void Destroy() {
                m_VertexBuffer->Destroy();
                m_NormalBuffer->Destroy();
                m_TexCrdBuffer->Destroy();
            }
            //SET
            void SetVertexStrideInBytes(size_t vertexStride)noexcept {m_VertexStrideInBytes = vertexStride;}
            void SetVertexFormat(OptixVertexFormat format)noexcept {m_VertexFormat = format;}
            //GET
            auto GetVertexStrideInBytes()const noexcept -> size_t { return m_VertexStrideInBytes; }
            auto GetVertexFormat()const noexcept -> OptixVertexFormat { return m_VertexFormat; }
            auto GetVertexCount()const noexcept -> size_t {
                if (m_VertexBuffer && m_VertexStrideInBytes > 0) { return m_VertexBuffer->GetSizeInBytes() / m_VertexStrideInBytes; }
                return 0;
            }
            auto GetVertexBufferGpuAddress()const noexcept -> CUdeviceptr { return CUDA::CUDANatives::GetCUdeviceptr(m_VertexBuffer.get()); }
            auto GetNormalBufferGpuAddress()const noexcept -> CUdeviceptr { return CUDA::CUDANatives::GetCUdeviceptr(m_NormalBuffer.get()); }
            auto GetTexCrdBufferGpuAddress()const noexcept -> CUdeviceptr { return CUDA::CUDANatives::GetCUdeviceptr(m_TexCrdBuffer.get()); }

            auto GetVertexBufferView()const noexcept -> CUDA::CUDABufferView
            {
                return CUDA::CUDABufferView(m_VertexBuffer.get());
            }
            auto GetNormalBufferView()const noexcept -> CUDA::CUDABufferView
            {
                return CUDA::CUDABufferView(m_NormalBuffer.get());
            }
            auto GetTexCrdBufferView()const noexcept -> CUDA::CUDABufferView
            {
                return CUDA::CUDABufferView(m_TexCrdBuffer.get());
            }
        private:
            OPX7MeshSharedResourceExtData(MeshSharedResource* pMeshSharedResource) noexcept :MeshSharedResourceExtData(pMeshSharedResource) {}
        private:
            std::unique_ptr<RTLib::Ext::CUDA::CUDABuffer> m_VertexBuffer = nullptr;
            std::unique_ptr<RTLib::Ext::CUDA::CUDABuffer> m_NormalBuffer = nullptr;
            std::unique_ptr<RTLib::Ext::CUDA::CUDABuffer> m_TexCrdBuffer = nullptr;
            size_t m_VertexStrideInBytes = 0;
            OptixVertexFormat m_VertexFormat = OPTIX_VERTEX_FORMAT_NONE;
        };

        class OPX7MeshUniqueResourceExtData : public RTLib::Core::MeshUniqueResourceExtData {
        public:
            static auto New(MeshUniqueResource* pMeshUniqueResource, OPX7::OPX7Context* context)noexcept->OPX7MeshUniqueResourceExtData* {
                auto extData = new OPX7MeshUniqueResourceExtData(pMeshUniqueResource);
                auto parent  = extData->GetParent();
                extData->m_TriIdxBuffer = std::unique_ptr<RTLib::Ext::CUDA::CUDABuffer>(context->CreateBuffer(
                    { RTLib::Ext::CUDA::CUDAMemoryFlags::eDefault, sizeof(uint32_t)*3 * std::size(parent->triIndBuffer), std::data(parent->triIndBuffer) }
                ));
                extData->m_MatIdxBuffer = std::unique_ptr<RTLib::Ext::CUDA::CUDABuffer>(context->CreateBuffer(
                    { RTLib::Ext::CUDA::CUDAMemoryFlags::eDefault, sizeof(uint32_t) * std::size(parent->matIndBuffer), std::data(parent->matIndBuffer) }
                ));

                return extData;
            }
            virtual ~OPX7MeshUniqueResourceExtData()noexcept {}
            void Destroy() {
                m_TriIdxBuffer.reset();
            }
            //SET
            void SetTriIdxStrideInBytes(size_t indexStride)noexcept { m_TriIdxStride = indexStride; }
            void SetTriIdxFormat(OptixIndicesFormat format)noexcept { m_IndicesFormat = format; }
            //GET
            auto GetTriIdxBufferGpuAddress()const noexcept -> CUdeviceptr { return CUDA::CUDANatives::GetCUdeviceptr(m_TriIdxBuffer.get()); }
            auto GetTriIdxCount()const noexcept -> size_t {
                if (m_TriIdxBuffer && m_TriIdxStride > 0) { return m_TriIdxBuffer->GetSizeInBytes() / m_TriIdxStride; }
                return 0;
            }
            auto GetTriIdxStrideInBytes()const noexcept -> size_t { return m_TriIdxStride; }
            auto GetTriIdxFormat()const noexcept -> OptixIndicesFormat { return m_IndicesFormat; }
            //
            auto GetMatIdxBufferGpuAddress()const noexcept -> CUdeviceptr { return CUDA::CUDANatives::GetCUdeviceptr(m_MatIdxBuffer.get()); };
            auto GetMatIdxCount()const noexcept -> size_t {
                if (m_MatIdxBuffer) { return m_MatIdxBuffer->GetSizeInBytes() / sizeof(uint32_t); }
                return 0;
            }
            //
            auto GetTriIdxBufferView()const noexcept -> CUDA::CUDABufferView
            {
                return CUDA::CUDABufferView(m_TriIdxBuffer.get());
            }
            auto GetMatIdxBufferView()const noexcept -> CUDA::CUDABufferView
            {
                return CUDA::CUDABufferView(m_MatIdxBuffer.get());
            }
        private:
            OPX7MeshUniqueResourceExtData(MeshUniqueResource* pMeshUniqueResource) noexcept :MeshUniqueResourceExtData(pMeshUniqueResource) {}
        private:
            std::unique_ptr<RTLib::Ext::CUDA::CUDABuffer> m_TriIdxBuffer = nullptr;
            std::unique_ptr<RTLib::Ext::CUDA::CUDABuffer> m_MatIdxBuffer = nullptr;
            OptixIndicesFormat m_IndicesFormat = OPTIX_INDICES_FORMAT_NONE;
            size_t m_TriIdxStride = 0;
        };

        void InitMeshGroupExtData(RTLib::Ext::OPX7::OPX7Context* opx7Context, RTLib::Core::MeshGroupPtr meshGroup)
        {
            {
                auto sharedResource = meshGroup->GetSharedResource();
                sharedResource->AddExtData<rtlib::test::OPX7MeshSharedResourceExtData>(opx7Context );
                auto extData = static_cast<rtlib::test::OPX7MeshSharedResourceExtData*>(sharedResource->extData.get());
                extData->SetVertexFormat(OPTIX_VERTEX_FORMAT_FLOAT3);
                extData->SetVertexStrideInBytes(sizeof(float) * 3);
            }
            for (auto& [name, uniqueResource] : meshGroup->GetUniqueResources())
            {
                uniqueResource->AddExtData<rtlib::test::OPX7MeshUniqueResourceExtData>(opx7Context );
                auto extData = static_cast<rtlib::test::OPX7MeshUniqueResourceExtData*>(uniqueResource->extData.get());
                extData->SetTriIdxFormat(OPTIX_INDICES_FORMAT_UNSIGNED_INT3);
                extData->SetTriIdxStrideInBytes(sizeof(uint32_t) * 3);
            }
        }
    }

}
#endif