#ifndef RTLIB_EXT_OPX7_TEST_H
#define RTLIB_EXT_OPX7_TEST_H
#include <RTLib/Core/Exceptions.h>
#include <RTLib/Core/World.h>
#include <RTLib/Core/BinaryReader.h>
#include <RTLib/Core/AABB.h>
#include <RTLib/Core/Camera.h>
#include <RTLib/Core/Scene.h>
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
#include <RTLib/Ext/CUDA/Math/Matrix.h>
#include <RTLib/Ext/CUGL/CUGLBuffer.h>
#include <RTLib/Ext/GLFW/GLFWContext.h>
#include <RTLib/Ext/GLFW/GL/GLFWOpenGLWindow.h>
#include <RTLib/Ext/GLFW/GL/GLFWOpenGLContext.h>
#include <RTLib/Ext/GL/GLRectRenderer.h>
#include <RTLib/Ext/GL/GLTexture.h>
#include <RTLib/Ext/GL/GLImage.h>
#include <RTLib/Ext/OPX7/Utils/OPX7UtilsPathGuiding.h>
#include <optix_stubs.h>
#include <GLFW/glfw3.h>
#include <stb_image.h>
#include <stb_image_write.h>
#include <tinyexr.h>
#include <cuda/SimpleKernel.h>
#include <RTLibExtOPX7TestConfig.h>
#include <memory>
#include <thread>
#include <iostream>
#include <filesystem>
#include <unordered_map>
#include <unordered_set>
#include <optional>
#include <random>
#include <fstream>
#include <ctime>
#include <stack>
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
        using namespace RTLib::Ext;
        using namespace RTLib::Ext::CUDA::Math;
        using BottomLevelAccelerationStructureData = RTLib::Ext::OPX7::OPX7Natives::AccelBuildOutput;

        inline auto TransformAABB(const AABB& inAABB,const RTLib::Ext::CUDA::Math::Matrix4x4& transform) noexcept->AABB
        {
            std::vector<float3> inPositions =
            {
                make_float3(inAABB.min[0] ,  inAABB.min[1] ,inAABB.min[2]),
                make_float3(inAABB.max[0] ,  inAABB.min[1] ,inAABB.min[2]),
                make_float3(inAABB.max[0] ,  inAABB.max[1] ,inAABB.min[2]),
                make_float3(inAABB.min[0] ,  inAABB.max[1] ,inAABB.min[2]),
                make_float3(inAABB.min[0] ,  inAABB.min[1] ,inAABB.max[2]),
                make_float3(inAABB.max[0] ,  inAABB.min[1] ,inAABB.max[2]),
                make_float3(inAABB.max[0] ,  inAABB.max[1] ,inAABB.max[2]),
                make_float3(inAABB.min[0] ,  inAABB.max[1] ,inAABB.max[2])
            };

            AABB aabb;
            for (auto& inPosition : inPositions)
            {
                float4 transformPos = transform * make_float4(inPosition, 1.0f);
                aabb.Update({ {transformPos.x / transformPos.w,transformPos.y / transformPos.w,transformPos.z / transformPos.w} });
            }
            return aabb;
        }

        struct GeometryAccelerationStructureData
        {
            std::unique_ptr<CUDA::CUDABuffer> buffer;
            OptixTraversableHandle            handle;
            std::array<float, 3>              aabbMin;
            std::array<float, 3>              aabbMax;
        };
        struct InstanceAccelerationStructureData 
        {
            std::unique_ptr<CUDA::CUDABuffer> buffer;
            OptixTraversableHandle            handle;
            std::unique_ptr<CUDA::CUDABuffer> instanceBuffer;
            std::vector<OptixInstance>        instanceArray;
            void Build(RTLib::Ext::OPX7::OPX7Context* context, const OptixAccelBuildOptions& options)
            {
                if (buffer) { return; }
                if (instanceBuffer) {
                    if (instanceBuffer->GetSizeInBytes() != instanceArray.size() * sizeof(instanceArray[0])) {
                        instanceBuffer->Destroy();
                        instanceBuffer = std::unique_ptr<RTLib::Ext::CUDA::CUDABuffer>(context->CreateBuffer(
                            RTLib::Ext::CUDA::CUDABufferCreateDesc{
                                RTLib::Ext::CUDA::CUDAMemoryFlags::eDefault,
                                sizeof(instanceArray[0])* instanceArray.size(),
                                instanceArray.data()
                            })
                        );
                    }
                    else {
                        RTLIB_CORE_ASSERT_IF_FAILED(context->CopyMemoryToBuffer(instanceBuffer.get(), { {instanceArray.data(),(size_t)0, instanceArray.size()*sizeof(instanceArray[0])}}));
                    }
                }
                else {
                    instanceBuffer = std::unique_ptr<RTLib::Ext::CUDA::CUDABuffer>(context->CreateBuffer(
                        RTLib::Ext::CUDA::CUDABufferCreateDesc{
                            RTLib::Ext::CUDA::CUDAMemoryFlags::eDefault,
                            sizeof(instanceArray[0]) * instanceArray.size(),
                            instanceArray.data()
                        })
                    );
                }

                auto buildInputs = std::vector<OptixBuildInput>(1);
                {
                    buildInputs[0].type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
                    buildInputs[0].instanceArray.instances = RTLib::Ext::CUDA::CUDANatives::GetCUdeviceptr(instanceBuffer.get());
                    buildInputs[0].instanceArray.numInstances = instanceArray.size();
                }

                auto [tmpBuffer, tmpHandle] = RTLib::Ext::OPX7::OPX7Natives::BuildAccelerationStructure(context, options, buildInputs);
                
                buffer = std::move(tmpBuffer);
                handle = tmpHandle;
            }
        };


        class OPX7MeshSharedResourceExtData :public RTLib::Core::MeshSharedResourceExtData {
         public:
            static auto New(MeshSharedResource* pMeshSharedResource, OPX7::OPX7Context* context)noexcept->OPX7MeshSharedResourceExtData* {
                auto extData = new OPX7MeshSharedResourceExtData(pMeshSharedResource);
                auto parent = extData->GetParent();
                extData->m_VertexBuffer = std::unique_ptr<RTLib::Ext::CUDA::CUDABuffer>(context->CreateBuffer(
                    { RTLib::Ext::CUDA::CUDAMemoryFlags::eDefault, sizeof(float3) * std::size(parent->vertexBuffer), std::data(parent->vertexBuffer) }
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
                if (m_VertexBuffer) {
                    m_VertexBuffer->Destroy();
                    m_VertexBuffer.reset();
                }
                if (m_NormalBuffer) {
                    m_NormalBuffer->Destroy();
                    m_NormalBuffer.reset();
                }
                if (m_TexCrdBuffer) {
                    m_TexCrdBuffer->Destroy();
                    m_TexCrdBuffer.reset();
                }
            }
            //SET
            void SetVertexStrideInBytes(size_t vertexStride)noexcept { m_VertexStrideInBytes = vertexStride; }
            void SetVertexFormat(OptixVertexFormat format)noexcept { m_VertexFormat = format; }
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
                auto parent = extData->GetParent();
                extData->m_TriIdxBuffer = std::unique_ptr<RTLib::Ext::CUDA::CUDABuffer>(context->CreateBuffer(
                    { RTLib::Ext::CUDA::CUDAMemoryFlags::eDefault, sizeof(uint32_t) * 3 * std::size(parent->triIndBuffer), std::data(parent->triIndBuffer) }
                ));
                extData->m_MatIdxBuffer = std::unique_ptr<RTLib::Ext::CUDA::CUDABuffer>(context->CreateBuffer(
                    { RTLib::Ext::CUDA::CUDAMemoryFlags::eDefault, sizeof(uint32_t) * std::size(parent->matIndBuffer), std::data(parent->matIndBuffer) }
                ));

                return extData;
            }
            virtual ~OPX7MeshUniqueResourceExtData()noexcept {}
            void Destroy() {
                if (m_MatIdxBuffer) {
                    m_MatIdxBuffer->Destroy();
                    m_MatIdxBuffer.reset();
                }
                if (m_TriIdxBuffer) {
                    m_TriIdxBuffer->Destroy();
                    m_TriIdxBuffer.reset();
                }
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

        inline void InitMeshGroupExtData(RTLib::Ext::OPX7::OPX7Context* opx7Context, RTLib::Core::MeshGroupPtr meshGroup)
        {
            {
                auto sharedResource = meshGroup->GetSharedResource();
                sharedResource->AddExtData<rtlib::test::OPX7MeshSharedResourceExtData>(opx7Context);
                auto extData = static_cast<rtlib::test::OPX7MeshSharedResourceExtData*>(sharedResource->extData.get());
                extData->SetVertexFormat(OPTIX_VERTEX_FORMAT_FLOAT3);
                extData->SetVertexStrideInBytes(sizeof(float) * 3);
            }
            for (auto& [name, uniqueResource] : meshGroup->GetUniqueResources())
            {
                uniqueResource->AddExtData<rtlib::test::OPX7MeshUniqueResourceExtData>(opx7Context);
                auto extData = static_cast<rtlib::test::OPX7MeshUniqueResourceExtData*>(uniqueResource->extData.get());
                extData->SetTriIdxFormat(OPTIX_INDICES_FORMAT_UNSIGNED_INT3);
                extData->SetTriIdxStrideInBytes(sizeof(uint32_t) * 3);
            }
        }

        class OPX7SphereResourceExtData:public SphereResourceExtData {
        public:
            static auto New(SphereResource* pSphereResource, OPX7::OPX7Context* context)noexcept->OPX7SphereResourceExtData* {
                auto extData = new OPX7SphereResourceExtData(pSphereResource);
                auto parent = extData->GetParent();
                extData->m_CenterBuffer = std::unique_ptr<RTLib::Ext::CUDA::CUDABuffer>(context->CreateBuffer(
                    { RTLib::Ext::CUDA::CUDAMemoryFlags::eDefault, sizeof(float) * 3 * std::size(parent->centerBuffer), std::data(parent->centerBuffer) }
                ));
                extData->m_RadiusBuffer = std::unique_ptr<RTLib::Ext::CUDA::CUDABuffer>(context->CreateBuffer(
                    { RTLib::Ext::CUDA::CUDAMemoryFlags::eDefault, sizeof(float)     * std::size(parent->radiusBuffer), std::data(parent->radiusBuffer) }
                ));
                extData->m_MatIdxBuffer = std::unique_ptr<RTLib::Ext::CUDA::CUDABuffer>(context->CreateBuffer(
                    { RTLib::Ext::CUDA::CUDAMemoryFlags::eDefault, sizeof(uint32_t)  * std::size(parent->matIndBuffer), std::data(parent->matIndBuffer) }
                ));

                return extData;
            }
            virtual ~OPX7SphereResourceExtData()noexcept {}
            void Destroy() {
                if (m_CenterBuffer) {
                    m_CenterBuffer->Destroy();
                    m_CenterBuffer.reset();
                }
                if (m_RadiusBuffer) {
                    m_RadiusBuffer->Destroy();
                    m_RadiusBuffer.reset();
                }
                if (m_MatIdxBuffer) {
                    m_MatIdxBuffer->Destroy();
                    m_MatIdxBuffer.reset();
                }
            }
            //SET
            void SetCenterStrideInBytes(size_t vertexStride)noexcept { m_CenterStrideInBytes = vertexStride; }
            void SetCenterFormat(OptixVertexFormat format)noexcept { m_CenterFormat = format; }
            void SetRadiusStrideInBytes(size_t radiusStride)noexcept { m_RadiusStrideInBytes = radiusStride; }
            //GET
            auto GetCenterStrideInBytes()const noexcept -> size_t { return m_CenterStrideInBytes; }
            auto GetCenterFormat()const noexcept -> OptixVertexFormat { return m_CenterFormat; }
            auto GetCenterCount()const noexcept -> size_t {
                if (m_CenterBuffer && m_CenterStrideInBytes > 0) { return m_CenterBuffer->GetSizeInBytes() / m_CenterStrideInBytes; }
                return 0;
            }
            auto GetCenterBufferGpuAddress()const noexcept -> CUdeviceptr { return CUDA::CUDANatives::GetCUdeviceptr(m_CenterBuffer.get()); }
            //
            auto GetRadiusStrideInBytes()const noexcept -> size_t { return m_RadiusStrideInBytes; }
            auto GetRadiusCount()const noexcept -> size_t {
                if (m_RadiusBuffer && m_RadiusStrideInBytes > 0) { return m_RadiusBuffer->GetSizeInBytes() / m_RadiusStrideInBytes; }
                return 0;
            }
            auto GetRadiusBufferGpuAddress()const noexcept -> CUdeviceptr { return CUDA::CUDANatives::GetCUdeviceptr(m_RadiusBuffer.get()); }
            //
            auto GetMatIdxBufferGpuAddress()const noexcept -> CUdeviceptr { return CUDA::CUDANatives::GetCUdeviceptr(m_MatIdxBuffer.get()); };
            auto GetMatIdxCount()const noexcept -> size_t {
                if (m_MatIdxBuffer) { return m_MatIdxBuffer->GetSizeInBytes() / sizeof(uint32_t); }
                return 0;
            }
            //
            auto GetCenterBufferView()const noexcept -> CUDA::CUDABufferView
            {
                return CUDA::CUDABufferView(m_CenterBuffer.get());
            }
            auto GetRadiusBufferView()const noexcept -> CUDA::CUDABufferView
            {
                return CUDA::CUDABufferView(m_RadiusBuffer.get());
            }
            auto GetMatIdxBufferView()const noexcept -> CUDA::CUDABufferView
            {
                return CUDA::CUDABufferView(m_MatIdxBuffer.get());
            }
        private:
            OPX7SphereResourceExtData(SphereResource* pSphereUniqueResource) noexcept :SphereResourceExtData(pSphereUniqueResource) {}
            std::unique_ptr<RTLib::Ext::CUDA::CUDABuffer> m_CenterBuffer = nullptr;
            std::unique_ptr<RTLib::Ext::CUDA::CUDABuffer> m_RadiusBuffer = nullptr;
            std::unique_ptr<RTLib::Ext::CUDA::CUDABuffer> m_MatIdxBuffer = nullptr;
            size_t m_CenterStrideInBytes = 0;
            size_t m_RadiusStrideInBytes = 0;
            OptixVertexFormat m_CenterFormat = OPTIX_VERTEX_FORMAT_NONE;
        };

        inline void InitSphereResExtData(RTLib::Ext::OPX7::OPX7Context* opx7Context, RTLib::Core::SphereResourcePtr sphereRes) {
            sphereRes->AddExtData<rtlib::test::OPX7SphereResourceExtData>(opx7Context);
            auto extData = sphereRes->GetExtData<rtlib::test::OPX7SphereResourceExtData>();
            extData->SetCenterFormat(OPTIX_VERTEX_FORMAT_FLOAT3);
            extData->SetCenterStrideInBytes(sizeof(float) * 3);
            extData->SetRadiusStrideInBytes(sizeof(float));
        }

        struct TextureData
        {
            std::unique_ptr<RTLib::Ext::CUDA::CUDATexture> handle;
            std::unique_ptr<RTLib::Ext::CUDA::CUDAImage>   image;

            auto GetCUtexObject()const noexcept -> CUtexObject
            {
                return RTLib::Ext::CUDA::CUDANatives::GetCUtexObject(handle.get());
            }

            void LoadFromPath(RTLib::Ext::CUDA::CUDAContext* context, const std::string& filePath)
            {
                int  texWid, texHei, texComp;
                auto pixelData = stbi_load(filePath.c_str(), &texWid, &texHei, &texComp, 4);
                auto imgData = std::vector<unsigned char>(texWid * texHei * 4, 255);
                {
                    for (size_t i = 0; i < texHei; ++i) {
                        auto srcData = pixelData + 4 * texWid * (texHei - 1 - i);
                        auto dstData = imgData.data() + 4 * texWid * i;
                        std::memcpy(dstData, srcData, 4 * texWid);
                    }
                }
                stbi_image_free(pixelData);

                {
                    auto imgDesc = RTLib::Ext::CUDA::CUDAImageCreateDesc();
                    imgDesc.imageType = RTLib::Ext::CUDA::CUDAImageType::e2D;
                    imgDesc.extent.width = texWid;
                    imgDesc.extent.height = texHei;
                    imgDesc.extent.depth = 0;
                    imgDesc.format = RTLib::Ext::CUDA::CUDAImageFormat::eUInt8X4;
                    imgDesc.mipLevels = 1;
                    imgDesc.arrayLayers = 1;
                    imgDesc.flags = RTLib::Ext::CUDA::CUDAImageCreateFlagBitsDefault;
                    image = std::unique_ptr<RTLib::Ext::CUDA::CUDAImage>(context->CreateImage(imgDesc));
                    std::cout << "Image: " << RTLib::Ext::CUDA::CUDANatives::GetCUarray(image.get()) << std::endl;
                }
                {
                    auto memoryImageCopy = RTLib::Ext::CUDA::CUDAMemoryImageCopy();
                    memoryImageCopy.srcData = imgData.data();
                    memoryImageCopy.dstImageExtent = { (uint32_t)texWid, (uint32_t)texHei, (uint32_t)0 };
                    memoryImageCopy.dstImageOffset = {};
                    memoryImageCopy.dstImageSubresources.baseArrayLayer = 0;
                    memoryImageCopy.dstImageSubresources.layerCount = 1;
                    memoryImageCopy.dstImageSubresources.mipLevel = 0;
                    RTLIB_CORE_ASSERT_IF_FAILED(context->CopyMemoryToImage(image.get(), { memoryImageCopy }));
                }
                {
                    auto texImgDesc = RTLib::Ext::CUDA::CUDATextureImageCreateDesc();
                    texImgDesc.image = image.get();
                    texImgDesc.sampler.addressMode[0] = RTLib::Ext::CUDA::CUDATextureAddressMode::eWarp;
                    texImgDesc.sampler.addressMode[1] = RTLib::Ext::CUDA::CUDATextureAddressMode::eWarp;
                    texImgDesc.sampler.addressMode[2] = RTLib::Ext::CUDA::CUDATextureAddressMode::eWarp;
                    texImgDesc.sampler.borderColor[0] = 1.0f;
                    texImgDesc.sampler.borderColor[1] = 1.0f;
                    texImgDesc.sampler.borderColor[2] = 1.0f;
                    texImgDesc.sampler.borderColor[3] = 1.0f;
                    texImgDesc.sampler.filterMode = RTLib::Ext::CUDA::CUDATextureFilterMode::eLinear;
                    texImgDesc.sampler.mipmapFilterMode = RTLib::Ext::CUDA::CUDATextureFilterMode::eLinear;
                    texImgDesc.sampler.mipmapLevelBias = 0.0f;
                    texImgDesc.sampler.maxMipmapLevelClamp = 99.0f;
                    texImgDesc.sampler.minMipmapLevelClamp = 0.0f;
                    texImgDesc.sampler.maxAnisotropy = 1;
                    texImgDesc.sampler.flags = RTLib::Ext::CUDA::CUDATextureFlagBitsNormalizedCordinates;
                    handle = std::unique_ptr<RTLib::Ext::CUDA::CUDATexture>(context->CreateTexture(texImgDesc));
                }
            }
            void Destroy() {
                if (handle) {
                    handle->Destroy();
                    handle = nullptr;
                }
                if (image) {
                    image->Destroy();
                    image = nullptr;
                }
            }

            static auto Color(RTLib::Ext::CUDA::CUDAContext* context, const uchar4 col)->TextureData
            {
                auto imgData = std::vector<uchar4>(32 * 32, col);
                auto texData = rtlib::test::TextureData();
                {
                    auto imgDesc = RTLib::Ext::CUDA::CUDAImageCreateDesc();
                    imgDesc.imageType = RTLib::Ext::CUDA::CUDAImageType::e2D;
                    imgDesc.extent.width = 32;
                    imgDesc.extent.height = 32;
                    imgDesc.extent.depth = 0;
                    imgDesc.format = RTLib::Ext::CUDA::CUDAImageFormat::eUInt8X4;
                    imgDesc.mipLevels = 1;
                    imgDesc.arrayLayers = 1;
                    imgDesc.flags = RTLib::Ext::CUDA::CUDAImageCreateFlagBitsDefault;
                    texData.image = std::unique_ptr<RTLib::Ext::CUDA::CUDAImage>(context->CreateImage(imgDesc));
                    std::cout << "Image: " << RTLib::Ext::CUDA::CUDANatives::GetCUarray(texData.image.get()) << std::endl;
                }
                {
                    auto memoryImageCopy = RTLib::Ext::CUDA::CUDAMemoryImageCopy();
                    memoryImageCopy.srcData = imgData.data();
                    memoryImageCopy.dstImageExtent = { (uint32_t)32, (uint32_t)32, (uint32_t)0 };
                    memoryImageCopy.dstImageOffset = {};
                    memoryImageCopy.dstImageSubresources.baseArrayLayer = 0;
                    memoryImageCopy.dstImageSubresources.layerCount = 1;
                    memoryImageCopy.dstImageSubresources.mipLevel = 0;
                    RTLIB_CORE_ASSERT_IF_FAILED(context->CopyMemoryToImage(texData.image.get(), { memoryImageCopy }));
                }
                {
                    auto texImgDesc = RTLib::Ext::CUDA::CUDATextureImageCreateDesc();
                    texImgDesc.image = texData.image.get();
                    texImgDesc.sampler.addressMode[0] = RTLib::Ext::CUDA::CUDATextureAddressMode::eClamp;
                    texImgDesc.sampler.addressMode[1] = RTLib::Ext::CUDA::CUDATextureAddressMode::eClamp;
                    texImgDesc.sampler.addressMode[2] = RTLib::Ext::CUDA::CUDATextureAddressMode::eClamp;
                    texImgDesc.sampler.borderColor[0] = 1.0f;
                    texImgDesc.sampler.borderColor[1] = 1.0f;
                    texImgDesc.sampler.borderColor[2] = 1.0f;
                    texImgDesc.sampler.borderColor[3] = 1.0f;
                    texImgDesc.sampler.filterMode = RTLib::Ext::CUDA::CUDATextureFilterMode::eLinear;
                    texImgDesc.sampler.mipmapFilterMode = RTLib::Ext::CUDA::CUDATextureFilterMode::eLinear;
                    texImgDesc.sampler.mipmapLevelBias = 0.0f;
                    texImgDesc.sampler.maxMipmapLevelClamp = 0.0f;
                    texImgDesc.sampler.minMipmapLevelClamp = 0.0f;
                    texImgDesc.sampler.maxAnisotropy = 0;
                    texImgDesc.sampler.flags = RTLib::Ext::CUDA::CUDATextureFlagBitsNormalizedCordinates;
                    texData.handle = std::unique_ptr<RTLib::Ext::CUDA::CUDATexture>(context->CreateTexture(texImgDesc));
                }
                return texData;
            }
            static auto Black(RTLib::Ext::CUDA::CUDAContext* context)->TextureData
            {
                return Color(context, make_uchar4(0, 0, 0, 0));
            }
            static auto White(RTLib::Ext::CUDA::CUDAContext* context)->TextureData
            {
                return Color(context, make_uchar4(255, 255, 255, 255));
            }
        };

        struct SceneData: public RTLib::Core::SceneData
        {
            void InitExtData(RTLib::Ext::OPX7::OPX7Context* opx7Context) {
                for (auto& [name, geometry] : world.geometryObjModels)
                {
                    rtlib::test::InitMeshGroupExtData(opx7Context, objAssetManager.GetAsset(geometry.base).meshGroup);
                }
                for (auto& [name, sphere] : sphereResources)
                {
                    rtlib::test::InitSphereResExtData(opx7Context, sphere);
                }
            }
            auto BuildGeometryASs(RTLib::Ext::OPX7::OPX7Context* opx7Context, const OptixAccelBuildOptions& accelBuildOptions)const noexcept -> std::unordered_map<std::string, GeometryAccelerationStructureData>
            {
                std::unordered_map<std::string, GeometryAccelerationStructureData> geometryASs = {};
                for (auto& [geometryASName,geometryASData] : world.geometryASs)
                {
                    AABB aabb;
                    auto buildInputs = std::vector<OptixBuildInput>();
                    auto holders = std::vector<std::unique_ptr<RTLib::Ext::OPX7::OPX7Geometry::Holder>>();
                    for (auto& geometryName : geometryASData.geometries)
                    {
                        if (world.geometryObjModels.count(geometryName) > 0) {
                            auto& geometryObjModel = world.geometryObjModels.at(geometryName);
                            auto& objAsset = objAssetManager.GetAsset(geometryObjModel.base);
                            for (auto& [meshName, meshData] : geometryObjModel.meshes)
                            {
                                auto geometry = RTLib::Ext::OPX7::OPX7GeometryTriangle();
                                {
                                    auto mesh = objAsset.meshGroup->LoadMesh(meshData.base);
                                    aabb.Update(mesh->GetUniqueResource()->variables.GetFloat3("aabb.min"));
                                    aabb.Update(mesh->GetUniqueResource()->variables.GetFloat3("aabb.max"));
                                    auto extSharedData = static_cast<rtlib::test::OPX7MeshSharedResourceExtData*>(mesh->GetSharedResource()->extData.get());
                                    auto extUniqueData = static_cast<rtlib::test::OPX7MeshUniqueResourceExtData*>(mesh->GetUniqueResource()->extData.get());
                                    geometry.SetVertexView({ extSharedData->GetVertexBufferView(),(unsigned int)extSharedData->GetVertexStrideInBytes(),RTLib::Ext::OPX7::OPX7VertexFormat::eFloat32x3 });
                                    geometry.SetTriIdxView({ extUniqueData->GetTriIdxBufferView(),(unsigned int)extUniqueData->GetTriIdxStrideInBytes(),RTLib::Ext::OPX7::OPX7TriIdxFormat::eUInt32x3 });
                                    geometry.SetSbtOffsetView({ extUniqueData->GetMatIdxBufferView(),(unsigned int)sizeof(uint32_t),RTLib::Ext::OPX7::OPX7SbtOffsetFormat::eUInt32,(unsigned int)meshData.materials.size() });
                                    geometry.SetGeometryFlags(OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT);
                                    geometry.SetTransformView({});
                                    geometry.SetPrimIdxOffset(0);
                                }
                                auto [buildInput, holder] = geometry.GetOptixBuildInputWithHolder();
                                buildInputs.push_back(buildInput);
                                holders.push_back(std::move(holder));
                            }
                        }
                        if (world.geometrySpheres.count(geometryName) > 0) {
                            auto  geometry = RTLib::Ext::OPX7::OPX7GeometrySphere();
                            {
                                auto& sphereResource = sphereResources.at(geometryName);
                                auto sphereResourceExtData = sphereResource->GetExtData<rtlib::test::OPX7SphereResourceExtData>();
                                aabb.Update(sphereResource->variables.GetFloat3("aabb.min"));
                                aabb.Update(sphereResource->variables.GetFloat3("aabb.max"));
                                geometry.SetVertexView({ sphereResourceExtData->GetCenterBufferView(),(unsigned int)sphereResourceExtData->GetCenterStrideInBytes(),RTLib::Ext::OPX7::OPX7VertexFormat::eFloat32x3 });
                                geometry.SetRadiusView({ sphereResourceExtData->GetRadiusBufferView(),(unsigned int)sphereResourceExtData->GetRadiusStrideInBytes(),false });
                                geometry.SetSbtOffsetView({ sphereResourceExtData->GetMatIdxBufferView(),(unsigned int)sizeof(uint32_t),RTLib::Ext::OPX7::OPX7SbtOffsetFormat::eUInt32,(unsigned int)sphereResource->materials.size() });
                                geometry.SetGeometryFlags(OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT);
                                geometry.SetPrimIdxOffset(0);
                            }
                            auto [buildInput, holder] = geometry.GetOptixBuildInputWithHolder();
                            buildInputs.push_back(buildInput);
                            holders.push_back(std::move(holder));
                        }
                    }
                    auto&& [buffer, handle] = RTLib::Ext::OPX7::OPX7Natives::BuildAccelerationStructure(opx7Context, accelBuildOptions, buildInputs);
                    geometryASs[geometryASName].buffer  = std::move(buffer);
                    geometryASs[geometryASName].handle  = handle;
                    geometryASs[geometryASName].aabbMin = aabb.min;
                    geometryASs[geometryASName].aabbMax = aabb.max;

                }

                return geometryASs;
            }
            auto BuildInstanceASs(RTLib::Ext::OPX7::OPX7Context* opx7Context, const OptixAccelBuildOptions& accelBuildOptions,const RTLib::Ext::OPX7::OPX7ShaderTableLayout* shaderTableLayout, std::unordered_map<std::string, GeometryAccelerationStructureData>& geometryASs)const ->std::unordered_map<std::string, InstanceAccelerationStructureData>
            {
                auto instanceASs = std::unordered_map<std::string, rtlib::test::InstanceAccelerationStructureData>();

                {
                    auto instIndices = std::unordered_map<std::string, unsigned int>();
                    {
                        unsigned int i = 0;
                        for (auto& name : shaderTableLayout->GetInstanceNames()) {
                            instIndices[name] = i;
                            ++i;
                        }
                    }
                    auto  dependentInstanceASs = std::unordered_map<std::string, std::pair<std::string, size_t>>();
                    auto instanceNamesWithLevels = std::vector<std::vector<std::string>>();
                    instanceNamesWithLevels.push_back(std::vector<std::string>{"Root"});
                    dependentInstanceASs["Root"] = std::make_pair("Root", 0);
                    {
                        auto tempInstanceASNames = std::stack<std::pair<std::string, std::string>>();
                        tempInstanceASNames.push({ "Root","Root" });
                        while (!tempInstanceASNames.empty()) {
                            auto [instanceASUrl, instanceASName] = tempInstanceASNames.top();
                            auto& instanceASElement = world.instanceASs.at(instanceASName);
                            tempInstanceASNames.pop();
                            for (auto& instanceName : instanceASElement.instances)
                            {
                                auto& instance = world.instances.at(instanceName);
                                if (instance.asType != "Geometry") {
                                    tempInstanceASNames.push({ instanceASUrl + "/" + instanceName, instance.base });
                                    dependentInstanceASs[instanceASUrl + "/" + instanceName] = std::make_pair(instance.base, dependentInstanceASs[instanceASUrl].second + 1);
                                    if (instanceNamesWithLevels.size() <= dependentInstanceASs[instanceASUrl].second + 1) {
                                        instanceNamesWithLevels.push_back(std::vector<std::string>{instanceASUrl + "/" + instanceName});
                                    }
                                    else {
                                        instanceNamesWithLevels[dependentInstanceASs[instanceASUrl].second + 1].push_back(instanceASUrl + "/" + instanceName);
                                    }
                                }
                            }
                        }
                        for (auto i = 0; i < instanceNamesWithLevels.size(); ++i) {
                            for (auto& instanceASName : instanceNamesWithLevels[instanceNamesWithLevels.size() - 1 - i]) {
                                auto& instanceASElement = world.instanceASs.at(dependentInstanceASs[instanceASName].first);
                                instanceASs[instanceASName].instanceArray.reserve(instanceASElement.instances.size());
                                for (auto& instanceName : instanceASElement.instances)
                                {
                                    auto& instanceElement = world.instances.at(instanceName);
                                    auto opx7Instance = OptixInstance();
                                    if (instanceElement.asType == "Geometry") {
                                        opx7Instance.traversableHandle = geometryASs.at(instanceElement.base).handle;
                                        opx7Instance.sbtOffset = shaderTableLayout->GetDesc(instanceASName + "/" + instanceName).recordOffset;
                                    }
                                    else {
                                        opx7Instance.traversableHandle = instanceASs.at(instanceASName + "/" + instanceName).handle;
                                        opx7Instance.sbtOffset = 0;
                                    }
                                    opx7Instance.instanceId = instIndices.at(instanceASName + "/" + instanceName);
                                    opx7Instance.visibilityMask = 255;
                                    opx7Instance.flags = OPTIX_INSTANCE_FLAG_NONE;

                                    auto transforms = instanceElement.transform;
                                    std::memcpy(opx7Instance.transform, transforms.data(), transforms.size() * sizeof(float));
                                    instanceASs[instanceASName].instanceArray.push_back(opx7Instance);
                                }
                                instanceASs[instanceASName].Build(opx7Context, accelBuildOptions);
                            }
                        }
                    }
                }
                return instanceASs;
            }
            auto LoadTextureMap(RTLib::Ext::CUDA::CUDAContext* cudaContext)const -> std::unordered_map<std::string, rtlib::test::TextureData> {
                std::unordered_map<std::string, rtlib::test::TextureData> cudaTextures = {};
                {
                    std::unordered_set< std::string> texturePathSet = {};
                    for (auto& [geometryName, geometryData] : world.geometryObjModels)
                    {
                        for (auto& [meshName, meshData] : geometryData.meshes)
                        {
                            for (auto& material : meshData.materials)
                            {
                                if (material.HasString("diffTex"))
                                {
                                    if (material.GetString("diffTex") != "") {
                                        texturePathSet.insert(material.GetString("diffTex"));
                                    }
                                }
                                if (material.HasString("specTex"))
                                {
                                    if (material.GetString("specTex") != "") {
                                        texturePathSet.insert(material.GetString("specTex"));
                                    }
                                }
                                if (material.HasString("emitTex"))
                                {
                                    if (material.GetString("emitTex") != "") {
                                        texturePathSet.insert(material.GetString("emitTex"));
                                    }
                                }
                            }
                        }
                    }
                    for (auto& texturePath : texturePathSet)
                    {
                        cudaTextures[texturePath].LoadFromPath(cudaContext, texturePath);
                    }
                    cudaTextures["White"] = rtlib::test::TextureData::White(cudaContext);
                    cudaTextures["Black"] = rtlib::test::TextureData::Black(cudaContext);
                }
                return cudaTextures;
            }
        };

        template<typename T>
        struct UploadBuffer
        {
            std::vector<T>                                cpuHandle;
            std::unique_ptr<RTLib::Ext::CUDA::CUDABuffer> gpuHandle;

            void Upload(RTLib::Ext::CUDA::CUDAContext* cudaContext)
            {
                if (!gpuHandle) {
                    auto desc = RTLib::Ext::CUDA::CUDABufferCreateDesc();
                    desc.flags = RTLib::Ext::CUDA::CUDAMemoryFlags::eDefault;
                    desc.sizeInBytes = sizeof(cpuHandle[0]) * std::size(cpuHandle);
                    desc.pData = cpuHandle.data();
                    gpuHandle = decltype(gpuHandle)(cudaContext->CreateBuffer(desc));
                }
                else {
                    auto copy = RTLib::Ext::CUDA::CUDAMemoryBufferCopy();
                    copy.dstOffset = 0;
                    copy.size = sizeof(cpuHandle[0]) * std::size(cpuHandle);
                    copy.srcData = cpuHandle.data();
                    RTLIB_CORE_ASSERT_IF_FAILED(cudaContext->CopyMemoryToBuffer(
                        gpuHandle.get(), { copy }
                    ));
                }
            }
        };

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
            auto cameraController = RTLib::Core::CameraController({ 0.0f, 1.0f, 5.0f });
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

        struct PressState
        {
            bool isPressed;
            bool isUpdated;
        };

        class KeyBoardStateManager
        {
        public:
            KeyBoardStateManager(RTLib::Ext::GLFW::GLFWWindow* window) :m_Window{ window }, m_States{}{}
            void Update() noexcept {
                for (auto& [keyCode, keyState] : m_States)
                {
                    (void)UpdateState(keyCode);
                }
            }
            bool UpdateState(int keyCode) noexcept {
                bool curPressed = glfwGetKey(m_Window->GetGLFWwindow(), keyCode) == GLFW_PRESS;
                if (!HasState(keyCode)) {
                    m_States[keyCode].isPressed = curPressed;
                    m_States[keyCode].isUpdated = false;
                    return true;
                }
                else {
                    auto prvPressed = m_States.at(keyCode).isPressed;
                    m_States[keyCode].isPressed = curPressed;
                    m_States[keyCode].isUpdated = (curPressed != prvPressed);
                    return m_States[keyCode].isUpdated;
                }
            }
            auto GetState(int keyCode) const noexcept -> std::optional<PressState>
            {
                if (m_States.count(keyCode) > 0) {
                    return std::optional<PressState>(m_States.at(keyCode));
                }
                else {
                    return std::optional<PressState>();
                }
            }
            bool HasState(int keyCode) const noexcept {
                return m_States.count(keyCode) > 0;
            }
        private:
            RTLib::Ext::GLFW::GLFWWindow* m_Window;
            std::unordered_map<int, PressState> m_States;
        };
        class MouseButtonStateManager
        {
        public:
            MouseButtonStateManager(RTLib::Ext::GLFW::GLFWWindow* window) :m_Window{ window }, m_States{}{}
            void Update() noexcept {
                for (auto& [keyCode, keyState] : m_States)
                {
                    (void)UpdateState(keyCode);
                }
            }
            bool UpdateState(int keyCode) noexcept {
                bool curPressed = glfwGetMouseButton(m_Window->GetGLFWwindow(), keyCode) == GLFW_PRESS;
                if (!HasState(keyCode)) {
                    m_States[keyCode].isPressed = curPressed;
                    m_States[keyCode].isUpdated = false;
                    return true;
                }
                else {
                    auto prvPressed = m_States.at(keyCode).isPressed;
                    m_States[keyCode].isPressed = curPressed;
                    m_States[keyCode].isUpdated = (curPressed != prvPressed);
                    return m_States[keyCode].isUpdated;
                }
            }
            auto GetState(int keyCode) const noexcept -> std::optional<PressState>
            {
                if (m_States.count(keyCode) > 0) {
                    return std::optional<PressState>(m_States.at(keyCode));
                }
                else {
                    return std::optional<PressState>();
                }
            }
            bool HasState(int keyCode) const noexcept {
                return m_States.count(keyCode) > 0;
            }
        private:
            RTLib::Ext::GLFW::GLFWWindow* m_Window;
            std::unordered_map<int, PressState> m_States;
        };

        inline auto UpdateCameraMovement(
            const KeyBoardStateManager* keyBoardManager,
            const MouseButtonStateManager* mouseButtonManager,
            RTLib::Core::CameraController& cameraController,
            float delTime,
            float delCurPosX, 
            float delCurPosY){
            bool isMovedCamera = false;
            const bool pressKeyLeft     = (keyBoardManager->GetState(GLFW_KEY_A)->isPressed) || (keyBoardManager->GetState( GLFW_KEY_LEFT)->isPressed);
            const bool pressKeyRight    = (keyBoardManager->GetState(GLFW_KEY_D)->isPressed) || (keyBoardManager->GetState(GLFW_KEY_RIGHT)->isPressed);
            const bool pressKeyForward  = (keyBoardManager->GetState(GLFW_KEY_W)->isPressed);
            const bool pressKeyBackward = (keyBoardManager->GetState(GLFW_KEY_S)->isPressed);
            const bool pressKeyUp       = (keyBoardManager->GetState(GLFW_KEY_UP)->isPressed);
            const bool pressKeyDown     = (keyBoardManager->GetState(GLFW_KEY_DOWN)->isPressed);
            if (pressKeyLeft    )
            {
                cameraController.ProcessKeyboard(RTLib::Core::CameraMovement::eLeft, delTime);
                isMovedCamera = true;
            }
            if (pressKeyRight   )
            {
                cameraController.ProcessKeyboard(RTLib::Core::CameraMovement::eRight, delTime);
                isMovedCamera = true;
            }
            if (pressKeyForward )
            {
                cameraController.ProcessKeyboard(RTLib::Core::CameraMovement::eForward, delTime);
                isMovedCamera = true;
            }
            if (pressKeyBackward)
            {
                cameraController.ProcessKeyboard(RTLib::Core::CameraMovement::eBackward,  delTime);
                isMovedCamera = true;
            }
            if (pressKeyUp      )
            {
                cameraController.ProcessKeyboard(RTLib::Core::CameraMovement::eUp,  delTime);
                isMovedCamera = true;
            }
            if (pressKeyDown    )
            {
                cameraController.ProcessKeyboard(RTLib::Core::CameraMovement::eDown,  delTime);
                isMovedCamera = true;
            }
            if (mouseButtonManager->GetState(GLFW_MOUSE_BUTTON_LEFT)->isPressed)
            {
                cameraController.ProcessMouseMovement(-delCurPosX, delCurPosY);
                isMovedCamera = true;
            }
            return isMovedCamera;
        }


        inline auto GetRaygenData(const Camera& camera) noexcept->RayGenData
        {
            RayGenData rgData;
            rgData.eye = camera.GetEyeAs<float3>();
            auto [u, v, w] = camera.GetUVW();
            rgData.u = make_float3(u[0],u[1],u[2]);
            rgData.v = make_float3(v[0],v[1],v[2]);
            rgData.w = make_float3(w[0],w[1],w[2]);
            return rgData;
        }

        inline auto GetShaderTableLayout(const RTLib::Core::WorldData& worldData, unsigned int stride = 1)->std::unique_ptr<RTLib::Ext::OPX7::OPX7ShaderTableLayout> {
            auto blasLayouts = std::unordered_map<std::string, std::unique_ptr<RTLib::Ext::OPX7::OPX7ShaderTableLayoutGeometryAS>>();
            for (auto& [geometryASName, geometryAS] : worldData.geometryASs)
            {
                blasLayouts[geometryASName] = std::make_unique<RTLib::Ext::OPX7::OPX7ShaderTableLayoutGeometryAS>(geometryASName);
                auto buildInputSize = size_t(0);
                for (auto& geometryName : geometryAS.geometries)
                {
                    if (worldData.geometryObjModels.count(geometryName) > 0) {
                        for (auto& [meshName, meshData] : worldData.geometryObjModels.at(geometryName).meshes)
                        {
                            blasLayouts[geometryASName]->SetDwGeometry(RTLib::Ext::OPX7::OPX7ShaderTableLayoutGeometry(meshName, meshData.materials.size()));
                        }
                    }
                    if (worldData.geometrySpheres.count(geometryName) > 0) {
                        blasLayouts[geometryASName]->SetDwGeometry(RTLib::Ext::OPX7::OPX7ShaderTableLayoutGeometry(geometryName, 1));
                    }
                }
            }
            auto instanceASs = std::unordered_map<std::string,std::unique_ptr<RTLib::Ext::OPX7::OPX7ShaderTableLayoutInstanceAS>>();
            {
                auto  dependentInstanceASs = std::unordered_map<std::string, std::pair<std::string, size_t>>();
                auto instanceNamesWithLevels = std::vector<std::vector<std::string>>();
                instanceNamesWithLevels.push_back(std::vector<std::string>{"Root"});
                dependentInstanceASs["Root"] = std::make_pair("Root", 0);
                {
                    auto tempInstanceASNames = std::stack<std::pair<std::string, std::string>>();
                    tempInstanceASNames.push({ "Root","Root" });
                    while (!tempInstanceASNames.empty()) {
                        auto [instanceASUrl, instanceASName] = tempInstanceASNames.top();
                        auto& instanceASElement = worldData.instanceASs.at(instanceASName);
                        tempInstanceASNames.pop();
                        for (auto& instanceName : instanceASElement.instances)
                        {
                            auto& instance = worldData.instances.at(instanceName);
                            if (instance.asType != "Geometry") {
                                tempInstanceASNames.push({ instanceASUrl + "/" + instanceName, instance.base });
                                dependentInstanceASs[instanceASUrl + "/" + instanceName] = std::make_pair(instance.base, dependentInstanceASs[instanceASUrl].second + 1);
                                if (instanceNamesWithLevels.size() <= dependentInstanceASs[instanceASUrl].second + 1) {
                                    instanceNamesWithLevels.push_back(std::vector<std::string>{instanceASUrl + "/" + instanceName});
                                }
                                else {
                                    instanceNamesWithLevels[dependentInstanceASs[instanceASUrl].second + 1].push_back(instanceASUrl + "/" + instanceName);
                                }
                            }
                        }
                    }
                    for (auto i = 0; i < instanceNamesWithLevels.size(); ++i) {
                        for (auto& instanceASName : instanceNamesWithLevels[instanceNamesWithLevels.size() - 1 - i]) {
                            auto& instanceASElement = worldData.instanceASs.at(dependentInstanceASs[instanceASName].first);
                            instanceASs[instanceASName] = std::make_unique<RTLib::Ext::OPX7::OPX7ShaderTableLayoutInstanceAS>();
                            for (auto& instanceName : instanceASElement.instances)
                            {
                                auto& instanceElement = worldData.instances.at(instanceName);
                                auto opx7Instance = OptixInstance();
                                if (instanceElement.asType == "Geometry") {
                                    instanceASs[instanceASName]->SetInstance(RTLib::Ext::OPX7::OPX7ShaderTableLayoutInstance(instanceName, blasLayouts.at(instanceElement.base).get()));
                                }
                                else {
                                    instanceASs[instanceASName]->SetInstance(RTLib::Ext::OPX7::OPX7ShaderTableLayoutInstance(instanceName, instanceASs.at(instanceASName+"/"+instanceName).get()));
                                }
                            }
                        }
                    }
                }
            }
            instanceASs["Root"]->SetRecordStride(stride);
            return std::make_unique<RTLib::Ext::OPX7::OPX7ShaderTableLayout>(*instanceASs["Root"].get());
        }

        inline auto CreateFrameTextureGL(RTLib::Ext::GL::GLContext* context, int width, int height)->std::unique_ptr<RTLib::Ext::GL::GLTexture>
        {
            auto frameTextureDesc = RTLib::Ext::GL::GLTextureCreateDesc();
            frameTextureDesc.image.imageType = RTLib::Ext::GL::GLImageType::e2D;
            frameTextureDesc.image.extent.width = width;
            frameTextureDesc.image.extent.height = height;
            frameTextureDesc.image.extent.depth = 0;
            frameTextureDesc.image.arrayLayers = 0;
            frameTextureDesc.image.mipLevels = 1;
            frameTextureDesc.image.format = RTLib::Ext::GL::GLFormat::eRGBA8;
            frameTextureDesc.sampler.magFilter = RTLib::Core::FilterMode::eLinear;
            frameTextureDesc.sampler.minFilter = RTLib::Core::FilterMode::eLinear;

            return std::unique_ptr<RTLib::Ext::GL::GLTexture>(context->CreateTexture(frameTextureDesc));
        }

        inline auto GetInstanceTransform(const WorldData& worldData, const std::string& instancePath)->RTLib::Ext::CUDA::Math::Matrix4x4
        {
            auto transform = RTLib::Ext::CUDA::Math::Matrix4x4::Identity();
            {
                auto tmpInstancePath = instancePath;
                while (tmpInstancePath != "Root")
                {
                    auto len = tmpInstancePath.find_last_of("/");
                    auto tmpInstanceName = std::string(tmpInstancePath.data() + len + 1, tmpInstancePath.data() + tmpInstancePath.size());
                    if (worldData.instances.count(tmpInstanceName) > 0) {
                        auto& tmpInstanceData = worldData.instances.at(tmpInstanceName);
                        auto  tmpTransform = RTLib::Ext::CUDA::Math::Matrix4x4(
                            make_float4(tmpInstanceData.transform[0], tmpInstanceData.transform[4], tmpInstanceData.transform[8], 0.0f),
                            make_float4(tmpInstanceData.transform[1], tmpInstanceData.transform[5], tmpInstanceData.transform[9], 0.0f),
                            make_float4(tmpInstanceData.transform[2], tmpInstanceData.transform[6], tmpInstanceData.transform[10], 0.0f),
                            make_float4(tmpInstanceData.transform[3], tmpInstanceData.transform[7], tmpInstanceData.transform[11], 1.0f)
                        );
                        transform *= tmpTransform;
                    }
                    tmpInstancePath = std::string(tmpInstancePath.data(), tmpInstancePath.data() + len);
                }
            }
            return transform;
        }

        inline auto LoadScene(std::string path)->SceneData {
            nlohmann::json sceneJson;
            std::ifstream sceneJsonFile(path, std::ios::binary);
            if (sceneJsonFile.is_open())
            {
                sceneJson = nlohmann::json::parse(sceneJsonFile);
                sceneJsonFile.close();
            }
            else
            {
                sceneJson = rtlib::test::GetDefaultSceneJson();
            }
            return sceneJson.get<rtlib::test::SceneData>();
        }
        inline void SaveScene(std::string path, const SceneData& sceneData) {
            nlohmann::json sceneJson(sceneData);
            auto sceneJsonFile = std::ofstream(path, std::ios::binary);
            sceneJsonFile << sceneJson;
            sceneJsonFile.close();
        }

        struct ModuleData
        {
            std::unique_ptr<RTLib::Ext::OPX7::OPX7Module> handle;
            RTLib::Ext::OPX7::OPX7ModuleCompileOptions    options;
        };

        struct PipelineData {
            RTLib::Ext::OPX7::OPX7PipelineLinkOptions                                            linkOptions;
            RTLib::Ext::OPX7::OPX7PipelineCompileOptions                                         compileOptions;
            std::unordered_map<std::string, ModuleData>                                          modules;
            std::unique_ptr<RTLib::Ext::OPX7::OPX7Pipeline>                                      handle;
            std::unique_ptr<RTLib::Ext::OPX7::OPX7ProgramGroup>                                  programGroupRG;
            std::unique_ptr<RTLib::Ext::OPX7::OPX7ProgramGroup>                                  programGroupEP;
            std::unordered_map<std::string, std::unique_ptr<RTLib::Ext::OPX7::OPX7ProgramGroup>> programGroupMSs;
            std::unordered_map<std::string, std::unique_ptr<RTLib::Ext::OPX7::OPX7ProgramGroup>> programGroupHGs;
            std::unique_ptr<RTLib::Ext::OPX7::OPX7ShaderTable>                                   shaderTable;

            void Free() {
                handle->Destroy();
                handle = nullptr;
                programGroupRG->Destroy();
                programGroupRG = nullptr;
                programGroupEP->Destroy();
                programGroupEP = nullptr;
                for (auto& [name, programGroupMS] : programGroupMSs) {
                    programGroupMS->Destroy();
                    programGroupMS = nullptr;
                }
                programGroupMSs.clear();
                for (auto& [name, programGroupHG] : programGroupHGs) {
                    programGroupHG->Destroy();
                    programGroupHG = nullptr;
                }
                programGroupHGs.clear();
                for (auto& [name, module] : modules) {
                    module.handle->Destroy();
                    module = {};
                }
                modules.clear(); 
                shaderTable->Destroy();
                shaderTable = nullptr;
            }

            void InitPipeline(RTLib::Ext::OPX7::OPX7Context* context)
            {
                auto pipelineCreateDesc = RTLib::Ext::OPX7::OPX7PipelineCreateDesc();
                pipelineCreateDesc.linkOptions = linkOptions;
                pipelineCreateDesc.compileOptions = compileOptions;
                pipelineCreateDesc.programGroups = {};
                pipelineCreateDesc.programGroups.push_back(programGroupRG.get());
                pipelineCreateDesc.programGroups.push_back(programGroupEP.get());
                for (auto& [name, programGroupMS] : programGroupMSs) {
                    pipelineCreateDesc.programGroups.push_back(programGroupMS.get());
                }
                for (auto& [name, programGroupHG] : programGroupHGs) {
                    pipelineCreateDesc.programGroups.push_back(programGroupHG.get());
                }
                handle = std::unique_ptr<RTLib::Ext::OPX7::OPX7Pipeline>(context->CreateOPXPipeline(pipelineCreateDesc));
            }

            void LoadModule(RTLib::Ext::OPX7::OPX7Context* context, std::string moduleName, const RTLib::Ext::OPX7::OPX7ModuleCompileOptions& moduleOptions, const std::vector<char>& ptxString)
            {
                auto createDesc = RTLib::Ext::OPX7::OPX7ModuleCreateDesc();
                {
                    createDesc.ptxBinary = ptxString;
                    createDesc.moduleOptions = moduleOptions;
                    createDesc.pipelineOptions = compileOptions;
                }
                modules[moduleName].handle = std::unique_ptr<RTLib::Ext::OPX7::OPX7Module>(context->CreateOPXModule(createDesc));
                modules[moduleName].options = moduleOptions;
            }
            void LoadBuiltInISTriangleModule(RTLib::Ext::OPX7::OPX7Context* context, std::string moduleName, const RTLib::Ext::OPX7::OPX7ModuleCompileOptions& moduleOptions, bool useMotionBlur = false) {
                auto createDesc = RTLib::Ext::OPX7::OPX7ModuleCreateDesc();
                {
                    createDesc.ptxBinary = {};
                    createDesc.moduleOptions = moduleOptions;
                    createDesc.pipelineOptions = compileOptions;
                }
                modules[moduleName].handle = std::unique_ptr<RTLib::Ext::OPX7::OPX7Module>(RTLib::Ext::OPX7::OPX7Module::BuiltInTriangleIS(context, createDesc, useMotionBlur));
                modules[moduleName].options = moduleOptions;
            }
            void LoadBuiltInISSphereModule(RTLib::Ext::OPX7::OPX7Context* context, std::string moduleName, const RTLib::Ext::OPX7::OPX7ModuleCompileOptions& moduleOptions, bool useMotionBlur = false) {
                auto createDesc = RTLib::Ext::OPX7::OPX7ModuleCreateDesc();
                {
                    createDesc.ptxBinary = {};
                    createDesc.moduleOptions = moduleOptions;
                    createDesc.pipelineOptions = compileOptions;
                }
                modules[moduleName].handle = std::unique_ptr<RTLib::Ext::OPX7::OPX7Module>(RTLib::Ext::OPX7::OPX7Module::BuiltInSphereIS(context, createDesc, useMotionBlur));
                modules[moduleName].options = moduleOptions;
            }

            void Launch(RTLib::Ext::CUDA::CUDAStream* stream, RTLib::Ext::CUDA::CUDABuffer* paramsBuffer, int width, int height,int depth=1)
            {

                handle->Launch(stream, RTLib::Ext::CUDA::CUDABufferView(paramsBuffer, 0, paramsBuffer->GetSizeInBytes()), shaderTable.get(), width, height, depth);
            }

            void SetProgramGroupRG(RTLib::Ext::OPX7::OPX7Context* context, std::string module_name, std::string func_name)
            {
                programGroupRG = std::unique_ptr<RTLib::Ext::OPX7::OPX7ProgramGroup>(context->CreateOPXProgramGroup(RTLib::Ext::OPX7::OPX7ProgramGroupCreateDesc::Raygen({ modules[module_name].handle.get(),func_name.c_str() })));
            }

            void SetProgramGroupMS(RTLib::Ext::OPX7::OPX7Context* context, std::string pg_name, std::string module_name, std::string func_name)
            {
                programGroupMSs[pg_name] = std::unique_ptr<RTLib::Ext::OPX7::OPX7ProgramGroup>(context->CreateOPXProgramGroup(RTLib::Ext::OPX7::OPX7ProgramGroupCreateDesc::Miss({ modules[module_name].handle.get(),func_name.c_str() })));
            }

            void SetProgramGroupHG(RTLib::Ext::OPX7::OPX7Context* context, std::string pg_name,
                std::string ch_module_name, std::string ch_func_name,
                std::string ah_module_name, std::string ah_func_name,
                std::string is_module_name, std::string is_func_name)
            {
                auto chSingleModule = RTLib::Ext::OPX7::OPX7ProgramGroupSingleModule();
                auto ahSingleModule = RTLib::Ext::OPX7::OPX7ProgramGroupSingleModule();
                auto isSingleModule = RTLib::Ext::OPX7::OPX7ProgramGroupSingleModule();
                if (modules.count(ch_module_name) > 0) {
                    chSingleModule.module = modules.at(ch_module_name).handle.get();
                    chSingleModule.entryFunctionName = ch_func_name.c_str();
                }
                else {
                    chSingleModule = {};
                }
                if (modules.count(ah_module_name) > 0) {
                    ahSingleModule.module = modules.at(ah_module_name).handle.get();
                    ahSingleModule.entryFunctionName = ah_func_name.c_str();
                }
                else {
                    ahSingleModule = {};
                }
                if (modules.count(is_module_name) > 0) {
                    isSingleModule.module = modules.at(is_module_name).handle.get();
                    if (!is_func_name.empty()) {
                        isSingleModule.entryFunctionName = is_func_name.c_str();
                    }
                }
                else {
                    isSingleModule = {};
                }

                programGroupHGs[pg_name] = std::unique_ptr<RTLib::Ext::OPX7::OPX7ProgramGroup>(context->CreateOPXProgramGroup(RTLib::Ext::OPX7::OPX7ProgramGroupCreateDesc::Hitgroup(
                    chSingleModule, ahSingleModule, isSingleModule
                )));
            }

            void SetProgramGroupEP(RTLib::Ext::OPX7::OPX7Context* context, std::string ep_module_name, std::string ep_function_name)
            {
                programGroupEP = std::unique_ptr<RTLib::Ext::OPX7::OPX7ProgramGroup>(context->CreateOPXProgramGroup(RTLib::Ext::OPX7::OPX7ProgramGroupCreateDesc::Exception({ modules[ep_module_name].handle.get(),ep_function_name.c_str() })));
            }

            template<typename T>
            void SetHostRayGenRecordTypeData( T data)
            {
                shaderTable->SetHostRaygenRecordTypeData(programGroupRG->GetRecord<T>(data));
            }
            template<typename T>
            void SetHostMissRecordTypeData(unsigned int index, std::string name, T data)
            {
                shaderTable->SetHostMissRecordTypeData(index, programGroupMSs[name]->GetRecord<T>(data));
            }
            template<typename T>
            void SetHostHitgroupRecordTypeData(unsigned int index, std::string name, T data)
            {
                shaderTable->SetHostHitgroupRecordTypeData(index, programGroupHGs[name]->GetRecord<T>(data));
            }
            template<typename T>
            void SetHostExceptionRecordTypeData(T data)
            {
                shaderTable->SetHostExceptionRecordTypeData(programGroupEP->GetRecord<T>(data));
            }

            void SetPipelineStackSize(unsigned int maxTraversableDepth) {
                auto cssRG = programGroupRG->GetStackSize().cssRG;
                unsigned int cssMS = 0;
                for (auto& [name, programGroup] : programGroupMSs) {
                    cssMS = std::max(cssMS, programGroup->GetStackSize().cssMS);
                }
                unsigned int cssCH = 0;
                unsigned int cssAH = 0;
                unsigned int cssIS = 0;
                for (auto& [name, programGroup] : programGroupHGs) {
                    auto cssHg = programGroup->GetStackSize();
                    cssCH = std::max(cssCH, cssHg.cssCH);
                    cssAH = std::max(cssCH, cssHg.cssAH);
                    cssIS = std::max(cssCH, cssHg.cssIS);
                }
                auto cssCCTree = static_cast<unsigned int>(0);
                auto cssCHOrMsPlusCCTree = std::max(cssMS, cssCH) + cssCCTree;
                auto continuationStackSizes = cssRG + cssCCTree +
                    (std::max<unsigned int>(1, linkOptions.maxTraceDepth) - 1) * cssCHOrMsPlusCCTree +
                    (std::min<unsigned int>(1, linkOptions.maxTraceDepth)) * std::max(cssCHOrMsPlusCCTree, cssIS + cssAH);

                handle->SetStackSize(0, 0, continuationStackSizes, maxTraversableDepth);

            }
        };
        struct TracerData
        {
            std::unordered_map<std::string, PipelineData>                                        pipelines;
            std::unique_ptr<RTLib::Ext::CUDA::CUDABuffer>                                        paramsBuffer;

            template<typename ParamsType>
            void InitParams(RTLib::Ext::OPX7::OPX7Context* context, ParamsType params) {
                auto desc = RTLib::Ext::CUDA::CUDABufferCreateDesc();
                desc.flags = RTLib::Ext::CUDA::CUDAMemoryFlags::eDefault;
                desc.sizeInBytes = sizeof(ParamsType);
                desc.pData = &params;
                paramsBuffer = std::unique_ptr<RTLib::Ext::CUDA::CUDABuffer>(context->CreateBuffer(desc));
            }

            void Free() {
                for (auto& [name, pipeline] : pipelines)
                {
                    pipeline.Free();
                }
                pipelines.clear();
                paramsBuffer->Destroy();
                paramsBuffer.reset();
            }

            void Launch(RTLib::Ext::CUDA::CUDAStream* stream, std::string pipelineName, int width, int height, int depth =1)
            {
                pipelines[pipelineName].Launch(stream, paramsBuffer.get(), width, height,depth);
            }
        };

        struct  EventState
        {
            bool isUpdated = false;
            bool isResized = false;
            bool isMovedCamera = false;
            bool isClearFrame = false;
        };
        struct WindowState
        {
            float curTime = 0.0f;
            float delTime = 0.0f;
            float2 curCurPos = {};
            float2 delCurPos = {};
            
        };

        inline bool SavePngImage(std::string path, int width, int height, const std::vector<unsigned char>& pixels)
        {
            return stbi_write_png(path.c_str(), width, height, 4, pixels.data(), width * 4);
        }
        inline bool SaveExrImage(std::string path, int width, int height, const std::vector<float>&        pixels)
        {
            EXRHeader header;
            InitEXRHeader(&header);

            EXRImage image;
            InitEXRImage(&image);

            image.num_channels = 3;

            std::vector<float> images[3];
            images[0].resize(width * height);
            images[1].resize(width * height);
            images[2].resize(width * height);

            // Split RGBRGBRGB... into R, G and B layer
            for (int i = 0; i < width * height; i++) {
                images[0][i] = pixels[3 * i + 0];
                images[1][i] = pixels[3 * i + 1];
                images[2][i] = pixels[3 * i + 2];
            }

            float* image_ptr[3];
            image_ptr[0] = &(images[2].at(0)); // B
            image_ptr[1] = &(images[1].at(0)); // G
            image_ptr[2] = &(images[0].at(0)); // R

            image.images = (unsigned char**)image_ptr;
            image.width = width;
            image.height = height;

            header.num_channels = 3;
            header.channels = (EXRChannelInfo*)malloc(sizeof(EXRChannelInfo) * header.num_channels);
            // Must be (A)BGR order, since most of EXR viewers expect this channel order.
            strncpy(header.channels[0].name, "B", 255); header.channels[0].name[strlen("B")] = '\0';
            strncpy(header.channels[1].name, "G", 255); header.channels[1].name[strlen("G")] = '\0';
            strncpy(header.channels[2].name, "R", 255); header.channels[2].name[strlen("R")] = '\0';

            header.pixel_types = (int*)malloc(sizeof(int) * header.num_channels);
            header.requested_pixel_types = (int*)malloc(sizeof(int) * header.num_channels);
            for (int i = 0; i < header.num_channels; i++) {
                header.pixel_types[i] = TINYEXR_PIXELTYPE_FLOAT; // pixel type of input image
                header.requested_pixel_types[i] = TINYEXR_PIXELTYPE_HALF; // pixel type of output image to be stored in .EXR
            }

            const char* err = NULL; // or nullptr in C++11 or later.
            int ret = SaveEXRImageToFile(&image, &header, path.c_str(), &err);
            if (ret != TINYEXR_SUCCESS) {
                fprintf(stderr, "Save EXR err: %s\n", err);
                FreeEXRErrorMessage(err); // free's buffer for an error message
                return ret;
            }
            //printf("Saved exr file. [ %s ] \n", path.c_str());

            free(header.channels);
            free(header.pixel_types);
            free(header.requested_pixel_types);
        }

        template<typename T>
        struct RegularGrid3TypeBuffer
        {
            float3                                        aabbMin;
            float3                                        aabbMax;
            uint3                                         bounds;
            std::vector<T>                                cpuHandle;
            std::unique_ptr<RTLib::Ext::CUDA::CUDABuffer> gpuHandle;
            void Alloc(uint3 bnds) {
                bounds = bnds;
                cpuHandle.resize(bounds.x * bounds.y * bounds.z);
            }
            void Download(RTLib::Ext::CUDA::CUDAContext* context) {
                auto desc = RTLib::Ext::CUDA::CUDABufferMemoryCopy();
                desc.srcOffset = 0;
                desc.dstData = cpuHandle.data();
                desc.size = cpuHandle.size() * sizeof(T);
                context->CopyBufferToMemory(gpuHandle.get(), { desc });
            }
            void Upload(RTLib::Ext::CUDA::CUDAContext* context) {
                if (!gpuHandle) {
                    auto desc = RTLib::Ext::CUDA::CUDABufferCreateDesc();
                    desc.flags = RTLib::Ext::CUDA::CUDAMemoryFlags::eDefault;
                    desc.pData = cpuHandle.data();
                    desc.sizeInBytes = cpuHandle.size() * sizeof(T);
                    gpuHandle = std::unique_ptr<RTLib::Ext::CUDA::CUDABuffer>(context->CreateBuffer(desc));
                }
                else {
                    auto desc = RTLib::Ext::CUDA::CUDAMemoryBufferCopy();
                    desc.dstOffset = 0;
                    desc.srcData = cpuHandle.data();
                    desc.size = cpuHandle.size() * sizeof(T);
                    context->CopyMemoryToBuffer(gpuHandle.get(), { desc });
                }
            }
            auto GetHandle()const noexcept -> RTLib::Ext::OPX7::Utils::RegularGrid3 
            {
                RegularGrid3 grid3;
                grid3.aabbOffset = aabbMin;
                grid3.aabbSize = aabbMax - aabbMin;
                grid3.bounds = bounds;
                return grid3;
            }
        };

        template<typename T>
        struct    HashGrid3TypeBuffer
        {
            float3                                        aabbMin;
            float3                                        aabbMax;
            uint3                                         bounds;
            std::vector<T>                                dataCpuHandle;
            std::unique_ptr<RTLib::Ext::CUDA::CUDABuffer> dataGpuHandle;
            std::vector<unsigned int>                     checkSumCpuHandle;
            std::unique_ptr<RTLib::Ext::CUDA::CUDABuffer> checkSumGpuHandle;
            void Alloc(uint3 bnds, unsigned int size) {
                bounds = bnds;
                dataCpuHandle.resize(size);
                checkSumCpuHandle.resize(size);
            }
            void Download(RTLib::Ext::CUDA::CUDAContext* context) {
                auto desc = RTLib::Ext::CUDA::CUDABufferMemoryCopy();
                desc.srcOffset = 0;
                desc.dstData = dataCpuHandle.data();
                desc.size = dataCpuHandle.size() * sizeof(dataCpuHandle[0]);
                context->CopyBufferToMemory(dataGpuHandle.get(), { desc });
            }
            void Upload(RTLib::Ext::CUDA::CUDAContext* context) {
                if (!dataGpuHandle) {
                    auto desc = RTLib::Ext::CUDA::CUDABufferCreateDesc();
                    desc.flags = RTLib::Ext::CUDA::CUDAMemoryFlags::eDefault;
                    
                    desc.pData = dataCpuHandle.data();
                    desc.sizeInBytes = dataCpuHandle.size() * sizeof(dataCpuHandle[0]);
                    dataGpuHandle = std::unique_ptr<RTLib::Ext::CUDA::CUDABuffer>(context->CreateBuffer(desc));

                    desc.pData = checkSumCpuHandle.data();
                    desc.sizeInBytes = checkSumCpuHandle.size() * sizeof(checkSumCpuHandle[0]);
                    checkSumGpuHandle = std::unique_ptr<RTLib::Ext::CUDA::CUDABuffer>(context->CreateBuffer(desc));
                }
                else {
                    auto desc = RTLib::Ext::CUDA::CUDAMemoryBufferCopy();
                    desc.dstOffset = 0;
                    desc.srcData = dataCpuHandle.data();
                    desc.size = dataCpuHandle.size() * sizeof(T);
                    context->CopyMemoryToBuffer(dataGpuHandle.get(), { desc });
                    desc.dstOffset = 0;
                    desc.srcData = checkSumCpuHandle.data();
                    desc.size = checkSumCpuHandle.size() * sizeof(checkSumCpuHandle[0]);
                    context->CopyMemoryToBuffer(checkSumGpuHandle.get(), { desc });

                }
            }
            auto GetHandle()const noexcept -> HashGrid3
            {
                HashGrid3 grid3;
                grid3.aabbOffset = aabbMin;
                grid3.aabbSize = aabbMax - aabbMin;
                grid3.bounds   = bounds;
                grid3.size     = dataCpuHandle.size();
                grid3.checkSums= reinterpret_cast<unsigned int*>(RTLib::Ext::CUDA::CUDANatives::GetCUdeviceptr(checkSumGpuHandle.get()));
                return grid3;
            }
        };

        struct DoubleBufferedHashGrid3Buffer
        {
            float3                                        aabbMin;
            float3                                        aabbMax;
            uint3                                         bounds;
            unsigned int                                  curIndex;
            std::vector<unsigned int>                     checkSumCpuHandle;
            std::unique_ptr<RTLib::Ext::CUDA::CUDABuffer> checkSumGpuHandles[2];

            auto GetCurCheckSumGpuHandle() noexcept -> RTLib::Ext::CUDA::CUDABuffer*
            {
                return checkSumGpuHandles[curIndex].get();
            }
            auto GetCurCheckSumGpuHandleMemoryFootPrint() const noexcept -> size_t {
                return checkSumGpuHandles[curIndex] ? checkSumGpuHandles[curIndex]->GetSizeInBytes() : 0;
            }

            auto GetMemoryFootPrint()const noexcept -> size_t
            {
                return sizeof(DoubleBufferedHashGrid3) + GetCurCheckSumGpuHandleMemoryFootPrint() + GetPrvCheckSumGpuHandleMemoryFootPrint();
            }
            auto GetPrvCheckSumGpuHandle() noexcept -> RTLib::Ext::CUDA::CUDABuffer*
            {
                return checkSumGpuHandles[1-curIndex].get();
            }
            auto GetPrvCheckSumGpuHandleMemoryFootPrint() const noexcept -> size_t {
                return checkSumGpuHandles[1 - curIndex] ? checkSumGpuHandles[1 - curIndex]->GetSizeInBytes() : 0;
            }

            void Alloc(uint3 bnds, unsigned int size) {
                bounds   = bnds;
                checkSumCpuHandle.resize(size);
                curIndex = 0;
            }
            void Download(RTLib::Ext::CUDA::CUDAContext* context) {
                auto desc = RTLib::Ext::CUDA::CUDABufferMemoryCopy();
                desc.srcOffset = 0;
                desc.dstData   = checkSumCpuHandle.data();
                desc.size      = checkSumCpuHandle.size() * sizeof(checkSumCpuHandle[0]);
                context->CopyBufferToMemory(GetCurCheckSumGpuHandle(), {desc});
            }
            void Upload(  RTLib::Ext::CUDA::CUDAContext* context) {
                if (!checkSumGpuHandles[0]) {
                    auto desc = RTLib::Ext::CUDA::CUDABufferCreateDesc();
                    desc.flags = RTLib::Ext::CUDA::CUDAMemoryFlags::eDefault;
                    desc.pData = checkSumCpuHandle.data();
                    desc.sizeInBytes = checkSumCpuHandle.size() * sizeof(checkSumCpuHandle[0]);
                    checkSumGpuHandles[0] = std::unique_ptr<RTLib::Ext::CUDA::CUDABuffer>(context->CreateBuffer(desc));
                    checkSumGpuHandles[1] = std::unique_ptr<RTLib::Ext::CUDA::CUDABuffer>(context->CreateBuffer(desc));
                }
                else {
                    auto desc = RTLib::Ext::CUDA::CUDAMemoryBufferCopy();
                    desc.dstOffset = 0;
                    desc.srcData = checkSumCpuHandle.data();
                    desc.size = checkSumCpuHandle.size() * sizeof(checkSumCpuHandle[0]);
                    context->CopyMemoryToBuffer(checkSumGpuHandles[0].get(), {desc});
                    context->CopyMemoryToBuffer(checkSumGpuHandles[1].get(), { desc });
                }
            }
            void Clear(   RTLib::Ext::CUDA::CUDAContext* context, RTLib::Ext::CUDA::CUDAStream* stream = nullptr) {
                auto curCheckSumGpuAddress = RTLib::Ext::CUDA::CUDANatives::GetCUdeviceptr(GetCurCheckSumGpuHandle());
                auto prvCheckSumGpuAddress = RTLib::Ext::CUDA::CUDANatives::GetCUdeviceptr(GetPrvCheckSumGpuHandle());
                if (stream) {
                    auto cuStream = RTLib::Ext::CUDA::CUDANatives::GetCUstream(stream);
                    cuMemsetD32Async(curCheckSumGpuAddress, checkSumCpuHandle.size(), 0, cuStream);
                    cuMemsetD32Async(prvCheckSumGpuAddress, checkSumCpuHandle.size(), 0, cuStream);
                }
                else {
                    cuMemsetD32(curCheckSumGpuAddress, checkSumCpuHandle.size(), 0);
                    cuMemsetD32(prvCheckSumGpuAddress, checkSumCpuHandle.size(), 0);
                }
            }
            void Update(  RTLib::Ext::CUDA::CUDAContext* context, RTLib::Ext::CUDA::CUDAStream* stream = nullptr) {
            #ifndef NDEBUG
                std::cout << "Update Double Buffered Hash Grid\n";
            #endif
                auto curCheckSumGpuAddress = RTLib::Ext::CUDA::CUDANatives::GetCUdeviceptr(GetCurCheckSumGpuHandle());
                auto prvCheckSumGpuAddress = RTLib::Ext::CUDA::CUDANatives::GetCUdeviceptr(GetPrvCheckSumGpuHandle());
                if (!stream) {
                    cuMemcpyDtoD(prvCheckSumGpuAddress, curCheckSumGpuAddress, checkSumCpuHandle.size() * sizeof(checkSumCpuHandle[0]));
                }
                else {
                    auto cuStream = RTLib::Ext::CUDA::CUDANatives::GetCUstream(stream);
                    cuMemcpyDtoDAsync(prvCheckSumGpuAddress, curCheckSumGpuAddress, checkSumCpuHandle.size() * sizeof(checkSumCpuHandle[0]), cuStream);
                }
                curIndex = (1 + curIndex)%2;
            }
            auto GetHandle() noexcept -> rtlib::test::DoubleBufferedHashGrid3
            {
                rtlib::test::DoubleBufferedHashGrid3 grid3;
                grid3.aabbOffset = aabbMin;
                grid3.aabbSize = aabbMax - aabbMin;
                grid3.bounds = bounds;
                grid3.size = checkSumCpuHandle.size();
                grid3.prvCheckSums = reinterpret_cast<unsigned int*>(RTLib::Ext::CUDA::CUDANatives::GetCUdeviceptr(GetPrvCheckSumGpuHandle()));
                grid3.curCheckSums = reinterpret_cast<unsigned int*>(RTLib::Ext::CUDA::CUDANatives::GetCUdeviceptr(GetCurCheckSumGpuHandle()));
                return grid3;
            }
        };

        inline auto SpecifyMaterialType(const RTLib::Core::VariableMap& material, const RTLib::Core::MeshPtr& mesh = nullptr) -> HitgroupType
        {
            auto emitCol = material.GetFloat3As<float3>("emitCol");
            auto specCol = material.GetFloat3As<float3>("specCol");
            auto tranCol = material.GetFloat3As<float3>("tranCol");
            auto refrIndx = material.GetFloat1("refrIndx");
            auto shinness = material.GetFloat1("shinness");
            auto illum    = material.GetUInt32("illum");
            if (emitCol.x + emitCol.y + emitCol.z > 0.0f)
            {
                if (mesh) {
                    if (mesh->GetUniqueResource()->variables.GetBoolOr("useNEE",false)) {
                        return HIT_GROUP_TYPE_NEE_LIGHT;
                    }
                    else {
                        return HIT_GROUP_TYPE_DEF_LIGHT;
                    }
                }
                else {
                    return HIT_GROUP_TYPE_DEF_LIGHT;
                }
            }
            else if (illum == 7)
            {
                return HIT_GROUP_TYPE_GLASS;
            }
            else {
                return HIT_GROUP_TYPE_PHONG;
            }
        }
    }
}
#endif