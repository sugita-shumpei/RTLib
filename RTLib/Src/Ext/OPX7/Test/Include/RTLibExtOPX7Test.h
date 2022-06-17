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
#include <RTLib/Ext/OPX7/OPX7Pipeline.h>
#include <RTLib/Ext/OPX7/OPX7Natives.h>
#include <RTLib/Ext/OPX7/OPX7Exceptions.h>
#include <RTLib/Ext/CUDA/CUDAExceptions.h>
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
#include <iostream>
#include <filesystem>
#include <fstream>
#include <unordered_map>
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
    }
    namespace ext
    {
        using namespace RTLib::Core;
        using namespace RTLib::Ext;
        using namespace RTLib::Ext::CUDA::Math;
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
            auto GetVertexBuffer()const noexcept -> CUdeviceptr { return CUDA::CUDANatives::GetCUdeviceptr(m_VertexBuffer.get()); }
            auto GetNormalBuffer()const noexcept -> CUdeviceptr { return CUDA::CUDANatives::GetCUdeviceptr(m_NormalBuffer.get()); }
            auto GetTexCrdBuffer()const noexcept -> CUdeviceptr { return CUDA::CUDANatives::GetCUdeviceptr(m_TexCrdBuffer.get()); }
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
            auto GetTriIdxBuffer()const noexcept -> CUdeviceptr { return CUDA::CUDANatives::GetCUdeviceptr(m_TriIdxBuffer.get()); }
            auto GetTriIdxCount()const noexcept -> size_t {
                if (m_TriIdxBuffer && m_TriIdxStride > 0) { return m_TriIdxBuffer->GetSizeInBytes() / m_TriIdxStride; }
                return 0;
            }
            auto GetTriIdxStrideInBytes()const noexcept -> size_t { return m_TriIdxStride; }
            auto GetTriIdxFormat()const noexcept -> OptixIndicesFormat { return m_IndicesFormat; }
            //
            auto GetMatIdxBuffer()const noexcept -> CUdeviceptr { return CUDA::CUDANatives::GetCUdeviceptr(m_MatIdxBuffer.get()); };
            auto GetMatIdxCount()const noexcept -> size_t {
                if (m_MatIdxBuffer) { return m_MatIdxBuffer->GetSizeInBytes() / sizeof(uint32_t); }
                return 0;
            }
        private:
            OPX7MeshUniqueResourceExtData(MeshUniqueResource* pMeshUniqueResource) noexcept :MeshUniqueResourceExtData(pMeshUniqueResource) {}
        private:
            std::unique_ptr<RTLib::Ext::CUDA::CUDABuffer> m_TriIdxBuffer = nullptr;
            std::unique_ptr<RTLib::Ext::CUDA::CUDABuffer> m_MatIdxBuffer = nullptr;
            OptixIndicesFormat m_IndicesFormat = OPTIX_INDICES_FORMAT_NONE;
            size_t m_TriIdxStride = 0;
        };
    }
    namespace utils {
        enum AxisFlag {
            AXIS_FLAG_YZ = 0,
            AXIS_FLAG_ZX = 1,
            AXIS_FLAG_XY = 2,
        };
        struct   Rect {
            float    x0;
            float    x1;
            float    y0;
            float    y1;
            float    z;
            AxisFlag axis;
        public:
            auto getVertices()const noexcept-> std::vector<float3> {
                std::vector<float3> vertices = {};
                vertices.resize(4);
                auto axisX = (axis + 1) % 3;
                auto axisY = (axis + 2) % 3;
                auto axisZ = (axis + 3) % 3;
                float vertex0[3] = {};
                vertex0[axisX] = x0;
                vertex0[axisY] = y0;
                vertex0[axisZ] = z;
                float vertexX[3] = {};
                vertexX[axisX] = x1;
                vertexX[axisY] = y0;
                vertexX[axisZ] = z;
                float vertexY[3] = {};
                vertexY[axisX] = x0;
                vertexY[axisY] = y1;
                vertexY[axisZ] = z;
                float vertex1[3] = {};
                vertex1[axisX] = x1;
                vertex1[axisY] = y1;
                vertex1[axisZ] = z;
                vertices[toIndex(0, 0)].x = vertex0[0];
                vertices[toIndex(0, 0)].y = vertex0[1];
                vertices[toIndex(0, 0)].z = vertex0[2];
                vertices[toIndex(1, 0)].x = vertexX[0];
                vertices[toIndex(1, 0)].y = vertexX[1];
                vertices[toIndex(1, 0)].z = vertexX[2];
                vertices[toIndex(1, 1)].x = vertex1[0];
                vertices[toIndex(1, 1)].y = vertex1[1];
                vertices[toIndex(1, 1)].z = vertex1[2];
                vertices[toIndex(0, 1)].x = vertexY[0];
                vertices[toIndex(0, 1)].y = vertexY[1];
                vertices[toIndex(0, 1)].z = vertexY[2];
                return vertices;
            }
            auto getIndices()const noexcept-> std::vector<uint3> {
                std::vector<uint3> indices = {};
                indices.resize(2);
                auto i00 = toIndex(0, 0);
                auto i10 = toIndex(1, 0);
                auto i11 = toIndex(1, 1);
                auto i01 = toIndex(0, 1);
                indices[0].x = i00;
                indices[0].y = i10;
                indices[0].z = i11;
                indices[1].x = i11;
                indices[1].y = i01;
                indices[1].z = i00;
                return indices;
            }
        private:
            static constexpr auto toIndex(uint32_t x, uint32_t y)noexcept->uint32_t {
                return 2 * y + x;
            }
        };
        struct    Box {
            float    x0;
            float    x1;
            float    y0;
            float    y1;
            float    z0;
            float    z1;
            auto getVertices()const noexcept-> std::vector<float3> {
                std::vector<float3> vertices = {};
                vertices.resize(8);
                //z: 0->
                vertices[toIndex(0, 0, 0)].x = x0;
                vertices[toIndex(0, 0, 0)].y = y0;
                vertices[toIndex(0, 0, 0)].z = z0;
                vertices[toIndex(1, 0, 0)].x = x1;
                vertices[toIndex(1, 0, 0)].y = y0;
                vertices[toIndex(1, 0, 0)].z = z0;
                vertices[toIndex(0, 1, 0)].x = x0;
                vertices[toIndex(0, 1, 0)].y = y1;
                vertices[toIndex(0, 1, 0)].z = z0;
                vertices[toIndex(1, 1, 0)].x = x1;
                vertices[toIndex(1, 1, 0)].y = y1;
                vertices[toIndex(1, 1, 0)].z = z0;
                //z: 1->
                vertices[toIndex(0, 0, 1)].x = x0;
                vertices[toIndex(0, 0, 1)].y = y0;
                vertices[toIndex(0, 0, 1)].z = z1;
                vertices[toIndex(1, 0, 1)].x = x1;
                vertices[toIndex(1, 0, 1)].y = y0;
                vertices[toIndex(1, 0, 1)].z = z1;
                vertices[toIndex(0, 1, 1)].x = x0;
                vertices[toIndex(0, 1, 1)].y = y1;
                vertices[toIndex(0, 1, 1)].z = z1;
                vertices[toIndex(1, 1, 1)].x = x1;
                vertices[toIndex(1, 1, 1)].y = y1;
                vertices[toIndex(1, 1, 1)].z = z1;
                return vertices;
            }
            auto  getIndices()const noexcept-> std::vector<uint3> {
                std::vector<uint3> indices = {};
                indices.resize(12);
                for (uint32_t i = 0; i < 3; ++i) {
                    for (uint32_t j = 0; j < 2; ++j) {
                        uint32_t index[3] = {};
                        //x...y...z
                        uint32_t iX = (i + 1) % 3;//x�ɑΉ�
                        uint32_t iY = (i + 2) % 3;//y�ɑΉ�
                        uint32_t iZ = (i + 3) % 3;//z�ɑΉ�
                        index[iX] = 0;
                        index[iY] = 0;
                        index[iZ] = j;
                        auto i00 = toIndex(index[0], index[1], index[2]);
                        index[iX] = 1;
                        index[iY] = 0;
                        index[iZ] = j;
                        auto i10 = toIndex(index[0], index[1], index[2]);
                        index[iX] = 1;
                        index[iY] = 1;
                        index[iZ] = j;
                        auto i11 = toIndex(index[0], index[1], index[2]);
                        index[iX] = 0;
                        index[iY] = 1;
                        index[iZ] = j;
                        auto i01 = toIndex(index[0], index[1], index[2]);
                        if (j == 0) {
                            indices[2 * (2 * i + j) + 0].x = i00;
                            indices[2 * (2 * i + j) + 0].y = i10;
                            indices[2 * (2 * i + j) + 0].z = i11;
                            indices[2 * (2 * i + j) + 1].x = i11;
                            indices[2 * (2 * i + j) + 1].y = i01;
                            indices[2 * (2 * i + j) + 1].z = i00;
                        }
                        else {
                            indices[2 * (2 * i + j) + 0].x = i00;
                            indices[2 * (2 * i + j) + 0].y = i01;
                            indices[2 * (2 * i + j) + 0].z = i11;
                            indices[2 * (2 * i + j) + 1].x = i11;
                            indices[2 * (2 * i + j) + 1].y = i10;
                            indices[2 * (2 * i + j) + 1].z = i00;
                        }

                    }
                }
                return indices;
            }
        private:
            static constexpr auto toIndex(uint32_t x, uint32_t y, uint32_t z)noexcept->uint32_t {
                return 4 * z + 2 * y + x;
            }
        };
        struct    AABB {
            float3 min = make_float3(FLT_MAX, FLT_MAX, FLT_MAX);
            float3 max = make_float3(-FLT_MAX, -FLT_MAX, -FLT_MAX);
        public:
            AABB()noexcept {}
            AABB(const AABB& aabb)noexcept = default;
            AABB& operator=(const AABB& aabb)noexcept = default;
            AABB(const float3& min, const float3& max)noexcept :min{ min }, max{ max }{}
            AABB(const std::vector<float3>& vertices)noexcept :AABB() {
                for (auto& vertex : vertices) {
                    this->Update(vertex);
                }
            }
            auto GetArea()const noexcept -> float {
                float3 range = max - min;
                return 2.0f * (range.x * range.y + range.y * range.z + range.z * range.x);
            }
            void Update(const float3& vertex)noexcept {
                min = RTLib::Ext::CUDA::Math::min(min, vertex);
                max = RTLib::Ext::CUDA::Math::max(max, vertex);
            }
        };

    }
}
#endif