#ifndef RTLIB_EXT_OPX7_TEST_H
#define RTLIB_EXT_OPX7_TEST_H
#include <RTLib/Core/BinaryReader.h>
#include <RTLib/Ext/OPX7/OPX7Context.h>
#include <RTLib/Ext/OPX7/OPX7Module.h>
#include <RTLib/Ext/OPX7/OPX7ProgramGroup.h>
#include <RTLib/Ext/OPX7/OPX7ShaderTable.h>
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
#include <fstream>
#include <unordered_map>
#include <array>
#include <string>
#include <string_view>
#include "cuda/SimpleKernel.h"
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
        class Camera
        {
            float3 m_Eye;
            float3 m_LookAt;
            float3 m_Vup;
            float m_FovY;
            float m_Aspect;

        public:
            Camera() noexcept : m_Eye{}, m_LookAt{}, m_Vup{}, m_FovY{}, m_Aspect{} {}
            Camera(const float3& eye,
                const float3& lookAt,
                const float3& vup,
                const float fovY,
                const float aspect) noexcept
                : m_Eye{ eye },
                m_LookAt{ lookAt },
                m_Vup{ vup },
                m_FovY{ fovY },
                m_Aspect{ aspect }
            {
                //std::cout << "Camera Eye (x:" << m_Eye.x    << " y:" <<m_Eye.y    << " z:" <<    m_Eye.z << ")" << std::endl;
                //std::cout << "Camera  At (x:" << m_LookAt.x << " y:" <<m_LookAt.y << " z:" << m_LookAt.z << ")" << std::endl;
                //std::cout << "Camera  up (x:" <<    m_Vup.x << " y:" <<   m_Vup.y << " z:" <<    m_Vup.z << ")" << std::endl;
            }
            //Direction
            inline float3 getDirection() const noexcept
            {
                return normalize(m_LookAt - m_Eye);
            }
            inline void setDirection(const float3& direction) noexcept
            {
                auto len = length(m_LookAt - m_Eye);
                m_Eye += len * normalize(direction);
            }
            //Get And Set
            RTLIB_DECLARE_GET_AND_SET_BY_REFERENCE(Camera, float3, Eye, m_Eye);
            RTLIB_DECLARE_GET_AND_SET_BY_REFERENCE(Camera, float3, LookAt, m_LookAt);
            RTLIB_DECLARE_GET_AND_SET_BY_REFERENCE(Camera, float3, Vup, m_Vup);
            RTLIB_DECLARE_GET_AND_SET_BY_VALUE(Camera, float, FovY, m_FovY);
            RTLIB_DECLARE_GET_AND_SET_BY_VALUE(Camera, float, Aspect, m_Aspect);
            //getUVW
            void getUVW(float3& u, float3& v, float3& w) const noexcept {
                w = m_LookAt - m_Eye;
                //front
                //u = rtlib::normalize(rtlib::cross(w,m_Vup));
                u = normalize(cross(m_Vup, w));
                v = normalize(cross(w, u));
                auto vlen = length(w) * std::tanf(RTLIB_M_PI * m_FovY / 360.0f);
                auto ulen = vlen * m_Aspect;
                u *= ulen;
                v *= vlen;
            }
            std::tuple<float3, float3, float3> getUVW() const noexcept {
                std::tuple<float3, float3, float3> uvw;
                this->getUVW(std::get<0>(uvw), std::get<1>(uvw), std::get<2>(uvw));
                return uvw;
            }
        };
        enum class CameraMovement : uint8_t
        {
            eForward = 0,
            eBackward = 1,
            eLeft = 2,
            eRight = 3,
            eUp = 4,
            eDown = 5,
        };
        struct CameraController
        {
        private:
            inline static constexpr float defaultYaw = -90.0f;
            inline static constexpr float defaultPitch = 0.0f;
            inline static constexpr float defaultSpeed = 1.0f;
            inline static constexpr float defaultSensitivity = 0.025f;
            inline static constexpr float defaultZoom = 45.0f;

        private:
            float3 m_Position;
            float3 m_Front;
            float3 m_Up;
            float3 m_Right;
            float m_Yaw;
            float m_Pitch;
            float m_MovementSpeed;
            float m_MouseSensitivity;
            float m_Zoom;

        public:
            CameraController(
                const float3& position = make_float3(0.0f, 0.0f, 0.0f),
                const float3& up = make_float3(0.0f, 1.0f, 0.0f),
                float yaw = defaultYaw,
                float pitch = defaultPitch) noexcept : m_Position{ position },
                m_Up{ up },
                m_Yaw{ yaw },
                m_Pitch{ pitch },
                m_MouseSensitivity{ defaultSensitivity },
                m_MovementSpeed{ defaultSpeed },
                m_Zoom{ defaultZoom }
            {
                UpdateCameraVectors();
            }
            void SetCamera(const Camera& camera) noexcept
            {
                m_Position = camera.getEye();
                m_Front = camera.getLookAt() - m_Position;
                m_Up = camera.getVup();
                m_Right = normalize(cross(m_Up, m_Front));
            }
            auto GetCamera(float fovY, float aspect) const noexcept -> Camera
            {
                return Camera(m_Position, m_Position + m_Front, m_Up, fovY, aspect);
            }
            auto GetCamera(float aspect)const noexcept->Camera {
                return GetCamera(m_Zoom, aspect);
            }
            auto GetZoom()const noexcept ->float {
                return m_Zoom;
            }
            void SetZoom(float zoom) noexcept {
                m_Zoom = zoom;
            }
            void ProcessKeyboard(CameraMovement mode, float deltaTime) noexcept
            {
                float velocity = m_MovementSpeed * deltaTime;
                if (mode == CameraMovement::eForward)
                {
                    m_Position += m_Front * velocity;
                }
                if (mode == CameraMovement::eBackward)
                {
                    m_Position -= m_Front * velocity;

                }
                if (mode == CameraMovement::eLeft)
                {
                    m_Position += m_Right * velocity;
                }
                if (mode == CameraMovement::eRight)
                {
                    m_Position -= m_Right * velocity;
                }
                if (mode == CameraMovement::eUp)
                {
                    m_Position -= m_Up * velocity;
                }
                if (mode == CameraMovement::eDown)
                {
                    m_Position += m_Up * velocity;
                }

            }
            void ProcessMouseMovement(float xoffset, float yoffset, bool constrainPitch = true) noexcept
            {
                xoffset *= m_MouseSensitivity;
                yoffset *= m_MouseSensitivity;
                m_Yaw -= xoffset;
                m_Pitch += yoffset;
                if (constrainPitch)
                {
                    if (m_Pitch > 89.0f)
                    {
                        m_Pitch = 89.0f;
                    }
                    if (m_Pitch < -89.0f)
                    {
                        m_Pitch = -89.0f;
                    }
                }
                UpdateCameraVectors();
            }
            void ProcessMouseScroll(float yoffset) noexcept
            {
                float next_zoom = m_Zoom - yoffset;
                if (next_zoom >= 1.0f && next_zoom <= 45.0f)
                    m_Zoom = next_zoom;
                if (next_zoom <= 1.0f)
                    m_Zoom = 1.0f;
                if (next_zoom >= 45.0f)
                    m_Zoom = 45.0f;
            }
            auto GetMouseSensitivity()const noexcept -> float { return m_MouseSensitivity; }
            void SetMouseSensitivity(float sensitivity) noexcept
            {
                m_MouseSensitivity = sensitivity;
            }
            auto GetMovementSpeed()const noexcept -> float { return m_MovementSpeed; }
            void SetMovementSpeed(float speed) noexcept
            {
                m_MovementSpeed = speed;
            }
        private:
            // Calculates the front vector from the Camera's (updated) Euler Angles
            void UpdateCameraVectors() noexcept
            {
                // Calculate the new Front vector
                float3 front;
                float yaw = RTLIB_M_PI * (m_Yaw) / 180.0f;
                float pitch = RTLIB_M_PI * (m_Pitch) / 180.0f;
                front.x = cos(yaw) * cos(pitch);
                front.y = sin(pitch);
                front.z = sin(yaw) * cos(pitch);
                m_Front = normalize(front);
                m_Right = normalize(cross(m_Up, m_Front));
            }
        };
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