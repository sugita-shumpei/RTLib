#ifndef RTLIB_CAMERA_CAMERA_H
#define RTLIB_CAMERA_CAMERA_H
#include <vector>
#include <memory>
#include <cassert>
#include <iostream>
#include <array>
#include <cmath>
#include <initializer_list>
#include <string>
#include <cstdlib>
#include <vector>
#include <tuple>
#define RTLIB_CAMERA_UTILS_MATH_CONSTANTS_PI 3.14159265
#define RTLIB_CAMERA_UTILS_CAMERA_MACRO_GETTER(TYPE,NAME, VARIABLE) \
    auto Get##NAME()const noexcept -> decltype(TYPE::VARIABLE) { return VARIABLE; }; \
    template<typename T> \
    auto Get##NAME##As()const noexcept -> T { static_assert(sizeof(T)==sizeof(TYPE::VARIABLE));\
        T res = T(); \
        auto v = Get##NAME(); std::memcpy(&res,&v, sizeof(v)); \
        return res;  \
    }
#define RTLIB_CAMERA_UTILS_CAMERA_MACRO_SETTER(TYPE, NAME, VARIABLE) \
    void Set##NAME(const decltype(TYPE::VARIABLE)& v) noexcept { VARIABLE = v; }; \
    template<typename T> \
    void Set##NAME##As(const T& v) noexcept { \
        static_assert(sizeof(T) == sizeof(VARIABLE)); \
        std::memcpy(&VARIABLE, &v, sizeof(v)); \
    }
#define RTLIB_CAMERA_UTILS_CAMERA_MACRO_GETTER_AND_SETTER(TYPE, NAME,VARIABLE) \
    RTLIB_CAMERA_UTILS_CAMERA_MACRO_GETTER(TYPE, NAME,VARIABLE); \
    RTLIB_CAMERA_UTILS_CAMERA_MACRO_SETTER(TYPE, NAME,VARIABLE)

namespace RTLib
{
    
    namespace CameraUtils
    {
        template<typename T, size_t N>
        inline constexpr auto IdentityMatrix()->std::array<T, N* N>{
            std::array<T, N* N> res = {};
            for (size_t i = 0; i < N; ++i) {
                res[N * i + i] = 1;
            }
            return res;
        }
        template<typename T, size_t N>
        inline constexpr auto Add(const std::array<T, N>& a, const std::array<T, N>& b)->std::array<T, N>
        {
            std::array<T, N> c;
            for (int i = 0; i < N; ++i) {
                c[i] = a[i] + b[i];
            }
            return c;
        }
        template<typename T, size_t N>
        inline constexpr auto Sub(const std::array<T, N>& a, const std::array<T, N>& b)->std::array<T, N>
        {
            std::array<T, N> c;
            for (int i = 0; i < N; ++i) {
                c[i] = a[i] - b[i];
            }
            return c;
        }
        template<typename T, size_t N>
        inline constexpr auto Div(const std::array<T, N>& a, const T& b)->std::array<T, N>
        {
            std::array<T, N> c;
            for (int i = 0; i < N; ++i) {
                c[i] = a[i] / b;
            }
            return c;
        }
        template<typename T, size_t N>
        inline constexpr auto Div(const std::array<T, N>& a, const std::array<T, N>& b)->std::array<T, N>
        {
            std::array<T, N> c;
            for (int i = 0; i < N; ++i) {
                c[i] = a[i] / b[i];
            }
            return c;
        }
        template<typename T, size_t N>
        inline constexpr auto Mul(const std::array<T, N>& a, const std::array<T, N>& b)->std::array<T, N>
        {
            std::array<T, N> c;
            for (int i = 0; i < N; ++i) {
                c[i] = a[i] * b[i];
            }
            return c;
        }
        template<typename T, size_t N>
        inline constexpr auto Mul(const std::array<T, N>& a, const T& b)->std::array<T, N>
        {
            std::array<T, N> c;
            for (int i = 0; i < N; ++i) {
                c[i] = a[i] * b;
            }
            return c;
        }
        template<typename T, size_t N>
        inline constexpr auto Dot(const std::array<T, N>& a, const std::array<T, N>& b)->T
        {
            T res = 0;
            for (int i = 0; i < N; ++i) {
                res += a[i] * b[i];
            }
            return res;
        }
        template<typename T, size_t N>
        inline constexpr auto LenSqr(const std::array<T, N>& a) -> T {
            T res = 0;

            for (int i = 0; i < N; ++i) {
                res += a[i] * a[i];
            }
            return res;
        }
        template<typename T, size_t N>
        inline constexpr auto Len(const std::array<T, N>& a) -> T {
            return std::sqrt(LenSqr(a));
        }
        template<typename T, size_t N>
        inline constexpr auto Normalize(const std::array<T, N>& a) -> std::array<T, N> {
            return Mul(a,std::sqrt(LenSqr(a)));
        }
        template<typename T>
        inline constexpr auto Cross(const std::array<T, 3>& a, const std::array<T, 3>& b)->std::array<T, 3>{
            return std::array<T, 3>{
                a[1] * b[2] - a[2] * b[1],
                a[2] * b[0] - a[0] * b[2],
                a[0] * b[1] - a[1] * b[0]
            };
        }
    }
    class Camera
    {
        std::array<float, 3> m_Eye;
        std::array<float, 3> m_LookAt;
        std::array<float, 3> m_Vup;
        float m_FovY;
        float m_Aspect;

    public:
        Camera() noexcept : m_Eye{}, m_LookAt{}, m_Vup{}, m_FovY{}, m_Aspect{} {}
        Camera(
            const std::array<float, 3>& eye,
            const std::array<float, 3>& lookAt,
            const std::array<float, 3>& vup,
            const float fovY,
            const float aspect) noexcept
            : m_Eye{ eye },
            m_LookAt{ lookAt },
            m_Vup{ vup },
            m_FovY{ fovY },
            m_Aspect{ aspect }
        {}
        //Direction
        inline std::array<float, 3> GetDirection() const noexcept
        {
            auto lens = RTLib::CameraUtils::Sub(m_LookAt, m_Eye);
            auto len  = RTLib::CameraUtils::Len(lens);
            return RTLib::CameraUtils::Div(lens, std::array<float, 3>{len, len, len});
        }
        inline void SetDirection(const std::array<float, 3>& direction) noexcept
        {
            auto lens  = RTLib::CameraUtils::Sub(m_LookAt, m_Eye);
            auto d_len = RTLib::CameraUtils::Len(direction);
            auto l_len = RTLib::CameraUtils::Len(lens);
            m_Eye = RTLib::CameraUtils::Sub(m_Eye, RTLib::CameraUtils::Mul(direction, std::array<float, 3>{l_len / d_len, l_len / d_len, l_len / d_len}));
        }
        //Get And Set
        RTLIB_CAMERA_UTILS_CAMERA_MACRO_GETTER_AND_SETTER(Camera, Eye, m_Eye);
        RTLIB_CAMERA_UTILS_CAMERA_MACRO_GETTER_AND_SETTER(Camera, LookAt, m_LookAt);
        RTLIB_CAMERA_UTILS_CAMERA_MACRO_GETTER_AND_SETTER(Camera, Vup, m_Vup);
        RTLIB_CAMERA_UTILS_CAMERA_MACRO_GETTER_AND_SETTER(Camera, FovY, m_FovY);
        RTLIB_CAMERA_UTILS_CAMERA_MACRO_GETTER_AND_SETTER(Camera, Aspect, m_Aspect);
        //getUVW
        void GetUVW(std::array<float, 3>& u, std::array<float, 3>& v, std::array<float, 3>& w) const noexcept {
            w = CameraUtils::Sub(m_LookAt, m_Eye);
            //front
            //u = rtlib::normalize(rtlib::cross(w,m_Vup));
            u = CameraUtils::Normalize(CameraUtils::Cross(m_Vup, w));
            v = CameraUtils::Normalize(CameraUtils::Cross(w, u));
            auto vlen = CameraUtils::Len(w) * std::tanf(RTLIB_CAMERA_UTILS_MATH_CONSTANTS_PI * m_FovY / 360.0f);
            auto ulen = vlen * m_Aspect;
            u[0] *= ulen;
            u[1] *= ulen;
            u[2] *= ulen;
            v[0] *= vlen;
            v[1] *= vlen;
            v[2] *= vlen;
        }
        std::tuple<std::array<float, 3>, std::array<float, 3>, std::array<float, 3>> GetUVW() const noexcept {
            std::tuple<std::array<float, 3>, std::array<float, 3>, std::array<float, 3>> uvw;
            this->GetUVW(std::get<0>(uvw), std::get<1>(uvw), std::get<2>(uvw));
            return uvw;
        }

        auto GetLookAtMatrixRH()const noexcept -> std::array<float, 16>{
            auto w = CameraUtils::Sub(m_LookAt, m_Eye);
            auto u = CameraUtils::Normalize(CameraUtils::Cross(m_Vup, w));
            auto v = CameraUtils::Normalize(CameraUtils::Cross(w, u));
            std::array<float, 16> res = {
                 u[0], u[1], u[2],0.0f,
                 v[0], v[1], v[2],0.0f,
                -w[0],-w[1],-w[2],0.0f,
                -CameraUtils::Dot(u,GetEye()),-CameraUtils::Dot(v,GetEye()), CameraUtils::Dot(w,GetEye()),1.0f,
            };
            return res;
        }
        auto GetLookAtMatrixLH()const noexcept -> std::array<float, 16>{
            auto w = CameraUtils::Sub(m_LookAt, m_Eye);
            auto u = CameraUtils::Normalize(CameraUtils::Cross(m_Vup, w));
            auto v = CameraUtils::Normalize(CameraUtils::Cross(w, u));
            std::array<float, 16> res = {
                 u[0], u[1], u[2],0.0f,
                 v[0], v[1], v[2],0.0f,
                 w[0], w[1], w[2],0.0f,
                -CameraUtils::Dot(u,GetEye()),-CameraUtils::Dot(v,GetEye()), -CameraUtils::Dot(w,GetEye()),1.0f,
            };
            return res;
        }
        auto GetPerspectiveMatrixRH(float zNear, float zFar)const noexcept -> std::array<float, 16>{
            assert(width > static_cast<T>(0));
            assert(height > static_cast<T>(0));
            assert(GetFovY() > static_cast<T>(0));

            const float rad = GetFovY() * RTLIB_CAMERA_UTILS_MATH_CONSTANTS_PI / 180.0f;
            const float h = cosf(0.5f * rad) / sinf(0.5f * rad);
            const float w = h / GetAspect();
            const float res2_2 = -zFar / (zFar - zNear);
            const float res3_2 = -(zFar * zNear) / (zFar - zNear);
            std::array<float, 16> res = {
                w   , 0.0f,  0.0f,  0.0f,
                0.0f,    h,  0.0f,  0.0f,
                0.0f, 0.0f,res2_2, -1.0f,
                0.0f, 0.0f,res3_2,  0.0f,
            };
            return res;
        }
        auto GetPerspectiveMatrixLH(float zNear, float zFar)const noexcept -> std::array<float, 16>{
            assert(width > static_cast<T>(0));
            assert(height > static_cast<T>(0));
            assert(GetFovY() > static_cast<T>(0));

            const float rad = GetFovY() * RTLIB_CAMERA_UTILS_MATH_CONSTANTS_PI / 180.0f;
            const float h = cosf(0.5f * rad) / sinf(0.5f * rad);
            const float w = h / GetAspect();
            const float res2_2 = zFar / (zFar - zNear);
            const float res3_2 = -(zFar * zNear) / (zFar - zNear);
            std::array<float, 16> res = {
                w   , 0.0f,  0.0f,  0.0f,
                0.0f,    h,  0.0f,  0.0f,
                0.0f, 0.0f,res2_2,  1.0f,
                0.0f, 0.0f,res3_2,  0.0f,
            };
            return res;
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
    struct     CameraController
    {
        inline static constexpr float defaultYaw = -90.0f;
        inline static constexpr float defaultPitch = 0.0f;
        inline static constexpr float defaultSpeed = 1.0f;
        inline static constexpr float defaultSensitivity = 0.025f;
        inline static constexpr float defaultZoom = 45.0f;

    private:
        std::array<float, 3> m_Position;
        std::array<float, 3> m_Front;
        std::array<float, 3> m_Up;
        std::array<float, 3> m_Right;
        float m_Yaw;
        float m_Pitch;
        float m_MovementSpeed;
        float m_MouseSensitivity;
        float m_Zoom;

    public:
        CameraController(
            const std::array<float, 3>& position = std::array<float, 3>{0.0f, 0.0f, 0.0f},
            const std::array<float, 3>& up = std::array<float, 3>{0.0f, 1.0f, 0.0f},
            float yaw = defaultYaw,
            float pitch = defaultPitch,
            float sensitivity = defaultSensitivity,
            float movementSpeed = defaultSpeed,
            float zoom = defaultZoom) noexcept : m_Position{ position },
            m_Up{ up },
            m_Yaw{ yaw },
            m_Pitch{ pitch },
            m_MouseSensitivity{ sensitivity },
            m_MovementSpeed{ movementSpeed },
            m_Zoom{ zoom }
        {
            UpdateCameraVectors();
        }
        void SetCamera(const Camera& camera) noexcept
        {
            m_Position = camera.GetEye();
            m_Front = CameraUtils::Sub(camera.GetLookAt(), m_Position);
            m_Up = camera.GetVup();
            m_Right = CameraUtils::Normalize(CameraUtils::Cross(m_Up, m_Front));
        }
        auto GetCamera(float fovY, float aspect) const noexcept -> Camera
        {
            return Camera(m_Position, CameraUtils::Add(m_Position, m_Front), m_Up, fovY, aspect);
        }
        auto GetCamera(float aspect)const noexcept->Camera {
            return GetCamera(m_Zoom, aspect);
        }
        void ProcessKeyboard(CameraMovement mode, float deltaTime) noexcept
        {
            float velocity = m_MovementSpeed * deltaTime;
            if (mode == CameraMovement::eForward)
            {
                m_Position[0] += m_Front[0] * velocity;
                m_Position[1] += m_Front[1] * velocity;
                m_Position[2] += m_Front[2] * velocity;
            }
            if (mode == CameraMovement::eBackward)
            {
                m_Position[0] -= m_Front[0] * velocity;
                m_Position[1] -= m_Front[1] * velocity;
                m_Position[2] -= m_Front[2] * velocity;

            }
            if (mode == CameraMovement::eLeft)
            {
                m_Position[0] += m_Right[0] * velocity;
                m_Position[1] += m_Right[1] * velocity;
                m_Position[2] += m_Right[2] * velocity;
            }
            if (mode == CameraMovement::eRight)
            {
                m_Position[0] -= m_Right[0] * velocity;
                m_Position[1] -= m_Right[1] * velocity;
                m_Position[2] -= m_Right[2] * velocity;
            }
            if (mode == CameraMovement::eUp)
            {
                m_Position[0] -= m_Up[0] * velocity;
                m_Position[1] -= m_Up[1] * velocity;
                m_Position[2] -= m_Up[2] * velocity;
            }
            if (mode == CameraMovement::eDown)
            {
                m_Position[0] += m_Up[0] * velocity;
                m_Position[1] += m_Up[1] * velocity;
                m_Position[2] += m_Up[2] * velocity;
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

        RTLIB_CAMERA_UTILS_CAMERA_MACRO_GETTER_AND_SETTER(CameraController, Position, m_Position);
        RTLIB_CAMERA_UTILS_CAMERA_MACRO_GETTER_AND_SETTER(CameraController, Up, m_Up);
        RTLIB_CAMERA_UTILS_CAMERA_MACRO_GETTER_AND_SETTER(CameraController, Yaw, m_Yaw);
        RTLIB_CAMERA_UTILS_CAMERA_MACRO_GETTER_AND_SETTER(CameraController, Pitch, m_Pitch);
        RTLIB_CAMERA_UTILS_CAMERA_MACRO_GETTER_AND_SETTER(CameraController, MouseSensitivity, m_MouseSensitivity);
        RTLIB_CAMERA_UTILS_CAMERA_MACRO_GETTER_AND_SETTER(CameraController, MovementSpeed, m_MovementSpeed);
        RTLIB_CAMERA_UTILS_CAMERA_MACRO_GETTER_AND_SETTER(CameraController, Zoom, m_Zoom);
    private:
        // Calculates the front vector from the Camera's (updated) Euler Angles
        void UpdateCameraVectors() noexcept
        {
            // Calculate the new Front vector
            std::array<float, 3> front = {};
            float yaw   = RTLIB_CAMERA_UTILS_MATH_CONSTANTS_PI * (m_Yaw) / 180.0f;
            float pitch = RTLIB_CAMERA_UTILS_MATH_CONSTANTS_PI * (m_Pitch) / 180.0f;
            front[0] = cos(yaw) * cos(pitch);
            front[1] = sin(pitch);
            front[2] = sin(yaw) * cos(pitch);
            m_Front = CameraUtils::Normalize(front);
            m_Right = CameraUtils::Normalize(CameraUtils::Cross(m_Up, m_Front));
        }
    };

}
#endif
