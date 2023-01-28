#ifndef RTLIB_UTILS_CAMERA_H
#define RTLIB_UTILS_CAMERA_H
#include <RTLib/Utils/Math.h>
#include <nlohmann/json.hpp>
#include <array>
#include <cmath>
#include <initializer_list>
#include <string>
#include <cstdlib>
#include <vector>
#include <tuple>
#define RTLIB_UTILS_CAMERA_MACRO_GETTER(TYPE,NAME, VARIABLE) \
    auto Get##NAME()const noexcept -> decltype(TYPE::VARIABLE) { return VARIABLE; }; \
    template<typename T> \
    auto Get##NAME##As()const noexcept -> T { static_assert(sizeof(T)==sizeof(TYPE::VARIABLE));\
        T res = T(); \
        auto v = Get##NAME(); std::memcpy(&res,&v, sizeof(v)); \
        return res;  \
    }
#define RTLIB_UTILS_CAMERA_MACRO_SETTER(TYPE, NAME, VARIABLE) \
    void Set##NAME(const decltype(TYPE::VARIABLE)& v) noexcept { VARIABLE = v; }; \
    template<typename T> \
    void Set##NAME##As(const T& v) noexcept { \
        static_assert(sizeof(T) == sizeof(VARIABLE)); \
        std::memcpy(&VARIABLE, &v, sizeof(v)); \
    }
#define RTLIB_UTILS_CAMERA_MACRO_GETTER_AND_SETTER(TYPE, NAME,VARIABLE) \
    RTLIB_UTILS_CAMERA_MACRO_GETTER(TYPE, NAME,VARIABLE); \
    RTLIB_UTILS_CAMERA_MACRO_SETTER(TYPE, NAME,VARIABLE)

namespace RTLib
{
    namespace Utils
    {
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
                auto lens = RTLib::Utils::Sub(m_LookAt, m_Eye);
                auto len = RTLib::Utils::Len(lens);
                return RTLib::Utils::Div(lens, std::array<float, 3>{len, len, len});
            }
            inline void SetDirection(const std::array<float, 3>& direction) noexcept
            {
                auto lens = RTLib::Utils::Sub(m_LookAt ,m_Eye);
                auto d_len = RTLib::Utils::Len(direction);
                auto l_len = RTLib::Utils::Len(lens);
                m_Eye = RTLib::Utils::Sub(m_Eye,RTLib::Utils::Mul(direction,std::array<float, 3>{l_len/ d_len, l_len / d_len, l_len / d_len}));
            }
            //Get And Set
            RTLIB_UTILS_CAMERA_MACRO_GETTER_AND_SETTER(Camera, Eye   , m_Eye   );
            RTLIB_UTILS_CAMERA_MACRO_GETTER_AND_SETTER(Camera, LookAt, m_LookAt);
            RTLIB_UTILS_CAMERA_MACRO_GETTER_AND_SETTER(Camera, Vup   , m_Vup   );
            RTLIB_UTILS_CAMERA_MACRO_GETTER_AND_SETTER(Camera, FovY  , m_FovY  );
            RTLIB_UTILS_CAMERA_MACRO_GETTER_AND_SETTER(Camera, Aspect, m_Aspect);
            //getUVW
            void GetUVW(std::array<float,3>& u, std::array<float, 3>& v, std::array<float, 3>& w) const noexcept {
                w = Sub(m_LookAt ,m_Eye);
                //front
                //u = rtlib::normalize(rtlib::cross(w,m_Vup));
                u = Normalize(Cross(m_Vup, w));
                v = Normalize(Cross(w, u));
                auto vlen = Len(w) * std::tanf(RTLIB_UTILS_MATH_CONSTANTS_PI * m_FovY / 360.0f);
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
                const std::array<float, 3>& up       = std::array<float, 3>{0.0f, 1.0f, 0.0f},
                float yaw   = defaultYaw,
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
                m_Front = Sub(camera.GetLookAt() , m_Position);
                m_Up    = camera.GetVup();
                m_Right = Normalize(Cross(m_Up, m_Front));
            }
            auto GetCamera(float fovY, float aspect) const noexcept -> Camera
            {
                return Camera(m_Position, Add(m_Position , m_Front), m_Up, fovY, aspect);
            }
            auto GetCamera(float aspect)const noexcept->Camera {
                return GetCamera(m_Zoom, aspect);
            }
            void ProcessKeyboard(CameraMovement mode, float deltaTime) noexcept
            {
                float velocity = m_MovementSpeed * deltaTime;
                if (mode == CameraMovement::eForward)
                {
                    m_Position[0] -= m_Front[0] * velocity;
                    m_Position[1] -= m_Front[1] * velocity;
                    m_Position[2] -= m_Front[2] * velocity;
                }
                if (mode == CameraMovement::eBackward)
                {
                    m_Position[0] += m_Front[0] * velocity;
                    m_Position[1] += m_Front[1] * velocity;
                    m_Position[2] += m_Front[2] * velocity;

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
            
            RTLIB_UTILS_CAMERA_MACRO_GETTER_AND_SETTER(CameraController, Position        , m_Position        );
            RTLIB_UTILS_CAMERA_MACRO_GETTER_AND_SETTER(CameraController, Up              , m_Up              );
            RTLIB_UTILS_CAMERA_MACRO_GETTER_AND_SETTER(CameraController, Yaw             , m_Yaw             );
            RTLIB_UTILS_CAMERA_MACRO_GETTER_AND_SETTER(CameraController, Pitch           , m_Pitch           );
            RTLIB_UTILS_CAMERA_MACRO_GETTER_AND_SETTER(CameraController, MouseSensitivity, m_MouseSensitivity);
            RTLIB_UTILS_CAMERA_MACRO_GETTER_AND_SETTER(CameraController, MovementSpeed   , m_MovementSpeed   );
            RTLIB_UTILS_CAMERA_MACRO_GETTER_AND_SETTER(CameraController, Zoom            , m_Zoom            );
        private:
            // Calculates the front vector from the Camera's (updated) Euler Angles
            void UpdateCameraVectors() noexcept
            {
                // Calculate the new Front vector
                std::array<float, 3> front = {};
                float yaw   = RTLIB_UTILS_MATH_CONSTANTS_PI * (m_Yaw) / 180.0f;
                float pitch = RTLIB_UTILS_MATH_CONSTANTS_PI * (m_Pitch) / 180.0f;
                front[0] = cos(yaw) * cos(pitch);
                front[1] = sin(pitch);
                front[2] = sin(yaw) * cos(pitch);
                m_Front = Normalize(front);
                m_Right = Normalize(Cross(m_Up, m_Front));
            }
        };

        void to_json(nlohmann::json& json, const CameraController& cameraController);
        void to_json(nlohmann::json& json, const Camera& camera);
        void from_json(const nlohmann::json& json, CameraController& cameraController);
        void from_json(const nlohmann::json& json, Camera& camera);
    }
}
#undef RTLIB_UTILS_CAMERA_MACRO_GETTER_AND_SETTER
#undef RTLIB_UTILS_CAMERA_MACRO_GETTER
#undef RTLIB_UTILS_CAMERA_MACRO_SETTER
#endif
