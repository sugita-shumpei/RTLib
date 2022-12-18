#include <RTLib/Core/Camera.h>
#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>
//Direction
static auto Internal_StdArrayF3ToGlmVec3(const std::array<float, 3>& v)-> glm::vec3 {
    return glm::vec3(v[0], v[1], v[2]);
}
static auto Internal_GlmVec3ToStdArrayF3(const glm::vec3& v)->std::array<float, 3> {
    return std::array<float, 3>{v[0], v[1], v[2]};
}
std::array<float, 3> RTLib::Core::Camera::GetDirection() const noexcept
{
    return Internal_GlmVec3ToStdArrayF3(glm::normalize(Internal_StdArrayF3ToGlmVec3(m_LookAt) - Internal_StdArrayF3ToGlmVec3(m_Eye)));
}

void RTLib::Core::Camera::SetDirection(const std::array<float, 3>& direction) noexcept
{
    auto dirs  = Internal_StdArrayF3ToGlmVec3(direction);
    auto dLen  = glm::length(dirs);
    auto lLen  = glm::length(Internal_StdArrayF3ToGlmVec3(m_LookAt) - Internal_StdArrayF3ToGlmVec3(m_Eye));
    m_Eye = Internal_GlmVec3ToStdArrayF3(Internal_StdArrayF3ToGlmVec3(m_Eye) - dirs * static_cast<float>(lLen / dLen));
}

//getUVW

void RTLib::Core::Camera::GetUVW(std::array<float, 3>& u, std::array<float, 3>& v, std::array<float, 3>& w) const noexcept {
    auto vecW = Internal_StdArrayF3ToGlmVec3(m_LookAt) - Internal_StdArrayF3ToGlmVec3(m_Eye);
    auto vecU = glm::normalize(glm::cross(Internal_StdArrayF3ToGlmVec3(m_Vup), vecW));
    auto vecV = glm::normalize(glm::cross(vecW, vecU));
    auto vLen = glm::length(vecW) * std::tan(glm::radians(m_FovY * 0.5f));
    auto uLen = vLen * m_Aspect;
    u = Internal_GlmVec3ToStdArrayF3(vecU*uLen);
    v = Internal_GlmVec3ToStdArrayF3(vecV*vLen);
    w = Internal_GlmVec3ToStdArrayF3(vecW);
}

void RTLib::Core::CameraController::SetCamera(const Camera& camera) noexcept
{
    m_Position  = camera.GetEye();
    auto vFront = Internal_StdArrayF3ToGlmVec3(camera.GetLookAt()) - Internal_StdArrayF3ToGlmVec3(m_Position);
    m_Front     = Internal_GlmVec3ToStdArrayF3(vFront);
    m_Up        = camera.GetVup();
    m_Right     = Internal_GlmVec3ToStdArrayF3(glm::normalize(glm::cross(Internal_StdArrayF3ToGlmVec3(m_Up),vFront)));
}

auto RTLib::Core::CameraController::GetCamera(float fovY, float aspect) const noexcept -> Camera
{
    return Camera(m_Position, Internal_GlmVec3ToStdArrayF3(Internal_StdArrayF3ToGlmVec3(m_Position)+Internal_StdArrayF3ToGlmVec3(m_Front)), m_Up, fovY, aspect);
}

// Calculates the front vector from the Camera's (updated) Euler Angles

void RTLib::Core::CameraController::UpdateCameraVectors() noexcept
{
    // Calculate the new Front vector
    float yaw   = glm::radians(m_Yaw);
    float pitch = glm::radians(m_Pitch);
    auto vFront = glm::normalize(glm::vec3(
        cos(yaw) * cos(pitch),
        sin(pitch),
        sin(yaw) * cos(pitch))
    );
    m_Front = Internal_GlmVec3ToStdArrayF3(vFront);
    m_Right = Internal_GlmVec3ToStdArrayF3(glm::normalize(glm::cross(Internal_StdArrayF3ToGlmVec3(m_Up), vFront)));
}

void RTLib::Core::to_json(nlohmann::json& json, const CameraController& cameraController)
{
    json["Position"] = cameraController.GetPosition();
    json["Yaw"] = cameraController.GetYaw();
    json["Pitch"] = cameraController.GetPitch();
    json["Zoom"] = cameraController.GetZoom();
    json["MouseSensitivity"] = cameraController.GetMouseSensitivity();
    json["MovementSpeed"] = cameraController.GetMovementSpeed();
}

void RTLib::Core::to_json(nlohmann::json& json, const Camera& camera)
{
    json["Eye"] = camera.GetEye();
    json["LookAt"] = camera.GetLookAt();
    json["Vup"] = camera.GetVup();
    json["Aspect"] = camera.GetAspect();
    json["FovY"] = camera.GetFovY();
}

void RTLib::Core::from_json(const nlohmann::json& json, CameraController& cameraController)
{
    cameraController = CameraController(
        json.at("Position").get<std::array<float, 3>>(),
        { 0.0f,1.0f,0.0f },
        json.at("Yaw").get<float>(),
        json.at("Pitch").get<float>(),
        json.at("MouseSensitivity").get<float>(),
        json.at("MovementSpeed").get<float>(),
        json.at("Zoom").get<float>()
    );
}

void RTLib::Core::from_json(const nlohmann::json& json, Camera& camera)
{
    camera.SetEye(json.at("Eye").get<std::array<float, 3>>());
    camera.SetLookAt(json.at("LookAt").get<std::array<float, 3>>());
    camera.SetVup(json.at("Vup").get<std::array<float, 3>>());
    camera.SetAspect(json.at("Aspect").get<float>());
    camera.SetFovY(json.at("FovY").get<float>());
}

