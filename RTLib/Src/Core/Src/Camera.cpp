#include <RTLib/Core/Camera.h>

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
