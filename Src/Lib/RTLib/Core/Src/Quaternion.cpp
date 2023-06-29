#include <RTLib/Core/Quaternion.h>

void glm::to_json(RTLib::Core::Json& json, const RTLib::Core::Quat& q)
{
	auto euler = glm::degrees(glm::eulerAngles(q));
	json = RTLib::Core::Json{
		{"x",euler.x},
		{"y",euler.y},
		{"z",euler.z}
	};
}

void glm::from_json(const RTLib::Core::Json& json, RTLib::Core::Quat& q)
{
	auto euler = glm::radians(
		RTLib::Core::Vector3(
			json.at("x").get<float>(),
			json.at("y").get<float>(),
			json.at("z").get<float>()
		)
	);
	q = RTLib::Core::Quat(euler);
}
