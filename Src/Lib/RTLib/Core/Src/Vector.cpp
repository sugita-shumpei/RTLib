#include <RTLib/Core/Vector.h>

void glm::to_json(RTLib::Core::Json& json, const RTLib::Core::Vector2& v)
{
	json = {
		{"x",v.x},
		{"y",v.y}
	};
}

void glm::to_json(RTLib::Core::Json& json, const RTLib::Core::Vector3& v)
{
	json = {
		{"x",v.x},
		{"y",v.y},
		{"z",v.z}
	};
}

void glm::to_json(RTLib::Core::Json& json, const RTLib::Core::Vector4& v)
{
	json = {
		{"x",v.x},
		{"y",v.y},
		{"z",v.z},
		{"w",v.w}
	};
}

void glm::from_json(const RTLib::Core::Json& json, RTLib::Core::Vector2& v)
{
	v.x = json.at("x").get<float>();
	v.y = json.at("y").get<float>();

}

void glm::from_json(const RTLib::Core::Json& json, RTLib::Core::Vector3& v)
{
	v.x = json.at("x").get<float>();
	v.y = json.at("y").get<float>();
	v.z = json.at("z").get<float>();
}

void glm::from_json(const RTLib::Core::Json& json, RTLib::Core::Vector4& v)
{
	v.x = json.at("x").get<float>();
	v.y = json.at("y").get<float>();
	v.z = json.at("z").get<float>();
	v.w = json.at("w").get<float>();
}
