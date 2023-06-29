#include <RTLib/Core/Matrix.h>
#include <vector>
#include <array>

void glm::to_json(RTLib::Core::Json& json, const RTLib::Core::Matrix2x2& m)
{
	json = nlohmann::json(
		std::vector<std::vector<float>>{
			{ m[0][0], m[0][1] },
			{ m[1][0], m[1][1] }
		}
	);
}

void glm::to_json(RTLib::Core::Json& json, const RTLib::Core::Matrix3x3& m)
{
	json = nlohmann::json(
		std::vector<std::vector<float>>{
			{ m[0][0], m[0][1], m[0][2] },
			{ m[1][0], m[1][1], m[1][2] },
			{ m[2][0], m[2][1], m[2][2] }
		}
	);
}

void glm::to_json(RTLib::Core::Json& json, const RTLib::Core::Matrix4x4& m)
{
	json = nlohmann::json(
		std::vector<std::vector<float>>{
			{ m[0][0], m[0][1], m[0][2], m[0][3] },
			{ m[1][0], m[1][1], m[1][2], m[1][3] },
			{ m[2][0], m[2][1], m[2][2], m[2][3] },
			{ m[3][0], m[3][1], m[3][2], m[3][3] }
		}
	);
}

void glm::from_json(const RTLib::Core::Json& json, RTLib::Core::Matrix2x2& m)
{
	auto v0 = json.at(0).get<std::vector<float>>();
	auto v1 = json.at(1).get<std::vector<float>>();

	m[0][0] = v0.at(0); m[0][1] = v0.at(1);
	m[1][0] = v1.at(0); m[1][1] = v1.at(1);
}

void glm::from_json(const RTLib::Core::Json& json, RTLib::Core::Matrix3x3& m)
{
	auto v0 = json.at(0).get<std::vector<float>>();
	auto v1 = json.at(1).get<std::vector<float>>();
	auto v2 = json.at(2).get<std::vector<float>>();

	m[0][0] = v0.at(0); m[0][1] = v0.at(1); m[0][2] = v0.at(2);
	m[1][0] = v1.at(0); m[1][1] = v1.at(1); m[1][2] = v1.at(2);
	m[2][0] = v2.at(0); m[2][1] = v2.at(1); m[2][2] = v2.at(2);
}

void glm::from_json(const RTLib::Core::Json& json, RTLib::Core::Matrix4x4& m)
{
	auto v0 = json.at(0).get<std::vector<float>>();
	auto v1 = json.at(1).get<std::vector<float>>();
	auto v2 = json.at(2).get<std::vector<float>>();
	auto v3 = json.at(3).get<std::vector<float>>();

	m[0][0] = v0.at(0); m[0][1] = v0.at(1); m[0][2] = v0.at(2); m[0][3] = v0.at(3);
	m[1][0] = v1.at(0); m[1][1] = v1.at(1); m[1][2] = v1.at(2); m[1][3] = v1.at(3);
	m[2][0] = v2.at(0); m[2][1] = v2.at(1); m[2][2] = v2.at(2); m[2][3] = v2.at(3);
	m[3][0] = v3.at(0); m[3][1] = v3.at(1); m[3][2] = v3.at(2); m[3][3] = v3.at(3);
}
