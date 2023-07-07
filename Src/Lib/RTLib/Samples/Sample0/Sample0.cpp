#include <RTLib/Core/Vector.h>
#include <RTLib/Core/Matrix.h>
#include <RTLib/Core/Camera.h>
#include <RTLib/Core/Json.h>
#include <RTLib/Core/Mesh.h>

#include <RTLib/Scene/Object.h>
#include <RTLib/Scene/Transform.h>
#include <RTLib/Scene/TransformGraph.h>
#include <RTLib/Scene/Camera.h>
#include <RTLib/Scene/Mesh.h>
#include <RTLib/Core/Quaternion.h>
//Assimp 
#include <assimp/config.h>
#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/GenericProperty.h>
#include <assimp/scene.h>
//glm
#include <glm/gtx/string_cast.hpp>
#include <glm/gtc/matrix_integer.hpp>
#include <glm/gtx/euler_angles.hpp>
//glfw
#include <glad/gl.h>
#include <GLFW/glfw3.h>
//Texture
#include <DirectXTex.h>
#include <Sample0Config.h>
#include <optional>
#include <random>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <cstdlib>
#include <stack>
#include <functional>
#include <unordered_set>

int main(int argc, const char** argv)
{
	auto importer = Assimp::Importer();
	//importer.SetPropertyBool(AI_CONFIG_IMPORT_FBX_PRESERVE_PIVOTS, false);
	importer.SetPropertyFloat(AI_CONFIG_GLOBAL_SCALE_FACTOR_KEY, 0.01f);
	auto data_path = SAMPLE_SAMPLE0_DATA_PATH"\\Models\\ZeroDay\\MEASURE_ONE\\MEASURE_ONE.fbx";
	//auto data_path = SAMPLE_SAMPLE0_DATA_PATH"\\Models\\Bistro_v5_2\\BistroExterior.fbx";
	auto data_root = SAMPLE_SAMPLE0_DATA_PATH"\\Models\\ZeroDay";

	unsigned int flag = 0;
	flag |= aiProcess_PreTransformVertices;
	flag |= aiProcess_Triangulate;
	//flag |= aiProcess_GenSmoothNormals;
	flag |= aiProcess_GenUVCoords;
	flag |= aiProcess_GenBoundingBoxes;
	//flag |= aiProcess_OptimizeMeshes;

	auto scene = importer.ReadFile(data_path, flag);

	RTLib::Int32 width = 1024;
	RTLib::Int32 height = 0;

	std::unordered_map<aiNode*, RTLib::Matrix4x4> modelMatrixMap = {};
	std::vector<GLuint> meshVaos = {};
	std::vector<GLuint> meshVbos = {};
	std::vector<std::pair<GLuint, GLsizei>> meshIbos = {};

	{
		std::stack<aiNode*> nodes = {};
		nodes.push(scene->mRootNode);
		modelMatrixMap[scene->mRootNode] = glm::transpose(std::_Bit_cast<RTLib::Matrix4x4>(scene->mRootNode->mTransformation));

		while (!nodes.empty()) {
			auto node = nodes.top();
			nodes.pop();
			auto numChildren = node->mNumChildren;
			for (auto i = 0; i < numChildren; ++i) {
				auto child = node->mChildren[i];
				auto parentTransform = modelMatrixMap.at(node);
				auto childTransform  = glm::transpose(std::_Bit_cast<RTLib::Matrix4x4>(child->mTransformation));
				modelMatrixMap.insert({ child, parentTransform * childTransform });
				nodes.push(child);
			}
		}


	}
	glfwInit();
	{
		auto camera = scene->mCameras[0];
		height = width / camera->mAspect;

		glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_API);
		glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
		glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GLFW_TRUE);
		glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
		glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
		glfwWindowHint(GLFW_VERSION_MAJOR, 3);
		glfwWindowHint(GLFW_VERSION_MINOR, 3);
		auto window = glfwCreateWindow(width, height, camera->mName.C_Str(), nullptr, nullptr);
		if (!window) { return -1; }

		glfwMakeContextCurrent(window);
		auto gl = std::make_unique<GladGLContext>();
		if (!gladLoadGLContext(gl.get(), (GLADloadfunc)glfwGetProcAddress)) {
			throw std::runtime_error("Failed To Load GL Context!");
		}
		{
			meshVaos.resize(scene->mNumMeshes);
			meshIbos.resize(scene->mNumMeshes);
			meshVbos.resize(scene->mNumMeshes);
			for (RTLib::UInt64 i = 0; i < scene->mNumMeshes; ++i) {
				auto mesh = scene->mMeshes[i];
				std::vector<RTLib::Float32> vertices(mesh->mNumVertices * 8);
				std::vector<RTLib::UInt32>  indices(mesh->mNumFaces * 3);
				for (RTLib::UInt64 j = 0; j < mesh->mNumVertices; ++j) {
					vertices[8 * j + 0] = mesh->mVertices[j][0];
					vertices[8 * j + 1] = mesh->mVertices[j][1];
					vertices[8 * j + 2] = mesh->mVertices[j][2];
					if (mesh->HasNormals())
					{
						vertices[8 * j + 3] = mesh->mNormals[j][0];
						vertices[8 * j + 4] = mesh->mNormals[j][1];
						vertices[8 * j + 5] = mesh->mNormals[j][2];
					}
					else {

						vertices[8 * j + 3] = 0.0f;
						vertices[8 * j + 4] = 0.0f;
						vertices[8 * j + 5] = 0.0f;
					}
					if (mesh->HasTextureCoords(0))
					{
						vertices[8 * j + 6] = mesh->mTextureCoords[0][j][0];
						vertices[8 * j + 7] = mesh->mTextureCoords[0][j][1];
					}
					else {

						vertices[8 * j + 6] = 0.5f;
						vertices[8 * j + 7] = 0.5f;
					}
					//std::cout << "V: " << vertices[8 * j + 0] << "," << vertices[8 * j + 1] << "," << vertices[8 * j + 2] << std::endl;
				}
				for (RTLib::UInt64 j = 0; j < mesh->mNumFaces; ++j) {
					indices[3 * j + 0] = mesh->mFaces[j].mIndices[0];
					indices[3 * j + 1] = mesh->mFaces[j].mIndices[1];
					indices[3 * j + 2] = mesh->mFaces[j].mIndices[2];
				}

				GLuint vao;
				gl->GenVertexArrays(1, &vao);
				GLuint bff[2];
				gl->GenBuffers(2, bff);

				meshVaos[i] = vao;
				meshVbos[i] = bff[0];
				meshIbos[i] = { bff[1],mesh->mNumFaces * 3 };

				gl->BindBuffer(GL_ARRAY_BUFFER, bff[0]);
				gl->BufferData(GL_ARRAY_BUFFER, mesh->mNumVertices * sizeof(RTLib::Float32) * 8, vertices.data(), GL_STATIC_DRAW);

				gl->BindBuffer(GL_ELEMENT_ARRAY_BUFFER, bff[1]);
				gl->BufferData(GL_ELEMENT_ARRAY_BUFFER, mesh->mNumFaces * 3 * sizeof(RTLib::UInt32), indices.data(), GL_STATIC_DRAW);

				gl->BindVertexArray(vao);
				gl->BindBuffer(GL_ARRAY_BUFFER, bff[0]);
				gl->BindBuffer(GL_ELEMENT_ARRAY_BUFFER, bff[1]);
				auto offset = static_cast<GLintptr>(0);
				gl->VertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(RTLib::Float32) * 8, reinterpret_cast<void*>(offset));
				gl->EnableVertexAttribArray(0);
				offset = static_cast<GLintptr>(sizeof(RTLib::Float32) * 3);
				gl->VertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(RTLib::Float32) * 8, reinterpret_cast<void*>(offset));
				gl->EnableVertexAttribArray(1);
				offset = static_cast<GLintptr>(sizeof(RTLib::Float32) * 6);
				gl->VertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(RTLib::Float32) * 8, reinterpret_cast<void*>(offset));
				gl->EnableVertexAttribArray(2);
				gl->BindVertexArray(0);
				gl->BindBuffer(GL_ARRAY_BUFFER, 0);
				gl->BindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
			}
		}

		auto prg = gl->CreateProgram();
		{
			constexpr RTLib::Char vsSource[] =
				"#version 460 core\n"
				"layout (location = 0) in vec3 position;\n"
				"layout (location = 1) in vec3 normal;\n"
				"layout (location = 2) in vec2 uv;\n"
				"uniform mat4 model;\n"
				"uniform mat4 viewProj;\n"
				"out vec3 outNormal;\n"
				"void main(){\n"
				"	gl_Position = viewProj * model * vec4(position,1.0);\n"
				"	outNormal = normalize(normal);\n"
				"}\n";
			constexpr RTLib::Char fsSource[] =
				"#version 460 core\n"
				"in vec3 outNormal;\n"
				"uniform vec4 color;\n"
				"layout (location = 0) out vec4 fragColor;\n"
				"void main(){\n"
				"	fragColor = vec4(0.5*outNormal+0.5,1.0);\n"
				"}\n";
			const RTLib::Char* pVsSource = vsSource;
			const RTLib::Char* pFsSource = fsSource;

			auto vs = gl->CreateShader(GL_VERTEX_SHADER);
			gl->ShaderSource(vs, 1, &pVsSource, nullptr);
			gl->CompileShader(vs);
			gl->AttachShader(prg, vs);
			{
				RTLib::Int32 status;
				RTLib::Int32 length;
				gl->GetShaderiv(vs, GL_COMPILE_STATUS, &status);
				gl->GetShaderiv(vs, GL_INFO_LOG_LENGTH, &length);
				std::vector<RTLib::Char> log(length + 1, '\0');
				gl->GetShaderInfoLog(vs, length, nullptr, log.data());
				if (status != GL_TRUE) {
					std::cerr << "VS Log: " << log.data() << std::endl;
				}
			}
			auto fs = gl->CreateShader(GL_FRAGMENT_SHADER);
			gl->ShaderSource(fs, 1, &pFsSource, nullptr);
			gl->CompileShader(fs);
			gl->AttachShader(prg, fs);
			{
				RTLib::Int32 status;
				RTLib::Int32 length;
				gl->GetShaderiv(fs, GL_COMPILE_STATUS, &status);
				gl->GetShaderiv(fs, GL_INFO_LOG_LENGTH, &length);
				std::vector<RTLib::Char> log(length + 1, '\0');
				gl->GetShaderInfoLog(fs, length, nullptr, log.data());
				if (status != GL_TRUE) {
					std::cerr << "FS Log: " << log.data() << std::endl;
				}
			}

			gl->LinkProgram(prg);
			{
				RTLib::Int32 status;
				RTLib::Int32 length;
				gl->GetProgramiv(prg, GL_LINK_STATUS, &status);
				gl->GetProgramiv(prg, GL_INFO_LOG_LENGTH, &length);
				std::vector<RTLib::Char> log(length + 1, '\0');
				gl->GetProgramInfoLog(prg, length, nullptr, log.data());
				if (status != GL_TRUE) {
					std::cerr << "Program Log: " << log.data() << std::endl;
				}
			}
			gl->DetachShader(prg, vs);
			gl->DetachShader(prg, fs);
		}

		auto modelPos    = gl->GetUniformLocation(prg, "model");
		auto viewProjPos = gl->GetUniformLocation(prg, "viewProj");
		auto colorPos    = gl->GetUniformLocation(prg, "color");

		auto cameraNode = scene->mRootNode->FindNode(scene->mCameras[0]->mName.C_Str());
		auto lookAtLen  = camera->mLookAt.Length();
		auto upLen      = camera->mUp.Length();
		auto viewMatrix = glm::lookAt(
			glm::vec3(camera->mPosition[0], camera->mPosition[1], camera->mPosition[2]),
			glm::vec3(camera->mPosition[0], camera->mPosition[1], camera->mPosition[2]) +
			glm::vec3(camera->mLookAt[0]  , camera->mLookAt[1]  , camera->mLookAt[2]  ),
			glm::vec3(camera->mUp[0]      , camera->mUp[1]      , camera->mUp[2]      )
		);
		auto cameraLocalMatrixXLen = glm::length(glm::vec3(viewMatrix[0]));
		auto cameraLocalMatrixYLen = glm::length(glm::vec3(viewMatrix[1]));
		auto cameraLocalMatrixZLen = glm::length(glm::vec3(viewMatrix[2]));

		auto viewMatrixBase = glm::inverse(modelMatrixMap.at(cameraNode));

		auto cameraPrarentMatrix = glm::transpose(std::_Bit_cast<RTLib::Core::Matrix4x4>(cameraNode->mTransformation));

		auto cameraParentMatrixXLen = glm::length(glm::vec3(cameraPrarentMatrix[0]));
		auto cameraParentMatrixYLen = glm::length(glm::vec3(cameraPrarentMatrix[1]));
		auto cameraParentMatrixZLen = glm::length(glm::vec3(cameraPrarentMatrix[2]));

		viewMatrix = viewMatrix * viewMatrixBase;

		auto cameraMatrixXLen = glm::length(glm::vec3(viewMatrix[0]));
		auto cameraMatrixYLen = glm::length(glm::vec3(viewMatrix[1]));
		auto cameraMatrixZLen = glm::length(glm::vec3(viewMatrix[2]));

		// OPENGLが想定しているFOVはY軸まわり, 一方でASSIMPが想定しているFOVはX軸まわり
		// なので結果を変える必要有
		// X = ASPECT * TAN_Y
		// Y = TAN_Y
		// 
		// X = TAN_X
		// Y = TAN_X / ASPECT
		// 注意: cameraのmClipPlaneNear, mClipPlaneFarは正規化された空間におけるFrustrum(0~1)ではなく
		// ビュー空間における値
		// そのため, View行列の深度方向のスケールをかける必要あり

		float fovy = std::atan(std::tan(camera->mHorizontalFOV / 2.0f) / camera->mAspect) * 2.0f;
		auto projMatrix = glm::perspective(fovy, camera->mAspect, camera->mClipPlaneNear, camera->mClipPlaneFar);

		auto viewProjMatrix = projMatrix * viewMatrix;
		{
			std::cout << glm::to_string(viewProjMatrix) << std::endl;
		}

		//{

		//	for (auto& [node, modelMatrix] : modelMatrixMap)
		//	{

		//		for (RTLib::UInt64 i = 0; i < node->mNumMeshes; ++i) {
		//			auto mesh = scene->mMeshes[node->mMeshes[i]];
		//			for (RTLib::UInt64 j = 0; j < mesh->mNumVertices; ++j) {
		//				auto vertex = RTLib::Vector3();
		//				vertex.x = mesh->mVertices[j][0];
		//				vertex.y = mesh->mVertices[j][1];
		//				vertex.z = mesh->mVertices[j][2];
		//				auto worldVertex = modelMatrix * RTLib::Vector4(vertex, 1.0f);
		//				std::cout << "VW: " << worldVertex.x << "," << worldVertex.y << "," << worldVertex.z << std::endl;

		//			}
		//		}
		//	}
		//	for (auto& [node, modelMatrix] : modelMatrixMap)
		//	{

		//		for (RTLib::UInt64 i = 0; i < node->mNumMeshes; ++i) {
		//			auto mesh = scene->mMeshes[node->mMeshes[i]];
		//			for (RTLib::UInt64 j = 0; j < mesh->mNumVertices; ++j) {
		//				auto vertex = RTLib::Vector3();
		//				vertex.x = mesh->mVertices[j][0];
		//				vertex.y = mesh->mVertices[j][1];
		//				vertex.z = mesh->mVertices[j][2];
		//				auto viewVertex = viewMatrix * modelMatrix * RTLib::Vector4(vertex, 1.0f);
		//				std::cout << "VV: " << viewVertex.x << "," << viewVertex.y << "," << viewVertex.z << std::endl;

		//			}
		//		}
		//	}
		//	for (auto& [node, modelMatrix] : modelMatrixMap)
		//	{

		//		for (RTLib::UInt64 i = 0; i < node->mNumMeshes; ++i) {
		//			auto mesh = scene->mMeshes[node->mMeshes[i]];
		//			for (RTLib::UInt64 j = 0; j < mesh->mNumVertices; ++j) {
		//				auto vertex = RTLib::Vector3();
		//				vertex.x = mesh->mVertices[j][0];
		//				vertex.y = mesh->mVertices[j][1];
		//				vertex.z = mesh->mVertices[j][2];
		//				auto viewVertex = viewMatrix * modelMatrix * RTLib::Vector4(vertex, 1.0f);
		//				auto projVertex = projMatrix * viewVertex;
		//				projVertex /= projVertex.w;
		//				std::cout << "VP: " << projVertex.x << "," << projVertex.y << "," << projVertex.z << std::endl;

		//			}
		//		}
		//	}
		//}
		gl->Enable(GL_DEPTH_TEST);
		gl->DepthFunc(GL_LESS);

		while (!glfwWindowShouldClose(window))
		{
			gl->Clear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
			gl->ClearColor(0.0f, 0.0f, 0.0f, 1.0f);
			gl->ClearDepth(1.0f);
			gl->Viewport(0, 0, width, height);
			gl->UseProgram(prg);
			gl->UniformMatrix4fv(viewProjPos, 1, GL_FALSE, &viewProjMatrix[0][0]);
			std::unordered_set<RTLib::UInt32> alreadyDrawSet = {};
			for (auto& [node, model] : modelMatrixMap)
			{
				gl->UniformMatrix4fv(modelPos, 1, GL_FALSE, &model[0][0]);
				for (auto i = 0; i < node->mNumMeshes; ++i) {

					RTLib::UInt32 meshIdx = node->mMeshes[i];
					if (alreadyDrawSet.count(meshIdx) > 0) { continue; }
					std::mt19937 mt1(meshIdx);
					std::mt19937 mt(mt1);
					std::uniform_real_distribution<float> uni(0.0f, 1.0f);
					RTLib::Vector4 vec(uni(mt), uni(mt), uni(mt), 1.0f);
					GLuint vao = meshVaos.at(meshIdx);
					GLuint cnt = meshIbos.at(meshIdx).second;
					gl->Uniform4fv(colorPos, 1, &vec[0]);
					gl->BindVertexArray(vao);
					gl->DrawElements(GL_TRIANGLES, scene->mMeshes[meshIdx]->mNumFaces * 3, GL_UNSIGNED_INT, 0);
					alreadyDrawSet.insert(meshIdx);
				}
			}
			glfwSwapBuffers(window);
			glfwPollEvents();
			{
				float cameraVelocity = 1.0f;
				if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
				{
					camera->mPosition += cameraVelocity * camera->mLookAt;

				}
				if ((glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) ||
					(glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_PRESS))
				{

					auto right = camera->mUp ^ camera->mLookAt;
					auto rightLen = std::sqrt(right.Length());
					right.Normalize();
					right *= rightLen;
					camera->mPosition -= cameraVelocity * right;

				}
				if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
				{
					camera->mPosition -= cameraVelocity * camera->mLookAt;
				}
				if ((glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) ||
					(glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS))
				{
					auto right = camera->mUp ^ camera->mLookAt;
					auto rightLen = std::sqrt(right.Length());
					right.Normalize();
					right *= rightLen;
					camera->mPosition += cameraVelocity * right;
				}
				if (glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS)
				{
					camera->mPosition += cameraVelocity * camera->mUp;
				}
				if (glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS)
				{
					camera->mPosition -= cameraVelocity * camera->mUp;
				}
				{
					viewMatrix = glm::lookAt(
						glm::vec3(camera->mPosition[0], camera->mPosition[1], camera->mPosition[2]),
						glm::vec3(camera->mPosition[0], camera->mPosition[1], camera->mPosition[2]) +
						glm::vec3(camera->mLookAt[0]  , camera->mLookAt[1]  , camera->mLookAt[2]),
						glm::vec3(camera->mUp[0], camera->mUp[1], camera->mUp[2])
					) * viewMatrixBase;
					// GLMが想定しているFOVはY軸まわり, 一方でASSIMPが想定しているFOVはX軸まわり
					// なので結果を変える必要有
					// X = ASPECT * TAN_Y
					// Y = TAN_Y
					// 
					// X = TAN_X
					// Y = TAN_X / ASPECT
					fovy = std::atan(std::tan(camera->mHorizontalFOV / 2.0f) / camera->mAspect) * 2.0f;
					projMatrix = glm::perspective(fovy, camera->mAspect, camera->mClipPlaneNear, camera->mClipPlaneFar);

					viewProjMatrix = projMatrix * viewMatrix;
				}
			}
		}
		{
			gl->DeleteVertexArrays(meshVaos.size(), meshVaos.data());
			gl->DeleteBuffers(meshVbos.size(), meshVbos.data());
			for (auto& [ibo, cnt] : meshIbos) {
				gl->DeleteBuffers(1, &ibo);
			}
		}
		gl->DeleteProgram(prg);
		glfwDestroyWindow(window);
		window = nullptr;
		glfwTerminate();
	}
	glfwTerminate();

	return 0;
}