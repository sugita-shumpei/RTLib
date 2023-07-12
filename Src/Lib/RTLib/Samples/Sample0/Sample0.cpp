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
#include <fstream>
#include <unordered_set>
#include "Sample0.h"

struct Vertex
{
	RTLib::Vector3 position;
	RTLib::Vector3 normal;
	RTLib::Vector2 uv;
};

int main(int argc, const char** argv)
{
#ifdef WIN32
	CoInitialize(0);
#endif
	auto importer = Assimp::Importer();
	//importer.SetPropertyBool(AI_CONFIG_IMPORT_FBX_PRESERVE_PIVOTS, false);
	//importer.SetPropertyFloat(AI_CONFIG_GLOBAL_SCALE_FACTOR_KEY, 10000.0f);
	auto data_path = SAMPLE_SAMPLE0_DATA_PATH"\\Models\\ZeroDay\\MEASURE_ONE\\MEASURE_ONE.fbx";
	//auto data_path = SAMPLE_SAMPLE0_DATA_PATH"\\Models\\Bistro_v5_2\\BistroExterior.fbx";
	auto data_root = std::filesystem::path(data_path).parent_path();

	unsigned int flag = 0;
	//flag |= aiProcess_PreTransformVertices;
	flag |= aiProcess_Triangulate;
	//flag |= aiProcess_GenSmoothNormals;
	flag |= aiProcess_GenUVCoords;
	flag |= aiProcess_GenBoundingBoxes;
	//flag |= aiProcess_OptimizeMeshes;

	auto scene = importer.ReadFile(data_path, flag);

	auto rootNode = std::shared_ptr<RootNode>(new RootNode(scene->mRootNode));
	{
		auto cameras = std::vector<aiCamera*>(scene->mCameras, scene->mCameras + scene->mNumCameras);
		//auto cameraNode = rootNode->find_node(cameras[0]->mName.C_Str());
		auto sceneMeta = MetaData(scene->mMetaData);

		for (auto camera : cameras) {
			auto cameraName = camera->mName;
			auto cameraNode = rootNode->find_node(cameraName.C_Str());
			auto nextKey = cameraNode.lock()->m_Parent.lock()->m_Node;
			bool isDone = false;
			while (!isDone) {
				auto& parentTrans = rootNode->nodeMap.at(nextKey);
				auto parentNode = parentTrans.node.lock();
				auto parentName = std::string(parentNode->m_Node->mName.C_Str());
				auto baseName   = std::string(cameraName.C_Str()) + "_$AssimpFbx$_";
				if (parentName.size() < baseName.size()) {
					isDone = true;
					break;
				}
				parentName = parentName.substr(0, baseName.size());
				if (parentName != baseName) {
					isDone = true;
					break;
				}
				parentNode->m_LocalPosition = RTLib::Vector3(0.0f);
				parentNode->m_LocalRotation = RTLib::Quat(1.0f,0.0f,0.0f,0.0f);
				parentNode->m_LocalScaling  = RTLib::Vector3(1.0f);
				parentNode->m_Dirty         = true;

				nextKey = parentNode->m_Parent.lock()->m_Node;
			}
		}
	}
	auto animation = std::shared_ptr<Animation>(new Animation(scene->mAnimations[0], rootNode));
	{
		//for (auto&bone : animation->m_Bones) {
		//	if (bone->get_node()->get_name() == "LOWER_ARM") {
		//		std::cout << "OK" << std::endl;
		//	}
		//}
	}

	animation->update_frames(0.0);

	scene->mCameras[0]->mAspect = 16.0f / 9.0f;

	RTLib::Int32 width = 2048;
	RTLib::Int32 height = 0;

	std::vector<GLuint>                       meshVaos = {};
	std::vector<GLuint>                       meshVbos = {};
	std::vector<std::pair<GLuint, GLsizei>>   meshIbos = {};
	std::vector<aiMaterial*>                  materials(scene->mMaterials, scene->mMaterials + scene->mNumMaterials);
	std::unordered_map<std::filesystem::path, std::vector<std::tuple<RTLib::UInt32,aiTextureType,RTLib::UInt32>>> textureInfoMap = {};
	std::vector<std::unordered_map<aiTextureType, std::vector<GLuint>>> texturesForMaterial(scene->mNumMaterials);
	std::cout << "NumMaterials: " << materials.size() << std::endl;
	for (auto i = 0; i < materials.size(); ++i) {
		std::string name = materials[i]->GetName().C_Str();
		//std::cout << "Materials[" << i << "].name=" << name << std::endl;
		//auto properties = std::vector<aiMaterialProperty*>(materials[i]->mProperties, materials[i]->mProperties + materials[i]->mNumProperties);
		//if (name == "Foliage_Leaves.DoubleSided") {
		//	for (auto& prop : properties) {
		//		std::cout << "Materials[" << i << "]." << prop->mKey.C_Str() << std::endl;
		//	}
		//}
		for (auto j = 0; j < aiTextureType_TRANSMISSION; ++j) {
			auto textureCount = materials[i]->GetTextureCount((aiTextureType)j);
			if (textureCount) {
				texturesForMaterial[i].insert({ (aiTextureType)j, std::vector<GLuint>{textureCount} });
			}
			for (auto k = 0; k < textureCount; ++k) {
				aiString aiPath;
				if (materials[i]->Get(AI_MATKEY_TEXTURE((aiTextureType)j, k), aiPath)==aiReturn_SUCCESS) {
					auto texPath = (data_root / aiPath.C_Str()).lexically_normal();
					if (textureInfoMap.count(texPath) == 0) {
						
						textureInfoMap.insert({ texPath ,{{i,(aiTextureType)j,k }} });
					}
					else {
						auto& textureInfos = textureInfoMap.at(texPath);
						textureInfos.push_back({ i,(aiTextureType)j,k });
					}
				}
				
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
				std::vector<Vertex> vertices(mesh->mNumVertices);
				std::vector<RTLib::UInt32>  indices(mesh->mNumFaces * 3);
				for (RTLib::UInt64 j = 0; j < mesh->mNumVertices; ++j) {
					vertices[j].position.x = mesh->mVertices[j][0];
					vertices[j].position.y = mesh->mVertices[j][1];
					vertices[j].position.z = mesh->mVertices[j][2];
					if (mesh->HasNormals())
					{
						vertices[j].normal.x = mesh->mNormals[j][0];
						vertices[j].normal.y = mesh->mNormals[j][1];
						vertices[j].normal.z = mesh->mNormals[j][2];
					}
					else {
						vertices[j].normal.x = 0.0f;
						vertices[j].normal.y = 0.0f;
						vertices[j].normal.z = 1.0f;
					}
					if (mesh->HasTextureCoords(0))
					{
						vertices[j].uv.x = mesh->mTextureCoords[0][j][0];
						vertices[j].uv.y = mesh->mTextureCoords[0][j][1];
					}
					else {

						vertices[j].uv.x = 0.5f;
						vertices[j].uv.y = 0.5f;
					}
					//std::cout << "V: " << vertices[8 * j + 0] << "," << vertices[8 * j + 1] << "," << vertices[8 * j + 2] << std::endl;
				}
				for (RTLib::UInt64 j = 0; j < mesh->mNumFaces; ++j) {
					assert(mesh->mFaces[j].mNumIndices == 3);
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

				gl->VertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof Vertex, reinterpret_cast<void*>(&static_cast<Vertex*>(nullptr)->position));
				gl->EnableVertexAttribArray(0);
				gl->VertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof Vertex, reinterpret_cast<void*>(&static_cast<Vertex*>(nullptr)->normal));
				gl->EnableVertexAttribArray(1);
				gl->VertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof Vertex, reinterpret_cast<void*>(&static_cast<Vertex*>(nullptr)->uv));
				gl->EnableVertexAttribArray(2);

				gl->BindVertexArray(0);
				gl->BindBuffer(GL_ARRAY_BUFFER, 0);
				gl->BindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

				gl->ObjectLabel(GL_VERTEX_ARRAY, vao, -1, mesh->mName.C_Str());
			}
		}
		{

			for (auto& [texPath, texInfos] : textureInfoMap) {
				auto image = DirectX::ScratchImage();
				auto meta = DirectX::TexMetadata();
				auto ext = texPath.extension();
				auto texPathWStr = texPath.wstring();
				if (ext == ".dds")
				{
					GLenum format;
					if (DirectX::LoadFromDDSFile(texPathWStr.data(), DirectX::DDS_FLAGS_NONE, &meta, image) >= 0) {
						std::cout << "Load Texture: DDS" << std::endl;
					}
					else {
						throw std::runtime_error("Error: Failed To Load Texture " + texPath.string());
					}
				}
				else {
					std::cout << "Load Texture: Not Supported" << std::endl;
				}
				for (auto& texInfo : texInfos) {
					auto& [matIdx, texType, texIdx] = texInfo;


					GLenum format;
					if (meta.format == DXGI_FORMAT_BC1_UNORM) {
						std::cout << "	DXGI Format: BC1 Unorm" << std::endl;
						format = GL_COMPRESSED_RGBA_S3TC_DXT1_EXT;
					}
					else if (meta.format == DXGI_FORMAT_BC2_UNORM) {
						std::cout << "	DXGI Format: BC2 Unorm" << std::endl;
						format = GL_COMPRESSED_RGBA_S3TC_DXT3_EXT;
					}
					else if (meta.format == DXGI_FORMAT_BC3_UNORM) {
						std::cout << "	DXGI Format: BC3 Unorm" << std::endl;
						format = GL_COMPRESSED_RGBA_S3TC_DXT5_EXT;
					}
					else if (meta.format == DXGI_FORMAT_BC4_UNORM) {
						std::cout << "	DXGI Format: BC4 Unorm" << std::endl;
						format = GL_COMPRESSED_RED_RGTC1_EXT;
					}
					else if (meta.format == DXGI_FORMAT_BC5_UNORM) {
						std::cout << "	DXGI Format: BC5 Unorm" << std::endl;
						format = GL_COMPRESSED_RED_GREEN_RGTC2_EXT;;
					}
					else {
						throw std::runtime_error("Failed To Support Format!");
					}

					GLuint tex;
					gl->GenTextures(1, &tex);
					// GL_TEXTURE_1D
					if (meta.dimension == DirectX::TEX_DIMENSION_TEXTURE1D)
					{
						if (meta.arraySize <= 1) {
							// Texture1D
							throw std::runtime_error("Error: Failed To Load Texture 1D!");
						}
						else {
							// Texture1DArray
							throw std::runtime_error("Error: Failed To Load Texture 1D Array!");
						}
					}
					if (meta.dimension == DirectX::TEX_DIMENSION_TEXTURE2D)
					{
						if (meta.arraySize <= 1) {
							auto pixelSize = image.GetPixelsSize();
							// Texture2D
							gl->BindTexture(GL_TEXTURE_2D, tex);
							for (int l = 0; l < meta.mipLevels; ++l) {
								const auto& mipImage = image.GetImage(l, 0, 0);
								gl->CompressedTexImage2D(GL_TEXTURE_2D, l, format, mipImage->width, mipImage->height, 0, mipImage->slicePitch, mipImage->pixels);

							}
							gl->TexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
							gl->TexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
						}
						else {
							if (meta.IsCubemap()) {
								if (meta.arraySize == 6)
								{
									// Cubemap
									throw std::runtime_error("Error: Failed To Load Texture Cubemap!");
								}
								else {
									// Cubemap Array
									throw std::runtime_error("Error: Failed To Load Texture Cubemap Array!");

								}
							}
							else {
								// Texture2DArray
								throw std::runtime_error("Error: Failed To Load Texture 2D Array!");
							}
							gl->TexImage3D;
						}
					}
					if (meta.dimension == DirectX::TEX_DIMENSION_TEXTURE3D)
					{
						//Texture3D
						throw std::runtime_error("Error: Failed To Load Texture 3D!");
					}

					texturesForMaterial[matIdx][texType][texIdx] = tex;

				}

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
				"uniform mat4 view;\n"
				"uniform mat4 proj;\n"
				"out vec3 outNormal;\n"
				"out vec2 outUv;\n"
				"void main(){\n"
				"	gl_Position = proj * view * model * vec4(position,1.0);\n"
				//"	if (gl_Position.z > gl_Position.w){ gl_Position.z = gl_Position.w; }\n"
				"	outNormal = normalize(transpose(inverse(mat3(model))) * normal);\n"
				"	outUv = vec2(1.0-uv.x,1.0-uv.y);\n"
				"}\n";
			constexpr RTLib::Char fsSource[] =
				"#version 460 core\n"
				"in vec3 outNormal;\n"
				"in vec2 outUv;\n"
				"uniform sampler2D tex;\n"
				"layout (location = 0) out vec4 fragColor;\n"
				"void main(){\n"
				"	fragColor = texture(tex,outUv);\n"
				"	if (fragColor.a < 0.5) { discard; }\n"
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
		auto viewPos     = gl->GetUniformLocation(prg, "view");
		auto projPos     = gl->GetUniformLocation(prg, "proj");
		auto texPos      = gl->GetUniformLocation(prg, "tex");
		auto cameraName = scene->mCameras[0]->mName;
		auto cameraNode = rootNode->find_node(cameraName.C_Str()).lock();
		auto cameraParent = cameraNode->m_Parent.lock();
		
		camera->mHorizontalFOV = glm::radians(90.0f);
		camera->mClipPlaneNear *= 100.0f;

		auto viewMatrix = RTLib::Matrix4x4();
		{
			auto cameraTransform = rootNode->find_node_transform(cameraNode->get_name()).transform;

			auto lookAtLen = camera->mLookAt.Length();
			auto upLen = camera->mUp.Length();

			auto viewMatrixLocal = glm::lookAt(
				glm::vec3(camera->mPosition[0], camera->mPosition[1], camera->mPosition[2]),
				glm::vec3(camera->mPosition[0], camera->mPosition[1], camera->mPosition[2]) +
				glm::vec3(camera->mLookAt[0], camera->mLookAt[1], camera->mLookAt[2]),
				glm::vec3(camera->mUp[0], camera->mUp[1], camera->mUp[2])
			);

			auto cameraToWorldBase = cameraTransform; // SCALEが含まれているので正規化する必要有
			cameraToWorldBase[0] = glm::normalize(cameraToWorldBase[0]);
			cameraToWorldBase[1] = glm::normalize(cameraToWorldBase[1]);
			cameraToWorldBase[2] = glm::normalize(cameraToWorldBase[2]);

			auto cameraScale    = cameraNode->m_LocalScaling;
			auto viewMatrixParent = glm::inverse(cameraToWorldBase);

			viewMatrix = viewMatrixLocal * viewMatrixParent;
		}
		auto projMatrix = RTLib::Matrix4x4();
		{

			float fovy = std::atan(std::tan(camera->mHorizontalFOV / 2.0f) / camera->mAspect) * 2.0f;
			projMatrix= glm::perspectiveLH(fovy, camera->mAspect,  camera->mClipPlaneNear, camera->mClipPlaneFar);

			projMatrix[0].x *= -1.0f;
			projMatrix[1].x *= -1.0f;
			projMatrix[2].x *= -1.0f;
			projMatrix[3].x *= -1.0f;
		}
			
		auto viewProjMatrix = projMatrix * viewMatrix;
		{
			std::cout << glm::to_string(viewProjMatrix) << std::endl;
		}

		gl->Enable(GL_DEPTH_TEST);
		gl->DepthFunc(GL_LESS);

		auto idx = 0;
		auto idx_delta = 0;
		auto tickPerSecond = animation->m_TickPerSecond;
		auto curTime = 0.0f;
		auto delTime = 0.0f;

		glfwSetTime(0.0f);
		while (!glfwWindowShouldClose(window))
		{

			double oldTime = curTime;
			curTime = glfwGetTime();
			delTime = curTime - oldTime;
			auto title = std::to_string(curTime * animation->m_TickPerSecond);
			glfwSetWindowTitle(window, title.c_str());

			animation->update_frames(curTime);

			{
				auto cameraTransform = rootNode->find_node_transform(cameraNode->get_name()).transform;

				auto lookAtLen = camera->mLookAt.Length();
				auto upLen = camera->mUp.Length();

				auto viewMatrixLocal = glm::lookAt(
					glm::vec3(camera->mPosition[0], camera->mPosition[1], camera->mPosition[2]),
					glm::vec3(camera->mPosition[0], camera->mPosition[1], camera->mPosition[2]) +
					glm::vec3(camera->mLookAt[0], camera->mLookAt[1], camera->mLookAt[2]),
					glm::vec3(camera->mUp[0], camera->mUp[1], camera->mUp[2])
				);

				auto cameraToWorldBase = cameraTransform; // SCALEが含まれているので正規化する必要有
				cameraToWorldBase[0] = glm::normalize(cameraToWorldBase[0]);
				cameraToWorldBase[1] = glm::normalize(cameraToWorldBase[1]);
				cameraToWorldBase[2] = glm::normalize(cameraToWorldBase[2]);

				auto cameraScale = cameraNode->m_LocalScaling;
				auto viewMatrixParent = glm::inverse(cameraToWorldBase);

				viewMatrix = viewMatrixLocal * viewMatrixParent;
			}
			{
				float fovy = std::atan(std::tan(camera->mHorizontalFOV / 2.0f) / camera->mAspect) * 2.0f;
				projMatrix = glm::perspectiveLH(fovy, camera->mAspect, camera->mClipPlaneNear, camera->mClipPlaneFar);

				projMatrix[0].x *= -1.0f;
				projMatrix[1].x *= -1.0f;
				projMatrix[2].x *= -1.0f;
				projMatrix[3].x *= -1.0f;

			}


			gl->ClearColor(0.0f, 0.0f, 0.0f, 1.0f);
			gl->ClearDepth(1.0f);
			gl->Clear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
			gl->Viewport(0, 0, width, height);
			gl->UseProgram(prg);
			{
				for (auto& [node, nodeTran] : rootNode->nodeMap)
				{

					for (auto i = 0; i < node->mNumMeshes; ++i) {

						RTLib::UInt32 meshIdx = node->mMeshes[i];

						std::mt19937 mt1(meshIdx);
						std::mt19937 mt(mt1);

						GLuint vao = meshVaos.at(meshIdx);
						GLuint cnt = meshIbos.at(meshIdx).second;

						gl->UniformMatrix4fv(modelPos, 1, GL_FALSE,&nodeTran.transform[0][0]);//おそらくあっている
						gl->UniformMatrix4fv(projPos, 1, GL_FALSE, &projMatrix[0][0]);
						gl->UniformMatrix4fv(viewPos, 1, GL_FALSE, &viewMatrix[0][0]);

						gl->ActiveTexture(GL_TEXTURE0);
						auto& texMap      = texturesForMaterial[scene->mMeshes[meshIdx]->mMaterialIndex];
						if (texMap.count(aiTextureType_DIFFUSE)){
							auto& texDiffuses = texMap.at(aiTextureType_DIFFUSE);
							if (!texDiffuses.empty()) {
								gl->BindTexture(GL_TEXTURE_2D, texDiffuses[0]);
							}
						}
						
						gl->Uniform1i(texPos, 0);

						gl->BindVertexArray(vao);
						gl->DrawElements(GL_TRIANGLES, scene->mMeshes[meshIdx]->mNumFaces * 3, GL_UNSIGNED_INT, 0);
					}
				}
			}
			glfwSwapBuffers(window);
			glfwPollEvents();


		}
		{
			gl->DeleteVertexArrays(meshVaos.size(), meshVaos.data());
			meshVaos.clear();
			gl->DeleteBuffers(meshVbos.size(), meshVbos.data());
			meshVbos.clear();
			for (auto& [ibo, cnt] : meshIbos) {
				gl->DeleteBuffers(1, &ibo);
			}
			meshIbos.clear();
			for (auto& textureForMaterial : texturesForMaterial) {
				for (auto& [type, texs] : textureForMaterial) {
					gl->DeleteTextures(texs.size(), texs.data());
				}
			}
			texturesForMaterial.clear();
		}
		gl->DeleteProgram(prg);
		glfwDestroyWindow(window);
		window = nullptr;
		glfwTerminate();
	}
	glfwTerminate();

	return 0;
}