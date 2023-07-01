#include <RTLib/Core/Vector.h>
#include <RTLib/Core/Matrix.h>
#include <RTLib/Core/Camera.h>
#include <RTLib/Core/Material.h>
#include <RTLib/Core/Json.h>
#include <glm/gtx/string_cast.hpp>
//Assimp 
#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/GenericProperty.h>
#include <assimp/scene.h>
//Texture Loading Library
#include <DirectXTex.h>
#include <Sample0Config.h>
#include <optional>
#include <filesystem>
#include <string>
#include <cstdlib>

constexpr char* aishading_model_to_string(aiShadingMode shadingModel)
{
	switch (shadingModel)
	{
	case aiShadingMode_Flat:
		return "ShadingMode::Flat";
	case aiShadingMode_Gouraud:
		return "ShadingMode::Gouraud";
	case aiShadingMode_Phong:
		return "ShadingMode::Phong";
	case aiShadingMode_Blinn:
		return "ShadingMode::Blinn";
	case aiShadingMode_Toon:
		return "ShadingMode::Toon";
	case aiShadingMode_OrenNayar:
		return "ShadingMode::OrenNayar";
	case aiShadingMode_Minnaert:
		return "ShadingMode::Minnaert";
	case aiShadingMode_CookTorrance:
		return "ShadingMode::CookTorrance";
	case aiShadingMode_NoShading:
		return "ShadingMode::NoShading";
	case aiShadingMode_Fresnel:
		return "ShadingMode::Fresnel";
	case aiShadingMode_PBR_BRDF:
		return "ShadingMode::PBR_BRDF";
	case _aiShadingMode_Force32Bit:
		return "ShadingMode::Force32Bit";
	default:
		return "ShadingMode::Unknown";
		break;
	}
}
constexpr char* aiblend_mode_to_string(aiBlendMode blendMode)
{
	switch (blendMode)
	{
	case aiBlendMode_Default: return "aiBlendMode::Default";
		break;
	case aiBlendMode_Additive:return "aiBlendMode::Additive";
		break;
	case _aiBlendMode_Force32Bit:return "aiBlendMode::Force32Bit";
		break; 
	default:return "aiBlendMode::Unknown";
		break;
	}
}
std::string     aiuvtransform_to_string(aiUVTransform transform)
{
	std::string res = "(";
	res = res + "rot=" + std::to_string(transform.mRotation) + ",";
	res = res + "scl=[" + std::to_string(transform.mScaling.x)+"," + std::to_string(transform.mScaling.y) + "],";
	res = res + "tra=[" + std::to_string(transform.mTranslation.x) + "," + std::to_string(transform.mTranslation.y) + "])";
	return res;
}
RTLib::Camera   aicamera_to_camera(const aiCamera& camera) {
	auto position = RTLib::Vector3(camera.mPosition.x, camera.mPosition.y, camera.mPosition.z);
	auto vup      = RTLib::Vector3(camera.mUp.x      , camera.mUp.y      , camera.mUp.z      );
	// T * R * S
	auto front    = RTLib::Vector3(camera.mLookAt.x  , camera.mLookAt.y  , camera.mLookAt.z  );
	auto right    = RTLib::cross(vup, front); // y * z -> x
	auto up       = vup;

	auto frontLen = RTLib::length(front);
	auto rightLen = RTLib::length(right);
	auto    upLen = RTLib::length(up);

	front         = RTLib::normalize(front);
	right         = RTLib::normalize(right);
	up            = RTLib::normalize(up);

	auto aspect   = camera.mAspect;
	auto fovY     = camera.mHorizontalFOV;
	auto zNear    = camera.mClipPlaneNear;
	auto zFar     = camera.mClipPlaneFar;

	auto rotation = RTLib::Core::toQuat(RTLib::Matrix3x3(right, up, front));
	auto scaling  = RTLib::Core::Vector3(1.0f);

	return RTLib::Camera(scaling, rotation, position, fovY, aspect, zNear, zFar);
}
struct Mesh
{

};
struct Scene
{
	RTLib::Camera camera;

};
int main()
{
	using namespace nlohmann;

	auto camera = RTLib::Camera()
		.set_position({ 1.0f,2.0f,3.0f })
		.set_fovy(30.0f)
		.set_aspect(1.0f)
		.set_front({ 0.0f,0.0f,1.0f });

	std::cout << RTLib::Core::Json(camera) << std::endl;

	auto data_path = SAMPLE_SAMPLE0_DATA_PATH"\\Models\\Bistro_v5_2\\BistroExterior.fbx";
	auto data_root = SAMPLE_SAMPLE0_DATA_PATH"\\Models\\Bistro_v5_2";
	auto importer = Assimp::Importer();
	unsigned int flag = 0;
	flag |= aiProcess_Triangulate;
	flag |= aiProcess_PreTransformVertices;
	flag |= aiProcess_CalcTangentSpace;
	flag |= aiProcess_GenSmoothNormals;
	flag |= aiProcess_GenUVCoords;
	flag |= aiProcess_GenBoundingBoxes;
	flag |= aiProcess_RemoveRedundantMaterials;
	flag |= aiProcess_OptimizeMeshes;

	auto scene = importer.ReadFile(data_path, flag);
	if (scene == nullptr)
	{
		std::cout << importer.GetErrorString() << std::endl;
	}
	else
	{
		std::cout << "scene.name="         << std::string(scene->mName.C_Str())     << std::endl;
		std::cout << "scene.numMeshes="    << std::to_string(scene->mNumMeshes)     << std::endl;
		std::cout << "scene.numCameras="   << std::to_string(scene->mNumCameras)    << std::endl;
		std::cout << "scene.numLights="    << std::to_string(scene->mNumLights)     << std::endl;
		std::cout << "scene.numMaterials=" << std::to_string(scene->mNumMaterials)  << std::endl;
		std::cout << "scene.numTextures="  << std::to_string(scene->mNumTextures)   << std::endl;
		std::cout << "scene.numSkeletons=" << std::to_string(scene->mNumSkeletons)  << std::endl;
		std::cout << "scene.numAnimations="<< std::to_string(scene->mNumAnimations)<< std::endl;

		for (size_t i = 0; i < scene->mNumMeshes; ++i)
		{
			const auto pMesh = scene->mMeshes[i];
			std::cout << "meshes[" << i << "].name="          << std::string(pMesh->mName.C_Str())        << std::endl;
			std::cout << "meshes[" << i << "].aabb=["         << pMesh->mAABB.mMin[0] << "," << pMesh->mAABB.mMin[1] \
													          << pMesh->mAABB.mMax[0] << "," << pMesh->mAABB.mMax[1] << "]" << std::endl;

			std::cout << "meshes[" << i << "].numVertices="   << std::to_string(pMesh->mNumVertices)      << std::endl;
			std::cout << "meshes[" << i << "].numFaces="      << std::to_string(pMesh->mNumFaces)         << std::endl;
			std::cout << "meshes[" << i << "].numBones="      << std::to_string(pMesh->mNumBones)         << std::endl;
			std::cout << "meshes[" << i << "].numAnimMeshes=" << std::to_string(pMesh->mNumAnimMeshes)    << std::endl;

			pMesh->mVertices[0];
			if (pMesh->HasNormals())
			{

				pMesh->mNormals[0];
			}
			if (pMesh->HasTangentsAndBitangents())
			{

				pMesh->mTangents[0];
				pMesh->mBitangents[0];
			}
			if (pMesh->HasTextureCoords(0))
			{
				pMesh->mTextureCoords[0][0];
			}
			if (pMesh->HasVertexColors(0))
			{
				pMesh->mColors[0];
			}
			if (pMesh->HasBones())
			{
				
			}
		}
	}
	
	return 0;
}