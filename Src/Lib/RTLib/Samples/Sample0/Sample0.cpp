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
std::string aiuvtransform_to_string(aiUVTransform transform)
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
		for (size_t i = 0; i < scene->mNumMeshes; ++i)
		{
			const auto pMesh = scene->mMeshes[i];

		}
		for (size_t i = 0; i < scene->mNumMaterials; ++i)
		{
			const auto pMaterial = scene->mMaterials[i];

			aiString       matName;
			aiShadingMode  shadingMode;
			aiBlendMode    blendFunc;

			aiColor3D      colorBase;
			aiColor3D      colorAmbi;
			aiColor3D      colorDiff;
			aiColor3D      colorSpec;
			aiColor3D      colorEmit;
			aiColor3D      colorTran;
			aiColor3D      colorRefl;
			RTLib::Float32 transmit;
			RTLib::Float32 shininess;
			RTLib::Float32 shininessStrength;
			RTLib::Float32 roughness;
			RTLib::Float32 reflectivity;
			RTLib::Float32 opacity;
			RTLib::Float32 ior;
			RTLib::Float32 bumpScaling;
			RTLib::Float32 dispScaling;

			aiString       texAmbi;
			RTLib::Float32 texBlendAmbi;
			RTLib::Int32   uvwSrcAmbi;
			aiUVTransform  uvTranAmbi;

			aiString       texDiff;
			RTLib::Float32 texBlendDiff;
			RTLib::Int32   uvwSrcDiff;
			aiUVTransform  uvTranDiff;

			aiString       texSpec;
			RTLib::Float32 texBlendSpec;
			RTLib::Int32   uvwSrcSpec;
			aiUVTransform  uvTranSpec;

			aiString       texEmit;
			RTLib::Float32 texBlendEmit;
			RTLib::Int32   uvwSrcEmit;
			aiUVTransform  uvTranEmit;

			aiString       texShin;
			RTLib::Float32 texBlendShin;
			RTLib::Int32   uvwSrcShin;
			aiUVTransform  uvTranShin;

			aiString       texRough;
			RTLib::Float32 texBlendRough;
			RTLib::Int32   uvwSrcRough;
			aiUVTransform  uvTranRough;

			aiString       texNorm;
			RTLib::Float32 texBlendNorm;
			RTLib::Int32   uvwSrcNorm;
			aiUVTransform  uvTranNorm;

			aiString       texRefl;
			RTLib::Float32 texBlendRefl;
			RTLib::Int32   uvwSrcRefl;
			aiUVTransform  uvTranRefl;

			if (pMaterial->Get(AI_MATKEY_NAME, matName) == aiReturn_SUCCESS) {
				std::cout << "materials[" << i << "].name = " << matName.C_Str() << std::endl;
			}

			if (pMaterial->Get(AI_MATKEY_SHADING_MODEL, shadingMode) == aiReturn_SUCCESS)
			{
				std::cout << "materials[" << i << "].shadingMode = " << aishading_model_to_string(shadingMode) << std::endl;
			}
			if (pMaterial->Get(AI_MATKEY_BLEND_FUNC, blendFunc) == aiReturn_SUCCESS)
			{
				std::cout << "materials[" << i << "].blendMode = " << aiblend_mode_to_string(blendFunc) << std::endl;
			}
			// COLOR
			if (pMaterial->Get(AI_MATKEY_BASE_COLOR, colorBase) == aiReturn_SUCCESS)
			{
				std::cout << "materials[" << i << "].base=[" << colorBase[0] << "," << colorBase[1] << "," << colorBase[2] << "]" << std::endl;
			}
			if (pMaterial->Get(AI_MATKEY_COLOR_AMBIENT, colorAmbi) == aiReturn_SUCCESS)
			{
				std::cout << "materials[" << i << "].ambi=[" << colorAmbi[0] << "," << colorAmbi[1] << "," << colorAmbi[2] << "]" << std::endl;
			}
			if (pMaterial->Get(AI_MATKEY_COLOR_DIFFUSE, colorDiff) == aiReturn_SUCCESS) {
				std::cout << "materials[" << i << "].diff=[" << colorDiff[0] << "," << colorDiff[1] << "," << colorDiff[2] << "]" << std::endl;
			}
			if (pMaterial->Get(AI_MATKEY_COLOR_SPECULAR, colorSpec) == aiReturn_SUCCESS) {
				std::cout << "materials[" << i << "].spec=[" << colorSpec[0] << "," << colorSpec[1] << "," << colorSpec[2] << "]" << std::endl;
			}
			if (pMaterial->Get(AI_MATKEY_COLOR_EMISSIVE, colorEmit) == aiReturn_SUCCESS) {
				std::cout << "materials[" << i << "].emit=[" << colorEmit[0] << "," << colorEmit[1] << "," << colorEmit[2] << "]" << std::endl;
			}
			if (pMaterial->Get(AI_MATKEY_COLOR_TRANSPARENT, colorTran) == aiReturn_SUCCESS) {
				std::cout << "materials[" << i << "].tran=[" << colorTran[0] << "," << colorTran[1] << "," << colorTran[2] << "]" << std::endl;
			}
			if (pMaterial->Get(AI_MATKEY_COLOR_REFLECTIVE, colorRefl) == aiReturn_SUCCESS) {
				std::cout << "materials[" << i << "].refl=[" << colorRefl[0] << "," << colorRefl[1] << "," << colorRefl[2] << "]" << std::endl;
			}
			if (pMaterial->Get(AI_MATKEY_TRANSMISSION_FACTOR, transmit) == aiReturn_SUCCESS) {
				std::cout << "materials[" << i << "].tranFact=" << transmit << std::endl;
			}

			if (pMaterial->Get(AI_MATKEY_SHININESS, shininess) == aiReturn_SUCCESS) {
				std::cout << "materials[" << i << "].shin=" << shininess << std::endl;
			}
			if (pMaterial->Get(AI_MATKEY_SHININESS_STRENGTH, shininessStrength) == aiReturn_SUCCESS) {
				std::cout << "materials[" << i << "].shinStrn=" << shininessStrength << std::endl;
			}
			if (pMaterial->Get(AI_MATKEY_ROUGHNESS_FACTOR, roughness) == aiReturn_SUCCESS) {
				std::cout << "materials[" << i << "].roug=" << roughness << std::endl;
			}
			if (pMaterial->Get(AI_MATKEY_OPACITY, opacity) == aiReturn_SUCCESS) {
				std::cout << "materials[" << i << "].opac=" << opacity << std::endl;
			}
			if (pMaterial->Get(AI_MATKEY_REFRACTI, ior) == aiReturn_SUCCESS) {
				std::cout << "materials[" << i << "].ior =" << ior << std::endl;
			}
			if (pMaterial->Get(AI_MATKEY_REFLECTIVITY, reflectivity) == aiReturn_SUCCESS) {
				std::cout << "materials[" << i << "].reflectivity =" << reflectivity << std::endl;
			}
			if (pMaterial->Get(AI_MATKEY_BUMPSCALING, bumpScaling) == aiReturn_SUCCESS) {
				std::cout << "materials[" << i << "].bumpScaling =" << bumpScaling << std::endl;
			}
			// TEX
			if (pMaterial->Get(AI_MATKEY_TEXTURE_AMBIENT(0), texAmbi) == aiReturn_SUCCESS) {
				std::cout << "materials[" << i << "].tex_ambi[0]=" << texAmbi.C_Str() << std::endl;
			}
			if (pMaterial->Get(AI_MATKEY_TEXTURE_DIFFUSE(0), texDiff) == aiReturn_SUCCESS) {
				std::cout << "materials[" << i << "].tex_diff[0]=" << texDiff.C_Str() << std::endl;
			}
			if (pMaterial->Get(AI_MATKEY_TEXTURE_SPECULAR(0), texSpec) == aiReturn_SUCCESS) {
				std::cout << "materials[" << i << "].tex_spec[0]=" << texSpec.C_Str() << std::endl;
			}
			if (pMaterial->Get(AI_MATKEY_TEXTURE_EMISSIVE(0), texEmit) == aiReturn_SUCCESS) {
				std::cout << "materials[" << i << "].tex_emit[0]=" << texEmit.C_Str() << std::endl;
			}
			if (pMaterial->Get(AI_MATKEY_TEXTURE_SHININESS(0), texShin) == aiReturn_SUCCESS) {
				std::cout << "materials[" << i << "].tex_shin[0]=" << texShin.C_Str() << std::endl;
			}
			if (pMaterial->Get(AI_MATKEY_TRANSMISSION_FACTOR, transmit) == aiReturn_SUCCESS) {
				std::cout << "materials[" << i << "].tex_shin[0]=" << texShin.C_Str() << std::endl;
			}
			if (pMaterial->Get(AI_MATKEY_TEXTURE_NORMALS(0), texNorm) == aiReturn_SUCCESS) {
				std::cout << "materials[" << i << "].tex_norm[0]=" << texNorm.C_Str() << std::endl;
			}
			// TEX_BLEND
			if (pMaterial->Get(AI_MATKEY_TEXBLEND_AMBIENT(0), texBlendAmbi) == aiReturn_SUCCESS) {
				std::cout << "materials[" << i << "].tex_blend_ambi[0]=" << texBlendAmbi << std::endl;
			}
			if (pMaterial->Get(AI_MATKEY_TEXBLEND_DIFFUSE(0), texBlendDiff) == aiReturn_SUCCESS) {
				std::cout << "materials[" << i << "].tex_blend_diff[0]=" << texBlendDiff << std::endl;
			}
			if (pMaterial->Get(AI_MATKEY_TEXBLEND_SPECULAR(0), texBlendSpec) == aiReturn_SUCCESS) {
				std::cout << "materials[" << i << "].tex_blend_spec[0]=" << texBlendSpec << std::endl;
			}
			if (pMaterial->Get(AI_MATKEY_TEXBLEND_EMISSIVE(0), texBlendEmit) == aiReturn_SUCCESS) {
				std::cout << "materials[" << i << "].tex_blend_emit[0]=" << texBlendEmit << std::endl;
			}
			if (pMaterial->Get(AI_MATKEY_TEXBLEND_SHININESS(0), texBlendShin) == aiReturn_SUCCESS) {
				std::cout << "materials[" << i << "].tex_blend_shin[0]=" << texBlendShin << std::endl;
			}
			if (pMaterial->Get(AI_MATKEY_TEXBLEND_NORMALS(0), texBlendNorm) == aiReturn_SUCCESS) {
				std::cout << "materials[" << i << "].tex_blend_norm[0]=" << texBlendNorm << std::endl;
			}
			if (pMaterial->Get(AI_MATKEY_TEXBLEND_REFLECTION(0), texBlendRefl) == aiReturn_SUCCESS) {
				std::cout << "materials[" << i << "].tex_blend_refl[0]=" << texBlendRefl << std::endl;
			}
			// UVW Src
			if (pMaterial->Get(AI_MATKEY_UVWSRC_AMBIENT(0), uvwSrcAmbi) == aiReturn_SUCCESS) {
				std::cout << "materials[" << i << "].uvw_src_ambi[0]=" << uvwSrcAmbi << std::endl;
			}
			if (pMaterial->Get(AI_MATKEY_UVWSRC_DIFFUSE(0), uvwSrcDiff) == aiReturn_SUCCESS) {
				std::cout << "materials[" << i << "].uvw_src_diff[0]=" << uvwSrcDiff << std::endl;
			}
			if (pMaterial->Get(AI_MATKEY_UVWSRC_SPECULAR(0), uvwSrcSpec) == aiReturn_SUCCESS) {
				std::cout << "materials[" << i << "].uvw_src_spec[0]=" << uvwSrcSpec << std::endl;
			}
			if (pMaterial->Get(AI_MATKEY_UVWSRC_EMISSIVE(0), uvwSrcEmit) == aiReturn_SUCCESS) {
				std::cout << "materials[" << i << "].uvw_src_emit[0]=" << uvwSrcEmit << std::endl;
			}
			if (pMaterial->Get(AI_MATKEY_UVWSRC_SHININESS(0), uvwSrcShin) == aiReturn_SUCCESS) {
				std::cout << "materials[" << i << "].uvw_src_shin[0]=" << uvwSrcShin << std::endl;
			}
			if (pMaterial->Get(AI_MATKEY_UVWSRC_NORMALS(0), uvwSrcNorm) == aiReturn_SUCCESS) {
				std::cout << "materials[" << i << "].uvw_src_norm[0]=" << uvwSrcNorm << std::endl;
			}
			if (pMaterial->Get(AI_MATKEY_UVWSRC_REFLECTION(0), uvwSrcRefl) == aiReturn_SUCCESS) {
				std::cout << "materials[" << i << "].uvw_src_refl[0]=" << uvwSrcRefl << std::endl;
			}
			// UVTransform
			if (pMaterial->Get(AI_MATKEY_UVTRANSFORM_AMBIENT(0), uvTranAmbi) == aiReturn_SUCCESS) {
				std::cout << "materials[" << i << "].uv_transform_ambi[0]=" << aiuvtransform_to_string(uvTranAmbi) << std::endl;
			}
			if (pMaterial->Get(AI_MATKEY_UVTRANSFORM_DIFFUSE(0), uvTranDiff) == aiReturn_SUCCESS) {
				std::cout << "materials[" << i << "].uv_transform_diff[0]=" << aiuvtransform_to_string(uvTranDiff) << std::endl;
			}
			if (pMaterial->Get(AI_MATKEY_UVTRANSFORM_SPECULAR(0), uvTranSpec) == aiReturn_SUCCESS) {
				std::cout << "materials[" << i << "].uv_transform_spec[0]=" << aiuvtransform_to_string(uvTranSpec) << std::endl;
			}
			if (pMaterial->Get(AI_MATKEY_UVTRANSFORM_EMISSIVE(0), uvTranEmit) == aiReturn_SUCCESS) {
				std::cout << "materials[" << i << "].uv_transform_emit[0]=" << aiuvtransform_to_string(uvTranEmit) << std::endl;
			}
			if (pMaterial->Get(AI_MATKEY_UVTRANSFORM_SHININESS(0), uvTranShin) == aiReturn_SUCCESS) {
				std::cout << "materials[" << i << "].uv_transform_shin[0]=" << aiuvtransform_to_string(uvTranShin) << std::endl;
			}
			if (pMaterial->Get(AI_MATKEY_UVTRANSFORM_NORMALS(0), uvTranNorm) == aiReturn_SUCCESS) {
				std::cout << "materials[" << i << "].uv_transform_norm[0]=" << aiuvtransform_to_string(uvTranNorm) << std::endl;
			}
			if (pMaterial->Get(AI_MATKEY_UVTRANSFORM_REFLECTION(0), uvTranRefl) == aiReturn_SUCCESS) {
				std::cout << "materials[" << i << "].uv_transform_refl[0]=" << aiuvtransform_to_string(uvTranRefl) << std::endl;
			}
			////for (auto i = 0; i < pMaterial->mNumProperties; ++i)
			////{
			////	std::cout << pMaterial->mProperties[i]->mKey.C_Str() << std::endl;
			////}
			std::cout << std::endl;

		}
	}

	{
		aiString texDiffPath;
		if (scene->mMaterials[0]->Get(AI_MATKEY_TEXTURE_DIFFUSE(0), texDiffPath) != aiReturn_SUCCESS) {
			return -1;
		}

		auto prev_loc = std::setlocale(LC_ALL, nullptr);
		if (!prev_loc) {
			std::cout << "Failed To Load " << std::endl;
			return -1;
		}
		auto cTexDiffPath = std::filesystem::path(std::string(data_path) + "\\" + std::string(texDiffPath.C_Str())).make_preferred();
		auto wTexDiffPath = cTexDiffPath.wstring();

		std::wcout << wTexDiffPath << std::endl;

		DirectX::TexMetadata metadata;
		DirectX::ScratchImage scratchImg;

		if (FAILED(DirectX::LoadFromDDSFile(cTexDiffPath.c_str(), DirectX::DDS_FLAGS_NONE, &metadata, scratchImg)))
		{
			return -1;
		}
	}
	
	return 0;
}