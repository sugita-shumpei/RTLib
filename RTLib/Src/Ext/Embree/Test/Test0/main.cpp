#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <RTLib/Core/Scene.h>
#include <RTLib/Core/AABB.h>
#include <embree3/rtcore.h>
#include <RTLibExtEmbreeTest0Config.h>
#include <memory>
#include <filesystem>
#include <stack>
#include <fstream>
#include <iostream>
#include <random>
#include <thread>
#include <stb_image_write.h>
template<typename T>
using InternalEmbreeUniquePointer = std::unique_ptr<std::remove_pointer_t<T>, void(*)(T)>;
template<typename T, void(*Deleter)(T)>
struct InternalEmbreeMakeHandleImpl {
    static void Del(T ptr) {
        if (!ptr) {
            return;
        }
        Deleter(ptr);
    }
	static auto Eval(T ptr) ->InternalEmbreeUniquePointer<T> {
		return InternalEmbreeUniquePointer<T>(ptr, Del);
	}
};
auto  Internal_MakeEmbreeDevice(const char* args)->InternalEmbreeUniquePointer<RTCDevice> {
	return InternalEmbreeMakeHandleImpl<RTCDevice, rtcReleaseDevice>::Eval(rtcNewDevice(args));
}
auto  Internal_MakeEmbreeScene(RTCDevice device)->InternalEmbreeUniquePointer<RTCScene> {
    if (!device) {
        return InternalEmbreeMakeHandleImpl<RTCScene, rtcReleaseScene>::Eval(nullptr);
    }
	return InternalEmbreeMakeHandleImpl<RTCScene, rtcReleaseScene>::Eval(rtcNewScene(device));
}
auto  Internal_MakeEmbreeGeometry(RTCDevice device, RTCGeometryType geoType)->InternalEmbreeUniquePointer<RTCGeometry> {
	return InternalEmbreeMakeHandleImpl<RTCGeometry, rtcReleaseGeometry>::Eval(rtcNewGeometry(device, geoType));
}
auto  Internal_MakeEmbreeBuffer(RTCDevice device, size_t sizeInBytes = 0) {
    if (sizeInBytes == 0){ 
        return InternalEmbreeMakeHandleImpl<RTCBuffer, rtcReleaseBuffer>::Eval(nullptr);
    }
	return InternalEmbreeMakeHandleImpl<RTCBuffer, rtcReleaseBuffer>::Eval(rtcNewBuffer(device, sizeInBytes));
}

namespace RTLib
{
	namespace Ext {
		namespace Embree {
            struct Embree3MeshSharedResourceExtData :public RTLib::Core::MeshSharedResourceExtData {

                Embree3MeshSharedResourceExtData(Core::MeshSharedResource* pMeshSharedResource, RTCDevice device)noexcept
                    :RTLib::Core::MeshSharedResourceExtData(pMeshSharedResource),
                    vertexBuffer{ Internal_MakeEmbreeBuffer(device, pMeshSharedResource->vertexBuffer.size() * sizeof(float) * 3) },
                    normalBuffer{ Internal_MakeEmbreeBuffer(device, pMeshSharedResource->normalBuffer.size() * sizeof(float) * 3) },
                    texCrdBuffer{ Internal_MakeEmbreeBuffer(device, pMeshSharedResource->texCrdBuffer.size() * sizeof(float) * 2) }{
                    if (!pMeshSharedResource->vertexBuffer.empty()) {
                        std::memcpy(rtcGetBufferData(vertexBuffer.get()), pMeshSharedResource->vertexBuffer.data(), pMeshSharedResource->vertexBuffer.size() * sizeof(float) * 3);
                    }
                    if (!pMeshSharedResource->normalBuffer.empty()) {
                        std::memcpy(rtcGetBufferData(normalBuffer.get()), pMeshSharedResource->normalBuffer.data(), pMeshSharedResource->normalBuffer.size() * sizeof(float) * 3);
                    }
                    if (!pMeshSharedResource->texCrdBuffer.empty()) {
                        std::memcpy(rtcGetBufferData(texCrdBuffer.get()), pMeshSharedResource->texCrdBuffer.data(), pMeshSharedResource->texCrdBuffer.size() * sizeof(float) * 2);
                    }
                }
            public:
                static auto New(Core::MeshSharedResource* pMeshSharedResource, RTCDevice device) noexcept -> Embree3MeshSharedResourceExtData* {
                    return new Embree3MeshSharedResourceExtData(pMeshSharedResource, device);
                }
                virtual ~Embree3MeshSharedResourceExtData()noexcept {
                    vertexBuffer.reset();
                    normalBuffer.reset();
                    texCrdBuffer.reset();
                }
                InternalEmbreeUniquePointer<RTCBuffer> vertexBuffer;
                InternalEmbreeUniquePointer<RTCBuffer> normalBuffer;
                InternalEmbreeUniquePointer<RTCBuffer> texCrdBuffer;
            };
            struct Embree3MeshUniqueResourceExtData :public RTLib::Core::MeshUniqueResourceExtData {
            public:
                static auto New(Core::MeshUniqueResource* pMeshUniqueResource, RTCDevice device) noexcept -> Embree3MeshUniqueResourceExtData* {
                    return new Embree3MeshUniqueResourceExtData(pMeshUniqueResource, device);
                }
                Embree3MeshUniqueResourceExtData(Core::MeshUniqueResource* pMeshUniqueResource, RTCDevice device)noexcept
                    :RTLib::Core::MeshUniqueResourceExtData(pMeshUniqueResource),
                    geometry{Internal_MakeEmbreeGeometry(device,RTC_GEOMETRY_TYPE_TRIANGLE)},
                    triIdxBuffer{ Internal_MakeEmbreeBuffer(device, pMeshUniqueResource->triIndBuffer.size() * sizeof(uint32_t) * 3) }{
                    if (!pMeshUniqueResource->triIndBuffer.empty()) {
                        std::memcpy(rtcGetBufferData(triIdxBuffer.get()), pMeshUniqueResource->triIndBuffer.data(), pMeshUniqueResource->triIndBuffer.size() * sizeof(uint32_t) * 3);
                    }
                    rtcSetGeometryBuffer(geometry.get(), RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3, triIdxBuffer.get(), 0, sizeof(uint32_t) * 3, pMeshUniqueResource->triIndBuffer.size());
                }
                virtual ~Embree3MeshUniqueResourceExtData()noexcept {
                    triIdxBuffer.reset();
                    geometry.reset();
                }
                InternalEmbreeUniquePointer<RTCGeometry> geometry;
                InternalEmbreeUniquePointer<RTCBuffer>   triIdxBuffer;
            };

            class Embree3SphereResourceExtData :public Core::SphereResourceExtData {
            public:
                static auto New(Core::SphereResource* pSphereResource, RTCDevice device)noexcept->Embree3SphereResourceExtData* {
                    auto extData = new Embree3SphereResourceExtData(pSphereResource,device);
                    return extData;
                }
                Embree3SphereResourceExtData(Core::SphereResource* pSphereResource, RTCDevice device)noexcept:
                    Core::SphereResourceExtData(pSphereResource), geometry{Internal_MakeEmbreeGeometry(device,RTC_GEOMETRY_TYPE_SPHERE_POINT)},
                    vertexBuffer{ Internal_MakeEmbreeBuffer(device,pSphereResource->centerBuffer.size()*sizeof(float)*4)}{
                    std::vector<float> vertices(pSphereResource->centerBuffer.size() * 4);
                    if (pSphereResource->centerBuffer.size() == 1) {
                        for (size_t i = 0; i < pSphereResource->centerBuffer.size(); ++i) {
                            vertices[4 * i + 0] = pSphereResource->centerBuffer[i][0];
                            vertices[4 * i + 1] = pSphereResource->centerBuffer[i][1];
                            vertices[4 * i + 2] = pSphereResource->centerBuffer[i][2];
                            vertices[4 * i + 2] = pSphereResource->radiusBuffer[0];
                        }
                    }
                    else {
                        for (size_t i = 0; i < pSphereResource->centerBuffer.size(); ++i) {
                            vertices[4 * i + 0] = pSphereResource->centerBuffer[i][0];
                            vertices[4 * i + 1] = pSphereResource->centerBuffer[i][1];
                            vertices[4 * i + 2] = pSphereResource->centerBuffer[i][2];
                            vertices[4 * i + 2] = pSphereResource->radiusBuffer[i];
                        }
                    }
                    std::memcpy(rtcGetBufferData(vertexBuffer.get()), vertices.data(), sizeof(float)* vertices.size());
                    rtcSetGeometryBuffer(geometry.get(), RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT4, vertexBuffer.get(), 0, sizeof(float) * 4, pSphereResource->centerBuffer.size());
                    rtcSetGeometryTimeStepCount(geometry.get(), 1);
                    rtcCommitGeometry(geometry.get());
                }
                virtual ~Embree3SphereResourceExtData()noexcept {}

                InternalEmbreeUniquePointer<RTCGeometry> geometry;
                InternalEmbreeUniquePointer<RTCBuffer>   vertexBuffer;
            };

            inline void InitMeshGroupExtData(RTCDevice device, RTLib::Core::MeshGroupPtr meshGroup)
            {
                auto sharedResource = meshGroup->GetSharedResource();
                sharedResource->AddExtData<Embree3MeshSharedResourceExtData>(device);
                auto shdExtData = static_cast<Embree3MeshSharedResourceExtData*>(sharedResource->extData.get());
                for (auto& [name, uniqueResource] : meshGroup->GetUniqueResources())
                {
                    uniqueResource->AddExtData<Embree3MeshUniqueResourceExtData>(device);
                    auto extData = static_cast<Embree3MeshUniqueResourceExtData*>(uniqueResource->extData.get());
                    rtcSetGeometryBuffer(extData->geometry.get(), RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3, shdExtData->vertexBuffer.get(), 0, sizeof(float) * 3, sharedResource->vertexBuffer.size());
                    if (shdExtData->normalBuffer) {
                        rtcSetGeometryBuffer(extData->geometry.get(), RTC_BUFFER_TYPE_NORMAL, 0, RTC_FORMAT_FLOAT3, shdExtData->normalBuffer.get(), 0, sizeof(float) * 3, sharedResource->normalBuffer.size());
                    }
                    if (shdExtData->texCrdBuffer) {
                        rtcSetGeometryBuffer(extData->geometry.get(), RTC_BUFFER_TYPE_VERTEX_ATTRIBUTE, 0, RTC_FORMAT_FLOAT2, shdExtData->texCrdBuffer.get(), 0, sizeof(float) * 2, sharedResource->texCrdBuffer.size());
                    }
                    rtcSetGeometryTimeStepCount(extData->geometry.get(), 1);
                    rtcCommitGeometry(extData->geometry.get());
                }
            }
            struct GeometryAccelerationStructureData;
            struct InstanceData {
                const GeometryAccelerationStructureData* gas = nullptr;
                std::array<float, 12> transforms = {};
                InternalEmbreeUniquePointer<RTCGeometry> handle = InternalEmbreeMakeHandleImpl<RTCGeometry,rtcReleaseGeometry>::Eval(nullptr);
            };

            struct GeometryAccelerationStructureData
            {
                enum GeometryType {
                    GeometryTypeTriangleArray = 1,
                    GeometryTypeSphereArray   = 2,
                };
                struct GeometryInfo {
                    GeometryType   type;
                    std::string    name;
                    std::string subname;
                };
                auto GenerateInstance(RTCDevice device, const std::array<float, 12>& transform)const->InstanceData;
                InternalEmbreeUniquePointer<RTCScene> handle = Internal_MakeEmbreeScene(nullptr);
                std::vector<GeometryInfo> geometries = {};
                std::array<float, 3> aabbMin = {+FLT_MAX,+FLT_MAX,+FLT_MAX};
                std::array<float, 3> aabbMax = {-FLT_MAX,-FLT_MAX,-FLT_MAX};
            };
            inline auto GeometryAccelerationStructureData::GenerateInstance(RTCDevice device, const std::array<float, 12>& transform)const  ->InstanceData {
                InstanceData instance = {};
                instance.gas = this;
                instance.transforms = transform;
                instance.handle = Internal_MakeEmbreeGeometry(device, RTC_GEOMETRY_TYPE_INSTANCE);
                rtcSetGeometryInstancedScene(instance.handle.get(), handle.get());
                rtcSetGeometryTransform(instance.handle.get(), 1, RTC_FORMAT_FLOAT12, transform.data());
            }
            struct InstanceAccelerationStructureData
            {
                InternalEmbreeUniquePointer<RTCScene> handle;
            };
            struct SceneData : public RTLib::Core::SceneData {
                void InitExtData(RTCDevice device) {
                    for (auto& [name, geometry] : world.geometryObjModels)
                    {
                        InitMeshGroupExtData(device, objAssetManager.GetAsset(geometry.base).meshGroup);
                    }
                    for (auto& [name, sphere] : sphereResources)
                    {
                        sphere->AddExtData<Embree3SphereResourceExtData>(device);
                    }
                }

                auto BuildGeometryASs(RTCDevice device)const noexcept -> std::unordered_map<std::string, GeometryAccelerationStructureData>
                {
                    std::unordered_map<std::string, GeometryAccelerationStructureData> geometryASs = {};
                    for (auto& [geometryASName, geometryASData] : world.geometryASs)
                    {
                        RTLib::Core::AABB aabb;
                        auto gasHandle = Internal_MakeEmbreeScene(device);
                        std::vector<GeometryAccelerationStructureData::GeometryInfo> geometries = {};
                        for (auto& geometryName : geometryASData.geometries)
                        {
                            if (world.geometryObjModels.count(geometryName) > 0) {
                                auto& geometryObjModel = world.geometryObjModels.at(geometryName);
                                auto& objAsset = objAssetManager.GetAsset(geometryObjModel.base);
                                for (auto& [meshName, meshData] : geometryObjModel.meshes)
                                {
                                    {
                                        auto mesh = objAsset.meshGroup->LoadMesh(meshData.base);
                                        aabb.Update(mesh->GetUniqueResource()->variables.GetFloat3("aabb.min"));
                                        aabb.Update(mesh->GetUniqueResource()->variables.GetFloat3("aabb.max"));
                                        auto extSharedData = static_cast<Embree3MeshSharedResourceExtData*>(mesh->GetSharedResource()->extData.get());
                                        auto extUniqueData = static_cast<Embree3MeshUniqueResourceExtData*>(mesh->GetUniqueResource()->extData.get());
                                        rtcAttachGeometry(gasHandle.get(), extUniqueData->geometry.get());
                                        geometries.push_back({ GeometryAccelerationStructureData::GeometryTypeTriangleArray,geometryName,meshName });
                                    }
                                }
                            }
                            if (world.geometrySpheres.count(geometryName) > 0) {
                                {
                                    auto& sphereResource = sphereResources.at(geometryName);
                                    auto sphereResourceExtData = sphereResource->GetExtData<Embree3SphereResourceExtData>();
                                    aabb.Update(sphereResource->variables.GetFloat3("aabb.min"));
                                    aabb.Update(sphereResource->variables.GetFloat3("aabb.max"));
                                    rtcAttachGeometry(gasHandle.get(), sphereResourceExtData->geometry.get());
                                    geometries.push_back({ GeometryAccelerationStructureData::GeometryTypeSphereArray,geometryName,"" });
                                }
                            }
                        }
                        rtcCommitScene(gasHandle.get());
                        geometryASs[geometryASName].handle = std::move(gasHandle);
                        geometryASs[geometryASName].aabbMin = aabb.min;
                        geometryASs[geometryASName].aabbMax = aabb.max;
                        geometryASs[geometryASName].geometries = std::move(geometries);
                    }

                    return geometryASs;
                }
            };
            inline auto GetDefaultSceneJson()->nlohmann::json {
                auto cameraController = RTLib::Core::CameraController({ 0.0f, 1.0f, 5.0f });
                cameraController.SetMovementSpeed(10.0f);
                return {
                    {"ObjModels",
                        {
                            {"CacheDir", std::filesystem::absolute(".").string()},
                            {"Assets",
                                {{"CornellBox-Original", RTLIB_EXT_EMBREE_TEST_TEST0_DATA_PATH "/Models/CornellBox/CornellBox-Original.obj"}}
                            }
                        }
                    },
                    {"World",
                     {
                         {"Geometries",
                            {{"CornellBox-Original", {{"Type", "ObjModel"},{"Base", "CornellBox-Original"}}}}
                         },
                         {"GeometryASs",
                            {{"CornellBox-Original", {{"Type", "GeometryAS"}, {"Geometries", std::vector<std::string>{"CornellBox-Original"}}}}}
                          },
                         {"Instances",
                            {{"Instance0",{{"Type", "Instance"}, {"ASType", "Geometry"},{"Base", "CornellBox-Original"},{"Transform",std::vector<float>{
                                1.0f,0.0f,0.0f,0.0f,
                                0.0f,1.0f,0.0f,0.0f,
                                0.0f,0.0f,1.0f,0.0f
                            }}}}}
                         },
                         {"InstanceASs",
                          {{"Root",{{"Type", "InstanceAS"},{"Instances", std::vector<std::string>{"Instance0"}}}}}
                         },
                     }},
                    {"CameraController", cameraController},
                    {"Config",{
                        {"Width" ,1024},
                        {"Height",1024},
                        {"Samples",1},
                        {"MaxSamples",10000},
                        {"ImagePath", RTLIB_EXT_EMBREE_TEST_TEST0_DATA_PATH"/../Result"},
                        {"SamplesPerSave",1000},
                        {"MaxDepth",4}
                    }}
                };
            }
            inline auto LoadScene(std::string path)->SceneData {
                nlohmann::json sceneJson;
                std::ifstream sceneJsonFile(path, std::ios::binary);
                if (sceneJsonFile.is_open())
                {
                    sceneJson = nlohmann::json::parse(sceneJsonFile);
                    sceneJsonFile.close();
                }
                else
                {
                    sceneJson = RTLib::Ext::Embree::GetDefaultSceneJson();
                }

                return sceneJson.get<RTLib::Ext::Embree::SceneData>();
            }
            inline void SaveScene(std::string path, const SceneData& sceneData) {
                nlohmann::json sceneJson(sceneData);
                auto sceneJsonFile = std::ofstream(path, std::ios::binary);
                sceneJsonFile << sceneJson;
                sceneJsonFile.close();
            }
		}
	}
}

int RenderTrace()
{
    auto device = Internal_MakeEmbreeDevice(nullptr);
    auto scene  = RTLib::Ext::Embree::LoadScene(RTLIB_EXT_EMBREE_TEST_TEST0_JSON_PATH"/scene4.json");
    scene.InitExtData(device.get());
    auto gasHandleMap = scene.BuildGeometryASs(device.get());
    auto& root = gasHandleMap.at("CornellBox-Water");
    std::random_device rd;
    std::vector<unsigned char> images(scene.config.width * scene.config.height * 4, 255);
    std::vector<std::thread> ths;
    size_t numThreads = std::thread::hardware_concurrency();
    for (int t = 0; t < numThreads; ++t) {

        ths.emplace_back([&scene, &root, &images,t, numThreads](size_t seed) {
            auto camera = scene.cameraController.GetCamera((float)scene.config.width / (float)scene.config.height);
            auto [u, v, w] = camera.GetUVW();
            RTCRayHit rayHit;
            std::mt19937_64 mt(seed);
            for (int j = t; j < scene.config.height; j+=numThreads) {
                for (int i = 0; i < scene.config.width; ++i) {

                    auto radiance = std::array<float, 3>{0.0f, 0.0f, 0.0f};
                    for (int s = 0; s < scene.config.maxSamples; ++s) {
                        rayHit.ray.org_x = camera.GetEye()[0];
                        rayHit.ray.org_y = camera.GetEye()[1];
                        rayHit.ray.org_z = camera.GetEye()[2];

                        const auto d = std::array<float, 2>{
                            2.0f * static_cast<float>(i + std::uniform_real_distribution<float>(0.0f, 1.0f)(mt)) / static_cast<float>(scene.config.width) - 1.0f,
                                2.0f * static_cast<float>(j + std::uniform_real_distribution<float>(0.0f, 1.0f)(mt)) / static_cast<float>(scene.config.height) - 1.0f};
                        auto dir = RTLib::Core::Normalize(
                            RTLib::Core::Add(
                                RTLib::Core::Sub(
                                    RTLib::Core::Mul({ -d[0],-d[0],-d[0] }, u),
                                    RTLib::Core::Mul({ d[1], d[1], d[1] }, v)
                                ),
                                w)
                        );


                        rayHit.ray.dir_x = dir[0];
                        rayHit.ray.dir_y = dir[1];
                        rayHit.ray.dir_z = dir[2];
                        rayHit.ray.tnear = 1e-10f;
                        rayHit.ray.tfar = FLT_MAX;
                        rayHit.hit.geomID = RTC_INVALID_GEOMETRY_ID;

                        auto color = std::array<float, 3>{0.0f, 0.0f, 0.0f};
                        auto atten = std::array<float, 3>{1.0f, 1.0f, 1.0f};
                        for (int d = 0; d < scene.config.maxDepth+1; ++d) {
                            
                            RTCIntersectContext ctx;
                            rtcInitIntersectContext(&ctx);
                            rtcIntersect1(root.handle.get(), &ctx, &rayHit);
                            if (rayHit.hit.geomID != RTC_INVALID_GEOMETRY_ID) {
                                auto& geoInfo = root.geometries.at(rayHit.hit.geomID);
                                if (geoInfo.type == RTLib::Ext::Embree::GeometryAccelerationStructureData::GeometryTypeTriangleArray) {
                                    const auto& geometry = scene.world.geometryObjModels.at(geoInfo.name);
                                    const auto& objModel = scene.objAssetManager.GetAsset(geometry.base);
                                    const auto& sharedRes = objModel.meshGroup->GetSharedResource();
                                    const auto& uniqueRes = objModel.meshGroup->GetUniqueResource(geoInfo.subname);
                                    const auto& matIndex = uniqueRes->materials[uniqueRes->matIndBuffer[rayHit.hit.primID]];
                                    
                                    auto  vNormal = RTLib::Core::Normalize(std::array<float, 3>{rayHit.hit.Ng_x, rayHit.hit.Ng_y, rayHit.hit.Ng_z});/*RTLib::Core::Normalize(RTLib::Core::Cross(RTLib::Core::Sub(vertex1, vertex0), RTLib::Core::Sub(vertex2, vertex0)));*/
                                    
                                    const auto  cosine   = vNormal[0] * rayHit.ray.dir_x + vNormal[1] * rayHit.ray.dir_y + vNormal[2] * rayHit.ray.dir_z;
                                    const auto& diffCol  = objModel.materials[matIndex].GetFloat3Or("diffCol", { 0.0f,0.0f,0.0f });
                                    const auto& specCol  = objModel.materials[matIndex].GetFloat3Or("specCol", { 0.0f,0.0f,0.0f });
                                    const auto& emitCol  = objModel.materials[matIndex].GetFloat3Or("emitCol", { 0.0f,0.0f,0.0f });
                                    const auto  illum    = objModel.materials[matIndex].GetUInt32Or("illum", 0);
                                    const auto  shinness = objModel.materials[matIndex].GetFloat1Or("shinness", 1.0f);
                                    const auto  refrIndx = objModel.materials[matIndex].GetFloat1Or("refrIndx", 1.0f);
                                    
                                    color[0] += atten[0] * emitCol[0];
                                    color[1] += atten[1] * emitCol[1];
                                    color[2] += atten[2] * emitCol[2];

                                    if (emitCol[0] + emitCol[1] + emitCol[2] > 0.0f) {
                                        break;
                                    }

                                    if (illum != 7) {
                                        //RANDOM_IN_UNIT_SPHERE
                                        //atten[0] *= (diffCol[0] * std::max(-cosine, 0.0f) * (4.0f * (float)RTLIB_CORE_MATH_CONSTANTS_PI) / (float)RTLIB_CORE_MATH_CONSTANTS_PI);
                                        //atten[1] *= (diffCol[1] * std::max(-cosine, 0.0f) * (4.0f * (float)RTLIB_CORE_MATH_CONSTANTS_PI) / (float)RTLIB_CORE_MATH_CONSTANTS_PI);
                                        //atten[2] *= (diffCol[2] * std::max(-cosine, 0.0f) * (4.0f * (float)RTLIB_CORE_MATH_CONSTANTS_PI) / (float)RTLIB_CORE_MATH_CONSTANTS_PI);
                                        //
                                        //auto rand0 = std::uniform_real_distribution<float>(0.0f, 2.0f)(mt);
                                        //auto rand1 = std::uniform_real_distribution<float>(0.0f, 2.0f)(mt);
                                        //const float r = rand0 - 1.0f;
                                        //const float z = sqrtf(1.0f - r * r);
                                        //const float t = rand1 * RTLIB_CORE_MATH_CONSTANTS_PI;

                                        atten[0] *= (diffCol[0] * static_cast<float>( cosine< 0.0f));
                                        atten[1] *= (diffCol[1] * static_cast<float>( cosine< 0.0f));
                                        atten[2] *= (diffCol[2] * static_cast<float>( cosine< 0.0f));

                                        auto rand0 = std::uniform_real_distribution<float>(0.0f, 1.0f)(mt);
                                        auto rand1 = std::uniform_real_distribution<float>(0.0f, 1.0f)(mt);

                                        const float r = sqrtf(rand0);
                                        const float z = sqrtf(1.0f - rand0);
                                        const float t = 2.0f * rand1 * (float)RTLIB_CORE_MATH_CONSTANTS_PI;
                                        //{r* ::cosf(t), r* ::sinf(t), z};
                                        auto norm = vNormal;
                                        auto binm = std::array<float, 3>{};
                                        if (fabsf(vNormal[0]) > fabsf(vNormal[2])) {
                                            binm[0] = -norm[1];
                                            binm[1] =  norm[0];
                                            binm[2] =     0.0f;
                                        }
                                        else {
                                            binm[0] =0.0f;
                                            binm[1] =-norm[2];
                                            binm[2] = norm[1];
                                        }
                                        binm = RTLib::Core::Normalize(binm);
                                        auto tagt = RTLib::Core::Normalize(RTLib::Core::Cross(norm, binm));

                                        rayHit.ray.org_x += (rayHit.ray.dir_x * rayHit.ray.tfar + 0.01f * vNormal[0]);
                                        rayHit.ray.org_y += (rayHit.ray.dir_y * rayHit.ray.tfar + 0.01f * vNormal[1]);
                                        rayHit.ray.org_z += (rayHit.ray.dir_z * rayHit.ray.tfar + 0.01f * vNormal[2]);

                                        auto dir = RTLib::Core::Normalize(std::array<float, 3>{

                                            z* ::cosf(t)* tagt[0] + z * ::sinf(t) * binm[0] + r * norm[0],
                                            z* ::cosf(t)* tagt[1] + z * ::sinf(t) * binm[1] + r * norm[1],
                                            z* ::cosf(t)* tagt[2] + z * ::sinf(t) * binm[2] + r * norm[2]
                                        });
                                        rayHit.ray.dir_x = dir[0];
                                        rayHit.ray.dir_y = dir[1];
                                        rayHit.ray.dir_z = dir[2];
                                    }
                                    else {
                                        if (!sharedRes->normalBuffer.empty()) {
                                            auto fNormal0 = sharedRes->normalBuffer[uniqueRes->triIndBuffer[rayHit.hit.primID][0]];
                                            auto fNormal1 = sharedRes->normalBuffer[uniqueRes->triIndBuffer[rayHit.hit.primID][1]];
                                            auto fNormal2 = sharedRes->normalBuffer[uniqueRes->triIndBuffer[rayHit.hit.primID][2]];
                                            
                                            auto dfNormal10 = RTLib::Core::Sub(fNormal1, fNormal0);
                                            auto dfNormal20 = RTLib::Core::Sub(fNormal2, fNormal0);

                                            vNormal = RTLib::Core::Normalize(
                                                std::array<float, 3>{
                                                    fNormal0[0] + rayHit.hit.u * dfNormal10[0] + rayHit.hit.v * dfNormal20[0],
                                                    fNormal0[1] + rayHit.hit.u * dfNormal10[1] + rayHit.hit.v * dfNormal20[1],
                                                    fNormal0[2] + rayHit.hit.u * dfNormal10[2] + rayHit.hit.v * dfNormal20[2],
                                                }
                                            );
                                            
                                        }
                                        std::array<float, 3> rNormal = {};
                                        float rRefIdx = 0.0f;
                                        float cosine_i = vNormal[0] * rayHit.ray.dir_x + vNormal[1] * rayHit.ray.dir_y + vNormal[2] * rayHit.ray.dir_z;
                                        if (cosine_i < 0.0f) {
                                            rNormal = vNormal;
                                            rRefIdx = 1.0f / refrIndx;
                                            cosine_i = -cosine_i;
                                        }
                                        else {
                                            rNormal = std::array<float, 3>{-vNormal[0] - vNormal[1], -vNormal[2]};
                                            rRefIdx = refrIndx;

                                        }
                                        auto sine_o_2 = (1.0f - cosine_i * cosine_i) * rRefIdx * rRefIdx;
                                        auto fresnell = 0.0f;
                                        {
                                            //float cosine_o = sqrtf(RTLib::Ext::CUDA::Math::max(1.0f - sine_o_2, 0.0f));
                                            //float r_p = (cosine_i - rRefIdx * cosine_o) / (cosine_i + rRefIdx * cosine_o);
                                            //float r_s = (rRefIdx * cosine_i - cosine_o) / (rRefIdx * cosine_i + cosine_o);
                                            //fresnell = (r_p * r_p + r_s * r_s) / 2.0f;

                                            float  f0 = ((1 - rRefIdx) / (1 + rRefIdx)) * ((1 - rRefIdx) / (1 + rRefIdx));
                                            fresnell = f0 + (1.0f - f0) * (1.0f - cosine_i) * (1.0f - cosine_i) * (1.0f - cosine_i) * (1.0f - cosine_i) * (1.0f - cosine_i);
                                        }
                                        float r_cosine_i = rNormal[0] * rayHit.ray.dir_x + rNormal[1] * rayHit.ray.dir_y + rNormal[2] * rayHit.ray.dir_z;
                                        auto reflDir = RTLib::Core::Normalize(std::array<float, 3>{
                                            rayHit.ray.dir_x - 2.0f * r_cosine_i * rNormal[0],
                                            rayHit.ray.dir_y - 2.0f * r_cosine_i * rNormal[1],
                                            rayHit.ray.dir_z - 2.0f * r_cosine_i * rNormal[2]
                                        });

                                        if (std::uniform_real_distribution<float>(0.0f, 1.0f)(mt) < fresnell || sine_o_2 > 1.0f) {
                                            rayHit.ray.org_x += (rayHit.ray.dir_x * rayHit.ray.tfar + 0.01f * rNormal[0]);
                                            rayHit.ray.org_y += (rayHit.ray.dir_y * rayHit.ray.tfar + 0.01f * rNormal[1]);
                                            rayHit.ray.org_z += (rayHit.ray.dir_z * rayHit.ray.tfar + 0.01f * rNormal[2]);
                                            rayHit.ray.dir_x = reflDir[0];
                                            rayHit.ray.dir_y = reflDir[1];
                                            rayHit.ray.dir_z = reflDir[2];
                                            /*currThroughput = prevThroughput * specular ;*/
                                            atten[0] *= specCol[0];
                                            atten[1] *= specCol[1];
                                            atten[2] *= specCol[2];
                                        }
                                        else {
                                            rayHit.ray.org_x += (rayHit.ray.dir_x * rayHit.ray.tfar - 0.01f * rNormal[0]);
                                            rayHit.ray.org_y += (rayHit.ray.dir_y * rayHit.ray.tfar - 0.01f * rNormal[1]);
                                            rayHit.ray.org_z += (rayHit.ray.dir_z * rayHit.ray.tfar - 0.01f * rNormal[2]);

                                            float  sine_i_2 = std::max(1.0f - cosine_i * cosine_i, 0.0f);
                                            float  cosine_o = sqrtf(1.0f - sine_o_2);
                                            if (sine_i_2 > 0.0f) {
                                                auto refrDir = std::array<float, 3>{};
                                                std::array<float, 3> k = std::array<float, 3>{
                                                    rayHit.ray.dir_x + cosine_i * rNormal[0], 
                                                    rayHit.ray.dir_y + cosine_i * rNormal[1], 
                                                    rayHit.ray.dir_z + cosine_i * rNormal[2]};
                                                auto tmp = sqrtf(sine_o_2) / sqrtf(sine_i_2);
                                                refrDir = RTLib::Core::Normalize(std::array<float, 3>{
                                                    tmp* k[0] - cosine_o * rNormal[0],
                                                    tmp* k[1] - cosine_o * rNormal[1],
                                                    tmp* k[2] - cosine_o * rNormal[2]
                                                });

                                                rayHit.ray.dir_x = refrDir[0];
                                                rayHit.ray.dir_y = refrDir[1];
                                                rayHit.ray.dir_z = refrDir[2];
                                            }
                                            /*currThroughput  = prevThroughput;*/
                                        }
                                    }

                                    rayHit.ray.tnear = 1e-10f;
                                    rayHit.ray.tfar = FLT_MAX;
                                    rayHit.hit.geomID = RTC_INVALID_GEOMETRY_ID;
                                }
                                if (geoInfo.type == RTLib::Ext::Embree::GeometryAccelerationStructureData::GeometryTypeSphereArray) {
                                    auto& geometry = scene.world.geometrySphereLists.at(geoInfo.name);
                                    break;
                                }
                            }
                            else {
                                break;
                            }
                        }
                        radiance[0] += color[0];
                        radiance[1] += color[1];
                        radiance[2] += color[2];
                    }
                    radiance[0] /= (float)scene.config.maxSamples;
                    radiance[1] /= (float)scene.config.maxSamples;
                    radiance[2] /= (float)scene.config.maxSamples;
                    radiance[0] = radiance[0] / (1.0f + radiance[0]);
                    radiance[1] = radiance[1] / (1.0f + radiance[1]);
                    radiance[2] = radiance[2] / (1.0f + radiance[2]);
                    images[4 * (j * scene.config.width + i) + 0] = 255 * powf(radiance[0], 1.0f / 2.2f);
                    images[4 * (j * scene.config.width + i) + 1] = 255 * powf(radiance[1], 1.0f / 2.2f);
                    images[4 * (j * scene.config.width + i) + 2] = 255 * powf(radiance[2], 1.0f / 2.2f);
                }
            }
            }, rd());
    }
    for (auto& thread : ths) {
        thread.join();
    }
    stbi_write_png(RTLIB_EXT_EMBREE_TEST_TEST0_JSON_PATH"/Result.png", scene.config.width, scene.config.height, 4,images.data(), 4 * scene.config.width);
}

int RenderDebug()
{
    auto device = Internal_MakeEmbreeDevice(nullptr);
    auto scene = RTLib::Ext::Embree::LoadScene(RTLIB_EXT_EMBREE_TEST_TEST0_JSON_PATH"/scene4.json");
    scene.InitExtData(device.get());
    auto gasHandleMap = scene.BuildGeometryASs(device.get());
    auto& root = gasHandleMap.at("CornellBox-Water");
    std::random_device rd;
    std::vector<unsigned char> images(scene.config.width * scene.config.height * 4, 255);
    std::vector<std::thread> ths;
    size_t numThreads = std::thread::hardware_concurrency();
    for (int t = 0; t < numThreads; ++t) {

        ths.emplace_back([&scene, &root, &images, t, numThreads](size_t seed) {
            auto camera = scene.cameraController.GetCamera((float)scene.config.width / (float)scene.config.height);
            auto [u, v, w] = camera.GetUVW();
            RTCRayHit rayHit;
            std::mt19937_64 mt(seed);
            for (int j = t; j < scene.config.height; j += numThreads) {
                for (int i = 0; i < scene.config.width; ++i) {

                    auto radiance = std::array<float, 3>{0.0f, 0.0f, 0.0f};
                    for (int s = 0; s < scene.config.maxSamples; ++s) {
                        rayHit.ray.org_x = camera.GetEye()[0];
                        rayHit.ray.org_y = camera.GetEye()[1];
                        rayHit.ray.org_z = camera.GetEye()[2];

                        const auto d = std::array<float, 2>{
                            2.0f * static_cast<float>(i + std::uniform_real_distribution<float>(0.0f, 1.0f)(mt)) / static_cast<float>(scene.config.width) - 1.0f,
                                2.0f * static_cast<float>(j + std::uniform_real_distribution<float>(0.0f, 1.0f)(mt)) / static_cast<float>(scene.config.height) - 1.0f};
                        auto dir = RTLib::Core::Normalize(
                            RTLib::Core::Add(
                                RTLib::Core::Sub(
                                    RTLib::Core::Mul({ -d[0],-d[0],-d[0] }, u),
                                    RTLib::Core::Mul({ d[1], d[1], d[1] }, v)
                                ),
                                w)
                        );


                        rayHit.ray.dir_x = dir[0];
                        rayHit.ray.dir_y = dir[1];
                        rayHit.ray.dir_z = dir[2];
                        rayHit.ray.tnear = 1e-10f;
                        rayHit.ray.tfar = FLT_MAX;
                        rayHit.hit.geomID = RTC_INVALID_GEOMETRY_ID;

                        auto color = std::array<float, 3>{0.0f, 0.0f, 0.0f};
                        auto atten = std::array<float, 3>{1.0f, 1.0f, 1.0f};
                        {

                            RTCIntersectContext ctx;
                            rtcInitIntersectContext(&ctx);
                            rtcIntersect1(root.handle.get(), &ctx, &rayHit);
                            if (rayHit.hit.geomID != RTC_INVALID_GEOMETRY_ID) {
                                auto& geoInfo = root.geometries.at(rayHit.hit.geomID);
                                if (geoInfo.type == RTLib::Ext::Embree::GeometryAccelerationStructureData::GeometryTypeTriangleArray) {
                                    const auto& geometry = scene.world.geometryObjModels.at(geoInfo.name);
                                    const auto& objModel = scene.objAssetManager.GetAsset(geometry.base);
                                    const auto& sharedRes = objModel.meshGroup->GetSharedResource();
                                    const auto& uniqueRes = objModel.meshGroup->GetUniqueResource(geoInfo.subname);
                                    const auto& matIndex = uniqueRes->materials[uniqueRes->matIndBuffer[rayHit.hit.primID]];

                                    auto  vNormal = RTLib::Core::Normalize(std::array<float, 3>{rayHit.hit.Ng_x, rayHit.hit.Ng_y, rayHit.hit.Ng_z});/*RTLib::Core::Normalize(RTLib::Core::Cross(RTLib::Core::Sub(vertex1, vertex0), RTLib::Core::Sub(vertex2, vertex0)));*/
                                    
                                    const auto  cosine = vNormal[0] * rayHit.ray.dir_x + vNormal[1] * rayHit.ray.dir_y + vNormal[2] * rayHit.ray.dir_z;
                                    const auto& diffCol = objModel.materials[matIndex].GetFloat3Or("diffCol", { 0.0f,0.0f,0.0f });
                                    const auto& specCol = objModel.materials[matIndex].GetFloat3Or("specCol", { 0.0f,0.0f,0.0f });
                                    const auto& emitCol = objModel.materials[matIndex].GetFloat3Or("emitCol", { 0.0f,0.0f,0.0f });
                                    const auto  illum = objModel.materials[matIndex].GetUInt32Or("illum", 0);
                                    const auto  shinness = objModel.materials[matIndex].GetFloat1Or("shinness", 1.0f);
                                    const auto  refrIndx = objModel.materials[matIndex].GetFloat1Or("refrIndx", 1.0f);
                                    if (illum == 7) {
                                        auto fNormal0 = sharedRes->normalBuffer[uniqueRes->triIndBuffer[rayHit.hit.primID][0]];
                                        auto fNormal1 = sharedRes->normalBuffer[uniqueRes->triIndBuffer[rayHit.hit.primID][1]];
                                        auto fNormal2 = sharedRes->normalBuffer[uniqueRes->triIndBuffer[rayHit.hit.primID][2]];

                                        auto dfNormal10 = RTLib::Core::Sub(fNormal1, fNormal0);
                                        auto dfNormal20 = RTLib::Core::Sub(fNormal2, fNormal0);

                                        vNormal = RTLib::Core::Normalize(
                                            std::array<float, 3>{
                                            fNormal0[0] + rayHit.hit.u * dfNormal10[0] + rayHit.hit.v * dfNormal20[0],
                                                fNormal0[1] + rayHit.hit.u * dfNormal10[1] + rayHit.hit.v * dfNormal20[1],
                                                fNormal0[2] + rayHit.hit.u * dfNormal10[2] + rayHit.hit.v * dfNormal20[2],
                                        }
                                        );

                                    }
                                    color[0] = (vNormal[0] + 1.0f) / 2.0f;
                                    color[1] = (vNormal[1] + 1.0f) / 2.0f;
                                    color[2] = (vNormal[2] + 1.0f) / 2.0f;
                                }
                                if (geoInfo.type == RTLib::Ext::Embree::GeometryAccelerationStructureData::GeometryTypeSphereArray) {
                                    auto& geometry = scene.world.geometrySphereLists.at(geoInfo.name);
                                    break;
                                }
                            }
                            else {
                                break;
                            }
                        }
                        radiance[0] += color[0];
                        radiance[1] += color[1];
                        radiance[2] += color[2];
                    }
                    radiance[0] /= (float)scene.config.maxSamples;
                    radiance[1] /= (float)scene.config.maxSamples;
                    radiance[2] /= (float)scene.config.maxSamples;
                    images[4 * (j * scene.config.width + i) + 0] = 255 * radiance[0];
                    images[4 * (j * scene.config.width + i) + 1] = 255 * radiance[1];
                    images[4 * (j * scene.config.width + i) + 2] = 255 * radiance[2];
                }
            }
            }, rd());
    }
    for (auto& thread : ths) {
        thread.join();
    }
    stbi_write_png(RTLIB_EXT_EMBREE_TEST_TEST0_JSON_PATH"/Result2.png", scene.config.width, scene.config.height, 4, images.data(), 4 * scene.config.width);
}
int main() {
    RenderTrace();
}