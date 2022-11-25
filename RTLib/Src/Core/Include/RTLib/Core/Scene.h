#ifndef RTLIB_CORE_SCENE_H
#define RTLIB_CORE_SCENE_H
#include <RTLib/Core/Camera.h>
#include <RTLib/Core/World.h>
#include <RTLib/Core/VariableMap.h>
#include <RTLib/Core/BinaryReader.h>
namespace RTLib
{
	namespace Core {
        struct TraceConfigData
        {
            std::string  imagePath;
            unsigned int width;
            unsigned int height;
            unsigned int samples;
            unsigned int samplesPerSave;
            unsigned int maxSamples;
            float        maxTimes;
            unsigned int maxDepth;
            bool         enableVis;
            std::string  defTracer;
            RTLib::Core::VariableMap custom;
            std::unordered_map<std::string, RTLib::Core::VariableMap> tracers;
        };
        
        template<typename JsonType>
        inline void   to_json(JsonType& j, const TraceConfigData& v) {
            j["ImagePath"] = v.imagePath;
            j["Width"] = v.width;
            j["Height"] = v.height;
            j["Samples"] = v.samples;
            j["SamplesPerSave"] = v.samplesPerSave;
            j["MaxSamples"] = v.maxSamples;
            j["MaxTimes"] = v.maxTimes;
            j["MaxDepth"] = v.maxDepth;
            j["EnableVis"] = v.enableVis;
            j["DefTracer"] = v.defTracer;
            if (!v.custom.IsEmpty())
            {
                j["Custom"] = v.custom;
            }
            for (auto& [name, tracer] : v.tracers)
            {
                if (tracer.IsEmpty()) {
                    j["Tracers"][name] = tracer;
                }
            }
        }
        template<typename JsonType>
        inline void from_json(const JsonType& j, TraceConfigData& v) {
            v.imagePath = j.at("ImagePath").get<std::string >();
            v.width = j.at("Width").get<unsigned int>();
            v.height = j.at("Height").get<unsigned int>();
            v.samples = j.at("Samples").get<unsigned int>();
            v.samplesPerSave = j.at("SamplesPerSave").get<unsigned int>();
            v.maxSamples = j.at("MaxSamples").get<unsigned int>();
            if (j.count("EnableVis") > 0) {
                v.enableVis = j.at("EnableVis").get<bool>();
            }
            else {
                v.enableVis = false;
            }
            if (j.count("DefTracer") > 0) {
                v.defTracer = j.at("DefTracer").get<std::string>();
            }
            else {
                v.defTracer = "NONE";
            }
            if (j.count("MaxTimes") > 0) {
                v.maxTimes = j.at("MaxTimes").get<float>();
            }
            else {
                v.maxTimes = FLT_MAX;
            }
            v.maxDepth = j.at("MaxDepth").get<unsigned int>();
            if (j.count("Custom") > 0) {
                v.custom = j.at("Custom").get<RTLib::Core::VariableMap>();
            }
            if (j.count("Tracers") > 0)
            {
                for (const auto& elem : j.at("Tracers").items())
                {
                    v.tracers[elem.key()] = elem.value().get<RTLib::Core::VariableMap>();
                }
            }
        }

        struct ImageConfigData
        {
            unsigned int       width;
            unsigned int       height;
            unsigned int       samples;
            unsigned long long time;
            bool               enableVis;
            std::string        pngFilePath;
            std::string        exrFilePath;
            std::string        binFilePath;
        };

        template<typename JsonType>
        inline void   to_json(JsonType& j, const ImageConfigData& v) {
            j["PngFilePath"] = v.pngFilePath;
            j["ExrFilePath"] = v.exrFilePath;
            j["BinFilePath"] = v.binFilePath;
            j["Width"] = v.width;
            j["Height"] = v.height;
            j["Samples"] = v.samples;
            j["Time"] = v.time;
            j["EnableVis"] = v.enableVis;
        }
        template<typename JsonType>
        inline void from_json(const JsonType& j, ImageConfigData& v) {
            v.pngFilePath = j.at("PngFilePath").get<std::string >();
            v.exrFilePath = j.at("ExrFilePath").get<std::string >();
            v.binFilePath = j.at("BinFilePath").get<std::string >();
            v.width = j.at("Width").get<unsigned int>();
            v.height = j.at("Height").get<unsigned int>();
            v.samples = j.at("Samples").get<unsigned int>();
            v.time = j.at("Time").get<unsigned long long>();
            v.enableVis = j.at("EnableVis").get<bool>();
        }

        struct SceneData
        {
            std::unordered_map<std::string, SphereResourcePtr> sphereResources;
            ObjModelAssetManager objAssetManager;
            CameraController     cameraController;
            WorldData            world;
            TraceConfigData      config;

            auto GetFrameBufferSizeInBytes()const noexcept -> size_t
            {
                return static_cast<size_t>(config.width * config.height);
            }
            auto GetCamera()const noexcept -> Camera
            {
                return cameraController.GetCamera(static_cast<float>(config.width) / static_cast<float>(config.height));
            }
        };

        template<typename JsonType>
        inline void   to_json(JsonType& j, const SceneData& v) {
            j["CameraController"] = v.cameraController;
            j["ObjModels"] = v.objAssetManager;
            j["World"] = v.world;
            j["Config"] = v.config;
        }

        template<typename JsonType>
        inline void from_json(const JsonType& j, SceneData& v) {
            v.cameraController = j.at("CameraController").get<RTLib::Core::CameraController>();
            v.objAssetManager = j.at("ObjModels").get<RTLib::Core::ObjModelAssetManager>();
            v.world = j.at("World").get<RTLib::Core::WorldData>();
            v.config = j.at("Config").get<TraceConfigData>();

            for (auto& [geometryASName, geometryAS] : v.world.geometryASs)
            {
                for (auto& geometryName : geometryAS.geometries) {
                    if (v.world.geometryObjModels.count(geometryName) > 0) {
                        auto& geometryObjModel = v.world.geometryObjModels.at(geometryName);
                        auto& objAsset = v.objAssetManager.GetAsset(geometryObjModel.base);
                        if (geometryObjModel.meshes.empty()) {
                            auto meshNames = objAsset.meshGroup->GetUniqueNames();
                            for (auto& meshName : meshNames)
                            {
                                auto mesh = objAsset.meshGroup->LoadMesh(meshName);
                                geometryObjModel.meshes[meshName].base = meshName;
                                geometryObjModel.meshes[meshName].transform = {};
                                geometryObjModel.meshes[meshName].materials.reserve(mesh->GetUniqueResource()->materials.size());
                                for (auto& matIdx : mesh->GetUniqueResource()->materials) {
                                    geometryObjModel.meshes[meshName].materials.push_back(objAsset.materials[matIdx]);
                                }
                            }
                        }
                        else {
                            for (auto& [meshName, meshData] : geometryObjModel.meshes)
                            {
                                auto mesh = objAsset.meshGroup->LoadMesh(meshData.base);
                                if (meshData.materials.empty()) {
                                    meshData.materials.reserve(mesh->GetUniqueResource()->materials.size());
                                    for (auto& matIdx : mesh->GetUniqueResource()->materials) {
                                        meshData.materials.push_back(objAsset.materials[matIdx]);
                                        if (meshData.materials.back().HasString("diffCol") ||
                                            meshData.materials.back().GetString("diffCol") == "") {
                                            meshData.materials.back().SetString("diffCol", "White");
                                        }
                                        if (meshData.materials.back().HasString("specCol") ||
                                            meshData.materials.back().GetString("specCol") == "") {
                                            meshData.materials.back().SetString("specCol", "White");
                                        }
                                        if (meshData.materials.back().HasString("emitCol") ||
                                            meshData.materials.back().GetString("emitCol") == "") {
                                            meshData.materials.back().SetString("emitCol", "White");
                                        }
                                    }
                                }

                            }
                        }
                    }
                    if (v.world.geometrySpheres.count(geometryName) > 0) {
                        auto& geometrySphere = v.world.geometrySpheres.at(geometryName);
                        v.sphereResources[geometryName] = RTLib::Core::SphereResource::New();
                        v.sphereResources[geometryName]->centerBuffer.push_back(geometrySphere.center);
                        v.sphereResources[geometryName]->radiusBuffer.push_back(geometrySphere.radius);
                        v.sphereResources[geometryName]->matIndBuffer.push_back(0);
                        v.sphereResources[geometryName]->materials.push_back(geometrySphere.material);
                        v.sphereResources[geometryName]->variables.SetFloat3("aabb.min",
                            {
                                geometrySphere.center[0] - geometrySphere.radius,
                                geometrySphere.center[1] - geometrySphere.radius,
                                geometrySphere.center[2] - geometrySphere.radius,
                            }
                        );
                        v.sphereResources[geometryName]->variables.SetFloat3("aabb.max",
                            {
                                geometrySphere.center[0] + geometrySphere.radius,
                                geometrySphere.center[1] + geometrySphere.radius,
                                geometrySphere.center[2] + geometrySphere.radius,
                            }
                        );
                    }
                }
            }

        }

	}
}
#endif