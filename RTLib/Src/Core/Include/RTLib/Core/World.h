#ifndef RTLIB_CORE_WORLD_H
#define RTLIB_CORE_WORLD_H
#include <RTLib/Core/VariableMap.h>
#include <string>
#include <iostream>
#include <array>
namespace RTLib
{
    namespace Core{
        struct WorldElementGeometryObjModelMesh
        {
            std::string base = {};
            std::vector<RTLib::Core::VariableMap> materials = {};
            std::array<float, 12> transform = {};
            bool useTransform = false;
            bool useMaterials = false;
        };
        //VariableArray
        template<typename JsonType>
        inline void   to_json(JsonType& j, const WorldElementGeometryObjModelMesh& v) {
            j["Base"] = v.base;
            if (v.useTransform) {
                j["Transform"] = v.transform;
            }
            if (v.useMaterials) {
                j["Materials"] = v.materials;
            }
        }
        template<typename JsonType>
        inline void from_json(const JsonType& j, WorldElementGeometryObjModelMesh& v) {
            v.base = j.at("Base").get<std::string>();
            v.useMaterials = j.count("Materials") > 0;
            v.useTransform = j.count("Transform") > 0;
            if (v.useMaterials) {
                v.materials = j.at("Materials").get<std::vector<RTLib::Core::VariableMap>>();
            }
            if (v.useTransform) {
                v.transform = j.at("Transform").get<std::array<float, 12>>();
            }
        }
        struct WorldElementGeometryObjModel
        {
            std::string base = "";
            std::unordered_map<std::string, WorldElementGeometryObjModelMesh> meshes = {};
            bool useMeshes = false;
        };
        template<typename JsonType>
        inline void   to_json(JsonType& j, const WorldElementGeometryObjModel& v) {
            j["Type"] = "ObjModel";
            j["Base"] = v.base;
            if (!v.meshes.empty() && v.useMeshes) {
                auto meshJsons = std::unordered_map<std::string,JsonType>();
                meshJsons.reserve(v.meshes.size());
                for (auto& [name, mesh] : v.meshes) {
                    meshJsons.insert({ name,nlohmann::json(mesh) });
                }
                j["Meshes"] = meshJsons;
            }
        }
        template<typename JsonType>
        inline void from_json(const JsonType& j, WorldElementGeometryObjModel& v) {
            v.base = j.at("Base").get<std::string>();
            v.meshes.clear();
            if ((j.count("Meshes") > 0)) {
                for (auto& meshElem : j.at("Meshes").items()) {
                    v.meshes[meshElem.key()] = meshElem.value().get<WorldElementGeometryObjModelMesh>();

                }
                v.useMeshes = true;
            }
            else {
                v.useMeshes = false;
            }
        }

        struct WorldElementGeometrySphere
        {
            std::array<float, 3> center;
            float                radius;
            VariableMap          material;
        };

        template<typename JsonType>
        inline void   to_json(JsonType& j, const WorldElementGeometrySphere& v) {
            j["Type"] = "Sphere";
            j["Center"] = v.center;
            j["Radius"] = v.radius;
            j["Material"] = v.material;
        }
        template<typename JsonType>
        inline void from_json(const JsonType& j, WorldElementGeometrySphere& v) {
            v.center = j.at("Center").get<std::array<float, 3>>();
            v.radius = j.at("Radius").get<float>();
            v.material = j.at("Material").get<VariableMap>();
        }

        struct WorldElementGeometrySphereList
        {
            using Float3 = std::array<float, 3>;
            std::vector<Float3>       centerBuffer = {};
            std::vector<float >       radiusBuffer = {};
            std::vector<unsigned int> matIndBuffer = {};
            std::vector<VariableMap>  materials = {};
        };

        template<typename JsonType>
        inline void   to_json(JsonType& j, const WorldElementGeometrySphereList& v) {
            auto tmpCntBuffer = std::vector<float>(v.centerBuffer.size() * 3);
            std::memcpy(tmpCntBuffer.data(), v.centerBuffer.data(), sizeof(v.centerBuffer[0]) * std::size(v.centerBuffer));
            j["Type"] = "SphereList";
            j["CenterBuffer"] = tmpCntBuffer;
            j["MatIndBuffer"] = v.matIndBuffer;
            j["RadiusBuffer"] = v.radiusBuffer;
            j["Materials"] = v.materials;
        }
        template<typename JsonType>
        inline void from_json(const JsonType& j, WorldElementGeometrySphereList& v) {
            auto tmpCntBuffer = j.at("CenterBuffer").get<std::vector<float>>();
            v.centerBuffer.resize(tmpCntBuffer.size() / 3);
            std::memcpy(v.centerBuffer.data(), tmpCntBuffer.data(), sizeof(v.centerBuffer[0]) * std::size(v.centerBuffer));
            v.radiusBuffer = j.at("RadiusBuffer").get<std::vector<float>>();
            v.matIndBuffer = j.at("MatIndBuffer").get<std::vector<unsigned int>>();
            v.materials = j.at("Materials").get<std::vector<VariableMap>>();
        }

        struct WorldElementGeometryAS
        {
            std::vector<std::string> geometries = {};
        };
        template<typename JsonType>
        inline void   to_json(JsonType& j, const WorldElementGeometryAS& v) {
            j["Type"] = "GeometryAS";
            j["Geometries"] = v.geometries;
        }
        template<typename JsonType>
        inline void from_json(const JsonType& j, WorldElementGeometryAS& v) {
            v.geometries = j.at("Geometries").get<std::vector<std::string>>();
        }
        struct WorldElementInstance
        {
            std::string base;
            std::string asType;
            std::array<float, 12> transform;

        };
        template<typename JsonType>
        inline void   to_json(JsonType& j, const WorldElementInstance& v) {
            j["Type"] = "Instance";
            j["Base"] = v.base;
            j["ASType"] = v.asType;
            j["Transform"] = v.transform;
        }
        template<typename JsonType>
        inline void from_json(const JsonType& j, WorldElementInstance& v) {
            v.base = j.at("Base").get<std::string>();
            v.asType = j.at("ASType").get<std::string>();
            v.transform = j.at("Transform").get<std::array<float, 12>>();
        }
        struct WorldElementInstanceAS
        {
            std::vector<std::string> instances = {};
        };
        template<typename JsonType>
        inline void   to_json(JsonType& j, const WorldElementInstanceAS& v) {
            j["Type"] = "InstanceAS";
            j["Instances"] = v.instances;
        }
        template<typename JsonType>
        inline void from_json(const JsonType& j, WorldElementInstanceAS& v) {
            v.instances = j.at("Instances").get<std::vector<std::string>>();
        }

        struct WorldData
        {
            std::unordered_map<std::string, WorldElementGeometrySphereList>   geometrySphereLists;
            std::unordered_map<std::string, WorldElementGeometrySphere>       geometrySpheres;
            std::unordered_map<std::string, WorldElementGeometryObjModel>     geometryObjModels;
            std::unordered_map<std::string, WorldElementGeometryAS>           geometryASs;
            std::unordered_map<std::string, WorldElementInstance>             instances;
            std::unordered_map<std::string, WorldElementInstanceAS>           instanceASs;
        };

        template<typename JsonType>
        inline void   to_json(JsonType& j, const WorldData& v) {
            for (auto& [name, geometry] : v.geometryObjModels) {
                j["Geometries"][name] = geometry;
            }
            for (auto& [name, geometry] : v.geometrySpheres) {
                j["Geometries"][name] = geometry;
            }
            for (auto& [name, geometry] : v.geometrySphereLists) {
                j["Geometries"][name] = geometry;
            }
            for (auto& [name, geometryAS] : v.geometryASs) {
                j["GeometryASs"][name] = geometryAS;
            }
            for (auto& [name, instance] : v.instances) {
                j["Instances"][name] = instance;
            }
            for (auto& [name, instanceAS] : v.instanceASs) {
                j["InstanceASs"][name] = instanceAS;
            }
        }
        template<typename JsonType>
        inline void from_json(const JsonType& j, WorldData& v) {
            for (auto& elem : j.at("Geometries").items()) {
                if (elem.value().at("Type").get<std::string>() == "ObjModel") {
                    v.geometryObjModels[elem.key()] = elem.value().get<WorldElementGeometryObjModel>();
                }
                if (elem.value().at("Type").get<std::string>() == "Sphere") {
                    v.geometrySpheres[elem.key()] = elem.value().get<WorldElementGeometrySphere>();
                }
                if (elem.value().at("Type").get<std::string>() == "SphereList") {
                    v.geometrySphereLists[elem.key()] = elem.value().get<WorldElementGeometrySphereList>();
                }
            }
            for (auto& elem : j.at("GeometryASs").items()) {
                v.geometryASs[elem.key()] = elem.value().get<WorldElementGeometryAS>();
            }
            for (auto& elem : j.at("Instances").items()) {
                v.instances[elem.key()] = elem.value().get<WorldElementInstance>();
            }
            for (auto& elem : j.at("InstanceASs").items()) {
                v.instanceASs[elem.key()] = elem.value().get<WorldElementInstanceAS>();
            }
        }
    }
}
#endif
