#ifndef TEST_TESTLIB_SHADER_BINDING_TABLE_LAYOUT__H
#define TEST_TESTLIB_SHADER_BINDING_TABLE_LAYOUT__H
#include <string>
#include <vector>
#include <memory>
#include <array>
#include <stack>
#include <queue>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <sstream>
#include <tuple>
namespace TestLib
{
	// OPTIX STYLE                                                                                      //
	// INSTANCE_AS_OR_ARRAY0-> INSTANCE0-> GEOMETRY_AS0         -> GEOMETRY0                            //
	//                                                             GEOMETRY1                            //
	//                      -> INSTANCE1-> INSTANCE_AS_OR_ARRAY1-> INSTANCE0-> GEOMETRY_AS1-> GEOMETRY0 //
	//                                                                                        GEOMETRY1 //
	//                                                                                        GEOMETRY2 //
	//                                                                                                  //
	//                                                                         GEOMETRY_AS2-> GEOMETRY0 //
	//                                                                                        GEOMETRY1 //
	//                                                                                                  //
	// VULKAN/D3D12 STYLE                                                                               //
	// INSTANCE_AS_OR_ARRAY0-> INSTANCE0-> GEOMETRY_AS0         -> GEOMETRY0                            //
	//                                                             GEOMETRY1                            //
	//                      -> INSTANCE1.                                                               //
	//                         INSTANCE_AS_OR_ARRAY1.                                                   //
	//                         INSTANCE0.                                                               //
	//                         GEOMETRY_AS1-> GEOMETRY_AS1      -> GEOMETRY0                            //
	//                                                             GEOMETRY1                            //
	//                                                             GEOMETRY2                            //
	//                      -> INSTANCE1.                                                               //
	//                         INSTANCE_AS_OR_ARRAY1.                                                   //
	//                         INSTANCE0.                                                               //
	//                         GEOMETRY_AS2-> GEOMETRY_AS2      -> GEOMETRY0                            //
	//                                                             GEOMETRY1                            //
	//                                                                                                  //
	// OPTIX 7.7 以前では Mutliple Level Instancing のOffsetは加算されない                              //
	// そのため, 7.7互換動作の実現には複数のInstance Acceleration Structureが必要になる                 //
	// 逆に7.7以降ではInstanceのOffsetが加算されるため, 従来より少ないInstanceで記述可能                //
	//                                                                                                  //
	// Geometry->Dirty                                                                                  //
	// GeometryAS->Dirty                                                                                //
	// Instance Of GeometryAS->Dirty                                                                    //
	// InstanceAS Of Instance->Dirty                                                                    //
	// Instance Of InstanceAS->Dirty                                                                    //
	// InstanceAS Of Instance->Dirty                                                                    //
	enum class AccelerationStructureType
	{
		eGeometry,
		eInstance
	};

	struct ShaderBindingTableLayoutGeometryDesc
	{
		ShaderBindingTableLayoutGeometryDesc() noexcept {}
		ShaderBindingTableLayoutGeometryDesc(
			std::string name_,
			unsigned int baseSbtCount_,
			const std::array<float, 12>& preTransform_ = {}
		) noexcept : 
			name{ name_ }, 
			baseSbtCount{ baseSbtCount_ }, 
			preTransform{ preTransform_ }
		{}

		std::string          name = "";
		unsigned int         baseSbtCount = 0;
		std::array<float,12> preTransform = {};
	};

	struct ShaderBindingTableLayoutGeometryAccelerationStructureDesc
	{
		ShaderBindingTableLayoutGeometryAccelerationStructureDesc() noexcept {}
		ShaderBindingTableLayoutGeometryAccelerationStructureDesc(
			std::string name_,
			const std::vector<ShaderBindingTableLayoutGeometryDesc>& geometries_ = {}
		) noexcept :
			name{ name_ },
			geometries{ geometries_ }
		{}

		std::string                                       name = "";
		std::vector<ShaderBindingTableLayoutGeometryDesc> geometries = {};
	};

	struct ShaderBindingTableLayoutInstanceDesc
	{
		ShaderBindingTableLayoutInstanceDesc() noexcept {}
		ShaderBindingTableLayoutInstanceDesc(
			std::string name_,
			AccelerationStructureType type_,
			unsigned int              baseIndex_, 
			const std::array<float, 12>& preTransform_ = {}
		) noexcept :
			name{ name_ },
			type{ type_ },
			baseIndex{ baseIndex_ },
			preTransform{ preTransform_ }
		{}

		std::string               name          = "";
		AccelerationStructureType type          = AccelerationStructureType::eGeometry;
		unsigned int              baseIndex     = 0;
		std::array<float, 12>     preTransform  = {};
	};

	struct ShaderBindingTableLayoutInstanceAccelerationStructureOrArrayDesc
	{
		ShaderBindingTableLayoutInstanceAccelerationStructureOrArrayDesc() noexcept {}
		ShaderBindingTableLayoutInstanceAccelerationStructureOrArrayDesc(
			std::string name_,
			const std::vector<ShaderBindingTableLayoutInstanceDesc>& instances_ = {}
		) noexcept :
			name{ name_ },
			instances{ instances_ }
		{}

		std::string                                       name      = "";
		std::vector<ShaderBindingTableLayoutInstanceDesc> instances = {};
	};

	struct ShaderBindingTableLayoutDesc
	{
		void normalize()
		{
			auto get_key = [](std::uint32_t instanceASIndex, std::uint32_t instanceIndex) -> std::uint64_t {
				return ((static_cast<uint64_t>(instanceASIndex) << static_cast<uint64_t>(32)) + static_cast<uint64_t>(instanceIndex));
			};

			std::unordered_map<std::uint64_t, std::uint32_t> instanceInfosOfInstnaceASOrArray = {};
			std::vector<ShaderBindingTableLayoutInstanceAccelerationStructureOrArrayDesc> newInstanceASs = {};
			std::vector<unsigned int> oldInstanceASIndices = {};

			instanceInfosOfInstnaceASOrArray[UINT64_MAX] = 0;
			newInstanceASs.push_back(instanceAccelerationStructureOrArrays[rootIndex]);
			oldInstanceASIndices.push_back(rootIndex);

			for (auto i = 0; i < instanceAccelerationStructureOrArrays.size(); ++i)
			{
				//if (rootIndex == i) {
				//	continue;
				//}
				auto& instanceAS = instanceAccelerationStructureOrArrays[i];
				for (auto j = 0; j < instanceAS.instances.size(); ++j)
				{
					auto& instance = instanceAS.instances[j];
					if (instance.type == AccelerationStructureType::eInstance)
					{
						uint64_t key = get_key(i, j);
						uint32_t val = instance.baseIndex;
						instanceInfosOfInstnaceASOrArray[key] = oldInstanceASIndices.size();
						newInstanceASs.push_back(instanceAccelerationStructureOrArrays[val]);
						oldInstanceASIndices.push_back(val);
					}
				}
			}
			for (auto i = 0; i < newInstanceASs.size(); ++i)
			{
				auto& newInstanceAS = newInstanceASs[i];
				auto  oldInstanceASIndex = oldInstanceASIndices[i];
				for (auto j = 0; j < newInstanceAS.instances.size(); ++j)
				{
					auto& instance = newInstanceAS.instances[j];
					if (instance.type == AccelerationStructureType::eInstance)
					{
						uint64_t key = get_key(oldInstanceASIndex, j);
						instance.baseIndex = instanceInfosOfInstnaceASOrArray.at(key);
					}
				}

			}

			instanceAccelerationStructureOrArrays = newInstanceASs;
		}
		std::vector<ShaderBindingTableLayoutGeometryAccelerationStructureDesc>        geometryAccelerationStructures = {};
		std::vector<ShaderBindingTableLayoutInstanceAccelerationStructureOrArrayDesc> instanceAccelerationStructureOrArrays  = {};
		unsigned int rootIndex = 0;
		unsigned int sbtStride = 0;
	};

	struct ShaderBindingTableLayout
	{
		struct Geometry;
		struct GeometryAccelerationStructure;
		struct Instance;
		struct InstanceAccelerationStructureOrArray;

		ShaderBindingTableLayout(ShaderBindingTableLayoutDesc desc) noexcept;

		auto find_geometry_acceleration_structure(std::string name) const -> const ShaderBindingTableLayout::GeometryAccelerationStructure*
		{
			for (auto& geometryAS : m_GeometryAccelerationStructures)
			{
				if (geometryAS.name == name) {
					return &geometryAS;
				}
			}
			return nullptr;
		}
		auto find_geometry(std::string name) const -> const ShaderBindingTableLayout::Geometry*
		{
			std::stringstream ss;
			ss << name;
			std::string geometryASName;
			std::getline(ss, geometryASName, '/');
			auto geometryAS = find_geometry_acceleration_structure(geometryASName);

			if (!geometryAS) { return nullptr; }

			std::string geometryName = "";
			std::getline(ss, geometryName, '/');

			for (auto& geometry : geometryAS->geometries)
			{
				if (geometry.name == geometryName) {
					return &geometry;
				}
			}
			return nullptr;
		}

		auto find_instance_acceleration_structure_or_arrays(std::string name) const  -> const ShaderBindingTableLayout::InstanceAccelerationStructureOrArray*
		{
			for (auto& instanceAS : m_InstanceAccelerationSturctureOrArrays)
			{
				if (instanceAS.name == name) {
					return &instanceAS;
				}
			}
			return nullptr;
		}
		auto find_instance(std::string name) const -> const ShaderBindingTableLayout::Instance*
		{
			std::stringstream ss;
			ss << name;

			auto* instanceAS = m_InstanceAccelerationSturctureOrArrays.data();
			const ShaderBindingTableLayout::Instance* res = nullptr;
			std::string instanceName;
			while (std::getline(ss, instanceName, '/'))
			{
				decltype(instanceAS) nextInstanceAS = nullptr;
				for (auto& instance : instanceAS->instances)
				{
					if (instance.type == AccelerationStructureType::eInstance)
					{
						if (instance.name == instanceName)
						{
							nextInstanceAS = &m_InstanceAccelerationSturctureOrArrays[instance.baseIndex];
							res = &instance;
						}
					}
				}
				if (!nextInstanceAS) {
					return nullptr;
				}
				instanceAS = nextInstanceAS;
			}
			
			return res;
		}

		struct Geometry
		{
			Geometry(ShaderBindingTableLayout* root_,
				const ShaderBindingTableLayoutGeometryDesc& desc) noexcept
				:root{root_},name{desc.name},baseSbtCount{desc.baseSbtCount},baseSbtOffset{0},preTransform{desc.preTransform}
			{}

			auto get_sbt_count() const noexcept -> unsigned int {
				return root->m_SbtStride * baseSbtCount;
			}

			ShaderBindingTableLayout* root          = nullptr;
			std::string               name          = "";
			unsigned int              baseSbtCount  = 0;
			unsigned int              baseSbtOffset = 0;
			std::array<float, 12>     preTransform = {};
		};
		struct GeometryAccelerationStructure
		{
			GeometryAccelerationStructure(ShaderBindingTableLayout* root_,
				const ShaderBindingTableLayoutGeometryAccelerationStructureDesc& desc)
				noexcept:root{root_},name{desc.name}
			{
				baseSbtCount = 0;
				geometries.reserve(desc.geometries.size());
				for (auto& geometry : desc.geometries){
					geometries.emplace_back(root_, geometry);
					geometries.back().baseSbtOffset = baseSbtCount;
					baseSbtCount += geometries.back().baseSbtCount;
				}
			}
			auto get_sbt_count() const noexcept -> unsigned int {
				return root->m_SbtStride * baseSbtCount;
			}

			ShaderBindingTableLayout* root         = nullptr;
			std::string               name = "";
			std::vector<Geometry>     geometries   = {};
			unsigned int              baseSbtCount = 0;
		};
		struct Instance
		{
			Instance(ShaderBindingTableLayout* root_,
				const ShaderBindingTableLayoutInstanceDesc& desc
			)noexcept
				:root{root_},
				type{desc.type},
				baseIndex{desc.baseIndex},
				name{desc.name},
				sbtOffsetInternalParent{ 0 },
				sbtOffsetExternalParent{ 0 },
				preTransform{desc.preTransform}
			{}


			auto get_sbt_count() const noexcept -> unsigned int {
				if (type == AccelerationStructureType::eGeometry)
				{
					return root->m_GeometryAccelerationStructures[baseIndex].baseSbtCount* root->m_SbtStride;
				}
				else {
					return root->m_InstanceAccelerationSturctureOrArrays[baseIndex].sbtCount;
				}
			}

			auto get_sbt_offset() const noexcept -> unsigned int { return sbtOffsetInternalParent + sbtOffsetExternalParent; }

			ShaderBindingTableLayout* root;
			AccelerationStructureType type      = AccelerationStructureType::eGeometry;
			unsigned int              baseIndex = 0;
			std::string               name      = "";
			unsigned int              sbtOffsetInternalParent = 0;
			unsigned int              sbtOffsetExternalParent = 0;
			std::array<float, 12>     preTransform = {};
		};
		struct InstanceAccelerationStructureOrArray
		{
			InstanceAccelerationStructureOrArray() noexcept {}
			InstanceAccelerationStructureOrArray(ShaderBindingTableLayout* root_,
				const ShaderBindingTableLayoutInstanceAccelerationStructureOrArrayDesc& desc
			)noexcept
				:root{ root_ }, 
				name{desc.name},
				sbtCount {0}, 
				sbtOffset{ 0 }
			{
				instances.reserve(desc.instances.size());
				for (auto& instance : desc.instances) {
					instances.emplace_back(root_, instance);
				}
			}

			ShaderBindingTableLayout* root          = nullptr;
			std::vector<Instance>     instances     = {};
			std::string               name          = "";
			unsigned int              sbtCount      = 0;
			unsigned int              sbtOffset     = 0;
		};
	private:
		unsigned int                                       m_SbtStride  = 0;
		std::vector<GeometryAccelerationStructure>         m_GeometryAccelerationStructures = {};
		std::vector<InstanceAccelerationStructureOrArray>  m_InstanceAccelerationSturctureOrArrays  = {};
	};
}
#endif
