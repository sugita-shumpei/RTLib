#ifndef RTLIB_CORE_SHADER_LAYOUT_H
#define RTLIB_CORE_SHADER_LAYOUT_H
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <array>
#include <memory>
#include <vector>
namespace RTLib {

	namespace Core
	{
		class ShaderTableLayout;
		class ShaderTableLayoutGeometryAS;
		class ShaderTableLayoutGeometry;
		class ShaderTableLayoutInstanceAS;
		class ShaderTableLayoutInstance;
		//GeometryAS
		class ShaderTableLayoutGeometry {
		public:
			friend class ShaderTableLayout;
			friend class ShaderTableLayoutGeometryAS;

			ShaderTableLayoutGeometry(const std::string& name, unsigned int baseRecordCount = 0)noexcept;
			ShaderTableLayoutGeometry(const ShaderTableLayoutGeometry& geometry)noexcept;
			auto operator=(const ShaderTableLayoutGeometry& geometry)noexcept->ShaderTableLayoutGeometry&;

			auto GetName()const noexcept -> std::string;

			auto GetBaseRecordOffset()const noexcept -> unsigned int;
			auto GetBaseRecordCount()const noexcept -> unsigned int;
		private:
			void Internal_SetUpGeometryAS(ShaderTableLayoutGeometryAS* gas)noexcept;
			void Internal_SetBaseRecordOffset(unsigned int offset)noexcept;
		private:
			std::string m_Name = "";
			ShaderTableLayoutGeometryAS* m_UpGeometryAS = nullptr;
			unsigned int m_BaseRecordCount = 0;
			unsigned int m_BaseRecordOffset = 0;
		};
		class ShaderTableLayoutGeometryAS {
			friend class ShaderTableLayoutGeometry;
			friend class ShaderTableLayout;
		private:
			using Geometry = ShaderTableLayoutGeometry;
		public:
			ShaderTableLayoutGeometryAS(const std::string& name)noexcept;
			ShaderTableLayoutGeometryAS(const ShaderTableLayoutGeometryAS& gas)noexcept;
			auto operator=(const ShaderTableLayoutGeometryAS& gas)noexcept->ShaderTableLayoutGeometryAS&;

			auto GetName()const noexcept -> std::string;
			auto GetBaseRecordCount() const noexcept -> unsigned int;
			auto GetDwGeometries() const noexcept ->const std::vector<Geometry>&;
			void SetDwGeometry(const ShaderTableLayoutGeometry& geometry)noexcept;

			void Update()noexcept;
		private:
			std::string m_Name = "";
			std::vector<Geometry> m_DwGeometries = {};
			unsigned int m_BaseRecordCount = 0;
		};
		//InstanceAS
		class ShaderTableLayoutInstance {
		private:
			using InstanceAS = ShaderTableLayoutInstanceAS;
			using GeometryAS = ShaderTableLayoutGeometryAS;
			friend class ShaderTableLayoutInstanceAS;
			friend class ShaderTableLayoutGeometryAS;
			friend class ShaderTableLayout;
		public:
			ShaderTableLayoutInstance(const std::string& name, GeometryAS* geometryAS)noexcept;
			ShaderTableLayoutInstance(const std::string& name, InstanceAS* instanceAS)noexcept;

			ShaderTableLayoutInstance(const ShaderTableLayoutInstance& instance)noexcept;
			auto operator=(const ShaderTableLayoutInstance& instance)noexcept->ShaderTableLayoutInstance&;

			auto GetName()const noexcept -> std::string;

			auto GetRecordStride()const noexcept -> unsigned int;
			void SetRecordStride(unsigned int recordStride)noexcept;

			auto GetRecordCount()const noexcept -> unsigned int;
			auto GetRecordOffset()const noexcept-> unsigned int;

			auto GetDwGeometryAS()const noexcept-> const GeometryAS*;
			auto GetDwInstanceAS()const noexcept-> const InstanceAS*;

			auto RootInstanceAS()      noexcept ->      InstanceAS*;
			auto RootInstanceAS()const noexcept ->const InstanceAS*;

			void Show()noexcept;
		private:
			void Internal_SetRecordStride(unsigned int recordStride)noexcept;
			void Internal_SetRecordOffset(unsigned int recordOffset)noexcept;
			void Internal_SetUpInstanceAS(InstanceAS* instanceAS)noexcept;
		private:
			std::string m_Name = "";
			InstanceAS* m_UpInstanceAS = nullptr;
			InstanceAS* m_DwInstanceAS = nullptr;
			const GeometryAS* m_DwGeometryAS = nullptr;
			unsigned int          m_RecordStride = 0;
			unsigned int          m_RecordCount = 0;
			unsigned int          m_RecordOffset = 0;
		};
		class ShaderTableLayoutInstanceAS {
		private:
			using Instance = ShaderTableLayoutInstance;
			using Instances = std::vector<Instance>;
			using InstanceAS = ShaderTableLayoutInstanceAS;
			using GeometryAS = ShaderTableLayoutGeometryAS;
			friend class ShaderTableLayoutInstance;
			friend class ShaderTableLayoutGeometryAS;
			friend class ShaderTableLayout;
		public:
			ShaderTableLayoutInstanceAS()noexcept;

			ShaderTableLayoutInstanceAS(const ShaderTableLayoutInstanceAS& instance)noexcept;
			auto operator=(const ShaderTableLayoutInstanceAS& instance)noexcept->ShaderTableLayoutInstanceAS&;

			auto GetInstanceCount()const noexcept -> unsigned int;
			auto GetInstances()const noexcept -> const Instances&;
			void SetInstance(const Instance& instance)noexcept;

			auto GetRecordStride()const noexcept -> unsigned int;
			void SetRecordStride(unsigned int recordStride)noexcept;

			auto GetRecordCount()const noexcept -> unsigned int;

			auto RootInstanceAS()      noexcept ->      InstanceAS*;
			auto RootInstanceAS()const noexcept ->const InstanceAS*;

			void Show()noexcept;
		private:
			void Internal_SetRecordStride(unsigned int recordStride)noexcept;
			void Internal_SetRecordOffset(unsigned int recordOffset)noexcept;
		private:
			Instance* m_UpInstance = nullptr;
			Instances    m_DwInstances = {};

			unsigned int m_RecordStride = 0;
			unsigned int m_RecordCount = 0;
		};

		struct ShaderTableLayoutDesc {
			const void* pData = nullptr;
			unsigned int recordStride;
			unsigned int recordCount;
			unsigned int recordOffset;
		};
		struct ShaderTableLayoutBaseDesc {
			const void* pData = nullptr;
			unsigned int baseRecordCount;
			unsigned int baseRecordOffset;
		};

		class ShaderTableLayout
		{
		public:
			ShaderTableLayout(const ShaderTableLayoutInstanceAS& tlas)noexcept;

			auto GetRecordCount() const noexcept -> unsigned int;
			auto GetRecordStride()const noexcept -> unsigned int;

			auto RootInstanceAS()      noexcept ->      ShaderTableLayoutInstanceAS*;
			auto RootInstanceAS()const noexcept ->const ShaderTableLayoutInstanceAS*;

			auto GetInstanceASs()const noexcept -> const std::vector<std::unique_ptr<ShaderTableLayoutInstanceAS>>& {
				return m_InstanceASLayouts;
			}
			auto GetGeometryASs()const noexcept -> const std::vector<std::unique_ptr<ShaderTableLayoutGeometryAS>>& {
				return m_GeometryASLayouts;
			}
			
			//NAME: "Root/L1-GASInstance/Geometry"
			//NAME: "Root/L1-IASInstance/L2-GASInstance/Geometry"
			//NAME: "Root/L1-IASInstance/L2-IASInstance/L3-GASInstance/Geometry"
			auto GetDesc(std::string name)const noexcept -> ShaderTableLayoutDesc {
				return m_Descs.at(name);
			}
			//NAME: "GAS/Geometry"
			auto GetBaseDesc(std::string name)const noexcept -> ShaderTableLayoutBaseDesc {
				return m_BaseDescs.at(name);
			}
			//NAME: "GAS"
			auto GetBaseGeometryNames()const noexcept   -> std::vector<std::string> { return m_BaseGeometryNames; }
			//NAME: "GAS/Geometry"
			auto GetBaseGeometryASNames()const noexcept -> std::vector<std::string> { return m_BaseGeometryASNames; }
			//NAME: "Root/L1-GASInstance"
			//NAME: "Root/L1-IASInstance/L2-GASInstance"
			//NAME: "Root/L1-IASInstance/L2-IASInstance/L3-GASInstance"
			auto GetGeometryNames()const noexcept -> std::vector<std::string> { return m_GeometryNames; }
			//NAME: "Root/L1-GASInstance/Geometry"
			//NAME: "Root/L1-IASInstance/L2-GASInstance/Geometry"
			//NAME: "Root/L1-IASInstance/L2-IASInstance/L3-GASInstance/Geometry"
			auto GetInstanceNames()const noexcept -> std::vector<std::string> { return m_InstanceNames; }
			auto GetMaxTraversableDepth()const noexcept -> unsigned int { return m_MaxTraversalDepth; }
		private:
			static auto SplitFirstOf(const std::string& name)->std::pair<std::string, std::string> {
				auto pos = name.find("/");
				if ((pos != std::string::npos) && (pos != name.size() - 1)) {
					auto fir = std::string(std::begin(name), std::begin(name) + pos);
					auto sec = std::string(std::begin(name) + pos + 1, std::end(name));
					return { fir,sec };
				}
				else {
					return { name,{} };
				}
			}
			static auto SplitLastOf(const std::string& name)->std::pair<std::string, std::string> {
				auto pos = name.find_last_of("/");
				if ((pos != std::string::npos) && (pos != name.size() - 1)) {
					auto fir = std::string(std::begin(name), std::begin(name) + pos);
					auto sec = std::string(std::begin(name) + pos + 1, std::end(name));
					return { fir,sec };
				}
				else {
					return { name,{} };
				}
			}
			static void EnumerateImpl(const ShaderTableLayoutInstanceAS* pIAS,
				std::unordered_set<const ShaderTableLayoutGeometryAS*>& gasSet,
				std::unordered_set<const ShaderTableLayoutInstanceAS*>& iasSet
			)noexcept {
				if (!pIAS) {
					return;
				}
				for (auto& instance : pIAS->GetInstances()) {
					auto gas = instance.GetDwGeometryAS();
					auto ias = instance.GetDwInstanceAS();
					if (gas) { gasSet.insert(gas); }
					if (ias) {
						iasSet.insert(ias);
						EnumerateImpl(ias, gasSet, iasSet);
					}
				}
			}

		private:
			std::vector<std::unique_ptr<ShaderTableLayoutGeometryAS>>  m_GeometryASLayouts;
			std::vector<std::unique_ptr<ShaderTableLayoutInstanceAS>>  m_InstanceASLayouts;
			std::unordered_map<std::string, ShaderTableLayoutDesc>     m_Descs;
			std::unordered_map<std::string, ShaderTableLayoutBaseDesc> m_BaseDescs;
			std::vector<std::string> m_BaseGeometryNames;
			std::vector<std::string> m_BaseGeometryASNames;
			std::vector<std::string> m_InstanceNames;
			std::vector<std::string> m_GeometryNames;
			unsigned int m_MaxTraversalDepth = 1;
		};
	}
}
#endif
