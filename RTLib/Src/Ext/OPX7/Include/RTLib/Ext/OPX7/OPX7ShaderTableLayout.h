#ifndef RTLIB_EXT_OPX7_OPX7_SHADER_LAYOUT_H
#define RTLIB_EXT_OPX7_OPX7_SHADER_LAYOUT_H
#include <unordered_map>
#include <unordered_set>
#include <memory>
#include <vector>
namespace RTLib {
	namespace Ext
	{
		namespace OPX7
		{
			class OPX7ShaderTableLayout;
			class OPX7ShaderTableLayoutGeometryAS;
			class OPX7ShaderTableLayoutGeometry;
			class OPX7ShaderTableLayoutInstanceAS;
			class OPX7ShaderTableLayoutInstance;
			//GeometryAS
			class OPX7ShaderTableLayoutGeometry {
			public:
				friend class OPX7ShaderTableLayout;
				friend class OPX7ShaderTableLayoutGeometryAS;
				
				OPX7ShaderTableLayoutGeometry(unsigned int baseRecordCount = 0)noexcept;
				OPX7ShaderTableLayoutGeometry(const OPX7ShaderTableLayoutGeometry& geometry)noexcept;
				auto operator=(const OPX7ShaderTableLayoutGeometry& geometry)noexcept->OPX7ShaderTableLayoutGeometry&;

				auto GetBaseRecordOffset()const noexcept -> unsigned int;
				auto GetBaseRecordCount()const noexcept -> unsigned int;
			private:
				void Internal_SetUpGeometryAS(OPX7ShaderTableLayoutGeometryAS* gas)noexcept;
				void Internal_SetBaseRecordOffset(unsigned int offset)noexcept;
			private:
				OPX7ShaderTableLayoutGeometryAS* m_UpGeometryAS = nullptr;
				unsigned int m_BaseRecordCount  = 0;
				unsigned int m_BaseRecordOffset = 0;
			};
			class OPX7ShaderTableLayoutGeometryAS {
				friend class OPX7ShaderTableLayoutGeometry;
				friend class OPX7ShaderTableLayout;
			private:
				using Geometry = OPX7ShaderTableLayoutGeometry;
			public:
				OPX7ShaderTableLayoutGeometryAS()noexcept;
				OPX7ShaderTableLayoutGeometryAS(const OPX7ShaderTableLayoutGeometryAS& gas)noexcept;
				auto operator=(const OPX7ShaderTableLayoutGeometryAS& gas)noexcept->OPX7ShaderTableLayoutGeometryAS&;

				auto GetBaseRecordCount() const noexcept -> unsigned int;
				auto GetDwGeometries() const noexcept ->const std::vector<Geometry>&;
				void SetDwGeometry(const OPX7ShaderTableLayoutGeometry& geometry)noexcept;

				void Update()noexcept;
			private:
				std::vector<Geometry> m_DwGeometries = {};
				unsigned int m_BaseRecordCount = 0;
			};
			//InstanceAS
			class OPX7ShaderTableLayoutInstance {
			private:
				using InstanceAS = OPX7ShaderTableLayoutInstanceAS;
				using GeometryAS = OPX7ShaderTableLayoutGeometryAS;
				friend class OPX7ShaderTableLayoutInstanceAS;
				friend class OPX7ShaderTableLayoutGeometryAS;
				friend class OPX7ShaderTableLayout;
			public:
				OPX7ShaderTableLayoutInstance(GeometryAS* geometryAS)noexcept;
				OPX7ShaderTableLayoutInstance(InstanceAS* instanceAS)noexcept;

				OPX7ShaderTableLayoutInstance(const OPX7ShaderTableLayoutInstance& instance)noexcept;
				auto operator=(const OPX7ShaderTableLayoutInstance& instance)noexcept->OPX7ShaderTableLayoutInstance&;

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
				void Internal_SetUpInstanceAS(InstanceAS*    instanceAS)noexcept;
			private:
				InstanceAS*       m_UpInstanceAS = nullptr;
				InstanceAS*       m_DwInstanceAS = nullptr;
				const GeometryAS* m_DwGeometryAS = nullptr;

				unsigned int      m_RecordStride = 0;
				unsigned int      m_RecordCount  = 0;
				unsigned int      m_RecordOffset = 0;
			};
			class OPX7ShaderTableLayoutInstanceAS {
			private:
				using Instance   = OPX7ShaderTableLayoutInstance;
				using Instances  = std::vector<Instance>;
				using InstanceAS = OPX7ShaderTableLayoutInstanceAS;
				using GeometryAS = OPX7ShaderTableLayoutGeometryAS;
				friend class OPX7ShaderTableLayoutInstance;
				friend class OPX7ShaderTableLayoutGeometryAS;
				friend class OPX7ShaderTableLayout;
			public:
				OPX7ShaderTableLayoutInstanceAS()noexcept;

				OPX7ShaderTableLayoutInstanceAS(const OPX7ShaderTableLayoutInstanceAS& instance)noexcept;
				auto operator=(const OPX7ShaderTableLayoutInstanceAS& instance)noexcept->OPX7ShaderTableLayoutInstanceAS&;

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
				Instance*    m_UpInstance   = nullptr;
				Instances    m_DwInstances  = {};

				unsigned int m_RecordStride = 0;
				unsigned int m_RecordCount  = 0;
			};

			class OPX7ShaderTableLayout
			{
			public: 
				OPX7ShaderTableLayout(const OPX7ShaderTableLayoutInstanceAS& tlas)noexcept;

				auto GetRecordCount() const noexcept -> unsigned int;
				auto GetRecordStride()const noexcept -> unsigned int;

				auto RootInstanceAS()      noexcept ->      OPX7ShaderTableLayoutInstanceAS*;
				auto RootInstanceAS()const noexcept ->const OPX7ShaderTableLayoutInstanceAS*;
			private:
				static void EnumerateImpl(const OPX7ShaderTableLayoutInstanceAS*  pIAS,
					std::unordered_set<const OPX7ShaderTableLayoutGeometryAS*>& gasSet,
					std::unordered_set<const OPX7ShaderTableLayoutInstanceAS*>& iasSet)noexcept {
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
				std::vector<std::unique_ptr<OPX7ShaderTableLayoutGeometryAS>> m_GeometryASLayouts;
				std::vector<std::unique_ptr<OPX7ShaderTableLayoutInstanceAS>> m_InstanceASLayouts;
			};
			
			
		}
	}
}
#endif
