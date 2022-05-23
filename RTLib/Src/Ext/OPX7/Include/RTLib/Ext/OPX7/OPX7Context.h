#ifndef RTLIB_EXT_OPX7_OPX7_CONTEXT_H
#define RTLIB_EXT_OPX7_OPX7_CONTEXT_H
#include <RTLib/Ext/CUDA/CUDAContext.h>
#include <RTLib/Ext/OPX7/OPX7Common.h>
#include <RTLib/Ext/OPX7/UuidDefinitions.h>
#include <vector>
#include <memory>
namespace RTLib
{
	namespace Ext
	{
		namespace OPX7
		{
			class OPX7Module;
			class OPX7Pipeline;
			class OPX7ProgramGroup;
			class OPX7ShaderTable;
			class OPX7AccelerationStructure;
			class OPX7AccelerationStructureInstance;
			class OPX7Context : public CUDA::CUDAContext
			{
				friend class OPX7Module;
				friend class OPX7Pipeline;
				friend class OPX7ProgramGroup;
				friend class OPX7ShaderTable;
				friend class OPX7Pipeline;
				friend class OPX7AccelerationStructure;
				friend class OPX7AccelerationStructureInstance;
			public:
				RTLIB_CORE_TYPE_OBJECT_DECLARE_DERIVED_METHOD(OPX7Context, CUDA::CUDAContext, RTLIB_TYPE_UUID_RTLIB_EXT_OPX7_OPX7_CONTEXT);
				OPX7Context(const OPX7ContextCreateDesc& desc) noexcept;
				virtual ~OPX7Context()noexcept;

				virtual bool Initialize() override;
				virtual void Terminate()  override;

				auto CreateOPXModule(const OPX7ModuleCreateDesc& desc) ->OPX7Module*;
				auto CreateOPXProgramGroups(const std::vector<OPX7ProgramGroupCreateDesc>& desc, const OPX7ProgramGroupOptions& options = {})->std::vector<OPX7ProgramGroup*>;
				auto CreateOPXProgramGroup(const OPX7ProgramGroupCreateDesc& desc, const OPX7ProgramGroupOptions& options = {})->OPX7ProgramGroup*;
				auto CreateOPXPipeline(const OPX7PipelineCreateDesc& desc)->OPX7Pipeline*;
				auto CreateOPXShaderTable(const OPX7ShaderTableCreateDesc& desc)->OPX7ShaderTable*;
			private:
				auto GetOptixDeviceContext() noexcept -> OptixDeviceContext;
				static void Launch(
					OPX7Pipeline* pipeline,
					CUDA::CUDAStream* stream,
					CUDA::CUDABufferView paramsBufferView,
					OPX7ShaderTable* shaderTable,
					unsigned int          width,
					unsigned int          height,
					unsigned int          depth
				);
				static void Launch(
					OPX7Pipeline*         pipeline,
					CUDA::CUDABufferView  paramsBufferView,
					OPX7ShaderTable*      shaderTable,
					unsigned int          width,
					unsigned int          height,
					unsigned int          depth
				);
			private:
				struct Impl;
				std::unique_ptr<Impl> m_Impl;
			};
		}
	}
}
#endif
