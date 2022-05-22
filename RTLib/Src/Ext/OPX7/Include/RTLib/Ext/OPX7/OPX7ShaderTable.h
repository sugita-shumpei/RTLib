#ifndef RTLIB_EXT_OPX7_OPX7_SHADER_TABLE_H
#define RTLIB_EXT_OPX7_OPX7_SHADER_TABLE_H
#include <RTLib/Core/BaseObject.h>
#include <RTLib/Ext/OPX7/OPX7Common.h>
#include <RTLib/Ext/OPX7/UuidDefinitions.h>
#include <RTLib/Ext/CUDA/CUDABuffer.h>
#include <memory>
namespace RTLib {
	namespace Ext
	{
		namespace OPX7
		{
			class OPX7Context;
			class OPX7ProgramGroup;
			class OPX7ShaderTable : public RTLib::Core::BaseObject
			{
			private:
				friend class OPX7ProgramGroup;
				RTLIB_CORE_TYPE_OBJECT_DECLARE_DERIVED_METHOD(OPX7ShaderTable, Core::BaseObject, RTLIB_TYPE_UUID_RTLIB_EXT_OPX7_OPX7_SHADER_TABLE);
			public:
				static auto New(OPX7Context* context, const OPX7ShaderTableCreateDesc& desc)->OPX7ShaderTable*;
				virtual ~OPX7ShaderTable()noexcept;
				//shaderTable
				auto GetBufferData()noexcept -> CUDA::CUDABuffer*;
				auto GetBufferSize()const noexcept -> size_t;
				auto GetRaygenRecordSizeInBytes()const noexcept -> unsigned int;
				auto GetRaygenRecordOffsetInBytes()const noexcept -> size_t;
				auto GetExceptionRecordSizeInBytes()const noexcept -> unsigned int;
				auto GetExceptionRecordOffsetInBytes()const noexcept -> size_t;
				auto GetMissRecordOffsetInBytes()const noexcept -> size_t;
				auto GetMissRecordSizeInBytes()const noexcept   -> size_t;
				auto GetMissRecordStrideInBytes()const noexcept -> unsigned int;
				auto GetMissRecordCount()const noexcept -> unsigned int;
				auto GetHitgroupRecordOffsetInBytes()const noexcept -> size_t;
				auto GetHitgroupRecordSizeInBytes()const noexcept -> size_t;
				auto GetHitgroupRecordStrideInBytes()const noexcept -> unsigned int;
				auto GetHitgroupRecordCount()const noexcept -> unsigned int;
				auto GetCallablesRecordOffsetInBytes()const noexcept -> size_t;
				auto GetCallablesRecordSizeInBytes()const noexcept   -> size_t;
				auto GetCallablesRecordStrideInBytes()const noexcept -> unsigned int;
				auto GetCallablesRecordCount()const noexcept -> unsigned int;
			private:
				OPX7ShaderTable(OPX7Context* context, const OPX7ShaderTableCreateDesc& desc)noexcept;
				auto GetOptixShaderBindingTable()const noexcept ->const OptixShaderBindingTable&;
			private:
				struct Impl;
				std::unique_ptr<Impl> m_Impl;
			};
		}
	}
}
#endif
