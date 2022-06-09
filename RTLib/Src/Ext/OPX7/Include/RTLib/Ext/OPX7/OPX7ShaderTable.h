#ifndef RTLIB_EXT_OPX7_OPX7_SHADER_TABLE_H
#define RTLIB_EXT_OPX7_OPX7_SHADER_TABLE_H
#include <RTLib/Core/BaseObject.h>
#include <RTLib/Ext/OPX7/OPX7Common.h>
#include <RTLib/Ext/OPX7/OPX7ShaderRecord.h>
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
				friend class OPX7Context;
				friend class OPX7ProgramGroup;
				RTLIB_CORE_TYPE_OBJECT_DECLARE_DERIVED_METHOD(OPX7ShaderTable, Core::BaseObject, RTLIB_TYPE_UUID_RTLIB_EXT_OPX7_OPX7_SHADER_TABLE);
			public:
				static auto Allocate(OPX7Context* context, const OPX7ShaderTableCreateDesc& desc)->OPX7ShaderTable*;
				virtual ~OPX7ShaderTable()noexcept;
				void Destroy()noexcept;
				//shaderTable
				auto GetBuffer()noexcept -> CUDA::CUDABuffer*;
				auto GetBufferSize()const noexcept -> size_t;
				//
				auto GetHostData()noexcept -> void*;
				auto GetHostData()const noexcept -> const void*;
				//Raygen
				auto GetHostRaygenRecordData()noexcept -> void*;
				auto GetHostRaygenRecordData()const noexcept -> const void*;
				template<typename T>
				auto GetHostRaygenRecordTypeData()noexcept -> OPX7::OPX7ShaderRecord<T>*;
				template<typename T>
				auto GetHostRaygenRecordTypeData()const noexcept -> const OPX7::OPX7ShaderRecord<T>*;
				template<typename T>
				void SetHostRaygenRecordTypeData(const OPX7::OPX7ShaderRecord<T>& record)noexcept;
				//Exception
				auto GetHostExceptionRecordData()noexcept -> void*;
				auto GetHostExceptionRecordData()const noexcept -> const void*;
				template<typename T>
				void SetHostExceptionRecordData(const OPX7::OPX7ShaderRecord<T>& record)noexcept;
				template<typename T>
				auto GetHostExceptionRecordTypeData()noexcept -> OPX7::OPX7ShaderRecord<T>*;
				template<typename T>
				auto GetHostExceptionRecordTypeData()const noexcept -> const OPX7::OPX7ShaderRecord<T>*;
				template<typename T>
				void SetHostExceptionRecordTypeData(const OPX7::OPX7ShaderRecord<T>& record)noexcept;
				//Miss
				auto GetHostMissRecordBase()noexcept -> void*;
				auto GetHostMissRecordBase()const noexcept -> const void*;
				auto GetHostMissRecordData(size_t index)noexcept -> void*;
				auto GetHostMissRecordData(size_t index)const noexcept -> const void*;
				template<typename T>
				auto GetHostMissRecordTypeData(size_t index)noexcept -> OPX7::OPX7ShaderRecord<T>*;
				template<typename T>
				auto GetHostMissRecordTypeData(size_t index)const noexcept -> const OPX7::OPX7ShaderRecord<T>*;
				template<typename T>
				void SetHostMissRecordTypeData(size_t index, const OPX7::OPX7ShaderRecord<T>& record)noexcept;
				//Hitgroup
				auto GetHostHitgroupRecordBase()noexcept -> void*;
				auto GetHostHitgroupRecordBase()const noexcept -> const void*;
				auto GetHostHitgroupRecordData(size_t index)noexcept -> void*;
				auto GetHostHitgroupRecordData(size_t index)const noexcept -> const void*;
				template<typename T>
				auto GetHostHitgroupRecordTypeData(size_t index)noexcept -> OPX7::OPX7ShaderRecord<T>*;
				template<typename T>
				auto GetHostHitgroupRecordTypeData(size_t index)const noexcept -> const OPX7::OPX7ShaderRecord<T>*;
				template<typename T>
				void SetHostHitgroupRecordTypeData(size_t index, const OPX7::OPX7ShaderRecord<T>& record)noexcept;
				//Callables
				auto GetHostCallablesRecordBase()noexcept -> void*;
				auto GetHostCallablesRecordBase()const noexcept -> const void*;
				auto GetHostCallablesRecordData(size_t index)noexcept -> void*;
				auto GetHostCallablesRecordData(size_t index)const noexcept -> const void*;
				template<typename T>
				auto GetHostCallablesRecordTypeData(size_t index)noexcept -> OPX7::OPX7ShaderRecord<T>*;
				template<typename T>
				auto GetHostCallablesRecordTypeData(size_t index)const noexcept -> const OPX7::OPX7ShaderRecord<T>*;
				template<typename T>
				void SetHostCallablesRecordTypeData(size_t index, const OPX7::OPX7ShaderRecord<T>& record)noexcept;
				//
				void Upload();
				void UploadRaygenRecord();
				void UploadExceptionRecord();
				void UploadMissRecord();
				void UploadHitgroupRecord();
				void UploadCallablesRecord();
				//Download
				void Download();
				void DownloadRaygenRecord();
				void DownloadExceptionRecord();
				void DownloadMissRecord();
				void DownloadHitgroupRecord();
				void DownloadCallablesRecord();
				//Raygen
				bool HasRaygenRecord()const noexcept;
				auto GetRaygenRecordSizeInBytes()const noexcept -> unsigned int;
				auto GetRaygenRecordOffsetInBytes()const noexcept -> size_t;
				//Exception
				bool HasExceptionRecord()const noexcept;
				auto GetExceptionRecordSizeInBytes()const noexcept -> unsigned int;
				auto GetExceptionRecordOffsetInBytes()const noexcept -> size_t;
				//Miss
				bool HasMissRecord()const noexcept;
				auto GetMissRecordOffsetInBytes()const noexcept -> size_t;
				auto GetMissRecordSizeInBytes()const noexcept   -> size_t;
				auto GetMissRecordStrideInBytes()const noexcept -> unsigned int;
				auto GetMissRecordCount()const noexcept -> unsigned int;
				//Hitgroup
				bool HasHitgroupRecord()const noexcept;
				auto GetHitgroupRecordOffsetInBytes()const noexcept -> size_t;
				auto GetHitgroupRecordSizeInBytes()const noexcept -> size_t;
				auto GetHitgroupRecordStrideInBytes()const noexcept -> unsigned int;
				auto GetHitgroupRecordCount()const noexcept -> unsigned int;
				//Callables
				bool HasCallablesRecord()const noexcept;
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
			template<typename T>
			inline auto OPX7ShaderTable::GetHostCallablesRecordTypeData(size_t index) noexcept -> OPX7::OPX7ShaderRecord<T>*
			{
				if (sizeof(OPX7::OPX7ShaderRecord<T>) != GetCallablesRecordSizeInBytes()) { return nullptr; }
				auto data = GetHostCallablesRecordData(index);
				return static_cast<OPX7::OPX7ShaderRecord<T>*>(data);
			}
			template<typename T>
			inline auto OPX7ShaderTable::GetHostCallablesRecordTypeData(size_t index) const noexcept -> const OPX7::OPX7ShaderRecord<T>*
			{
				if (sizeof(OPX7::OPX7ShaderRecord<T>) != GetCallablesRecordSizeInBytes()) { return nullptr; }
				auto data = GetHostCallablesRecordData(index);
				return static_cast<const OPX7::OPX7ShaderRecord<T>*>(data);
			}
			template<typename T>
			inline void OPX7ShaderTable::SetHostCallablesRecordTypeData(size_t index, const OPX7::OPX7ShaderRecord<T>& record) noexcept
			{
				auto* pHostData = GetHostCallablesRecordTypeData<T>(index);
				if (!pHostData) { return; }
				*pHostData = record;
			}
			template<typename T>
			inline auto OPX7ShaderTable::GetHostRaygenRecordTypeData() noexcept -> OPX7::OPX7ShaderRecord<T>*
			{
				if (sizeof(OPX7::OPX7ShaderRecord<T>) != GetRaygenRecordSizeInBytes()) { return nullptr; }
				auto data = GetHostRaygenRecordData();
				return static_cast< OPX7::OPX7ShaderRecord<T>*>(data);
			}
			template<typename T>
			inline auto OPX7ShaderTable::GetHostRaygenRecordTypeData() const noexcept -> const OPX7::OPX7ShaderRecord<T>*
			{
				if (sizeof(OPX7::OPX7ShaderRecord<T>) != GetRaygenRecordSizeInBytes()) { return nullptr; }
				auto data = GetHostRaygenRecordData();
				return static_cast<const OPX7::OPX7ShaderRecord<T>*>(data);
			}
			template<typename T>
			inline void OPX7ShaderTable::SetHostRaygenRecordTypeData(const OPX7::OPX7ShaderRecord<T>& record) noexcept
			{
				auto* pHostData = GetHostRaygenRecordTypeData<T>();
				if (!pHostData) { return; }
				*pHostData = record;
			}
			template<typename T>
			inline void OPX7ShaderTable::SetHostExceptionRecordData(const OPX7::OPX7ShaderRecord<T>& record) noexcept
			{
				auto* pHostData = GetHostExceptionRecordTypeData<T>();
				if (!pHostData) { return; }
				*pHostData = record;
			}
			template<typename T>
			inline auto OPX7ShaderTable::GetHostExceptionRecordTypeData() noexcept -> OPX7::OPX7ShaderRecord<T>*
			{
				if (sizeof(OPX7::OPX7ShaderRecord<T>) != GetExcepitonRecordSizeInBytes()) { return nullptr; }
				auto data = GetHostExceptionRecordData();
				return static_cast< OPX7::OPX7ShaderRecord<T>*>(data);
			}
			template<typename T>
			inline auto OPX7ShaderTable::GetHostExceptionRecordTypeData() const noexcept -> const OPX7::OPX7ShaderRecord<T>*
			{
				if (sizeof(OPX7::OPX7ShaderRecord<T>) != GetExcepitonRecordSizeInBytes()) { return nullptr; }
				auto data = GetHostExceptionRecordData();
				return static_cast<const OPX7::OPX7ShaderRecord<T>*>(data);
			}
			template<typename T>
			inline void OPX7ShaderTable::SetHostExceptionRecordTypeData(const OPX7::OPX7ShaderRecord<T>& record) noexcept
			{
				auto* pHostData = GetHostExceptionRecordTypeData<T>();
				if (!pHostData) { return; }
				*pHostData = record;
			}
			template<typename T>
			inline auto OPX7ShaderTable::GetHostMissRecordTypeData(size_t index) noexcept -> OPX7::OPX7ShaderRecord<T>*
			{
				if (sizeof(OPX7::OPX7ShaderRecord<T>) != GetMissRecordStrideInBytes()) { return nullptr; }
				auto data = GetHostMissRecordData(index);
				return static_cast< OPX7::OPX7ShaderRecord<T>*>(data);
			}
			template<typename T>
			inline auto OPX7ShaderTable::GetHostMissRecordTypeData(size_t index) const noexcept -> const OPX7::OPX7ShaderRecord<T>*
			{
				if (sizeof(OPX7::OPX7ShaderRecord<T>) != GetMissRecordStrideInBytes()) { return nullptr; }
				auto data = GetHostMissRecordData(index);
				return static_cast<const OPX7::OPX7ShaderRecord<T>*>(data);
			}
			template<typename T>
			inline void OPX7ShaderTable::SetHostMissRecordTypeData(size_t index, const OPX7::OPX7ShaderRecord<T>& record) noexcept
			{
				auto* pHostData = GetHostMissRecordTypeData<T>(index);
				if (!pHostData) { return; }
				*pHostData = record;
			}
			template<typename T>
			inline auto OPX7ShaderTable::GetHostHitgroupRecordTypeData(size_t index) noexcept -> OPX7::OPX7ShaderRecord<T>*
			{
				if (sizeof(OPX7::OPX7ShaderRecord<T>) != GetHitgroupRecordStrideInBytes()) { return nullptr; }
				auto data = GetHostHitgroupRecordData(index);
				return static_cast< OPX7::OPX7ShaderRecord<T>*>(data);
			}
			template<typename T>
			inline auto OPX7ShaderTable::GetHostHitgroupRecordTypeData(size_t index) const noexcept -> const OPX7::OPX7ShaderRecord<T>*
			{
				if (sizeof(OPX7::OPX7ShaderRecord<T>) != GetHitgroupRecordStrideInBytes()) { return nullptr; }
				auto data = GetHostHitgroupRecordData(index);
				return static_cast<const OPX7::OPX7ShaderRecord<T>*>(data);
			}
			template<typename T>
			inline void OPX7ShaderTable::SetHostHitgroupRecordTypeData(size_t index, const OPX7::OPX7ShaderRecord<T>& record) noexcept
			{
				auto* pHostData = GetHostHitgroupRecordTypeData<T>(index);
				if (!pHostData) { return; }
				*pHostData = record;
			}
}
	}
}
#endif
