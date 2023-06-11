#ifndef TEST1_SHADER_BINDING_TABLE__H
#define TEST1_SHADER_BINDING_TABLE__H
#include <OptiXToolkit/OptixMemory/SyncRecord.h>
#include <OptiXToolkit/OptixMemory/Record.h>
#include <Test1_ProgramGroup.h>
#include <optix.h>
#include <cuda.h>
#include <memory>
namespace Test1
{
	struct ProgramGroup;
	struct ShaderRecord
	{
		ShaderRecord(){}

		virtual ~ShaderRecord() {}

		virtual auto at(size_t idx) -> void* = 0;

		virtual auto get_stride_in_bytes() const->size_t = 0;

		virtual auto get_count() const->size_t = 0;

		virtual void resize(size_t size) = 0;

		virtual void pack_header(ProgramGroup* pg) = 0;

		virtual void pack_header(size_t index, ProgramGroup* pg) = 0;

		virtual auto get_device_ptr() noexcept -> CUdeviceptr = 0;

		virtual void copy_to_device() = 0;

		virtual void copy_to_device_async(CUstream stream) = 0;

	};

	template<typename T>
	struct TypeShaderRecord: public ShaderRecord
	{
		TypeShaderRecord(size_t size):data{size}{}

		virtual ~TypeShaderRecord(){}

		virtual auto at(size_t idx) -> void* override;

		virtual auto get_stride_in_bytes() const->size_t override;

		virtual auto get_count() const->size_t override;

		virtual void resize(size_t size) override;

		virtual void pack_header(ProgramGroup* pg) override;

		virtual void pack_header(size_t index, ProgramGroup* pg) override;

		virtual auto get_device_ptr() noexcept -> CUdeviceptr override;

		virtual void copy_to_device() override;

		virtual void copy_to_device_async(CUstream stream) override;

		otk::SyncRecord<T> data;
	};

	struct ShaderBindingTable
	{
		std::shared_ptr<ShaderRecord> raygen   = nullptr;
		std::shared_ptr<ShaderRecord> miss     = nullptr;
		std::shared_ptr<ShaderRecord> hitgroup = nullptr;
		std::shared_ptr<ShaderRecord> callables = nullptr;

		auto get_opx7_shader_binding_table() const noexcept -> OptixShaderBindingTable
		{
			OptixShaderBindingTable sbt = {};
			if (raygen)   { 
				sbt.raygenRecord = raygen->get_device_ptr(); 
			}
			if (miss)     {
				sbt.missRecordBase          = miss->get_device_ptr();
				sbt.missRecordStrideInBytes = miss->get_stride_in_bytes();
				sbt.missRecordCount         = miss->get_count();
			}
			if (hitgroup) { 
				sbt.hitgroupRecordBase          = hitgroup->get_device_ptr();
				sbt.hitgroupRecordStrideInBytes = hitgroup->get_stride_in_bytes();
				sbt.hitgroupRecordCount         = hitgroup->get_count();
			}
			if (callables) {
				sbt.callablesRecordBase          = callables->get_device_ptr();
				sbt.callablesRecordStrideInBytes = callables->get_stride_in_bytes();
				sbt.callablesRecordCount         = callables->get_count();
			}
			return sbt;
		}
	};

	template<typename T>
	inline auto TypeShaderRecord<T>::at(size_t idx) -> void* 
	{
		return &data[idx];
	}

	template<typename T>
	inline auto TypeShaderRecord<T>::get_stride_in_bytes() const -> size_t 
	{
		return sizeof(otk::Record<T>);
	}

	template<typename T>
	inline auto TypeShaderRecord<T>::get_count() const -> size_t 
	{
		return data.size();
	}

	template<typename T>
	inline void TypeShaderRecord<T>::resize(size_t size)
	{
		data.resize(size);
	}

	template<typename T>
	inline void TypeShaderRecord<T>::pack_header(ProgramGroup* pg)
	{
		if (!pg) { return; }
		data.packHeader(pg->get_opx7_program_group());

	}

	template<typename T>
	inline void TypeShaderRecord<T>::pack_header(size_t index, ProgramGroup* pg)
	{
		if (!pg) { return; }
		data.packHeader(index,pg->get_opx7_program_group());
	}

	template<typename T>
	inline auto TypeShaderRecord<T>::get_device_ptr()  noexcept -> CUdeviceptr 
	{
		return CUdeviceptr(data);
	}

	template<typename T>
	inline void TypeShaderRecord<T>::copy_to_device()
	{
		data.copyToDevice();
	}

	template<typename T>
	inline void TypeShaderRecord<T>::copy_to_device_async(CUstream stream)
	{
		data.copyToDeviceAsync(stream);
	}
}
#endif
