#include <RTLib/Ext/OPX7/OPX7ShaderTable.h>
#include <RTLib/Ext/OPX7/OPX7Context.h>
#include <RTLib/Ext/CUDA/CUDABuffer.h>
#include <RTLib/Ext/CUDA/CUDACommon.h>
#include <RTLib/Ext/CUDA/CUDANatives.h>
#include <RTLib/Ext/CUDA/CUDAExceptions.h>
#include <cuda.h>
#include <cassert>
#include <iostream>
struct RTLib::Ext::OPX7::OPX7ShaderTable::Impl
{
	Impl(OPX7Context* ctx, const OPX7ShaderTableCreateDesc& desc)noexcept
	{
		context = ctx;
		shaderBindingTable.missRecordStrideInBytes      = desc.missRecordStrideInBytes;
		shaderBindingTable.missRecordCount              = desc.missRecordCount;
		shaderBindingTable.hitgroupRecordStrideInBytes  = desc.hitgroupRecordStrideInBytes;
		shaderBindingTable.hitgroupRecordCount          = desc.hitgroupRecordCount;
		shaderBindingTable.callablesRecordStrideInBytes = desc.callablesRecordStrideInBytes;
		shaderBindingTable.callablesRecordCount         = desc.callablesRecordCount;
		raygenRecordSizeInBytes    = desc.raygenRecordSizeInBytes;
		exceptionRecordSizeInBytes = desc.exceptionRecordSizeInBytes;
		missRecordSizeInBytes      = static_cast<size_t>(shaderBindingTable.missRecordStrideInBytes)      * shaderBindingTable.missRecordCount;
		hitgroupRecordSizeInBytes  = static_cast<size_t>(shaderBindingTable.hitgroupRecordStrideInBytes)  * shaderBindingTable.hitgroupRecordCount;
		callablesRecordSizeInBytes = static_cast<size_t>(shaderBindingTable.callablesRecordStrideInBytes) * shaderBindingTable.callablesRecordCount;
		shaderBindingTableSizeInBytes  = raygenRecordSizeInBytes;
		if (exceptionRecordSizeInBytes > 0) {
			exceptionRecordOffsetInBytes = shaderBindingTableSizeInBytes;
		}
		shaderBindingTableSizeInBytes += exceptionRecordSizeInBytes;
		if (missRecordSizeInBytes      > 0) {
			missRecordOffsetInBytes      = shaderBindingTableSizeInBytes;
		}
		shaderBindingTableSizeInBytes += missRecordSizeInBytes;
		if (hitgroupRecordSizeInBytes > 0) {
			hitgroupRecordOffsetInBytes  = shaderBindingTableSizeInBytes;
		}
		shaderBindingTableSizeInBytes += hitgroupRecordSizeInBytes;
		if (callablesRecordSizeInBytes > 0) {
			callablesRecordOffsetInBytes = shaderBindingTableSizeInBytes;
		}
		shaderBindingTableSizeInBytes += callablesRecordSizeInBytes;
	}
	void Allocate() {
		auto buffDesc        = RTLib::Ext::CUDA::CUDABufferCreateDesc();
		buffDesc.sizeInBytes = shaderBindingTableSizeInBytes;
		buffDesc.flags	     = CUDA::CUDAMemoryFlags::eDefault;
		buffer               = std::unique_ptr<CUDA::CUDABuffer>(context->CreateBuffer(buffDesc));
		RTLIB_EXT_CUDA_THROW_IF_FAILED(cuMemHostAlloc(&pHostData, shaderBindingTableSizeInBytes, 0));
		auto baseAddress = CUDA::CUDANatives::GetCUdeviceptr(buffer.get());
		if (raygenRecordSizeInBytes > 0) {
			shaderBindingTable.raygenRecord = baseAddress;
		}
		if (exceptionRecordSizeInBytes > 0) {
			shaderBindingTable.exceptionRecord = baseAddress+ exceptionRecordOffsetInBytes;
		}
		if (missRecordSizeInBytes > 0) {
			shaderBindingTable.missRecordBase = baseAddress + missRecordOffsetInBytes;
		}
		if (hitgroupRecordSizeInBytes > 0) {
			shaderBindingTable.hitgroupRecordBase = baseAddress + hitgroupRecordOffsetInBytes;
		}
		if (callablesRecordSizeInBytes > 0) {
			shaderBindingTable.callablesRecordBase = baseAddress + callablesRecordOffsetInBytes;
		}
	}
	OPX7Context*					  context                      = nullptr;
	std::unique_ptr<CUDA::CUDABuffer> buffer                       = nullptr;
	OptixShaderBindingTable           shaderBindingTable           = {};
	void*                             pHostData                    = nullptr;
	size_t                            shaderBindingTableSizeInBytes= 0;
	size_t                            raygenRecordSizeInBytes      = 0;
	size_t                            exceptionRecordSizeInBytes   = 0;
	size_t                            exceptionRecordOffsetInBytes = 0;
	size_t                            missRecordSizeInBytes        = 0;
	size_t                            missRecordOffsetInBytes      = 0;
	size_t                            hitgroupRecordSizeInBytes    = 0;
	size_t                            hitgroupRecordOffsetInBytes  = 0;
	size_t                            callablesRecordSizeInBytes   = 0;
	size_t                            callablesRecordOffsetInBytes = 0;
};

auto RTLib::Ext::OPX7::OPX7ShaderTable::Allocate(OPX7Context* context, const OPX7ShaderTableCreateDesc& desc) -> OPX7ShaderTable*
{
	auto shaderTable = new OPX7ShaderTable(context, desc);
	shaderTable->m_Impl->Allocate();
	return shaderTable;
}

RTLib::Ext::OPX7::OPX7ShaderTable::~OPX7ShaderTable() noexcept
{
	m_Impl.reset();
}

void RTLib::Ext::OPX7::OPX7ShaderTable::Destroy() noexcept
{
	if (!m_Impl) { return; }
	if (!m_Impl->buffer) { return; }
	m_Impl->buffer->Destroy();
	m_Impl->buffer.reset();
	if (m_Impl->pHostData) {
		try {
			RTLIB_EXT_CUDA_THROW_IF_FAILED(cuMemFreeHost(m_Impl->pHostData));
		}
		catch (CUDA::CUDAException& err) {
			std::cerr << err.what() << std::endl;
		}
	}
	m_Impl->pHostData = nullptr;
}

auto RTLib::Ext::OPX7::OPX7ShaderTable::GetBuffer() noexcept -> CUDA::CUDABuffer*
{
	assert(m_Impl != nullptr);
	return m_Impl->buffer.get();
}

auto RTLib::Ext::OPX7::OPX7ShaderTable::GetBufferSize() const noexcept -> size_t
{
	assert(m_Impl != nullptr);
	return m_Impl->shaderBindingTableSizeInBytes;
}

auto RTLib::Ext::OPX7::OPX7ShaderTable::GetHostData() noexcept -> void*
{
	return m_Impl?m_Impl->pHostData:nullptr;
}

auto RTLib::Ext::OPX7::OPX7ShaderTable::GetHostData() const noexcept -> const void*
{
	return m_Impl ? m_Impl->pHostData : nullptr;
}

auto RTLib::Ext::OPX7::OPX7ShaderTable::GetHostRaygenRecordData() noexcept -> void*
{
	return ((char*)GetHostData()) + GetRaygenRecordOffsetInBytes();
}

auto RTLib::Ext::OPX7::OPX7ShaderTable::GetHostRaygenRecordData() const noexcept -> const void*
{
	if (!HasRaygenRecord()) { return nullptr; }
	return ((char*)GetHostData()) + GetRaygenRecordOffsetInBytes();
}

auto RTLib::Ext::OPX7::OPX7ShaderTable::GetHostExceptionRecordData() noexcept -> void*
{
	if (!HasExceptionRecord()) { return nullptr; }
	return ((char*)GetHostData()) + GetExceptionRecordOffsetInBytes();
}

auto RTLib::Ext::OPX7::OPX7ShaderTable::GetHostExceptionRecordData() const noexcept -> const void*
{
	if (!HasExceptionRecord()) { return nullptr; }
	return ((char*)GetHostData()) + GetExceptionRecordOffsetInBytes();
}

auto RTLib::Ext::OPX7::OPX7ShaderTable::GetHostMissRecordBase() noexcept -> void*
{
	if (!HasMissRecord()) { return nullptr; }
	return ((char*)GetHostData()) + GetMissRecordOffsetInBytes();
}

auto RTLib::Ext::OPX7::OPX7ShaderTable::GetHostMissRecordBase() const noexcept -> const void*
{
	if (!HasMissRecord()) { return nullptr; }
	return ((char*)GetHostData()) + GetMissRecordOffsetInBytes();
}

auto RTLib::Ext::OPX7::OPX7ShaderTable::GetHostMissRecordData(size_t index) noexcept -> void*
{
	auto baseData = GetHostMissRecordBase();
	if (!baseData) { return nullptr; }
	if (index >= GetMissRecordCount()) { return nullptr; }
	return ((char*)baseData)+index * GetMissRecordSizeInBytes();
}

auto RTLib::Ext::OPX7::OPX7ShaderTable::GetHostMissRecordData(size_t index) const noexcept -> const void*
{
	auto baseData = GetHostMissRecordBase();
	if (!baseData) { return nullptr; }
	if (index >= GetMissRecordCount()) { return nullptr; }
	return ((char*)baseData) + index * GetMissRecordSizeInBytes();
}

auto RTLib::Ext::OPX7::OPX7ShaderTable::GetHostHitgroupRecordBase() noexcept -> void*
{
	if (!HasHitgroupRecord()) { return nullptr; }
	return ((char*)GetHostData()) + GetHitgroupRecordOffsetInBytes();
}

auto RTLib::Ext::OPX7::OPX7ShaderTable::GetHostHitgroupRecordBase() const noexcept -> const void*
{
	if (!HasHitgroupRecord()) { return nullptr; }
	return ((char*)GetHostData()) + GetHitgroupRecordOffsetInBytes();
}

auto RTLib::Ext::OPX7::OPX7ShaderTable::GetHostHitgroupRecordData(size_t index) noexcept -> void*
{
	auto baseData = GetHostHitgroupRecordBase();
	if (!baseData) { return nullptr; }
	if (index >= GetHitgroupRecordCount()) { return nullptr; }
	return ((char*)baseData) + index * GetHitgroupRecordSizeInBytes();
}

auto RTLib::Ext::OPX7::OPX7ShaderTable::GetHostHitgroupRecordData(size_t index) const noexcept -> const void*
{
	auto baseData = GetHostHitgroupRecordBase();
	if (!baseData) { return nullptr; }
	if (index >= GetHitgroupRecordCount()) { return nullptr; }
	return ((char*)baseData) + index * GetHitgroupRecordSizeInBytes();
}

auto RTLib::Ext::OPX7::OPX7ShaderTable::GetHostCallablesRecordBase() noexcept -> void*
{
	if (!HasCallablesRecord()) { return nullptr; }
	return ((char*)GetHostData()) + GetCallablesRecordOffsetInBytes();
}

auto RTLib::Ext::OPX7::OPX7ShaderTable::GetHostCallablesRecordBase() const noexcept -> const void*
{
	if (!HasCallablesRecord()) { return nullptr; }
	return ((char*)GetHostData()) + GetCallablesRecordOffsetInBytes();
}

auto RTLib::Ext::OPX7::OPX7ShaderTable::GetHostCallablesRecordData(size_t index) noexcept -> void*
{
	auto baseData = GetHostCallablesRecordBase();
	if (!baseData) { return nullptr; }
	if (index >= GetCallablesRecordCount()) { return nullptr; }
	return ((char*)baseData) + index * GetCallablesRecordSizeInBytes();
}

auto RTLib::Ext::OPX7::OPX7ShaderTable::GetHostCallablesRecordData(size_t index) const noexcept -> const void*
{
	auto baseData = GetHostCallablesRecordBase();
	if (!baseData) { return nullptr; }
	if (index >= GetCallablesRecordCount()) { return nullptr; }
	return ((char*)baseData) + index * GetCallablesRecordSizeInBytes();
}

void RTLib::Ext::OPX7::OPX7ShaderTable::Upload()
{
	assert(m_Impl != nullptr);
	assert(m_Impl->context->CopyMemoryToBuffer(m_Impl->buffer.get(), {
		{m_Impl->pHostData, 0, m_Impl->shaderBindingTableSizeInBytes}
	}));
}

void RTLib::Ext::OPX7::OPX7ShaderTable::UploadRaygenRecord()
{
	assert(m_Impl != nullptr);
	if (m_Impl->raygenRecordSizeInBytes == 0) { return; }
	auto offsetSizeInBytes = GetRaygenRecordOffsetInBytes();
	auto sizeInBytes = GetRaygenRecordSizeInBytes();
	assert(m_Impl->context->CopyMemoryToBuffer(m_Impl->buffer.get(), {
		{(char*)m_Impl->pHostData, offsetSizeInBytes,sizeInBytes}
	}));
}

void RTLib::Ext::OPX7::OPX7ShaderTable::UploadExceptionRecord()
{
	assert(m_Impl != nullptr);
	if (m_Impl->exceptionRecordOffsetInBytes == 0) { return; }
	auto offsetSizeInBytes = GetExceptionRecordOffsetInBytes();
	auto sizeInBytes = GetExceptionRecordSizeInBytes();
	assert(m_Impl->context->CopyMemoryToBuffer(m_Impl->buffer.get(), {
		{(char*)m_Impl->pHostData, offsetSizeInBytes,sizeInBytes}
	}));
}

void RTLib::Ext::OPX7::OPX7ShaderTable::UploadMissRecord()
{
	assert(m_Impl != nullptr);
	if (m_Impl->exceptionRecordOffsetInBytes == 0) { return; }
	auto offsetSizeInBytes = GetMissRecordOffsetInBytes();
	auto sizeInBytes = GetMissRecordSizeInBytes();
	assert(m_Impl->context->CopyMemoryToBuffer(m_Impl->buffer.get(), {
		{(char*)m_Impl->pHostData, offsetSizeInBytes,sizeInBytes}
	}));
}

void RTLib::Ext::OPX7::OPX7ShaderTable::UploadHitgroupRecord()
{
	assert(m_Impl != nullptr);
	if (m_Impl->exceptionRecordOffsetInBytes == 0) { return; }
	auto offsetSizeInBytes = GetHitgroupRecordOffsetInBytes();
	auto sizeInBytes = GetHitgroupRecordSizeInBytes();
	assert(m_Impl->context->CopyMemoryToBuffer(m_Impl->buffer.get(), {
		{(char*)m_Impl->pHostData, offsetSizeInBytes,sizeInBytes}
	}));
}

void RTLib::Ext::OPX7::OPX7ShaderTable::UploadCallablesRecord()
{	
	assert(m_Impl != nullptr);
	if (m_Impl->exceptionRecordOffsetInBytes == 0) { return; }
	auto offsetSizeInBytes = GetCallablesRecordOffsetInBytes();
	auto sizeInBytes = GetCallablesRecordSizeInBytes();
	assert(m_Impl->context->CopyMemoryToBuffer(m_Impl->buffer.get(), {
		{(char*)m_Impl->pHostData, offsetSizeInBytes,sizeInBytes}
	}));
}

void RTLib::Ext::OPX7::OPX7ShaderTable::Download()
{
	assert(m_Impl != nullptr);
	assert(m_Impl->context->CopyBufferToMemory(m_Impl->buffer.get(), {
		{m_Impl->pHostData, 0, m_Impl->shaderBindingTableSizeInBytes}
		}));
}

void RTLib::Ext::OPX7::OPX7ShaderTable::DownloadRaygenRecord()
{
	assert(m_Impl != nullptr);
	if (m_Impl->raygenRecordSizeInBytes == 0) { return; }
	auto offsetSizeInBytes = GetRaygenRecordOffsetInBytes();
	auto sizeInBytes = GetRaygenRecordSizeInBytes();
	assert(m_Impl->context->CopyBufferToMemory(m_Impl->buffer.get(), {
		{(char*)m_Impl->pHostData, offsetSizeInBytes,sizeInBytes}
		}));
}

void RTLib::Ext::OPX7::OPX7ShaderTable::DownloadExceptionRecord()
{
	assert(m_Impl != nullptr);
	if (m_Impl->exceptionRecordOffsetInBytes == 0) { return; }
	auto offsetSizeInBytes = GetExceptionRecordOffsetInBytes();
	auto sizeInBytes = GetExceptionRecordSizeInBytes();
	assert(m_Impl->context->CopyBufferToMemory(m_Impl->buffer.get(), {
		{(char*)m_Impl->pHostData, offsetSizeInBytes,sizeInBytes}
		}));
}

void RTLib::Ext::OPX7::OPX7ShaderTable::DownloadMissRecord()
{
	assert(m_Impl != nullptr);
	if (m_Impl->exceptionRecordOffsetInBytes == 0) { return; }
	auto offsetSizeInBytes = GetMissRecordOffsetInBytes();
	auto sizeInBytes = GetMissRecordSizeInBytes();
	assert(m_Impl->context->CopyBufferToMemory(m_Impl->buffer.get(), {
		{(char*)m_Impl->pHostData, offsetSizeInBytes,sizeInBytes}
		}));
}

void RTLib::Ext::OPX7::OPX7ShaderTable::DownloadHitgroupRecord()
{
	assert(m_Impl != nullptr);
	if (m_Impl->exceptionRecordOffsetInBytes == 0) { return; }
	auto offsetSizeInBytes = GetHitgroupRecordOffsetInBytes();
	auto sizeInBytes = GetHitgroupRecordSizeInBytes();
	assert(m_Impl->context->CopyBufferToMemory(m_Impl->buffer.get(), {
		{(char*)m_Impl->pHostData, offsetSizeInBytes,sizeInBytes}
		}));
}

void RTLib::Ext::OPX7::OPX7ShaderTable::DownloadCallablesRecord()
{
	assert(m_Impl != nullptr);
	if (m_Impl->exceptionRecordOffsetInBytes == 0) { return; }
	auto offsetSizeInBytes = GetCallablesRecordOffsetInBytes();
	auto sizeInBytes = GetCallablesRecordSizeInBytes();
	assert(m_Impl->context->CopyBufferToMemory(m_Impl->buffer.get(), {
		{(char*)m_Impl->pHostData, offsetSizeInBytes,sizeInBytes}
		}));
}

bool RTLib::Ext::OPX7::OPX7ShaderTable::HasRaygenRecord() const noexcept
{
	assert(m_Impl != nullptr);
	return m_Impl->raygenRecordSizeInBytes > 0;
}

auto RTLib::Ext::OPX7::OPX7ShaderTable::GetRaygenRecordSizeInBytes() const noexcept -> unsigned int
{
	assert(m_Impl != nullptr);
	return m_Impl->raygenRecordSizeInBytes;
}

auto RTLib::Ext::OPX7::OPX7ShaderTable::GetRaygenRecordOffsetInBytes() const noexcept -> size_t
{
	assert(m_Impl != nullptr);
	return 0;
}

bool RTLib::Ext::OPX7::OPX7ShaderTable::HasExceptionRecord() const noexcept
{
	assert(m_Impl != nullptr);
	return m_Impl->exceptionRecordSizeInBytes > 0;
}

auto RTLib::Ext::OPX7::OPX7ShaderTable::GetExceptionRecordSizeInBytes() const noexcept -> unsigned int
{
	assert(m_Impl != nullptr);
	return m_Impl->exceptionRecordSizeInBytes;
}

auto RTLib::Ext::OPX7::OPX7ShaderTable::GetExceptionRecordOffsetInBytes() const noexcept -> size_t
{
	assert(m_Impl != nullptr);
	return m_Impl->exceptionRecordOffsetInBytes;
}

bool RTLib::Ext::OPX7::OPX7ShaderTable::HasMissRecord() const noexcept
{
	assert(m_Impl != nullptr);
	return m_Impl->missRecordSizeInBytes > 0;
}

auto RTLib::Ext::OPX7::OPX7ShaderTable::GetMissRecordOffsetInBytes() const noexcept -> size_t
{
	assert(m_Impl != nullptr);
	return m_Impl->missRecordOffsetInBytes;
}

auto RTLib::Ext::OPX7::OPX7ShaderTable::GetMissRecordSizeInBytes() const noexcept -> size_t
{
	assert(m_Impl != nullptr);
	return m_Impl->missRecordSizeInBytes;
}

auto RTLib::Ext::OPX7::OPX7ShaderTable::GetMissRecordStrideInBytes() const noexcept -> unsigned int
{
	assert(m_Impl != nullptr);
	return m_Impl->shaderBindingTable.missRecordStrideInBytes;
}

auto RTLib::Ext::OPX7::OPX7ShaderTable::GetMissRecordCount() const noexcept -> unsigned int
{
	assert(m_Impl != nullptr);
	return m_Impl->shaderBindingTable.missRecordCount;
}

bool RTLib::Ext::OPX7::OPX7ShaderTable::HasHitgroupRecord() const noexcept
{
	assert(m_Impl != nullptr);
	return m_Impl->hitgroupRecordSizeInBytes > 0;
}

auto RTLib::Ext::OPX7::OPX7ShaderTable::GetHitgroupRecordOffsetInBytes() const noexcept -> size_t
{
	assert(m_Impl != nullptr);
	return m_Impl->hitgroupRecordOffsetInBytes;
}

auto RTLib::Ext::OPX7::OPX7ShaderTable::GetHitgroupRecordSizeInBytes() const noexcept -> size_t
{
	assert(m_Impl != nullptr);
	return m_Impl->hitgroupRecordSizeInBytes;
}

auto RTLib::Ext::OPX7::OPX7ShaderTable::GetHitgroupRecordStrideInBytes() const noexcept -> unsigned int
{
	assert(m_Impl != nullptr);
	return m_Impl->shaderBindingTable.hitgroupRecordStrideInBytes;
}

auto RTLib::Ext::OPX7::OPX7ShaderTable::GetHitgroupRecordCount() const noexcept -> unsigned int
{
	assert(m_Impl != nullptr);
	return m_Impl->shaderBindingTable.hitgroupRecordCount;
}

bool RTLib::Ext::OPX7::OPX7ShaderTable::HasCallablesRecord() const noexcept
{
	assert(m_Impl != nullptr);
	return m_Impl->callablesRecordSizeInBytes > 0;
}

auto RTLib::Ext::OPX7::OPX7ShaderTable::GetCallablesRecordOffsetInBytes() const noexcept -> size_t
{
	assert(m_Impl != nullptr);
	return m_Impl->callablesRecordOffsetInBytes;
}

auto RTLib::Ext::OPX7::OPX7ShaderTable::GetCallablesRecordSizeInBytes() const noexcept -> size_t
{
	assert(m_Impl != nullptr);
	return m_Impl->callablesRecordSizeInBytes;
}

auto RTLib::Ext::OPX7::OPX7ShaderTable::GetCallablesRecordStrideInBytes() const noexcept -> unsigned int
{
	assert(m_Impl != nullptr);
	return m_Impl->shaderBindingTable.callablesRecordStrideInBytes;
}

auto RTLib::Ext::OPX7::OPX7ShaderTable::GetCallablesRecordCount() const noexcept -> unsigned int
{
	assert(m_Impl != nullptr);
	return m_Impl->shaderBindingTable.callablesRecordCount;
}

RTLib::Ext::OPX7::OPX7ShaderTable::OPX7ShaderTable(OPX7Context* context, const OPX7ShaderTableCreateDesc& desc) noexcept:
	m_Impl{new Impl(context,desc)}
{

}

auto RTLib::Ext::OPX7::OPX7ShaderTable::GetOptixShaderBindingTable() const noexcept -> const OptixShaderBindingTable&
{
	// TODO: return ステートメントをここに挿入します
	assert(m_Impl != nullptr);
	return m_Impl->shaderBindingTable;
}
