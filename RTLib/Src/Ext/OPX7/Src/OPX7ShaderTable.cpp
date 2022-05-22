#include <RTLib/Ext/OPX7/OPX7ShaderTable.h>
#include <RTLib/Ext/OPX7/OPX7Context.h>
#include <RTLib/Ext/CUDA/CUDABuffer.h>
#include <RTLib/Ext/CUDA/CUDACommon.h>
#include <cassert>

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
	}
	OPX7Context*					  context                      = nullptr;
	std::unique_ptr<CUDA::CUDABuffer> buffer                       = nullptr;
	OptixShaderBindingTable           shaderBindingTable           = {};
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

auto RTLib::Ext::OPX7::OPX7ShaderTable::New(OPX7Context* context, const OPX7ShaderTableCreateDesc& desc) -> OPX7ShaderTable*
{
	auto shaderTable = new OPX7ShaderTable(context, desc);

	return nullptr;
}

RTLib::Ext::OPX7::OPX7ShaderTable::~OPX7ShaderTable() noexcept
{
	m_Impl.reset();
}

auto RTLib::Ext::OPX7::OPX7ShaderTable::GetBufferData() noexcept -> CUDA::CUDABuffer*
{
	assert(m_Impl != nullptr);
	return m_Impl->buffer.get();
}

auto RTLib::Ext::OPX7::OPX7ShaderTable::GetBufferSize() const noexcept -> size_t
{
	assert(m_Impl != nullptr);
	return m_Impl->shaderBindingTableSizeInBytes;
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
