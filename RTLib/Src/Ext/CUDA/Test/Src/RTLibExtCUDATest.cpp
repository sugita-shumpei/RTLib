#include <RTLib/Ext/CUDA/CUDAContext.h>
#include <RTLib/Ext/CUDA/CUDABuffer.h>
int main(int argc, const char* argv)
{
	auto ctx = RTLib::Ext::CUDA::CUDAContext();
	ctx.Initialize();
	{
		auto bffDesc  = RTLib::Ext::CUDA::CUDABufferDesc();
		bffDesc.flags = RTLib::Ext::CUDA::CUDAMemoryFlags::ePageLocked;
		bffDesc.sizeInBytes = 1024;
		auto bff = ctx.CreateBuffer(bffDesc);
		bff->Destroy();
		delete bff;
	}
	ctx.Terminate();
}