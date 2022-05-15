#include <RTLib/Ext/CUDA/CUDAContext.h>
#include <RTLib/Ext/CUDA/CUDABuffer.h>
#include <RTLib/Ext/CUDA/CUDAImage.h>
#include <memory>
#include <cassert>
#include <iostream>
template<typename T>
void Show(const std::vector<T>& data){
	for (auto i = 0; i < 32; ++i) {
		for (auto j = 0; j < 32; ++j) {
			std::cout << data[32 * i + j] << " ";
		} 
		std::cout << std::endl;
	}
}
int main(int argc, const char* argv)
{
	auto ctx = RTLib::Ext::CUDA::CUDAContext();
	ctx.Initialize();
	{
		auto bffDesc = RTLib::Ext::CUDA::CUDABufferDesc();
		{
			bffDesc.flags       = RTLib::Ext::CUDA::CUDAMemoryFlags::ePageLocked;
			bffDesc.sizeInBytes = 32 * 32 * 4;
		}
		auto bff1 = std::unique_ptr<RTLib::Ext::CUDA::CUDABuffer>(ctx.CreateBuffer(bffDesc));
		auto bff2 = std::unique_ptr<RTLib::Ext::CUDA::CUDABuffer>(ctx.CreateBuffer(bffDesc));
		auto imgDesc = RTLib::Ext::CUDA::CUDAImageDesc();
		{
			imgDesc.imageType= RTLib::Ext::CUDA::CUDAImageType::e2D;
			imgDesc.width    = 32;
			imgDesc.height   = 32;
			imgDesc.layers   = 4;
			imgDesc.levels   = 4;
			imgDesc.format   = RTLib::Ext::CUDA::CUDAImageDataType::eFloat32;
			imgDesc.channels = 1;
		}
		auto img     = std::unique_ptr<RTLib::Ext::CUDA::CUDAImage>(ctx.CreateImage(imgDesc));
		auto mipImg0 = std::unique_ptr<RTLib::Ext::CUDA::CUDAImage>(img->GetMipImage(0));
		auto mipImg1 = std::unique_ptr<RTLib::Ext::CUDA::CUDAImage>(img->GetMipImage(1));
		auto mipImg2 = std::unique_ptr<RTLib::Ext::CUDA::CUDAImage>(img->GetMipImage(2));
		auto mipImg3 = std::unique_ptr<RTLib::Ext::CUDA::CUDAImage>(img->GetMipImage(3));
		{
			std::vector<float> srcData(32*32);
			for (auto i = 0; i < srcData.size(); ++i) {
				srcData[i] = i;
			}
			std::vector<float> dstData(32 * 32);
			assert(ctx.CopyMemoryToBuffer(bff1.get(), { {srcData.data(),0,sizeof(srcData[0])*std::size(srcData)}}));
			assert(ctx.CopyBuffer(bff1.get(), bff2.get(), { {0,0,32*sizeof(float)}}));
			assert(ctx.CopyBufferToMemory(bff2.get(), { {dstData.data(),0,sizeof(dstData[0]) * std::size(dstData)} }));
			Show(dstData);
			assert(ctx.CopyImageToBuffer(img.get(), bff2.get(), { {0,0,0,{0,0,1},{0,0,0},{32,32,0}} }));
		}
		bff1->Destroy();
		bff2->Destroy();
		img->Destroy();
	}
	ctx.Terminate();
}