#include <RTLib/Ext/CUDA/CUDAContext.h>
#include <RTLib/Ext/CUDA/CUDABuffer.h>
#include <RTLib/Ext/CUDA/CUDAImage.h>
#include <memory>
#include <cassert>
#include <iostream>
template<typename T>
void Show(const std::vector<T>& data){
	for (auto i = 0; i < 16; ++i) {
		for (auto j = 0; j < 16; ++j) {
			std::cout << data[16 * i + j] << " ";
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
			bffDesc.sizeInBytes = 16 * 16 * 4;
		}
		auto bff1 = std::unique_ptr<RTLib::Ext::CUDA::CUDABuffer>(ctx.CreateBuffer(bffDesc));
		auto bff2 = std::unique_ptr<RTLib::Ext::CUDA::CUDABuffer>(ctx.CreateBuffer(bffDesc));
		auto imgDesc = RTLib::Ext::CUDA::CUDAImageDesc();
		{
			imgDesc.imageType= RTLib::Ext::CUDA::CUDAImageType::e2D;
			imgDesc.width    = 16;
			imgDesc.height   = 16;
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
			std::vector<float> srcData0(16 * 16);
			for (auto i = 0; i < srcData0.size(); ++i) {
				srcData0[i] = i;
			}
			std::vector<float> srcData1(16 * 16);
			for (auto i = 0; i < srcData0.size(); ++i) {
				srcData1[i] = i*2;
			}
			std::vector<float> srcData2(16 * 16);
			for (auto i = 0; i < srcData0.size(); ++i) {
				srcData2[i] = i * 3;
			}
			std::vector<float> srcData3(16 * 16);
			for (auto i = 0; i < srcData0.size(); ++i) {
				srcData3[i] = i * 4;
			}

			std::vector<float> dstData0(16 * 16);
			std::vector<float> dstData1(16 * 16);
			std::vector<float> dstData2(16 * 16);
			std::vector<float> dstData3(16 * 16);

			assert(ctx.CopyMemoryToBuffer(bff1.get(), { {srcData0.data(),0,sizeof(srcData0[0]) * std::size(srcData0)}}));
			assert(ctx.CopyBuffer(bff1.get(), bff2.get(), { {0,0,16 *sizeof(float)}}));
			assert(ctx.CopyBufferToMemory(bff2.get(), { {dstData0.data(),0,sizeof(dstData0[0]) * std::size(dstData0)} }));
			assert(ctx.CopyMemoryToImage(  img.get(), { {srcData0.data() ,{0,0,1},{0,0,0},{16,16,0} } }));
			assert(ctx.CopyImageToMemory(  img.get(), { {dstData0.data() ,{0,0,1},{0,0,0},{16,16,0} } }));
			assert(ctx.CopyMemoryToImage(  img.get(), { {srcData1.data() ,{0,1,1},{0,0,0},{16,16,0} } }));
			assert(ctx.CopyImageToMemory(  img.get(), { {dstData1.data() ,{0,1,1},{0,0,0},{16,16,0} } }));
			assert(ctx.CopyMemoryToImage(  img.get(), { {srcData2.data() ,{0,2,1},{0,0,0},{16,16,0} } }));
			assert(ctx.CopyImageToMemory(  img.get(), { {dstData2.data() ,{0,2,1},{0,0,0},{16,16,0} } }));
			assert(ctx.CopyMemoryToImage(  img.get(), { {srcData3.data() ,{0,3,1},{0,0,0},{16,16,0} } }));
			assert(ctx.CopyImageToMemory(  img.get(), { {dstData3.data() ,{0,3,1},{0,0,0},{16,16,0} } }));

			Show(dstData0);
			Show(dstData1);
			Show(dstData2);
		}
		bff1->Destroy();
		bff2->Destroy();
		img->Destroy();
	}
	ctx.Terminate();
}