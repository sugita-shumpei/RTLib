#include <RTLib/Ext/CUDA/CUDAContext.h>
#include <RTLib/Ext/CUDA/CUDABuffer.h>
#include <RTLib/Ext/CUDA/CUDAImage.h>
#include <RTLib/Ext/CUDA/CUDATexture.h>
#include <RTLib/Ext/CUDA/CUDAModule.h>
#include <RTLib/Ext/CUDA/CUDAFunction.h>
#include <RTLib/Ext/CUDA/CUDAStream.h>
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
	{;
		auto stream  = std::unique_ptr<RTLib::Ext::CUDA::CUDAStream>(ctx.CreateStream());
		auto bffDesc = RTLib::Ext::CUDA::CUDABufferDesc();
		{
			bffDesc.flags       = RTLib::Ext::CUDA::CUDAMemoryFlags::ePageLocked;
			bffDesc.sizeInBytes = 16 * 16 * 4;
		}
		auto bff1     = std::unique_ptr<RTLib::Ext::CUDA::CUDABuffer>(ctx.CreateBuffer(bffDesc));
		auto bff2     = std::unique_ptr<RTLib::Ext::CUDA::CUDABuffer>(ctx.CreateBuffer(bffDesc));
		auto imgDesc0 = RTLib::Ext::CUDA::CUDAImageDesc();
		{
			imgDesc0.imageType= RTLib::Ext::CUDA::CUDAImageType::e2D;
			imgDesc0.width    = 16;
			imgDesc0.height   = 16;
			imgDesc0.layers   = 4;
			imgDesc0.levels   = 4;
			imgDesc0.format   = RTLib::Ext::CUDA::CUDAImageDataType::eFloat32;
			imgDesc0.channels = 1;
		}
		auto imgDesc1 = RTLib::Ext::CUDA::CUDAImageDesc();
		{
			imgDesc1.imageType = RTLib::Ext::CUDA::CUDAImageType::e2D;
			imgDesc1.width     = 16;
			imgDesc1.height    = 16;
			imgDesc1.layers    = 2;
			imgDesc1.levels    = 1;
			imgDesc1.format    = RTLib::Ext::CUDA::CUDAImageDataType::eUInt16;
			imgDesc1.channels  = 2;
		}
		auto img0    = std::unique_ptr<RTLib::Ext::CUDA::CUDAImage>(ctx.CreateImage(imgDesc0));
		auto mipImg0 = std::unique_ptr<RTLib::Ext::CUDA::CUDAImage>(img0->GetMipImage(0));
		auto mipImg1 = std::unique_ptr<RTLib::Ext::CUDA::CUDAImage>(img0->GetMipImage(1));
		auto mipImg2 = std::unique_ptr<RTLib::Ext::CUDA::CUDAImage>(img0->GetMipImage(2));
		auto mipImg3 = std::unique_ptr<RTLib::Ext::CUDA::CUDAImage>(img0->GetMipImage(3));
		auto img1    = std::unique_ptr<RTLib::Ext::CUDA::CUDAImage>(ctx.CreateImage(imgDesc1));
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

			assert(stream->CopyMemoryToBuffer( bff1.get(), { {srcData0.data(),0,sizeof(srcData0[0]) * std::size(srcData0)}}));
			assert(stream->CopyBuffer(         bff1.get(), bff2.get(), { {0,0,16 *sizeof(float)}}));
			assert(stream->CopyBufferToMemory( bff2.get(), { {dstData0.data(),0,sizeof(dstData0[0]) * std::size(dstData0)} }));
			assert(stream->CopyMemoryToImage(  img0.get(), { {srcData0.data() ,{0,0,1},{0,0,0},{16,16,0} } }));
			assert(stream->CopyImageToMemory(  img0.get(), { {dstData0.data() ,{0,0,1},{0,0,0},{16,16,0} } }));
			assert(stream->CopyMemoryToImage(  img0.get(), { {srcData1.data() ,{0,1,1},{0,0,0},{16,16,0} } }));
			assert(stream->CopyImageToMemory(  img0.get(), { {dstData1.data() ,{0,1,1},{0,0,0},{16,16,0} } }));
			assert(stream->CopyMemoryToImage(  img0.get(), { {srcData2.data() ,{0,2,1},{0,0,0},{16,16,0} } }));
			assert(stream->CopyImageToMemory(  img0.get(), { {dstData2.data() ,{0,2,1},{0,0,0},{16,16,0} } }));
			assert(stream->CopyMemoryToImage(  img0.get(), { {srcData3.data() ,{0,3,1},{0,0,0},{16,16,0} } }));
			assert(stream->CopyImageToMemory(  img0.get(), { {dstData3.data() ,{0,3,1},{0,0,0},{16,16,0} } }));

			Show(dstData0);
			Show(dstData1);
			Show(dstData2);
		}
		auto texDesc = RTLib::Ext::CUDA::CUDATextureImageDesc();
		{
			texDesc.image                       = img1.get();
			texDesc.sampler.addressMode[0]      = RTLib::Ext::CUDA::CUDATextureAddressMode::eClamp;
			texDesc.sampler.addressMode[1]      = RTLib::Ext::CUDA::CUDATextureAddressMode::eClamp;
			texDesc.sampler.addressMode[2]      = RTLib::Ext::CUDA::CUDATextureAddressMode::eClamp;
			texDesc.sampler.borderColor[0]      = 0.0f;
			texDesc.sampler.borderColor[1]      = 0.0f;
			texDesc.sampler.borderColor[2]      = 0.0f;
			texDesc.sampler.borderColor[3]      = 0.0f;
			texDesc.sampler.filterMode          = RTLib::Ext::CUDA::CUDATextureFilterMode::eLinear;
			texDesc.sampler.mipmapFilterMode    = RTLib::Ext::CUDA::CUDATextureFilterMode::eLinear;
			texDesc.sampler.mipmapLevelBias     = 0.0f;
			texDesc.sampler.maxMipmapLevelClamp = 0.0f;
			texDesc.sampler.minMipmapLevelClamp = 0.0f;
			texDesc.sampler.maxAnisotropy       = 0;
			texDesc.sampler.flags               = 0;
		}
		auto tex = std::unique_ptr<RTLib::Ext::CUDA::CUDATexture>(ctx.CreateTexture(texDesc));
		stream->Destroy();
		 tex->Destroy();
		bff1->Destroy();
		bff2->Destroy();
		img0->Destroy();
		img1->Destroy();
	}
	ctx.Terminate();
}