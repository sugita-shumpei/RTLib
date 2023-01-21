#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <RTLib/Core/Exceptions.h>
#include <RTLib/Ext/CUDA/CUDAContext.h>
#include <RTLib/Ext/CUDA/CUDABuffer.h>
#include <RTLib/Ext/CUDA/CUDAImage.h>
#include <RTLib/Ext/CUDA/CUDATexture.h>
#include <RTLib/Ext/CUDA/CUDAModule.h>
#include <RTLib/Ext/CUDA/CUDAFunction.h>
#include <RTLib/Ext/CUDA/CUDAStream.h>
#include <RTLib/Ext/CUDA/CUDANatives.h>
#include <RTLibExtCUDATestConfig.h>
#include <RTLibExtCUDATest.h>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <memory>
#include <numeric>
#include <cassert>
#include <iostream>
#include <fstream>
template<typename T>
void Show(const std::vector<T>& data){
	for (auto i = 0; i < 16; ++i) {
		for (auto j = 0; j < 16; ++j) {
			std::cout << data[16 * i + j] << " ";
		} 
		std::cout << std::endl;
	}
}
class PrefixScan
{
public:
	PrefixScan(RTLib::Ext::CUDA::CUDAContext* ctx, unsigned int middleBufferMaxCapacity) noexcept
	{
		m_Context = ctx;
		m_MiddleBufferMaxCapacity = middleBufferMaxCapacity;
	}
	~PrefixScan() {}

	void Init()
	{
		m_Module = std::unique_ptr<RTLib::Ext::CUDA::CUDAModule>(m_Context->LoadModuleFromFile(RTLIB_EXT_CUDA_TEST_CUDA_PATH"/simpleKernel.ptx"));
		m_ScanPerThreadsKernel = std::unique_ptr<RTLib::Ext::CUDA::CUDAFunction>(m_Module->LoadFunction("naiveScanKernel_Scan"));
		m_AddPerThreadsKernel  = std::unique_ptr<RTLib::Ext::CUDA::CUDAFunction>(m_Module->LoadFunction("naiveScanKernel_Add"));
		m_ScanPerThreadsEffectiveKernel = std::unique_ptr<RTLib::Ext::CUDA::CUDAFunction>(m_Module->LoadFunction("downSweepScanKernel_Scan"));
		auto obffDesc = RTLib::Ext::CUDA::CUDABufferCreateDesc();
		{
			obffDesc.flags = RTLib::Ext::CUDA::CUDAMemoryFlags::eDefault;
			obffDesc.sizeInBytes = m_MiddleBufferMaxCapacity;
			obffDesc.pData = nullptr;
		}
		m_MiddleBuffer = std::unique_ptr<RTLib::Ext::CUDA::CUDABuffer>(m_Context->CreateBuffer(obffDesc));
	}
	void Free()
	{
		m_MiddleBuffer->Destroy();
		m_ScanPerThreadsKernel->Destory();
		m_AddPerThreadsKernel->Destory();
		m_ScanPerThreadsEffectiveKernel->Destory();
		m_Module->Destory();
	}

	void Execute_Naive(
		RTLib::Ext::CUDA::CUDAStream*  stream,
		RTLib::Ext::CUDA::CUDABuffer*  inBuffer,
		RTLib::Ext::CUDA::CUDABuffer*  outBuffer,
		unsigned int                   numElement,
		unsigned int                   maxBlocks,
		unsigned int                   maxSharedMemorySizesPerBlock = 32*1024) noexcept {
		//1023 -> 1024
		//16
		//| | | | | |...| |
		maxBlocks = (maxBlocks / 2) * 2;
		unsigned int numGrids = (numElement + maxBlocks - 1) / (maxBlocks);

		RTLIB_CORE_ASSERT_IF_FAILED(LaunchScan(
			stream,
			RTLib::Ext::CUDA::CUDANatives::GetCUdeviceptr(inBuffer),
			RTLib::Ext::CUDA::CUDANatives::GetCUdeviceptr(outBuffer),
			1,
			0,
			numElement,
			numGrids,
			maxBlocks,
			sizeof(unsigned int) * 2 * maxBlocks
		));

		RTLIB_CORE_ASSERT_IF_FAILED(LaunchScan(
			stream,
			RTLib::Ext::CUDA::CUDANatives::GetCUdeviceptr(outBuffer),
			RTLib::Ext::CUDA::CUDANatives::GetCUdeviceptr(m_MiddleBuffer.get()),
			maxBlocks,
			maxBlocks-1,
			numElement,
			1,
			numGrids,
			sizeof(unsigned int) * 2 * numGrids
		));

		RTLIB_CORE_ASSERT_IF_FAILED(LaunchAdd(
			stream,
			RTLib::Ext::CUDA::CUDANatives::GetCUdeviceptr(m_MiddleBuffer.get()),
			RTLib::Ext::CUDA::CUDANatives::GetCUdeviceptr(outBuffer),
			numElement,
			numGrids,
			maxBlocks
		));

	}

	void Execute_Effective(
		RTLib::Ext::CUDA::CUDAStream* stream,
		RTLib::Ext::CUDA::CUDABuffer* inBuffer,
		RTLib::Ext::CUDA::CUDABuffer* outBuffer,
		unsigned int                   numElement,
		unsigned int                   maxBlocks,
		unsigned int                   maxSharedMemorySizesPerBlock = 32 * 1024) noexcept {
		//1023 -> 1024
		//16
		//| | | | | |...| |
		maxBlocks = (maxBlocks / 2) * 2;
		unsigned int numGrids = (numElement + 2 * maxBlocks - 1) / (2 * maxBlocks);// numGrids*maxBlocks ~numElement/2

		RTLIB_CORE_ASSERT_IF_FAILED(LaunchScan_Effective(
			stream,
			RTLib::Ext::CUDA::CUDANatives::GetCUdeviceptr(inBuffer),
			RTLib::Ext::CUDA::CUDANatives::GetCUdeviceptr(outBuffer),
			1,
			0,
			numElement,
			numGrids,
			maxBlocks,
			sizeof(unsigned int) * 2 * maxBlocks
		));

		RTLIB_CORE_ASSERT_IF_FAILED(LaunchScan_Effective(
			stream,
			RTLib::Ext::CUDA::CUDANatives::GetCUdeviceptr(outBuffer),
			RTLib::Ext::CUDA::CUDANatives::GetCUdeviceptr(m_MiddleBuffer.get()),
			maxBlocks,
			maxBlocks - 1,
			numElement,
			1,
			(numGrids/2),
			sizeof(unsigned int) * numGrids
		));

		RTLIB_CORE_ASSERT_IF_FAILED(LaunchAdd(
			stream,
			RTLib::Ext::CUDA::CUDANatives::GetCUdeviceptr(m_MiddleBuffer.get()),
			RTLib::Ext::CUDA::CUDANatives::GetCUdeviceptr(outBuffer),
			numElement,
			numGrids,
			maxBlocks
		));

	}
private:
	bool LaunchScan(
		RTLib::Ext::CUDA::CUDAStream* stream,
		CUdeviceptr         inBufferAddress,
		CUdeviceptr         outBufferAddress,
		unsigned int        stride,
		unsigned int        offset,
		unsigned int        maxRangeInBuffers,
		unsigned int        numGrids,
		unsigned int        numBlocks,
		unsigned int        sharedMemorySizePerGrid)
	{
		return m_ScanPerThreadsKernel->Launch({ numGrids,1,1,numBlocks,1,1,sharedMemorySizePerGrid ,
			{
				&inBufferAddress,
				&outBufferAddress,
				&stride,
				&offset,
				&numBlocks,
				&maxRangeInBuffers
			},
			stream
		});
	}

	bool LaunchScan_Effective(
		RTLib::Ext::CUDA::CUDAStream* stream,
		CUdeviceptr         inBufferAddress,
		CUdeviceptr         outBufferAddress,
		unsigned int        stride,
		unsigned int        offset,
		unsigned int        maxRangeInBuffers,
		unsigned int        numGrids,
		unsigned int        numBlocks,
		unsigned int        sharedMemorySizePerGrid)
	{
		return m_ScanPerThreadsKernel->Launch({ numGrids,1,1,numBlocks,1,1,sharedMemorySizePerGrid ,
			{
				&inBufferAddress,
				&outBufferAddress,
				&stride,
				&offset,
				&numBlocks,
				&maxRangeInBuffers
			},
				stream
			});
	}
	bool LaunchAdd(
		RTLib::Ext::CUDA::CUDAStream* stream,
		CUdeviceptr          inBufferAddress,
		CUdeviceptr         outBufferAddress,
		unsigned int      maxRangeOutBuffers,
		unsigned int                numGrids,
		unsigned int               numBlocks
	) {
		return m_AddPerThreadsKernel->Launch({ numGrids,1,1,numBlocks,1,1,0 ,
			{
				&inBufferAddress,
				&outBufferAddress,
				&numBlocks,
				&maxRangeOutBuffers,
			},
			stream
		});
	}
private:
	RTLib::Ext::CUDA::CUDAContext*                   m_Context;
	std::unique_ptr<RTLib::Ext::CUDA::CUDAModule>    m_Module;
	std::unique_ptr<RTLib::Ext::CUDA::CUDABuffer>    m_MiddleBuffer;
	std::unique_ptr<RTLib::Ext::CUDA::CUDAFunction>  m_ScanPerThreadsKernel;
	std::unique_ptr<RTLib::Ext::CUDA::CUDAFunction>  m_ScanPerThreadsEffectiveKernel;
	std::unique_ptr<RTLib::Ext::CUDA::CUDAFunction>  m_AddPerThreadsKernel;
	size_t                                           m_MiddleBufferMaxCapacity;

};
int main(int argc, const char* argv)
{
	auto ctx = RTLib::Ext::CUDA::CUDAContext();
	ctx.Initialize();
	{
		std::cout << "ThisTypeId: " << ctx.GetTypeIdString()     << std::endl;
		std::cout << "BaseTypeId: " << ctx.GetBaseTypeIdString() << std::endl;

		auto stream  = std::unique_ptr<RTLib::Ext::CUDA::CUDAStream>(ctx.CreateStream());
		auto bffDesc = RTLib::Ext::CUDA::CUDABufferCreateDesc();
		{
			bffDesc.flags       = RTLib::Ext::CUDA::CUDAMemoryFlags::eDefault;
			bffDesc.sizeInBytes = 16 * 16 * 4;
		}
		auto bff1     = std::unique_ptr<RTLib::Ext::CUDA::CUDABuffer>(ctx.CreateBuffer(bffDesc));
		auto bff2     = std::unique_ptr<RTLib::Ext::CUDA::CUDABuffer>(ctx.CreateBuffer(bffDesc));
		auto imgDesc0 = RTLib::Ext::CUDA::CUDAImageCreateDesc();
		{
			imgDesc0.imageType= RTLib::Ext::CUDA::CUDAImageType::e2D;
			imgDesc0.extent.width    = 16;
			imgDesc0.extent.height   = 16;
			imgDesc0.arrayLayers     = 4;
			imgDesc0.mipLevels       = 4;
			imgDesc0.format          = RTLib::Ext::CUDA::CUDAImageFormat::eFloat32X1;
		}
		auto imgDesc1 = RTLib::Ext::CUDA::CUDAImageCreateDesc();
		{
			imgDesc1.imageType = RTLib::Ext::CUDA::CUDAImageType::e2D;
			imgDesc1.extent.width  = 16;
			imgDesc1.extent.height = 16;
			imgDesc1.arrayLayers   = 2;
			imgDesc1.mipLevels     = 1;
			imgDesc1.format        = RTLib::Ext::CUDA::CUDAImageFormat::eUInt16X2;
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

			RTLIB_CORE_ASSERT_IF_FAILED(stream->CopyMemoryToBuffer( bff1.get(), { {srcData0.data(),0,sizeof(srcData0[0]) * std::size(srcData0)}}));
			RTLIB_CORE_ASSERT_IF_FAILED(stream->CopyBuffer(         bff1.get(), bff2.get(), { {0,0,16 *sizeof(float)}}));
			RTLIB_CORE_ASSERT_IF_FAILED(stream->CopyBufferToMemory( bff2.get(), { {dstData0.data(),0,sizeof(dstData0[0]) * std::size(dstData0)} }));
			Show(dstData0);
			RTLIB_CORE_ASSERT_IF_FAILED(stream->CopyMemoryToImage(  img0.get(), { {srcData0.data() ,{0,0,1},{0,0,0},{16,16,0} } }));
			RTLIB_CORE_ASSERT_IF_FAILED(stream->CopyImageToMemory(  img0.get(), { {dstData0.data() ,{0,0,1},{0,0,0},{16,16,0} } }));
			Show(dstData0);
			RTLIB_CORE_ASSERT_IF_FAILED(stream->CopyMemoryToImage(  img0.get(), { {srcData1.data() ,{0,1,1},{0,0,0},{16,16,0} } }));
			RTLIB_CORE_ASSERT_IF_FAILED(stream->CopyImageToMemory(  img0.get(), { {dstData1.data() ,{0,1,1},{0,0,0},{16,16,0} } }));
			Show(dstData1);
			RTLIB_CORE_ASSERT_IF_FAILED(stream->CopyMemoryToImage(  img0.get(), { {srcData2.data() ,{0,2,1},{0,0,0},{16,16,0} } }));
			//assert(stream->CopyImageToMemory(  img0.get(), { {dstData2.data() ,{0,2,1},{0,0,0},{16,16,0} } }));
			Show(dstData2);
			RTLIB_CORE_ASSERT_IF_FAILED(stream->CopyMemoryToImage(  img0.get(), { {srcData3.data() ,{0,3,1},{0,0,0},{16,16,0} } }));
			RTLIB_CORE_ASSERT_IF_FAILED(stream->CopyImageToMemory(  img0.get(), { {dstData3.data() ,{0,3,1},{0,0,0},{16,16,0} } }));
			Show(dstData3);
		}
		auto texDesc = RTLib::Ext::CUDA::CUDATextureImageCreateDesc();
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
		auto mod = std::unique_ptr<RTLib::Ext::CUDA::CUDAModule>( ctx.LoadModuleFromFile(RTLIB_EXT_CUDA_TEST_CUDA_PATH"/simpleKernel.ptx"));
		auto fnc = std::unique_ptr<RTLib::Ext::CUDA::CUDAFunction>(mod->LoadFunction("blurKernel"));
		{
			int x, y, comp;
			auto iImgData = stbi_load(RTLIB_EXT_CUDA_TEST_DATA_PATH"/Textures/sample.png", &x, &y, &comp, 4);
			auto ibffDesc = RTLib::Ext::CUDA::CUDABufferCreateDesc();
			{
				ibffDesc.flags = RTLib::Ext::CUDA::CUDAMemoryFlags::eDefault;
				ibffDesc.sizeInBytes = x * y * comp;
			}
			auto oImgData = std::vector<unsigned char>(x * y * comp);
			auto obffDesc = RTLib::Ext::CUDA::CUDABufferCreateDesc();
			{
				obffDesc.flags = RTLib::Ext::CUDA::CUDAMemoryFlags::eDefault;
				obffDesc.sizeInBytes = x * y * comp;
			}
			auto ibff = std::unique_ptr<RTLib::Ext::CUDA::CUDABuffer>(ctx.CreateBuffer(ibffDesc));
			RTLIB_CORE_ASSERT_IF_FAILED(ctx.CopyMemoryToBuffer(ibff.get(), { {static_cast<const void*>(iImgData),static_cast<size_t>(0),static_cast<size_t>(x * y * comp)}}));
			auto obff = std::unique_ptr<RTLib::Ext::CUDA::CUDABuffer>(ctx.CreateBuffer(obffDesc));

			auto ipixel = RTLib::Ext::CUDA::CUDANatives::GetCUdeviceptr(ibff.get());
			auto opixel = RTLib::Ext::CUDA::CUDANatives::GetCUdeviceptr(obff.get());
			int width  = x;
			int height = y;
			fnc->Launch({ 1024,1024,1,32,32,1,0,{
				&ipixel,
				&opixel,
				&x,
				&y
			}, nullptr});
			RTLIB_CORE_ASSERT_IF_FAILED(ctx.CopyBufferToMemory(obff.get(), { {static_cast<void*>(oImgData.data()),static_cast<size_t>(0),static_cast<size_t>(x * y * comp)} }));
			stbi_write_png(RTLIB_EXT_CUDA_TEST_CUDA_PATH"/../Result.png", x, y, comp, oImgData.data(), 4 * x);

			ibff->Destroy();
			obff->Destroy();
		}
		unsigned int numBlock   = 512;
		unsigned int numElement = 65536;
		std::vector<unsigned int>  inArray(numElement, 1);//128 * 512
		std::vector<unsigned int> outArray(numElement, 0);//128 * 512
		std::vector<unsigned int> midArray(inArray.size()/ numBlock, 0);
		std::iota(std::begin(inArray), std::end(inArray),0);
		std::for_each(std::begin(inArray), std::end(inArray), [](unsigned int& i) { if (i != 0) { i = 2 * i - 1; } });
		{
			{
				auto ibffDesc = RTLib::Ext::CUDA::CUDABufferCreateDesc();
				{
					ibffDesc.flags = RTLib::Ext::CUDA::CUDAMemoryFlags::eDefault;
					ibffDesc.sizeInBytes = std::size(inArray) * sizeof(inArray[0]);
					ibffDesc.pData = std::data(inArray);
				}
				auto ibff = std::unique_ptr<RTLib::Ext::CUDA::CUDABuffer>(ctx.CreateBuffer(ibffDesc));
				auto obffDesc = RTLib::Ext::CUDA::CUDABufferCreateDesc();
				{
					obffDesc.flags = RTLib::Ext::CUDA::CUDAMemoryFlags::eDefault;
					obffDesc.sizeInBytes = std::size(outArray) * sizeof(outArray[0]);
					obffDesc.pData = nullptr;
				}
				auto obff = std::unique_ptr<RTLib::Ext::CUDA::CUDABuffer>(ctx.CreateBuffer(obffDesc));
				PrefixScan scan(&ctx, 1024 * 1024);
				
				scan.Init();
				auto naiveArray = std::vector<unsigned long long>();
				auto effecArray = std::vector<unsigned long long>();
				for (int i = 0; i < 100;++i) {
					
					{
						auto beg = std::chrono::system_clock::now();
						scan.Execute_Naive(stream.get(), ibff.get(), obff.get(), numElement, 512);
						RTLIB_CORE_ASSERT_IF_FAILED(stream->Synchronize());
						auto end = std::chrono::system_clock::now();
						naiveArray.push_back(std::chrono::duration_cast<std::chrono::microseconds>(end - beg).count());
						RTLIB_CORE_ASSERT_IF_FAILED(ctx.CopyBufferToMemory(obff.get(), { {static_cast<void*>(outArray.data()),static_cast<size_t>(0),outArray.size() * sizeof(unsigned)} }));
						ctx.Synchronize();
					}
					{
						auto beg = std::chrono::system_clock::now();
						scan.Execute_Effective(stream.get(), ibff.get(), obff.get(), numElement, 512);
						RTLIB_CORE_ASSERT_IF_FAILED(stream->Synchronize());
						auto end = std::chrono::system_clock::now();
						effecArray.push_back(std::chrono::duration_cast<std::chrono::microseconds>(end - beg).count());
						RTLIB_CORE_ASSERT_IF_FAILED(ctx.CopyBufferToMemory(obff.get(), { {static_cast<void*>(outArray.data()),static_cast<size_t>(0),outArray.size() * sizeof(unsigned)} }));
						ctx.Synchronize();
					}


				}
				for (auto& naive : naiveArray) {
					std::cout << "Naive: " << naive << std::endl;
				}
				for (auto& effec : effecArray) {
					std::cout << "Effec: " << effec << std::endl;
				}

				
				scan.Free();
			}
			{

				std::ofstream file(RTLIB_EXT_CUDA_TEST_CUDA_PATH"\\result.bin");
				if (file.is_open()) {
					for (auto& val : outArray)
					{

						file << std::sqrtf(val) << std::endl;
					}
					file.close();
				}
			}
		}
		fnc->Destory();
		mod->Destory();
		stream->Destroy();
		 tex->Destroy();
		bff1->Destroy();
		bff2->Destroy();
		img0->Destroy();
		img1->Destroy();
	}
	ctx.Terminate();
}