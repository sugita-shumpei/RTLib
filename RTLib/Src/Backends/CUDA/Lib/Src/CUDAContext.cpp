#include <RTLib/Backends/CUDA/CUDAContext.h>
#include <RTLib/Backends/CUDA/CUDAEntry.h>
#include <RTLib/Backends/CUDA/CUDAStream.h>
#include <RTLib/Backends/CUDA/CUDALinearMemory.h>
#include <RTLib/Backends/CUDA/CUDAPinnedHostMemory.h>
#include <RTLib/Backends/CUDA/CUDAArray.h>
#include <RTLib/Backends/CUDA/CUDAMipmappedArray.h>
#include <RTLib/Backends/CUDA/CUDATexture.h>
#include <RTLib/Backends/CUDA/CUDAModule.h>
#include <RTLib/Backends/CUDA/CUDAFunction.h>
#include "CUDAInternals.h"
#include <unordered_map>
#include <unordered_set>
#include <iostream>

struct RTLib::Backends::Cuda::Context::Impl{
	Impl(const Device& device, ContextCreateFlags flags) noexcept :isBinded{ true }, streams{} {
		auto&  entry = Entry::Handle();
		unsigned int tmpFlags = 0;
		if (flags & ContextCreateScheduleBlockingSync) {
			tmpFlags |= CU_CTX_SCHED_BLOCKING_SYNC;
		}
		if (flags & ContextCreateScheduleYield) {
			tmpFlags |= CU_CTX_SCHED_YIELD;
		}
		if (flags & ContextCreateScheduleSpin) {
			tmpFlags |= CU_CTX_SCHED_SPIN;
		}
		if (flags & ContextCreateMapHost) {
			tmpFlags |= CU_CTX_MAP_HOST;
		}
		if (flags & ContextCreateLmemResizeToMax) {
			tmpFlags |= CU_CTX_LMEM_RESIZE_TO_MAX;
		}
		//RTLIB_BACKENDS_CUDA_DEBUG_ASSERT(cuCtxCreate(&context, tmpFlags,Internals::GetCUdevice(device)));
		RTLIB_BACKENDS_CUDA_DEBUG_ASSERT(cuCtxCreate(&context, tmpFlags, Internals::GetCUdevice(device)));
		auto pDefaultStream = std::shared_ptr<void>(nullptr);
		streams.insert(pDefaultStream);
		defaultStream = std::unique_ptr<Stream>(new Stream(pDefaultStream));
	}
	~Impl()noexcept {
		std::cout << "Destroy!\n";
		auto& entry = Entry::Handle();
		streams.clear();
		defaultStream.reset();
		RTLIB_BACKENDS_CUDA_DEBUG_ASSERT(cuCtxDestroy(context));
	}

	static void StreamDeleter(void* p) {
		if (!p) {
			return;
		}
		RTLIB_BACKENDS_CUDA_DEBUG_ASSERT(cuStreamDestroy(static_cast<CUstream>(p)));
	}
	auto NewStream(StreamCreateFlags flags) -> std::shared_ptr<void> {
		unsigned int tmpFlags = 0;
		if (flags & StreamCreateNonBlocking) {
			tmpFlags |= CU_STREAM_NON_BLOCKING;
		}
		CUstream stream;
		RTLIB_BACKENDS_CUDA_DEBUG_ASSERT(cuStreamCreate(&stream, tmpFlags));
		auto ptr = std::shared_ptr<void>(stream, StreamDeleter);
		streams.insert(ptr);
		return ptr;
	}
	CUcontext                                 context;
	std::unordered_set<std::shared_ptr<void>> streams;
	std::unique_ptr<Stream>             defaultStream;
	bool                                     isBinded;
};

RTLib::Backends::Cuda::Context::Context(const Device& device, ContextCreateFlags flags) noexcept:m_Impl{new Impl{device,flags}}
{
	CurrentContext::Handle().Set(this);
}

RTLib::Backends::Cuda::Context::~Context() noexcept
{
	CurrentContext::Handle().PopFromStack(this, true);
	m_Impl.reset();
}

bool RTLib::Backends::Cuda::Context::operator==(const Context& ctx) const noexcept
{
	return this->m_Impl->context == ctx.m_Impl->context;
}

bool RTLib::Backends::Cuda::Context::operator!=(const Context& ctx) const noexcept
{
	return this->m_Impl->context != ctx.m_Impl->context;
}

auto RTLib::Backends::Cuda::Context::GetCurrent() noexcept -> Context*
{
	return CurrentContext::Handle().Get();
}

void RTLib::Backends::Cuda::Context::SetCurrent() noexcept
{
	CurrentContext::Handle().Set(this);
}

void RTLib::Backends::Cuda::Context::PopFromStack()
{
	CurrentContext::Handle().PopFromStack(this);
}


auto RTLib::Backends::Cuda::Context::GetHandle() const noexcept -> void*
{
	return m_Impl->context;
}

bool RTLib::Backends::Cuda::Context::IsBindedStack() const noexcept
{
	return m_Impl->isBinded;
}

auto RTLib::Backends::Cuda::Context::GetDefaultStream() const noexcept -> Stream*
{
	return m_Impl->defaultStream.get();
}

struct RTLib::Backends::Cuda::CurrentContext::Impl {
	Impl() noexcept  {}
	~Impl() noexcept {}

	Context* context = nullptr; 
};

RTLib::Backends::Cuda::CurrentContext::CurrentContext() noexcept:m_Impl{new Impl()}{}


auto RTLib::Backends::Cuda::CurrentContext::Handle() noexcept -> CurrentContext&
{
	// TODO: return ステートメントをここに挿入します
	static thread_local auto current = CurrentContext();
	return current;
}

RTLib::Backends::Cuda::CurrentContext::~CurrentContext() noexcept = default;

auto RTLib::Backends::Cuda::CurrentContext::CreateStream(StreamCreateFlags flags) const noexcept -> Stream*
{
	assert(Get());
	return new Stream(Get()->m_Impl->NewStream(flags));
}

auto RTLib::Backends::Cuda::CurrentContext::CreateLinearMemory1D(size_t sizeInBytes) const noexcept -> LinearMemory1D*
{
	return new LinearMemory1D(sizeInBytes);
}

auto RTLib::Backends::Cuda::CurrentContext::CreateLinearMemory2D(size_t width, size_t height, unsigned int elementSizeInBytes) const noexcept -> LinearMemory2D*
{
	return new LinearMemory2D(width,height,elementSizeInBytes);
}

auto RTLib::Backends::Cuda::CurrentContext::CreatePinnedHostMemory(size_t sizeInBytes) const noexcept -> PinnedHostMemory*
{
	return new PinnedHostMemory(sizeInBytes);
}

auto RTLib::Backends::Cuda::CurrentContext::CreateStreamUnique(StreamCreateFlags flags) const noexcept -> std::unique_ptr<Stream>
{
	return std::unique_ptr<Stream>(CreateStream(flags));
}

auto RTLib::Backends::Cuda::CurrentContext::CreateLinearMemory1DUnique(size_t sizeInBytes) const noexcept -> std::unique_ptr<LinearMemory1D>
{
	return std::unique_ptr<LinearMemory1D>(CreateLinearMemory1D(sizeInBytes));
}

auto RTLib::Backends::Cuda::CurrentContext::CreateLinearMemory2DUnique(size_t width, size_t height, unsigned int elementSizeInBytes) const noexcept -> std::unique_ptr<LinearMemory2D>
{
	return std::unique_ptr<LinearMemory2D>(CreateLinearMemory2D(width,height,elementSizeInBytes));
}

auto RTLib::Backends::Cuda::CurrentContext::CreatePinnedHostMemoryUnique(size_t sizeInBytes) const noexcept -> std::unique_ptr<PinnedHostMemory>
{
	return std::unique_ptr<PinnedHostMemory>(CreatePinnedHostMemory(sizeInBytes));
}

auto RTLib::Backends::Cuda::CurrentContext::GetDefaultStream() const noexcept -> Stream*
{
	auto ptr = Get();
	return ptr?ptr->GetDefaultStream():nullptr;
}

void RTLib::Backends::Cuda::CurrentContext::Set(Context* ctx, bool sysOp) noexcept
{
	if (ctx || m_Impl->context) {
		CUcontext context = ctx ? static_cast<CUcontext>(ctx->GetHandle()) : nullptr;
		if (!sysOp) {
			auto&  entry  = Entry::Handle();
			RTLIB_BACKENDS_CUDA_DEBUG_ASSERT(cuCtxSetCurrent(reinterpret_cast<CUcontext>(context)));
		}
		m_Impl->context = ctx;
		if (ctx) {
			ctx->m_Impl->isBinded = true;
		}
	}
	else {
		m_Impl->context = nullptr;
	}
}

auto RTLib::Backends::Cuda::CurrentContext::Pop(bool sysOp) noexcept->Context*
{
	CUcontext context = m_Impl->context ? static_cast<CUcontext>(m_Impl->context->GetHandle()) : nullptr;
	if (context && m_Impl->context) {
		auto&  entry = Entry::Handle();
		if (!sysOp) {
			RTLIB_BACKENDS_CUDA_DEBUG_ASSERT(cuCtxPopCurrent(nullptr));
		}
		auto res = m_Impl->context;
		res->m_Impl->isBinded = false;
		m_Impl->context = nullptr;
		return res;
	}
	else {
		return nullptr;
	}
}

void RTLib::Backends::Cuda::CurrentContext::PopFromStack(Context* ctx, bool sysOp)
{
	if (!ctx) {
		return;
	}
	ctx->m_Impl->isBinded = false;
	auto& entry = Entry::Handle();
	if (!sysOp) {
		if (ctx != m_Impl->context)
		{
			RTLIB_BACKENDS_CUDA_DEBUG_ASSERT(cuCtxSetCurrent(static_cast<CUcontext>(ctx->GetHandle())));
			RTLIB_BACKENDS_CUDA_DEBUG_ASSERT(cuCtxPopCurrent(nullptr));
			RTLIB_BACKENDS_CUDA_DEBUG_ASSERT(cuCtxSetCurrent(static_cast<CUcontext>(m_Impl->context->GetHandle())));
			
		}
		else {
			RTLIB_BACKENDS_CUDA_DEBUG_ASSERT(cuCtxPopCurrent(nullptr));
			m_Impl->context = nullptr;
		}
	}
	else {
		if (ctx == m_Impl->context) {
			m_Impl->context       = nullptr;
		}
	}
}

auto RTLib::Backends::Cuda::CurrentContext::Get() const noexcept -> Context*
{
	return m_Impl->context;
}


void RTLib::Backends::Cuda::CurrentContext::Push(Context* ctx) noexcept
{
	if (!ctx) {
		return;
	}
	auto& entry = Entry::Handle();
	if (ctx->m_Impl->isBinded) {
		return;
	}
	RTLIB_BACKENDS_CUDA_DEBUG_ASSERT(cuCtxPushCurrent(ctx->m_Impl->context));
	m_Impl->context       = ctx;
	ctx->m_Impl->isBinded = true;
}

auto RTLib::Backends::Cuda::CurrentContext::CreateArray1D(unsigned int count, unsigned int numChannels, ArrayFormat format,bool useSurface)const noexcept -> Array1D* {
	assert(Get());
	return new Array1D(count, numChannels, format, useSurface);
}

auto RTLib::Backends::Cuda::CurrentContext::CreateArray2D(unsigned int width, unsigned int height, unsigned int numChannels, ArrayFormat format, bool useSurface)const noexcept -> Array2D* {
	assert(Get());
	return new Array2D(width, height, numChannels, format, useSurface);
}

auto RTLib::Backends::Cuda::CurrentContext::CreateArray3D(unsigned int width, unsigned int height, unsigned int depth, unsigned int numChannels, ArrayFormat format, bool useSurface)const noexcept -> Array3D* {
	assert(Get());	
	return new Array3D(width, height, depth, numChannels, format, useSurface);
}

auto RTLib::Backends::Cuda::CurrentContext::CreateMipmappedArray1D(unsigned int levels, unsigned int count, unsigned int numChannels, ArrayFormat format, bool useSurface)const noexcept -> MipmappedArray1D* {
	assert(Get());
	return new MipmappedArray1D(levels, count, numChannels, format, useSurface);
}

auto RTLib::Backends::Cuda::CurrentContext::CreateMipmappedArray2D(unsigned int levels, unsigned int width, unsigned int height, unsigned int numChannels, ArrayFormat format, bool useSurface)const noexcept -> MipmappedArray2D* {
	assert(Get());
	return new MipmappedArray2D(levels, width, height, numChannels, format, useSurface);
}

auto RTLib::Backends::Cuda::CurrentContext::CreateMipmappedArray3D(unsigned int levels, unsigned int width, unsigned int height, unsigned int depth, unsigned int numChannels, ArrayFormat format, bool useSurface)const noexcept -> MipmappedArray3D* {
	assert(Get());
	return new MipmappedArray3D(levels, width, height, depth, numChannels, format, useSurface);
}

auto RTLib::Backends::Cuda::CurrentContext::CreateTextureFromArray(const Array* arr, const TextureDesc& desc) const noexcept -> Texture*
{
	assert(Get());
	return new Texture(arr, desc);
}

auto RTLib::Backends::Cuda::CurrentContext::CreateTextureFromMipmappedArray(const MipmappedArray* mipmapped, const TextureDesc& desc) const noexcept -> Texture*
{
	assert(Get());
	return new Texture(mipmapped,desc);
}

auto RTLib::Backends::Cuda::CurrentContext::CreateModuleFromFile(const char* filename) const noexcept -> Module*
{
	return new Module(filename);
}

auto RTLib::Backends::Cuda::CurrentContext::CreateModuleFromData(const std::vector<char>& image) const noexcept -> Module*
{
	return new Module(image);
}

auto RTLib::Backends::Cuda::CurrentContext::CreateModuleFromData2(const std::vector<char>& image, const std::unordered_map<JitOption, void*>& options) const noexcept -> Module*
{
	return new Module(image,options);
}

auto RTLib::Backends::Cuda::CurrentContext::CreateArray1DUnique(unsigned int count, unsigned int numChannels, ArrayFormat format, bool useSurface)const noexcept -> std::unique_ptr < Array1D> {
	assert(Get());
	return std::unique_ptr<Array1D>(CreateArray1D(count, numChannels, format, useSurface));
}

auto RTLib::Backends::Cuda::CurrentContext::CreateArray2DUnique(unsigned int width, unsigned int height, unsigned int numChannels, ArrayFormat format, bool useSurface)const noexcept ->  std::unique_ptr < Array2D> {
	assert(Get());
	return std::unique_ptr<Array2D>(CreateArray2D(width, height, numChannels, format, useSurface));
}

auto RTLib::Backends::Cuda::CurrentContext::CreateArray3DUnique(unsigned int width, unsigned int height, unsigned int depth, unsigned int numChannels, ArrayFormat format, bool useSurface)const noexcept ->  std::unique_ptr < Array3D> {
	assert(Get());
	return std::unique_ptr<Array3D>(CreateArray3D(width, height, depth, numChannels, format, useSurface));
}

auto RTLib::Backends::Cuda::CurrentContext::CreateMipmappedArray1DUnique(unsigned int levels, unsigned int count, unsigned int numChannels, ArrayFormat format, bool useSurface)const noexcept -> std::unique_ptr < MipmappedArray1D> {
	assert(Get());
	return std::unique_ptr<MipmappedArray1D>(CreateMipmappedArray1D(levels, count, numChannels, format, useSurface));
}

auto RTLib::Backends::Cuda::CurrentContext::CreateMipmappedArray2DUnique(unsigned int levels, unsigned int width, unsigned int height, unsigned int numChannels, ArrayFormat format, bool useSurface)const noexcept ->  std::unique_ptr < MipmappedArray2D> {
	assert(Get());
	return std::unique_ptr<MipmappedArray2D>(CreateMipmappedArray2D(levels, width, height, numChannels, format, useSurface));
}

auto RTLib::Backends::Cuda::CurrentContext::CreateMipmappedArray3DUnique(unsigned int levels, unsigned int width, unsigned int height, unsigned int depth, unsigned int numChannels, ArrayFormat format, bool useSurface)const noexcept ->  std::unique_ptr < MipmappedArray3D> {
	assert(Get());
	return std::unique_ptr<MipmappedArray3D>(CreateMipmappedArray3D(levels, width, height, depth, numChannels, format, useSurface));
}

auto RTLib::Backends::Cuda::CurrentContext::CreateTextureUniqueFromArray(const Array* arr, const TextureDesc& desc) const noexcept -> std::unique_ptr<Texture>
{
	assert(Get());
	return std::unique_ptr<Texture>(CreateTextureFromArray(arr,desc));
}

auto RTLib::Backends::Cuda::CurrentContext::CreateTextureUniqueFromMipmappedArray(const MipmappedArray* mipmapped, const TextureDesc& desc) const noexcept -> std::unique_ptr<Texture>
{
	assert(Get());
	return std::unique_ptr<Texture>(CreateTextureFromMipmappedArray(mipmapped,desc));
}

auto RTLib::Backends::Cuda::CurrentContext::CreateModuleUniqueFromFile(const char* filename) const noexcept -> std::unique_ptr<Module>
{
	assert(Get());
	return std::unique_ptr<Module>(CreateModuleFromFile(filename));
}

auto RTLib::Backends::Cuda::CurrentContext::CreateModuleUniqueFromData(const std::vector<char>& image) const noexcept -> std::unique_ptr<Module>
{
	assert(Get());
	return std::unique_ptr<Module>(CreateModuleFromData(image));
}

auto RTLib::Backends::Cuda::CurrentContext::CreateModuleUniqueFromData2(const std::vector<char>& image, const std::unordered_map<JitOption, void*>& options) const noexcept -> std::unique_ptr<Module>
{
	assert(Get());
	return std::unique_ptr<Module>(CreateModuleFromData2(image,options));
}

void RTLib::Backends::Cuda::CurrentContext::Copy1DLinearMemory(const LinearMemory* dstMemory, const LinearMemory* srcMemory, const Memory1DCopy& copy)const noexcept {
	assert(Get()&&(dstMemory!=nullptr)&&(srcMemory!=nullptr));
	RTLIB_BACKENDS_CUDA_DEBUG_ASSERT(cuMemcpy(Internals::GetCUdeviceptr(dstMemory)+copy.dstOffsetInBytes, Internals::GetCUdeviceptr(srcMemory)+copy.srcOffsetInBytes,copy.sizeInBytes));
}

void RTLib::Backends::Cuda::CurrentContext::Copy1DFromLinearMemoryToHost(void* dstMemory, const LinearMemory* srcMemory, const Memory1DCopy& copy)const noexcept {
	assert(Get() && (dstMemory != nullptr) && (srcMemory != nullptr));
	RTLIB_BACKENDS_CUDA_DEBUG_ASSERT(cuMemcpyDtoH(static_cast<char*>(dstMemory) + copy.dstOffsetInBytes, Internals::GetCUdeviceptr(srcMemory) + copy.srcOffsetInBytes, copy.sizeInBytes));
}

void RTLib::Backends::Cuda::CurrentContext::Copy1DFromHostToLinearMemory(const LinearMemory* dstMemory,const void* srcMemory, const Memory1DCopy& copy)const noexcept {
	assert(Get() && (dstMemory != nullptr) && (srcMemory != nullptr));
	RTLIB_BACKENDS_CUDA_DEBUG_ASSERT(cuMemcpyHtoD(Internals::GetCUdeviceptr(dstMemory) + copy.dstOffsetInBytes, static_cast<const char*>(srcMemory) + copy.srcOffsetInBytes, copy.sizeInBytes));
}

void RTLib::Backends::Cuda::CurrentContext::Copy2DLinearMemory(const LinearMemory* dstMemory, const LinearMemory* srcMemory, const Memory2DCopy& copy)const noexcept {
	assert(Get() && (dstMemory != nullptr) && (srcMemory != nullptr));
	CUDA_MEMCPY2D memCpy2D;
	Internals::SetCudaMemcpy2DMemoryCopy2D(memCpy2D, copy);
	Internals::SetCudaMemcpy2DSrcLinearMemory(memCpy2D, srcMemory);
	Internals::SetCudaMemcpy2DDstLinearMemory(memCpy2D, dstMemory);
	RTLIB_BACKENDS_CUDA_DEBUG_ASSERT(cuMemcpy2D(&memCpy2D));
}

void RTLib::Backends::Cuda::CurrentContext::Copy2DFromLinearMemoryToHost(void* dstMemory, const LinearMemory* srcMemory, const Memory2DCopy& copy)const noexcept {
	assert(Get() && (dstMemory != nullptr) && (srcMemory != nullptr));
	CUDA_MEMCPY2D memCpy2D;
	Internals::SetCudaMemcpy2DMemoryCopy2D(memCpy2D, copy);
	Internals::SetCudaMemcpy2DSrcLinearMemory(memCpy2D, srcMemory);
	Internals::SetCudaMemcpy2DDstHost(memCpy2D, dstMemory);
	RTLIB_BACKENDS_CUDA_DEBUG_ASSERT(cuMemcpy2D(&memCpy2D));
}

void RTLib::Backends::Cuda::CurrentContext::Copy2DFromHostToLinearMemory(const LinearMemory* dstMemory, const void* srcMemory, const Memory2DCopy& copy)const noexcept {
	assert(Get() && (dstMemory != nullptr) && (srcMemory != nullptr));
	CUDA_MEMCPY2D memCpy2D;
	Internals::SetCudaMemcpy2DMemoryCopy2D(memCpy2D, copy);
	Internals::SetCudaMemcpy2DSrcHost(memCpy2D, srcMemory);
	Internals::SetCudaMemcpy2DDstLinearMemory(memCpy2D, dstMemory);
	RTLIB_BACKENDS_CUDA_DEBUG_ASSERT(cuMemcpy2D(&memCpy2D));
}

void RTLib::Backends::Cuda::CurrentContext::Copy2DLinearMemory2D(const LinearMemory2D* dstMemory, const LinearMemory2D* srcMemory, const Memory2DCopy& copy)const noexcept {
	assert(Get() && (dstMemory != nullptr) && (srcMemory != nullptr));
	CUDA_MEMCPY2D memCpy2D;
	Internals::SetCudaMemcpy2DMemoryCopy2D(memCpy2D, copy);
	Internals::SetCudaMemcpy2DSrcLinearMemory2D(memCpy2D, srcMemory);
	Internals::SetCudaMemcpy2DDstLinearMemory2D(memCpy2D, dstMemory);
	RTLIB_BACKENDS_CUDA_DEBUG_ASSERT(cuMemcpy2D(&memCpy2D));
}

void RTLib::Backends::Cuda::CurrentContext::Copy2DFromLinearMemory2DToLinearMemory(const LinearMemory* dstMemory, const LinearMemory2D* srcMemory, const Memory2DCopy& copy)const noexcept {
	assert(Get() && (dstMemory != nullptr) && (srcMemory != nullptr));
	CUDA_MEMCPY2D memCpy2D;
	Internals::SetCudaMemcpy2DMemoryCopy2D(memCpy2D, copy);
	Internals::SetCudaMemcpy2DSrcLinearMemory2D(memCpy2D, srcMemory);
	Internals::SetCudaMemcpy2DDstLinearMemory(memCpy2D, dstMemory);
	RTLIB_BACKENDS_CUDA_DEBUG_ASSERT(cuMemcpy2D(&memCpy2D));
}

void RTLib::Backends::Cuda::CurrentContext::Copy2DFromLinearMemoryToLinearMemory2D(const LinearMemory2D* dstMemory, const LinearMemory* srcMemory, const Memory2DCopy& copy)const noexcept {
	assert(Get() && (dstMemory != nullptr) && (srcMemory != nullptr));
	CUDA_MEMCPY2D memCpy2D;
	Internals::SetCudaMemcpy2DMemoryCopy2D(memCpy2D, copy);
	Internals::SetCudaMemcpy2DSrcLinearMemory(memCpy2D, srcMemory);
	Internals::SetCudaMemcpy2DDstLinearMemory2D(memCpy2D, dstMemory);
	RTLIB_BACKENDS_CUDA_DEBUG_ASSERT(cuMemcpy2D(&memCpy2D));
}

void RTLib::Backends::Cuda::CurrentContext::Copy2DFromLinearMemory2DToHost(void* dstMemory, const LinearMemory2D* srcMemory, const Memory2DCopy& copy)const noexcept {
	assert(Get() && (dstMemory != nullptr) && (srcMemory != nullptr));
	CUDA_MEMCPY2D memCpy2D;
	Internals::SetCudaMemcpy2DMemoryCopy2D(memCpy2D, copy);
	Internals::SetCudaMemcpy2DSrcLinearMemory2D(memCpy2D, srcMemory);
	Internals::SetCudaMemcpy2DDstHost(memCpy2D, dstMemory);
	RTLIB_BACKENDS_CUDA_DEBUG_ASSERT(cuMemcpy2D(&memCpy2D));
}

void RTLib::Backends::Cuda::CurrentContext::Copy2DFromHostToLinearMemory2D(const LinearMemory2D* dstMemory,const void* srcMemory, const Memory2DCopy& copy)const noexcept {
	assert(Get() && (dstMemory != nullptr) && (srcMemory != nullptr));
	CUDA_MEMCPY2D memCpy2D;
	Internals::SetCudaMemcpy2DMemoryCopy2D(memCpy2D, copy);
	Internals::SetCudaMemcpy2DSrcHost(memCpy2D, srcMemory);
	Internals::SetCudaMemcpy2DDstLinearMemory2D(memCpy2D, dstMemory);
	RTLIB_BACKENDS_CUDA_DEBUG_ASSERT(cuMemcpy2D(&memCpy2D));
}

void RTLib::Backends::Cuda::CurrentContext::Copy2DArray(const Array* dstArray, const Array* srcArray, const Memory2DCopy& copy)const noexcept {
	assert(Get() && (dstArray != nullptr) && (srcArray != nullptr));
	CUDA_MEMCPY2D memCpy2D;
	Internals::SetCudaMemcpy2DMemoryCopy2D(memCpy2D, copy);
	Internals::SetCudaMemcpy2DSrcArray(memCpy2D, srcArray);
	Internals::SetCudaMemcpy2DDstArray(memCpy2D, dstArray);
	RTLIB_BACKENDS_CUDA_DEBUG_ASSERT(cuMemcpy2D(&memCpy2D));
}

void RTLib::Backends::Cuda::CurrentContext::Copy2DFromArrayToHost(void* dstMemory, const Array* srcArray, const Memory2DCopy& copy)const noexcept {
	assert(Get() && (dstMemory != nullptr) && (srcArray != nullptr));
	CUDA_MEMCPY2D memCpy2D;
	Internals::SetCudaMemcpy2DMemoryCopy2D(memCpy2D, copy);
	Internals::SetCudaMemcpy2DSrcArray(memCpy2D, srcArray);
	Internals::SetCudaMemcpy2DDstHost(memCpy2D, dstMemory);
	RTLIB_BACKENDS_CUDA_DEBUG_ASSERT(cuMemcpy2D(&memCpy2D));
}

void RTLib::Backends::Cuda::CurrentContext::Copy2DFromHostToArray(const Array* dstArray, const void* srcMemory, const Memory2DCopy& copy)const noexcept {
	assert(Get() && (dstArray != nullptr) && (srcMemory != nullptr));
	CUDA_MEMCPY2D memCpy2D;
	Internals::SetCudaMemcpy2DMemoryCopy2D(memCpy2D, copy);
	Internals::SetCudaMemcpy2DSrcHost(memCpy2D, srcMemory);
	Internals::SetCudaMemcpy2DDstArray(memCpy2D, dstArray);
	RTLIB_BACKENDS_CUDA_DEBUG_ASSERT(cuMemcpy2D(&memCpy2D));
}

void RTLib::Backends::Cuda::CurrentContext::Copy2DFromArrayToLinearMemory(const LinearMemory* dstMemory, const Array* srcArray, const Memory2DCopy& copy)const noexcept {
	assert(Get() && (dstMemory != nullptr) && (srcArray != nullptr));
	CUDA_MEMCPY2D memCpy2D;
	Internals::SetCudaMemcpy2DMemoryCopy2D(memCpy2D, copy);
	Internals::SetCudaMemcpy2DSrcArray(memCpy2D, srcArray);
	Internals::SetCudaMemcpy2DDstLinearMemory(memCpy2D,dstMemory);
	RTLIB_BACKENDS_CUDA_DEBUG_ASSERT(cuMemcpy2D(&memCpy2D));
}

void RTLib::Backends::Cuda::CurrentContext::Copy2DFromLinearMemoryToArray(const Array* dstArray, const LinearMemory* srcMemory, const Memory2DCopy& copy)const noexcept {
	assert(Get() && (dstArray != nullptr) && (srcMemory != nullptr));
	CUDA_MEMCPY2D memCpy2D;
	Internals::SetCudaMemcpy2DMemoryCopy2D(memCpy2D, copy);
	Internals::SetCudaMemcpy2DSrcLinearMemory(memCpy2D, srcMemory);
	Internals::SetCudaMemcpy2DDstArray(memCpy2D, dstArray);
	RTLIB_BACKENDS_CUDA_DEBUG_ASSERT(cuMemcpy2D(&memCpy2D));
}

void RTLib::Backends::Cuda::CurrentContext::Copy2DFromArrayToLinearMemory2D(const LinearMemory2D* dstMemory, const Array* srcArray, const Memory2DCopy& copy)const noexcept {
	assert(Get() && (dstMemory != nullptr) && (srcArray != nullptr));
	CUDA_MEMCPY2D memCpy2D;
	Internals::SetCudaMemcpy2DMemoryCopy2D(memCpy2D, copy);
	Internals::SetCudaMemcpy2DSrcArray(memCpy2D, srcArray);
	Internals::SetCudaMemcpy2DDstLinearMemory2D(memCpy2D, dstMemory);
	RTLIB_BACKENDS_CUDA_DEBUG_ASSERT(cuMemcpy2D(&memCpy2D));
}

void RTLib::Backends::Cuda::CurrentContext::Copy2DFromLinearMemory2DToArray(const Array* dstArray, const LinearMemory2D* srcMemory, const Memory2DCopy& copy)const noexcept {
	assert(Get() && (dstArray != nullptr) && (srcMemory != nullptr));
	CUDA_MEMCPY2D memCpy2D;
	Internals::SetCudaMemcpy2DMemoryCopy2D(memCpy2D, copy);
	Internals::SetCudaMemcpy2DSrcLinearMemory2D(memCpy2D, srcMemory);
	Internals::SetCudaMemcpy2DDstArray(memCpy2D, dstArray);
	RTLIB_BACKENDS_CUDA_DEBUG_ASSERT(cuMemcpy2D(&memCpy2D));
}

