#include <RTLib/Ext/OPX7/OPX7Natives.h>
#include <RTLib/Ext/OPX7/OPX7Context.h>
#include <RTLib/Ext/OPX7/OPX7Module.h>
#include <RTLib/Ext/OPX7/OPX7ProgramGroup.h>
#include <RTLib/Ext/OPX7/OPX7Pipeline.h>
#include <RTLib/Ext/OPX7/OPX7ShaderTable.h>
#include <RTLib/Ext/OPX7/OPX7Exceptions.h>
#include <RTLib/Ext/CUDA/CUDABuffer.h>
#include <optix_stubs.h>

auto RTLib::Ext::OPX7::OPX7Natives::GetOptixDeviceContext(OPX7Context* context) -> OptixDeviceContext
{
    return context ? context->GetOptixDeviceContext() : nullptr;
}

auto RTLib::Ext::OPX7::OPX7Natives::GetOptixModule(OPX7Module* module) -> OptixModule
{
    return module ? module->GetOptixModule() : nullptr;
}

auto RTLib::Ext::OPX7::OPX7Natives::GetOptixProgramGroup(OPX7ProgramGroup* programGroup) -> OptixProgramGroup
{
    return programGroup ? programGroup->GetOptixProgramGroup() : nullptr;
}

auto RTLib::Ext::OPX7::OPX7Natives::GetOptixPipeline(OPX7Pipeline* pipeline) -> OptixPipeline
{
    return pipeline ? pipeline->GetOptixPipeline() : nullptr;
}

auto RTLib::Ext::OPX7::OPX7Natives::BuildAccelerationStructure(RTLib::Ext::OPX7::OPX7Context* context, const OptixAccelBuildOptions& accelBuildOptions, const std::vector<OptixBuildInput>& buildInputs) -> AccelBuildOutput
{
    OptixAccelBufferSizes bufferSizes = {};
    RTLIB_EXT_OPX7_THROW_IF_FAILED(optixAccelComputeMemoryUsage(OPX7::OPX7Natives::GetOptixDeviceContext(context), &accelBuildOptions, buildInputs.data(), buildInputs.size(), &bufferSizes));
    if ((accelBuildOptions.buildFlags & OPTIX_BUILD_FLAG_ALLOW_COMPACTION) != OPTIX_BUILD_FLAG_ALLOW_COMPACTION) {
        AccelBuildOutput accel = {};
        accel.buffer = std::unique_ptr<RTLib::Ext::CUDA::CUDABuffer>(context->CreateBuffer({ CUDA::CUDAMemoryFlags::eDefault, bufferSizes.outputSizeInBytes }));
        auto  tempBuffer = std::unique_ptr<CUDA::CUDABuffer>(context->CreateBuffer({ CUDA::CUDAMemoryFlags::eDefault, bufferSizes.tempSizeInBytes }));
        RTLIB_EXT_OPX7_THROW_IF_FAILED(optixAccelBuild(
            OPX7::OPX7Natives::GetOptixDeviceContext(context), 0, &accelBuildOptions,
            buildInputs.data(), buildInputs.size(),
            CUDA::CUDANatives::GetCUdeviceptr(tempBuffer.get()), tempBuffer->GetSizeInBytes(),
            CUDA::CUDANatives::GetCUdeviceptr(accel.buffer.get()), accel.buffer->GetSizeInBytes(),
            &accel.handle, nullptr, 0));
        tempBuffer->Destroy();
        return accel;
    }
    else {
        AccelBuildOutput tempAccel = {};
        tempAccel.buffer = std::unique_ptr<CUDA::CUDABuffer>(context->CreateBuffer({ CUDA::CUDAMemoryFlags::eDefault, bufferSizes.outputSizeInBytes }));
        auto compactSizeBuffer = std::unique_ptr<CUDA::CUDABuffer>(context->CreateBuffer({ CUDA::CUDAMemoryFlags::eDefault, sizeof(size_t) }));
        auto  tempBuffer = std::unique_ptr<CUDA::CUDABuffer>(context->CreateBuffer({ CUDA::CUDAMemoryFlags::eDefault, bufferSizes.tempSizeInBytes }));
        OptixAccelEmitDesc compactDesc = {};
        compactDesc.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
        compactDesc.result = CUDA::CUDANatives::GetCUdeviceptr(compactSizeBuffer.get());
        RTLIB_EXT_OPX7_THROW_IF_FAILED(optixAccelBuild(
            OPX7::OPX7Natives::GetOptixDeviceContext(context), nullptr, &accelBuildOptions,
            buildInputs.data(), buildInputs.size(),
            CUDA::CUDANatives::GetCUdeviceptr(tempBuffer.get()), tempBuffer->GetSizeInBytes(),
            CUDA::CUDANatives::GetCUdeviceptr(tempAccel.buffer.get()), tempAccel.buffer->GetSizeInBytes(),
            &tempAccel.handle, &compactDesc, 1));
        size_t compactSize = {};
        context->CopyBufferToMemory(compactSizeBuffer.get(), { { &compactSize,0, sizeof(size_t) } });
        compactSizeBuffer->Destroy();
        std::cout << compactSize << "vs" << bufferSizes.outputSizeInBytes << std::endl;
        if (compactSize < bufferSizes.outputSizeInBytes) {
            AccelBuildOutput accel = {};
            accel.buffer = std::unique_ptr<RTLib::Ext::CUDA::CUDABuffer>(context->CreateBuffer({ CUDA::CUDAMemoryFlags::eDefault, compactSize }));
            RTLIB_EXT_OPX7_THROW_IF_FAILED(optixAccelCompact(OPX7::OPX7Natives::GetOptixDeviceContext(context), nullptr, tempAccel.handle,
                CUDA::CUDANatives::GetCUdeviceptr(accel.buffer.get()),
                accel.buffer->GetSizeInBytes(), &accel.handle));
            tempBuffer->Destroy();
            tempAccel.buffer->Destroy();
            return accel;
        }
        else {
            return tempAccel;
        }
    }
}
