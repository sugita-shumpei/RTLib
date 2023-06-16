#include <TestLib/AccelerationStructure.h>
#include <TestLib/Utils.h>
#include <OptiXToolkit/Error/ErrorCheck.h>
#include <optix_stubs.h>
struct TestLib::AccelerationStructure::Impl
{
    struct BuildInputStorageTriangleArray
    {
        std::vector<CUdeviceptr>  vertexBuffers;
        std::vector<unsigned int> flags;
    };
    struct BuildInputStorageSphereArray
    {
        std::vector<CUdeviceptr>  vertexBuffers;
        std::vector<CUdeviceptr>  radiusBuffers;
        std::vector<unsigned int> flags;
    };
    struct BuildInputStorageCustomPrimitiveArray
    {
        std::vector<CUdeviceptr> aabbBuffers;
        std::vector<unsigned int> flags;
    };
    struct BuildInputStorageCurveArray
    {
        std::vector<CUdeviceptr>  vertexBuffers;
        std::vector<CUdeviceptr>  widthBuffers;
    };
    struct BuildInputStorageInstanceArray
    {

    };

    Impl(TestLib::Context* ctx) :      context{ ctx } {}
    TestLib::Context*                  context = nullptr;
    OptixAccelBuildOptions             options = {};
    std::vector<OptixBuildInput>       buildInputs = {};
    std::vector<std::shared_ptr<void>> buildInputStorages = {};
    otk::DeviceBuffer                  outputBuffer = {};
    OptixTraversableHandle             outputHandle = {};
};

TestLib::AccelerationStructure::AccelerationStructure(TestLib::Context* context):
    m_Impl{new Impl(context)}
{}

TestLib::AccelerationStructure::~AccelerationStructure()
{
    m_Impl.reset();
}

auto TestLib::AccelerationStructure::get_options() const noexcept -> const OptixAccelBuildOptions&
{
    // TODO: return ステートメントをここに挿入します
    return m_Impl->options;
}

void TestLib::AccelerationStructure::set_options(const OptixAccelBuildOptions& options) noexcept
{
    m_Impl->options = options;
}

auto TestLib::AccelerationStructure::get_build_inputs() const noexcept -> const std::vector<OptixBuildInput>&
{
    // TODO: return ステートメントをここに挿入します
    return m_Impl->buildInputs;
}

void TestLib::AccelerationStructure::set_build_inputs(const std::vector<OptixBuildInput>& buildInputs) noexcept
{
    set_num_build_inputs(buildInputs.size());
    for (size_t i = 0; i < buildInputs.size(); ++i) {
        set_build_input(i, buildInputs[i]);
    }
}

auto TestLib::AccelerationStructure::get_num_build_inputs() const noexcept -> size_t {
    return m_Impl->buildInputs.size();
}

void TestLib::AccelerationStructure::set_num_build_inputs(size_t numBuildInputs) {
    m_Impl->buildInputs.resize(numBuildInputs);
    m_Impl->buildInputStorages.resize(numBuildInputs);
}

auto TestLib::AccelerationStructure::get_build_input(size_t idx) const noexcept -> const OptixBuildInput&
{
    // TODO: return ステートメントをここに挿入します
    return m_Impl->buildInputs[idx];
}

void TestLib::AccelerationStructure::set_build_input(size_t idx, const OptixBuildInput& buildInput) noexcept
{
    m_Impl->buildInputs[idx] = buildInput;
    switch(buildInput.type)
    {
    case OPTIX_BUILD_INPUT_TYPE_TRIANGLES:
        {
        auto buildInputStorage = new Impl::BuildInputStorageTriangleArray();
        auto& triangleArray = buildInput.triangleArray;
        auto numKeys = std::max<size_t>(1, m_Impl->options.motionOptions.numKeys);
        auto numSbtRecords = std::max<size_t>(triangleArray.numSbtRecords, 1);
        buildInputStorage->flags = std::vector<unsigned int>(numSbtRecords);
        buildInputStorage->vertexBuffers = std::vector<CUdeviceptr>(numKeys);
        for (size_t i = 0; i < numSbtRecords; ++i)
        {
            buildInputStorage->flags[i] = triangleArray.flags[i];
        }
        for (size_t i = 0; i < numKeys; ++i)
        {
            buildInputStorage->vertexBuffers[i] = triangleArray.vertexBuffers[i];
        }
        m_Impl->buildInputs[idx].triangleArray.flags = buildInputStorage->flags.data();
        m_Impl->buildInputs[idx].triangleArray.vertexBuffers = buildInputStorage->vertexBuffers.data();
        m_Impl->buildInputStorages[idx].reset(buildInputStorage);
        }
        break;
    case OPTIX_BUILD_INPUT_TYPE_SPHERES:
        {
        auto buildInputStorage = new Impl::BuildInputStorageSphereArray();
        auto& sphereArray = buildInput.sphereArray;
        auto numKeys = std::max<size_t>(1, m_Impl->options.motionOptions.numKeys);
        auto numSbtRecords = std::max<size_t>(sphereArray.numSbtRecords, 1);
        auto singleRadius = sphereArray.singleRadius;

        buildInputStorage->flags = std::vector<unsigned int>(numSbtRecords);
        buildInputStorage->vertexBuffers = std::vector<CUdeviceptr>(numKeys);
        buildInputStorage->radiusBuffers = std::vector<CUdeviceptr>(singleRadius ? 1 : numKeys);

        for (size_t i = 0; i < numSbtRecords; ++i)
        {
            buildInputStorage->flags[i] = sphereArray.flags[i];
        }
        for (size_t i = 0; i < numKeys; ++i)
        {
            buildInputStorage->vertexBuffers[i] = sphereArray.vertexBuffers[i];
        }
        if (!singleRadius)
        {
            for (size_t i = 0; i < numKeys; ++i)
            {
                buildInputStorage->radiusBuffers[i] = sphereArray.radiusBuffers[i];
            }
        }
        else {
            buildInputStorage->radiusBuffers[0] = sphereArray.radiusBuffers[0];
        }

        m_Impl->buildInputs[idx].sphereArray.flags = buildInputStorage->flags.data();
        m_Impl->buildInputs[idx].sphereArray.vertexBuffers = buildInputStorage->vertexBuffers.data();
        m_Impl->buildInputs[idx].sphereArray.radiusBuffers = buildInputStorage->radiusBuffers.data();
        m_Impl->buildInputStorages[idx].reset(buildInputStorage);
        }
        break;
    case OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES:
        {
        auto buildInputStorage = new Impl::BuildInputStorageCustomPrimitiveArray();
        auto& customPrimitiveArray = buildInput.customPrimitiveArray;
        auto numKeys = std::max<size_t>(1, m_Impl->options.motionOptions.numKeys);
        auto numSbtRecords = std::max<size_t>(customPrimitiveArray.numSbtRecords, 1);

        buildInputStorage->flags = std::vector<unsigned int>(numSbtRecords);
        buildInputStorage->aabbBuffers = std::vector<CUdeviceptr>(numKeys);

        for (size_t i = 0; i < numSbtRecords; ++i)
        {
            buildInputStorage->flags[i] = customPrimitiveArray.flags[i];
        }
        for (size_t i = 0; i < numKeys; ++i)
        {
            buildInputStorage->aabbBuffers[i] = customPrimitiveArray.aabbBuffers[i];
        }
        m_Impl->buildInputs[idx].customPrimitiveArray.flags = buildInputStorage->flags.data();
        m_Impl->buildInputs[idx].customPrimitiveArray.aabbBuffers = buildInputStorage->aabbBuffers.data();
        m_Impl->buildInputStorages[idx].reset(buildInputStorage);
        }
        break;
    case OPTIX_BUILD_INPUT_TYPE_CURVES:
        {
        auto buildInputStorage = new Impl::BuildInputStorageCurveArray();
        auto& curveArray = buildInput.curveArray;
        auto numKeys = std::max<size_t>(1, m_Impl->options.motionOptions.numKeys);

        buildInputStorage->vertexBuffers = std::vector<CUdeviceptr>(numKeys);
        buildInputStorage->widthBuffers  = std::vector<CUdeviceptr>(numKeys);

        for (size_t i = 0; i < numKeys; ++i)
        {
            buildInputStorage->vertexBuffers[i] = curveArray.vertexBuffers[i];
            buildInputStorage->widthBuffers[i]  = curveArray.widthBuffers[i];
        }
        m_Impl->buildInputs[idx].curveArray.vertexBuffers = buildInputStorage->vertexBuffers.data();
        //m_Impl->buildInputs[idx].curveArray.normalBuffers = buildInputStorage->normalBuffers.data();
        m_Impl->buildInputs[idx].curveArray.widthBuffers  = buildInputStorage->widthBuffers.data();
        m_Impl->buildInputStorages[idx].reset(buildInputStorage);
        }
        break;
    }
}

auto TestLib::AccelerationStructure::get_opx7_traversable_handle() const noexcept -> OptixTraversableHandle
{
    return m_Impl->outputHandle;
}

void TestLib::AccelerationStructure::build(otk::DeviceBuffer* tempBuffer)
{
    build_async(nullptr, tempBuffer);
}

void TestLib::AccelerationStructure::build_async(CUstream stream, otk::DeviceBuffer* tempBuffer)
{
    auto accelOptions = m_Impl->options;
    accelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes bufferSizes = {};
    OTK_ERROR_CHECK(optixAccelComputeMemoryUsage(
        m_Impl->context->get_opx7_device_context(), &accelOptions, m_Impl->buildInputs.data(), m_Impl->buildInputs.size(), &bufferSizes
    ));

    bool allowCompaction = (accelOptions.buildFlags&OPTIX_BUILD_FLAG_ALLOW_COMPACTION);
    size_t tempBufferSize = compute_aligned_size(bufferSizes.tempSizeInBytes, OPTIX_ACCEL_BUFFER_BYTE_ALIGNMENT);
    size_t outputBufferSize = compute_aligned_size(bufferSizes.outputSizeInBytes, OPTIX_ACCEL_BUFFER_BYTE_ALIGNMENT);
    size_t descBufferSize = sizeof(uint64_t);
    size_t totalTempBufferSize = tempBufferSize;

    if (!allowCompaction) {
        m_Impl->outputBuffer.resize(outputBufferSize);
    }
    else {
        totalTempBufferSize += (outputBufferSize + descBufferSize);
    }

    bool allocTempBuffer = !tempBuffer;
    if (allocTempBuffer) {
        tempBuffer = new otk::DeviceBuffer(totalTempBufferSize);
    }
    else {
        if (tempBuffer->size() < totalTempBufferSize) {
            tempBuffer->resize(totalTempBufferSize);
        }
    }

    auto outputBufferPtr = static_cast<CUdeviceptr>(0);
    auto tempBufferPtr = static_cast<CUdeviceptr>(0);
    auto descBufferPtr = static_cast<CUdeviceptr>(0);

    if (allowCompaction)
    {
        tempBufferPtr = reinterpret_cast<CUdeviceptr>(tempBuffer->devicePtr());
        outputBufferPtr = tempBufferPtr + static_cast<CUdeviceptr>(tempBufferSize);
        descBufferPtr = outputBufferPtr + static_cast<CUdeviceptr>(outputBufferSize);
    }
    else {
        tempBufferPtr = reinterpret_cast<CUdeviceptr>(tempBuffer->devicePtr());
        outputBufferPtr = reinterpret_cast<CUdeviceptr>(m_Impl->outputBuffer.devicePtr());
        descBufferPtr = 0;

    }

    OptixAccelEmitDesc emitDescs[1] = {};
    emitDescs[0].type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    emitDescs[0].result = descBufferPtr;

    OptixTraversableHandle tempHandle = 0;

    OTK_ERROR_CHECK(optixAccelBuild(
        m_Impl->context->get_opx7_device_context(),
        stream,
        &accelOptions,
        m_Impl->buildInputs.data(), m_Impl->buildInputs.size(),
        tempBufferPtr,
        bufferSizes.tempSizeInBytes,
        outputBufferPtr,
        bufferSizes.outputSizeInBytes,
        &tempHandle,
        allowCompaction ? emitDescs : nullptr,
        allowCompaction ? 1 : 0
    ));

    if (allowCompaction)
    {
        uint64_t compactionSize = 0;
        OTK_ERROR_CHECK(cuMemcpyDtoHAsync(&compactionSize, descBufferPtr, sizeof(uint64_t), stream));
        if (compactionSize < outputBufferSize)
        {
            m_Impl->outputBuffer.resize(compactionSize);
            outputBufferPtr = reinterpret_cast<CUdeviceptr>(m_Impl->outputBuffer.devicePtr());

            OTK_ERROR_CHECK(optixAccelCompact(
                m_Impl->context->get_opx7_device_context(),
                stream,
                tempHandle,
                outputBufferPtr,
                outputBufferSize,
                &m_Impl->outputHandle
            ));
        }
    }
    else
    {
        m_Impl->outputHandle = tempHandle;
    }
    if (allocTempBuffer) {
        delete tempBuffer;
    }
}
