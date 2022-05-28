#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define TINYOBJLOADER_IMPLEMENTATION
#include <RTLibExtOPX7Test.h>
struct Vertex
{
    float3 position;
};
struct ShapeInfo
{
    std::string name = {};
    std::vector<uint3> indices = {};
    uint32_t matID = 0;
};
struct MaterialInfo
{
    std::string diffTexName = {};
    float3 diffColor = {};
    std::string specTexName = {};
    float3 specColor = {};
    std::string emitTexName = {};
    float3 emitColor = {};
    float shinness = 0.0f;
    float refractiveID = 1.0f;
    unsigned int illum = 0;
};
struct WindowState
{
    float curTime = 0.0f;
    float delTime = 0.0f;
    float2 curCurPos = {};
    float2 delCurPos = {};
};
int OfflineSample()
{
    using namespace RTLib::Ext::CUDA;
    using namespace RTLib::Ext::OPX7;
    static constexpr float3 vertices[] = {float3{-0.5f, -0.5f, 0.0f}, float3{0.5f, -0.5f, 0.0f}, float3{0.0f, 0.5f, 0.0f}};
    static constexpr uint3 indices[] = {{0, 1, 2}};
    auto box = rtlib::utils::Box{};
    box.x0 = -0.5f;
    box.y0 = -0.5f;
    box.z0 = -0.5f;
    box.x1 = 0.5f;
    box.y1 = 0.5f;
    box.z1 = 0.5f;
    // auto vertices = box.getVertices();
    // auto indices = box.getIndices();
    try
    {
        int width = 1024;
        int height = 1024;
        auto camera = rtlib::ext::Camera({0.0f, 0.0f, 2.0f}, {0.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 3.0f}, 45.0f, 1.0f);
        // contextはcopy/move不可
        auto contextCreateDesc = OPX7ContextCreateDesc();
        contextCreateDesc.validationMode = OPX7ContextValidationMode::eALL;
        contextCreateDesc.level = OPX7ContextValidationLogLevel::ePrint;
        auto context = std::make_unique<OPX7Context>(contextCreateDesc);
        context->Initialize();
        auto d_vertices = std::unique_ptr<CUDABuffer>(context->CreateBuffer({CUDAMemoryFlags::eDefault, sizeof(vertices[0]) * std::size(vertices), (void *)std::data(vertices)}));
        auto d_pVertices = CUDANatives::GetCUdeviceptr(d_vertices.get());
        auto d_indices = std::unique_ptr<CUDABuffer>(context->CreateBuffer({CUDAMemoryFlags::eDefault, sizeof(indices[0]) * std::size(indices), (void *)std::data(indices)}));
        auto accelBuildOptions = OptixAccelBuildOptions();
        accelBuildOptions.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE | OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
        accelBuildOptions.motionOptions = {};
        accelBuildOptions.operation = OPTIX_BUILD_OPERATION_BUILD;
        auto geometryFlags = std::vector<unsigned int>{
            OPTIX_GEOMETRY_FLAG_NONE};
        auto buildInput = OptixBuildInput();
        buildInput.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
        buildInput.triangleArray.vertexBuffers = &d_pVertices;
        buildInput.triangleArray.numVertices = std::size(vertices);
        buildInput.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
        buildInput.triangleArray.vertexStrideInBytes = sizeof(float3);
        buildInput.triangleArray.indexBuffer = CUDANatives::GetCUdeviceptr(d_indices.get());
        buildInput.triangleArray.numIndexTriplets = std::size(indices);
        buildInput.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
        buildInput.triangleArray.indexStrideInBytes = sizeof(uint3);
        buildInput.triangleArray.numSbtRecords = 1;
        buildInput.triangleArray.flags = geometryFlags.data();
        auto pipelineCompileOptions = OPX7PipelineCompileOptions{};
        auto [outputBuffer, traversableHandle] = OPX7Natives::BuildAccelerationStructure(context.get(), accelBuildOptions, {buildInput});
        {
            pipelineCompileOptions.usesMotionBlur = false;
            pipelineCompileOptions.traversableGraphFlags = OPX7TraversableGraphFlagsAllowSingleGAS;
            pipelineCompileOptions.numAttributeValues = 3;
            pipelineCompileOptions.numPayloadValues = 3;
            pipelineCompileOptions.launchParamsVariableNames = "params";
            pipelineCompileOptions.usesPrimitiveTypeFlags = OPX7PrimitiveTypeFlagsTriangle;
            pipelineCompileOptions.exceptionFlags = OPX7ExceptionFlagBits::OPX7ExceptionFlagsNone;
        }
        auto ptxSource = std::vector<char>();
        {
            auto cuFile = std::ifstream(RTLIB_EXT_OPX7_TEST_CUDA_PATH "/SimpleKernel.ptx", std::ios::binary);
            ptxSource = std::vector<char>((std::istreambuf_iterator<char>(cuFile)), (std::istreambuf_iterator<char>()));
        }
        // contextはcopy不可
        auto moduleCreateDesc = OPX7ModuleCreateDesc{};
        {
            moduleCreateDesc.ptxBinary = ptxSource;
            moduleCreateDesc.pipelineOptions = pipelineCompileOptions;
            moduleCreateDesc.moduleOptions.optLevel = OPX7CompileOptimizationLevel::eDefault;
            moduleCreateDesc.moduleOptions.debugLevel = OPX7CompileDebugLevel::eMinimal;
            moduleCreateDesc.moduleOptions.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
            moduleCreateDesc.moduleOptions.payloadTypes = {};
            moduleCreateDesc.moduleOptions.boundValueEntries = {};
        }
        auto module = std::unique_ptr<RTLib::Ext::OPX7::OPX7Module>(context->CreateOPXModule(moduleCreateDesc));
        auto programGroups = context->CreateOPXProgramGroups({OPX7ProgramGroupCreateDesc::Raygen({module.get(), "__raygen__rg"}),
                                                              OPX7ProgramGroupCreateDesc::Miss({module.get(), "__miss__ms"}),
                                                              OPX7ProgramGroupCreateDesc::Hitgroup({module.get(), "__closesthit__ch"})});
        auto raygenPG = std::unique_ptr<OPX7ProgramGroup>(programGroups[0]);
        auto missPG = std::unique_ptr<OPX7ProgramGroup>(programGroups[1]);
        auto hitgroupPG = std::unique_ptr<OPX7ProgramGroup>(programGroups[2]);
        auto pipelineCreateDesc = OPX7PipelineCreateDesc{};
        {
            pipelineCreateDesc.linkOptions.maxTraceDepth = 1;
            pipelineCreateDesc.linkOptions.debugLevel = OPX7CompileDebugLevel::eMinimal;
            pipelineCreateDesc.compileOptions = pipelineCompileOptions;
            pipelineCreateDesc.programGroups = {
                raygenPG.get(), missPG.get(), hitgroupPG.get()};
        }
        auto pipeline = std::unique_ptr<OPX7Pipeline>(context->CreateOPXPipeline(pipelineCreateDesc));
        auto raygenRecord = raygenPG->GetRecord<RayGenData>();
        raygenRecord.data.eye = camera.getEye();
        auto [u, v, w] = camera.getUVW();
        raygenRecord.data.u = u;
        raygenRecord.data.v = v;
        raygenRecord.data.w = w;
        auto missRecord = missPG->GetRecord<MissData>();
        missRecord.data.bgColor = float4{1.0f, 0.0f, 0.0f, 1.0f};
        auto hitgroupRecord = hitgroupPG->GetRecord<HitgroupData>();
        hitgroupRecord.data.vertices = reinterpret_cast<float3 *>(CUDANatives::GetCUdeviceptr(d_vertices.get()));
        hitgroupRecord.data.indices = reinterpret_cast<uint3 *>(CUDANatives::GetCUdeviceptr(d_indices.get()));
        hitgroupRecord.data.diffuse = make_float3(1.0f, 1.0f, 1.0f);
        hitgroupRecord.data.emission = make_float3(0.3f, 0.3f, 0.3f);
        auto shaderTableDesc = OPX7ShaderTableCreateDesc();
        {
            shaderTableDesc.raygenRecordSizeInBytes = sizeof(raygenRecord);
            shaderTableDesc.missRecordStrideInBytes = sizeof(missRecord);
            shaderTableDesc.missRecordCount = 1;
            shaderTableDesc.hitgroupRecordStrideInBytes = sizeof(hitgroupRecord);
            shaderTableDesc.hitgroupRecordCount = 1;
        }
        auto shaderTable = std::unique_ptr<OPX7ShaderTable>(context->CreateOPXShaderTable(shaderTableDesc));
        {
            shaderTable->SetHostRaygenRecordTypeData(raygenRecord);
            shaderTable->SetHostMissRecordTypeData(    0, missRecord);
            shaderTable->SetHostHitgroupRecordTypeData(0, hitgroupRecord);
            shaderTable->Upload();
        }
        auto d_pixel = std::unique_ptr<CUDABuffer>(context->CreateBuffer({CUDAMemoryFlags::eDefault, static_cast<size_t>(sizeof(uchar4) * width * height), nullptr}));
        auto params = Params();
        params.image = reinterpret_cast<uchar4 *>(CUDANatives::GetCUdeviceptr(d_pixel.get()));
        params.width = width;
        params.height = height;
        params.gasHandle = traversableHandle;
        auto d_params = std::unique_ptr<CUDABuffer>(context->CreateBuffer({CUDAMemoryFlags::eDefault, static_cast<size_t>(sizeof(Params)), &params}));
        auto img_pixels = std::vector<uchar4>(width * height);
        auto stream = std::unique_ptr<CUDAStream>(context->CreateStream());
        pipeline->Launch(stream.get(),CUDABufferView(d_params.get(), 0, d_params->GetSizeInBytes()), shaderTable.get(), width, height, 1);
        stream->CopyBufferToMemory(d_pixel.get(), {{img_pixels.data(), 0, width * height * sizeof(uchar4)}});
        stream->Synchronize();
        stbi_write_bmp("tekitou.bmp", width, height, 4, img_pixels.data());
        stream->Destroy();
    }
    catch (std::runtime_error &err)
    {
        std::cerr << err.what() << std::endl;
    }
    return 0;
}
int main()
{
    OfflineSample();
}
