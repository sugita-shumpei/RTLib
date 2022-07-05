#define __CUDACC__
#include "SimpleKernel.h"
struct HitRecordUserData
{
    float3 radiance;
    float3 throughPut;
    float3 bsdfVal;
    float  bsdfPdf;
    unsigned int depth;
};
struct HitRecordUserDebugData
{
    float3 diffuse;
    float3 specular;
    float3 emission;
    uint3  gridIndex;
    float3 gridValue;
    float  shinness;
    float  refrIndx;
};
extern "C" {
    __constant__ Params params;
}
namespace rtlib = RTLib::Ext::CUDA::Math;
template<typename RNG>
static __forceinline__ __device__ float3       sampleCosinePDF(const float3& normal, RNG& rng)
{
    rtlib::ONB onb(normal);
    return onb.local(rtlib::random_cosine_direction(rng));
}
template<typename RNG>
static __forceinline__ __device__ float3       samplePhongPDF(const float3& reflectDir, float shinness, RNG& rng)
{
    rtlib::ONB onb(reflectDir);
    const auto cosTht = powf(rtlib::random_float1(0.0f, 1.0f, rng), 1.0f / (shinness + 1.0f));
    const auto sinTht = sqrtf(1.0f - cosTht * cosTht);
    const auto phi = rtlib::random_float1(0.0f, RTLIB_M_2PI, rng);
    return onb.local(make_float3(sinTht * cosf(phi), sinTht * sinf(phi), cosTht));
}
static __forceinline__ __device__ float        getValPhongPDF(const float3& direction, const float3& reflectDir, float shinness)
{

    const auto reflCos = rtlib::max(rtlib::dot(reflectDir, direction), 0.0f);
    return (shinness + 2.0f) * powf(reflCos, shinness) / RTLIB_M_2PI;
}
extern "C" __global__ void     __raygen__default () {
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();
    auto* rgData    = reinterpret_cast<RayGenData*>(optixGetSbtDataPointer());
    const auto seed = params.seedBuffer[params.width * idx.y + idx.x];
    float3 result   = params.accumBuffer[params.width * idx.y + idx.x];
    BasicHitRecord<HitRecordUserData> hrec; 
    hrec.seed = seed;

    float3 color = make_float3(0.0f);
    rtlib::Xorshift32 xor32(hrec.seed);

    const auto gitter = rtlib::random_float2(xor32);
    const float2 d = make_float2(
        (2.0f * static_cast<float>(idx.x + gitter.x) / static_cast<float>(dim.x)) - 1.0,
        (2.0f * static_cast<float>(idx.y + gitter.y) / static_cast<float>(dim.y)) - 1.0);

    hrec.rayOrigin    = rgData->GetRayOrigin();
    hrec.rayDirection = rgData->GetRayDirection(d);
    hrec.rayDistance  = 0.0f;
    hrec.cosine       = 0.0f;
    hrec.seed         = xor32.m_seed;
    hrec.flags        = HIT_RECORD_FLAG_COUNT_EMITTED;

    hrec.userData.throughPut = make_float3(1.0f);
    hrec.userData.radiance   = make_float3(0.0f);
    hrec.userData.bsdfVal    = make_float3(0.0f);
    hrec.userData.bsdfPdf    = 0.0f;
    hrec.userData.depth      = 0;

    while (true) {
        TraceRadiance(params.gasHandle, hrec.rayOrigin, hrec.rayDirection, 0.0001f, 1.0e20f, hrec);

        color += hrec.userData.radiance;

        ++hrec.userData.depth;

        if (isnan(hrec.rayDirection.x)|| isnan(hrec.rayDirection.y)|| isnan(hrec.rayDirection.z)||
            isnan(hrec.userData.radiance.x) || isnan(hrec.userData.radiance.y) || isnan(hrec.userData.radiance.z)) {
            printf("error\n");
            break;
        }

        if ((hrec.flags & HIT_RECORD_FLAG_FINISH) || (hrec.userData.depth >= params.maxDepth)) {
            break;
        }
    }
    
    result += color;
    
    // printf("%f, %lf\n", texCoord.x, texCoord.y);
    params.accumBuffer[params.width * idx.y + idx.x] = result;
    result =  result  / static_cast<float>(params.samplesForLaunch + params.samplesForAccum);
    result = (result) / (make_float3(1.0f) + result);
    params.frameBuffer[params.width * idx.y + idx.x] = make_uchar4(
        rtlib::min(static_cast<int>(rtlib::linear_to_gamma(result.x) * 255.99f), 255),
        rtlib::min(static_cast<int>(rtlib::linear_to_gamma(result.y) * 255.99f), 255),
        rtlib::min(static_cast<int>(rtlib::linear_to_gamma(result.z) * 255.99f), 255),
        255
    );
    params.seedBuffer [params.width * idx.y + idx.x] = hrec.seed;
}
extern "C" __global__ void     __raygen__debug   () {
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();
    auto* rgData = reinterpret_cast<RayGenData*>(optixGetSbtDataPointer());
    const auto seed = params.seedBuffer[params.width * idx.y + idx.x];
    BasicHitRecord<HitRecordUserDebugData> hrec;
    hrec.seed = seed;

    float3 color = make_float3(0.0f);
    rtlib::Xorshift32 xor32(hrec.seed);

    const auto gitter = rtlib::random_float2(xor32);
    const float2 d = make_float2(
        (2.0f * static_cast<float>(idx.x + gitter.x) / static_cast<float>(dim.x)) - 1.0,
        (2.0f * static_cast<float>(idx.y + gitter.y) / static_cast<float>(dim.y)) - 1.0);

    hrec.rayOrigin          = rgData->GetRayOrigin();
    hrec.rayDirection       = rgData->GetRayDirection(d);
    hrec.rayDistance        = 0.0f;
    hrec.cosine             = 0.0f;
    hrec.seed               = xor32.m_seed;
    hrec.userData.gridValue = make_float3(0.0f);
    hrec.userData.diffuse   = make_float3(1.0f);
    hrec.userData.emission  = make_float3(0.0f);
    hrec.userData.specular  = make_float3(0.0f);
    hrec.userData.shinness  = 0.0f;
    hrec.userData.refrIndx  = 0;

    TraceRadiance(params.gasHandle, hrec.rayOrigin, hrec.rayDirection, 0.0001f, 1.0e20f, hrec);
    // printf("%f, %lf\n", texCoord.x, texCoord.y);
    //float3 result = make_float3(hrec.userData.gridValue /static_cast<float>(params.samplesForAccum));
    if (params.debugFrameType == DEBUG_FRAME_TYPE_NORMAL)
    {
        auto color = (hrec.normal + make_float3(1.0f)) / 2.0f;
        params.frameBuffer[params.width * idx.y + idx.x] = make_uchar4(
            rtlib::min(static_cast<int>(rtlib::linear_to_gamma(color.x) * 255.99f), 255),
            rtlib::min(static_cast<int>(rtlib::linear_to_gamma(color.y) * 255.99f), 255),
            rtlib::min(static_cast<int>(rtlib::linear_to_gamma(color.z) * 255.99f), 255),
            255
        );
    }
    if (params.debugFrameType == DEBUG_FRAME_TYPE_DEPTH)
    {
        auto color = make_float3(hrec.rayDistance / (1.0f + hrec.rayDistance));
        params.frameBuffer[params.width * idx.y + idx.x] = make_uchar4(
            rtlib::min(static_cast<int>(rtlib::linear_to_gamma(color.x) * 255.99f), 255),
            rtlib::min(static_cast<int>(rtlib::linear_to_gamma(color.y) * 255.99f), 255),
            rtlib::min(static_cast<int>(rtlib::linear_to_gamma(color.z) * 255.99f), 255),
            255
        );
    }
    if (params.debugFrameType == DEBUG_FRAME_TYPE_DIFFUSE)
    {
        params.frameBuffer[params.width * idx.y + idx.x] = make_uchar4(
            rtlib::min(static_cast<int>(rtlib::linear_to_gamma(hrec.userData.diffuse.x) * 255.99f), 255),
            rtlib::min(static_cast<int>(rtlib::linear_to_gamma(hrec.userData.diffuse.y) * 255.99f), 255),
            rtlib::min(static_cast<int>(rtlib::linear_to_gamma(hrec.userData.diffuse.z) * 255.99f), 255),
            255
        );
    }
    if (params.debugFrameType == DEBUG_FRAME_TYPE_SPECULAR)
    {
        params.frameBuffer[params.width * idx.y + idx.x] = make_uchar4(
            rtlib::min(static_cast<int>(rtlib::linear_to_gamma(hrec.userData.specular.x) * 255.99f), 255),
            rtlib::min(static_cast<int>(rtlib::linear_to_gamma(hrec.userData.specular.y) * 255.99f), 255),
            rtlib::min(static_cast<int>(rtlib::linear_to_gamma(hrec.userData.specular.z) * 255.99f), 255),
            255
        );
    }
    if (params.debugFrameType == DEBUG_FRAME_TYPE_EMISSION)
    {
        params.frameBuffer[params.width * idx.y + idx.x] = make_uchar4(
            rtlib::min(static_cast<int>(rtlib::linear_to_gamma(hrec.userData.emission.x) * 255.99f), 255),
            rtlib::min(static_cast<int>(rtlib::linear_to_gamma(hrec.userData.emission.y) * 255.99f), 255),
            rtlib::min(static_cast<int>(rtlib::linear_to_gamma(hrec.userData.emission.z) * 255.99f), 255),
            255
        );
    }
    if (params.debugFrameType == DEBUG_FRAME_TYPE_SHINNESS)
    {
        auto color = make_float3(hrec.userData.shinness / (1.0f + hrec.userData.shinness));
        params.frameBuffer[params.width * idx.y + idx.x] = make_uchar4(
            rtlib::min(static_cast<int>(rtlib::linear_to_gamma(color.x) * 255.99f), 255),
            rtlib::min(static_cast<int>(rtlib::linear_to_gamma(color.y) * 255.99f), 255),
            rtlib::min(static_cast<int>(rtlib::linear_to_gamma(color.z) * 255.99f), 255),
            255
        );
    }
    if (params.debugFrameType == DEBUG_FRAME_TYPE_REFR_INDEX)
    {
        auto color = make_float3(hrec.userData.refrIndx / (1.0f + hrec.userData.refrIndx));
        params.frameBuffer[params.width * idx.y + idx.x] = make_uchar4(
            rtlib::min(static_cast<int>(rtlib::linear_to_gamma(color.x) * 255.99f), 255),
            rtlib::min(static_cast<int>(rtlib::linear_to_gamma(color.y) * 255.99f), 255),
            rtlib::min(static_cast<int>(rtlib::linear_to_gamma(color.z) * 255.99f), 255),
            255
        );
    }
    if (params.debugFrameType == DEBUG_FRAME_TYPE_GRID_VALUE)
    {

        params.frameBuffer[params.width * idx.y + idx.x] = make_uchar4(
            rtlib::min(static_cast<int>(rtlib::linear_to_gamma(hrec.userData.gridValue.x) * 255.99f), 255),
            rtlib::min(static_cast<int>(rtlib::linear_to_gamma(hrec.userData.gridValue.y) * 255.99f), 255),
            rtlib::min(static_cast<int>(rtlib::linear_to_gamma(hrec.userData.gridValue.z) * 255.99f), 255),
            255
        );
    }
    params.seedBuffer[params.width * idx.y + idx.x] = hrec.seed;
}
extern "C" __global__ void       __miss__radiance() {
    auto* hrec = BasicHitRecord<HitRecordUserData>::GetGlobalPointer();
    auto* msData = reinterpret_cast<MissData*>(optixGetSbtDataPointer());

    hrec->SetGlobalRayOrigin(optixGetWorldRayOrigin());
    hrec->SetGlobalRayDirAndTmax(make_float4(optixGetWorldRayDirection(), optixGetRayTmax()));

    hrec->cosine = 0.0f;
    hrec->flags |= HIT_RECORD_FLAG_FINISH;

    hrec->userData.radiance = hrec->userData.throughPut * make_float3(msData->bgColor.x, msData->bgColor.y, msData->bgColor.z);
    hrec->userData.bsdfVal  = make_float3(1.0f);
}
extern "C" __global__ void __closesthit__radiance() {
    auto* hrec   = BasicHitRecord<HitRecordUserData>::GetGlobalPointer();
    auto* hgData = reinterpret_cast<HitgroupData*>(optixGetSbtDataPointer());
    auto primitiveId = optixGetPrimitiveIndex();
    auto uv = optixGetTriangleBarycentrics();

    auto distance  = optixGetRayTmax();
    auto texCrd   = hgData->GetTexCrd(uv,primitiveId);
    auto normal   = hgData->GetNormal(uv,primitiveId);
    auto position = optixGetWorldRayOrigin() + distance * optixGetWorldRayDirection();
    auto reflDir  = rtlib::normalize(rtlib::reflect(optixGetWorldRayDirection(), normal));

    auto diffuse   = hgData->SampleDiffuse(texCrd);
    auto specular  = hgData->SampleSpecular(texCrd);
    auto emission  = hgData->SampleEmission(texCrd);
    auto shinness = hgData->shinness;

    auto xor32 = rtlib::Xorshift32(hrec->seed);

    auto  direction       = make_float3(0.0f);
    auto   cosine          = float(0.0f);
    auto  bsdfVal         = make_float3(0.0f);
    auto   bsdfPdf         = float(0.0f);
    auto  prevThroughput  = hrec->userData.throughPut;

    if (emission.x * emission.y * emission.z <= 0.0f) {
        auto direction0 = sampleCosinePDF(normal, xor32);
        auto direction1 = samplePhongPDF(reflDir, shinness, xor32);
        auto cosine0     = rtlib::dot(direction0, normal);
        auto cosine1     = rtlib::dot(direction1, normal);
        auto cosinePdf0  = rtlib::max(cosine0 * RTLIB_M_INV_PI, 0.0f);
        auto cosinePdf1  = rtlib::max(cosine1 * RTLIB_M_INV_PI, 0.0f);
        auto phongPdf0   = getValPhongPDF(direction0, reflDir, shinness);
        auto phongPdf1   = getValPhongPDF(direction1, reflDir, shinness);
        auto aver_diff   = ( diffuse.x +  diffuse.y +  diffuse.z) / 3.0f;
        auto aver_spec   = (specular.x + specular.y + specular.z) / 3.0f;
        auto select_prob = (aver_diff) / (aver_diff + aver_spec);
        if (rtlib::random_float1(xor32) < select_prob) {
            auto reflCos = rtlib::dot(reflDir, direction0);
            direction    = direction0;
            cosine       = cosine0;
            bsdfVal      = diffuse * RTLIB_M_INV_PI + specular * phongPdf0;
            bsdfPdf      = (select_prob *cosinePdf0 + (1.0f-select_prob)*phongPdf0);
        }
        else {
            auto reflCos = rtlib::dot(reflDir, direction1);
            direction    = direction1;
            cosine       = cosine1;
            bsdfVal      = diffuse * RTLIB_M_INV_PI + specular * phongPdf1;
            bsdfPdf      = (select_prob * cosinePdf1 + (1.0f - select_prob) * phongPdf1);
        }
        hrec->userData.throughPut = prevThroughput * ((bsdfPdf> 0.0f)?(bsdfVal * fabsf(cosine) / bsdfPdf):make_float3(0.0f));
    }


    hrec->SetGlobalRayOrigin(position);
    hrec->SetGlobalRayDirAndTmax(make_float4(direction, distance));

    hrec->userData.radiance = make_float3(0.0f);
    if (hrec->flags & HIT_RECORD_FLAG_COUNT_EMITTED)
    {
        hrec->userData.radiance += prevThroughput * emission * static_cast<float>(rtlib::dot(optixGetWorldRayDirection(),normal)<0.0f);
    }

    if (params.flags & PARAM_FLAG_NEE) {
        if (hrec->userData.depth < params.maxDepth-1) {
            auto lRec = params.lights.Sample(position, xor32);
            if (!TraceOccluded(params.gasHandle, position, lRec.direction, 0.0001f, lRec.distance - 0.0001f)) {
                auto e = lRec.emission;
                auto b = diffuse * RTLIB_M_INV_PI + specular* getValPhongPDF(lRec.direction, reflDir, shinness);
                auto  g = rtlib::max(-rtlib::dot(lRec.direction, lRec.normal), 0.0f) * fabsf(rtlib::dot(lRec.direction, normal)) / (lRec.distance * lRec.distance);
                hrec->userData.radiance += prevThroughput * b * e * g * lRec.invPdf;
            }
        }
    }

    auto& val = params.grid.Find(position);
    atomicAdd(&val.x, diffuse.x);
    atomicAdd(&val.y, diffuse.y);
    atomicAdd(&val.z, diffuse.z);
    atomicAdd(&val.w, 1.0f);

    hrec->normal = normal;
    hrec->seed   = xor32.m_seed;
    hrec->cosine = cosine;
    hrec->flags  = 0;

    if (!(params.flags & PARAM_FLAG_NEE)) {
        hrec->flags |= HIT_RECORD_FLAG_COUNT_EMITTED;
    }

    if (emission.x * emission.y * emission.z > 0.0f) {
        hrec->flags |= HIT_RECORD_FLAG_FINISH;
    }

}
extern "C" __global__ void       __miss__occluded() {
    optixSetPayload_0(false);
}
extern "C" __global__ void __closesthit__occluded() {
    optixSetPayload_0(true);
}
extern "C" __global__ void    __closesthit__debug() {
    auto* hrec = BasicHitRecord<HitRecordUserDebugData>::GetGlobalPointer();
    auto* hgData = reinterpret_cast<HitgroupData*>(optixGetSbtDataPointer());
    auto primitiveId = optixGetPrimitiveIndex();
    auto uv = optixGetTriangleBarycentrics();

    auto distance  = optixGetRayTmax();
    auto texCrd   = hgData->GetTexCrd(uv, primitiveId);
    auto normal   = hgData->GetNormal(uv, primitiveId);
    auto position = optixGetWorldRayOrigin() + distance * optixGetWorldRayDirection();
    auto reflDir  = rtlib::normalize(rtlib::reflect(optixGetWorldRayDirection(), normal));

    auto  direction = make_float3(0.0f);

    hrec->SetGlobalRayOrigin(position);
    hrec->SetGlobalRayDirAndTmax(make_float4(direction, distance));
    hrec->normal = normal;
    hrec->cosine = 0.0f;
    hrec->flags  = 0;
    float3 gridIndexF = (position - params.grid.aabbOffset) / params.grid.aabbSize;
    hrec->userData.gridIndex = rtlib::clamp(make_uint3(params.grid.bounds.x *gridIndexF.x, params.grid.bounds.y * gridIndexF.y, params.grid.bounds.z * gridIndexF.z),make_uint3(0),params.grid.bounds-make_uint3(1));
    auto gridValue     = params.grid.Find(position);
    hrec->userData.gridValue = (gridValue.w > 0.0f) ? make_float3(gridValue.x, gridValue.y, gridValue.z) / gridValue.w:make_float3(0.0f);
    hrec->userData.diffuse   = hgData->SampleDiffuse(texCrd);
    hrec->userData.specular  = hgData->SampleSpecular(texCrd);
    hrec->userData.emission  = hgData->SampleEmission(texCrd);
    hrec->userData.shinness  = hgData->shinness;
    hrec->userData.refrIndx  = hgData->refIndex;
}
extern "C" __global__ void       __miss__debug(){
    auto* hrec = BasicHitRecord<HitRecordUserDebugData>::GetGlobalPointer();
    auto* msData = reinterpret_cast<MissData*>(optixGetSbtDataPointer());

    hrec->SetGlobalRayOrigin(optixGetWorldRayOrigin());
    hrec->SetGlobalRayDirAndTmax(make_float4(optixGetWorldRayDirection(), optixGetRayTmax()));
    hrec->normal = make_float3(1.0f,0.0f,0.0f);
    hrec->cosine = 0.0f;
    hrec->flags  = 0;
    hrec->userData.gridIndex = make_uint3(0, 0, 0);
    hrec->userData.gridValue = make_float3(0.0f);
    hrec->userData.diffuse   = make_float3(0.0f);
    hrec->userData.specular  = make_float3(0.0f);
    hrec->userData.emission  = make_float3(0.0f);
    hrec->userData.shinness  = 0.0f;
    hrec->userData.refrIndx  = 0.0f;

}
extern "C" __global__ void     __anyhit__ah() {
    auto* hgData = reinterpret_cast<HitgroupData*>(optixGetSbtDataPointer());
}
extern "C" __global__ void __exception__ep() {
    auto code = optixGetExceptionCode();
    if (code == OPTIX_EXCEPTION_CODE_TRAVERSAL_DEPTH_EXCEEDED)
    {
        printf("%d\n", optixGetTransformListSize());
    }
}