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
extern "C" {
    __constant__ Params params;
}
namespace rtlib = RTLib::Ext::CUDA::Math;
extern "C" __global__ void       __raygen__rg() {
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();
    auto* rgData = reinterpret_cast<RayGenData*>(optixGetSbtDataPointer());
    const auto seed = params.seedBuffer[params.width * idx.y + idx.x];
    float3 result = params.accumBuffer[params.width * idx.y + idx.x]*static_cast<float>(params.samplesForAccum);
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
    hrec.flags        = 0;

    hrec.userData.throughPut = make_float3(1.0f);
    hrec.userData.radiance   = make_float3(0.0f);
    hrec.userData.bsdfVal    = make_float3(0.0f);
    hrec.userData.bsdfPdf    = 0.0f;
    hrec.userData.depth      = 0;
    while (true) {
        TraceRadiance(params.gasHandle, hrec.rayOrigin, hrec.rayDirection, 0.01f, 1.0e20f, hrec);
        color += hrec.userData.radiance;
        ++hrec.userData.depth;

        if (isnan(hrec.rayDirection.x)|| isnan(hrec.rayDirection.y)|| isnan(hrec.rayDirection.z)) {
            printf("error\n");
            break;
        }

        if ((hrec.flags & HIT_RECORD_FLAG_FINISH) || (hrec.userData.depth > 10)) {
            break;
        }
    }
    
    result += color;
    result /= static_cast<float>(params.samplesForLaunch + params.samplesForAccum);

    // printf("%f, %lf\n", texCoord.x, texCoord.y);
    params.accumBuffer[params.width * idx.y + idx.x] = result;
    params.frameBuffer[params.width * idx.y + idx.x] = rtlib::rgba_to_srgb(make_uchar4(static_cast<unsigned char>(255.99 * result.x), static_cast<unsigned char>(255.99 * result.y), static_cast<unsigned char>(255.99 * result.z), 255));
    params.seedBuffer [params.width * idx.y + idx.x] = hrec.seed;
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
    auto position  = optixGetWorldRayOrigin() + distance * optixGetWorldRayDirection();
    auto texCrd    = hgData->GetTexCrd(uv,primitiveId);
    auto normal    = hgData->GetNormal(uv,primitiveId);

    auto diffuse   = hgData->SampleDiffuse(texCrd);
    //auto specualr = hgData->SampleSpecular(texCrd);
    auto emission  = hgData->SampleEmission(texCrd);

    auto xor32 = rtlib::Xorshift32(hrec->seed);
    auto   onb  = rtlib::ONB(normal);

    auto direction = rtlib::normalize(onb.local(rtlib::random_cosine_direction(xor32)));
    auto  cosine    = rtlib::dot(direction, normal);
    auto bsdfVal   = diffuse * RTLIB_M_INV_PI;
    auto bsdfPdf   = cosine  * RTLIB_M_INV_PI;

    hrec->SetGlobalRayOrigin(position + 0.01f * normal);
    hrec->SetGlobalRayDirAndTmax(make_float4(direction, distance));

    hrec->normal = normal;
    hrec->seed   = xor32.m_seed;
    hrec->cosine = cosine;
    hrec->flags |= HIT_RECORD_FLAG_COUNT_EMITTED;

    if (emission.x * emission.y * emission.z > 0.0f) {
        hrec->flags |= HIT_RECORD_FLAG_FINISH;
    }

    hrec->userData.radiance    = hrec->userData.throughPut * emission;
    hrec->userData.throughPut *= diffuse;
}
extern "C" __global__ void       __miss__occluded() {
    optixSetPayload_0(false);
}
extern "C" __global__ void __closesthit__occluded() {
    optixSetPayload_0(true);
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