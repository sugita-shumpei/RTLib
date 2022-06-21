#define __CUDACC__
#include "SimpleKernel.h"
extern "C" {
    __constant__ Params params;
}
namespace rtlib = RTLib::Ext::CUDA::Math;
static __forceinline__  __device__ void trace(OptixTraversableHandle handle, const float3& rayOrigin, const float3& rayDirection, float tmin, float tmax, float3& color) {
    unsigned int p0, p1, p2;
    optixTrace(handle, rayOrigin, rayDirection, tmin, tmax, 0.0f, OptixVisibilityMask(255), OPTIX_RAY_FLAG_NONE, 0, 1, 0, p0, p1, p2);
    color.x = __int_as_float(p0);
    color.y = __int_as_float(p1);
    color.z = __int_as_float(p2);
}
extern "C" __global__ void     __raygen__rg() {
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();
    auto* rgData = reinterpret_cast<RayGenData*>(optixGetSbtDataPointer());
    const float3 u = rgData->u;
    const float3 v = rgData->v;
    const float3 w = rgData->w;
    const auto seed = params.seedBuffer[params.width * idx.y + idx.x];
    rtlib::Xorshift32 xor32(seed);
    const auto gitter = rtlib::random_float2(xor32);
    const float2 d = make_float2(
        (2.0f * static_cast<float>(idx.x+ gitter.x) / static_cast<float>(dim.x)) - 1.0,
        (2.0f * static_cast<float>(idx.y+ gitter.y) / static_cast<float>(dim.y)) - 1.0);
    const float3 origin = rgData->eye;
    const float3 direction = rtlib::normalize(d.x * u + d.y * v + w);
    //printf("%f, %lf, %lf\n", direction.x, direction.y, direction.z);
    float3 result = params.accumBuffer[params.width * idx.y + idx.x]*static_cast<float>(params.samplesForAccum);
    float3 color = make_float3(0.0f);
    for (unsigned int i = 0; i < params.samplesForLaunch; ++i) {
        trace(params.gasHandle, origin, direction, 0.0f, 1e16f, color);
        result += color;
    }
    result /= static_cast<float>(params.samplesForLaunch + params.samplesForAccum);
    // printf("%f, %lf\n", texCoord.x, texCoord.y);
    params.accumBuffer[params.width * idx.y + idx.x] = result;
    params.frameBuffer[params.width * idx.y + idx.x] = make_uchar4(static_cast<unsigned char>(255.99 * result.x), static_cast<unsigned char>(255.99 * result.y), static_cast<unsigned char>(255.99 * result.z), 255);
    params.seedBuffer[ params.width * idx.y + idx.x] = xor32.m_seed;
}
extern "C" __global__ void       __miss__ms() {
    auto* msData = reinterpret_cast<MissData*>(optixGetSbtDataPointer());
    optixSetPayload_0(__float_as_int(msData->bgColor.x));
    optixSetPayload_1(__float_as_int(msData->bgColor.y));
    optixSetPayload_2(__float_as_int(msData->bgColor.z));
}
extern "C" __global__ void __closesthit__ch() {
    auto* hgData = reinterpret_cast<HitgroupData*>(optixGetSbtDataPointer());
    float2 texCoord  = optixGetTriangleBarycentrics();
    auto primitiveId = optixGetPrimitiveIndex();
    auto p0 = hgData->vertices[hgData->indices[primitiveId].x];
    auto p1 = hgData->vertices[hgData->indices[primitiveId].y];
    auto p2 = hgData->vertices[hgData->indices[primitiveId].z];
    auto normal = rtlib::normalize(rtlib::cross(p1 - p0, p2 - p0));
    optixSetPayload_0(__float_as_int((0.5f + 0.5f * normal.x)));
    optixSetPayload_1(__float_as_int((0.5f + 0.5f * normal.y)));
    optixSetPayload_2(__float_as_int((0.5f + 0.5f * normal.z)));
}
extern "C" __global__ void     __anyhit__ah() {
    auto* hgData = reinterpret_cast<HitgroupData*>(optixGetSbtDataPointer());
}
