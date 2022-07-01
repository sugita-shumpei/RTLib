#ifndef  RTLIB_EXT_OPX7_TEST_SIMPLE_KERNEL_H
#define  RTLIB_EXT_OPX7_TEST_SIMPLE_KERNEL_H
#include <cuda_runtime.h>
#include <optix.h>
#include <RTLib/Ext/CUDA/Math/Math.h>
#include <RTLib/Ext/CUDA/Math/Random.h>
#include <RTLib/Ext/CUDA/Math/VectorFunction.h>
#include <RTLib/Ext/OPX7/OPX7Payload.h>
//#define TEST_SKIP_TEXTURE_SAMPLE
//#define   TEST11_SHOW_EMISSON_COLOR
//#define TEST11_SHOW_NORMAL
struct ParallelLight {
    float3   corner;
    float3   v1, v2;
    float3   normal;
    float3 emission;
};
struct Params {
    unsigned int*           seedBuffer;
    float3*                accumBuffer;
    uchar4*                frameBuffer;
    unsigned int           width;
    unsigned int           height;
    unsigned int           samplesForLaunch;
    unsigned int           samplesForAccum;
    OptixTraversableHandle gasHandle;
};
struct RayGenData {
    float3 u, v, w;
    float3 eye;
    RTLIB_HOST_DEVICE auto GetRayOrigin()const->float3 { return eye; }
    RTLIB_HOST_DEVICE auto GetRayDirection(float2 barycentrics)const->float3
    {
        namespace rtlib = RTLib::Ext::CUDA::Math;
        return rtlib::normalize(barycentrics.x * u + barycentrics.y * v + w);
    }
};
struct MissData {
    float4  bgColor;
};
struct HitgroupData {
    float3*             vertices;
    float2*             texCrds;
    uint3*              indices;
    float3              diffuse;
    float3              specular;
    float3              emission;
    float               shinness;
    cudaTextureObject_t diffuseTex;
    cudaTextureObject_t specularTex;
    cudaTextureObject_t emissionTex;
#ifdef __CUDACC__
    RTLIB_DEVICE auto GetNormal(float2 barycentrics, unsigned int primIdx)const noexcept -> float3
    {
        namespace rtlib = RTLib::Ext::CUDA::Math;
        auto p0 = vertices[indices[primIdx].x];
        auto p1 = vertices[indices[primIdx].y];
        auto p2 = vertices[indices[primIdx].z];
        return rtlib::normalize(rtlib::cross(p1-p0,p2-p0));
    }
    RTLIB_DEVICE auto GetTexCrd(float2 barycentrics,unsigned int primIdx)const noexcept -> float2
    {
        auto t0 = texCrds[indices[primIdx].x];
        auto t1 = texCrds[indices[primIdx].y];
        auto t2 = texCrds[indices[primIdx].z];
        return (1.0f - barycentrics.x - barycentrics.y) * t0+barycentrics.x * t1 + barycentrics.y * t2;
    }
    RTLIB_DEVICE auto SampleDiffuse(float2 uv)const noexcept -> float3
    {
        auto diffuseCol = tex2D<float4>(diffuseTex, uv.x, uv.y);
        return make_float3(diffuseCol.x, diffuseCol.y, diffuseCol.z) * diffuse;
    }
    RTLIB_DEVICE auto SampleSpecular(float2 uv)const noexcept -> float3
    {
        auto specularCol = tex2D<float4>(specularTex, uv.x, uv.y);
        return make_float3(specularCol.x, specularCol.y, specularCol.z) * specular;
    }
    RTLIB_DEVICE auto SampleEmission(float2 uv)const noexcept -> float3
    {
        auto emissionCol = tex2D<float4>(emissionTex, uv.x, uv.y);
        return make_float3(emissionCol.x, emissionCol.y, emissionCol.z) * emission;
    }
#endif
};
enum   RayType
{
    RAY_TYPE_RADIANCE = 0,
    RAY_TYPE_OCCLUDED= 1,
    RAY_TYPE_COUNT
};
enum   HitRecordFlag
{
    HIT_RECORD_FLAG_NONE = 0,
    HIT_RECORD_FLAG_MISS  = 1,
    HIT_RECORD_FLAG_DELTA_MATERIAL = 2,
    HIT_RECORD_FLAG_COUNT_EMITTED = 4,
    HIT_RECORD_FLAG_FINISH = 8
};

template<typename UserData>
struct BasicHitRecord
{
    float3       rayOrigin;
    float        rayDistance;
    float3       rayDirection;
    float        cosine;
    float3       normal;
    unsigned int seed;
    unsigned int flags;
    UserData     userData;

#ifdef __CUDACC__
    static __forceinline__ __device__ auto GetGlobalPointer()->BasicHitRecord<UserData>*
    {
        namespace rtlib = RTLib::Ext::CUDA::Math;
        return reinterpret_cast<BasicHitRecord<UserData>*>(
            rtlib::to_combine(
                optixGetPayload_6(), optixGetPayload_7()
            )
            );
    }
    static __forceinline__ __device__ void SetGlobalRayOrigin(float3 rayOrg)
    {
        namespace rtlib = RTLib::Ext::CUDA::Math;
        optixSetPayload_0(__float_as_uint(rayOrg.x));
        optixSetPayload_1(__float_as_uint(rayOrg.y));
        optixSetPayload_2(__float_as_uint(rayOrg.z));
    }
    static __forceinline__ __device__ auto  AsRayOrigin(unsigned int p0, unsigned int p1, unsigned int p2)->float3
    {
        namespace rtlib = RTLib::Ext::CUDA::Math;
        return make_float3(
            __uint_as_float(p0),
            __uint_as_float(p1),
            __uint_as_float(p2)
        );
    }
    static __forceinline__ __device__ void SetGlobalRayDirAndTmax(float4 rayDirAndTmax)
    {
        namespace rtlib = RTLib::Ext::CUDA::Math;
        optixSetPayload_3(__float_as_uint(rayDirAndTmax.x));
        optixSetPayload_4(__float_as_uint(rayDirAndTmax.y));
        float sign = (rayDirAndTmax.z > 0.0f);
        optixSetPayload_5(__float_as_uint(sign * rayDirAndTmax.w));
    }
    static __forceinline__ __device__ auto  AsRayDirAndTmax(unsigned int p3, unsigned int p4, unsigned int p5)->float4 {
        namespace rtlib = RTLib::Ext::CUDA::Math;
        float rayDirX = __uint_as_float(p3);
        float rayDirY = __uint_as_float(p4);
        float aRayDirZ = sqrtf(rtlib::max(1.0f - rayDirX * rayDirX - rayDirY * rayDirY,0.0f));
        float sRayTmax = __uint_as_float(p5);
        float rayDirZ  = (sRayTmax > 0.0f) ? aRayDirZ : -aRayDirZ;
        return make_float4(rayDirX, rayDirY, rayDirZ, fabsf(sRayTmax));
    }
#endif
};
struct BasicHitRecordDefaultUserData {};
using  HitRecord = BasicHitRecord<BasicHitRecordDefaultUserData>;
#ifdef __CUDACC__

template<typename UserData>
static __forceinline__ __device__ void TraceRadiance(OptixTraversableHandle tlasHandle,   float3  rayOrigin,   float3  rayDirection, float rayTmin, float rayTmax, BasicHitRecord<UserData>& hrec)
{
    unsigned int p0, p1,  p2, p3, p4, p5, p6, p7;
    unsigned long long iptr = reinterpret_cast<unsigned long long>(&hrec);
    p6 = rtlib::to_upper(iptr);
    p7 = rtlib::to_lower(iptr);

    optixTrace(tlasHandle, rayOrigin, rayDirection, rayTmin, rayTmax, 0.0f, OptixVisibilityMask(255), OPTIX_RAY_FLAG_NONE, RAY_TYPE_RADIANCE, RAY_TYPE_COUNT, RAY_TYPE_RADIANCE, p0, p1, p2, p3, p4, p5, p6, p7);
    hrec.rayOrigin       = BasicHitRecord<UserData>::AsRayOrigin    (p0, p1, p2);
    float4 rayDirAndTmax = BasicHitRecord<UserData>::AsRayDirAndTmax(p3, p4, p5);
    hrec.rayDirection    = rtlib::normalize(make_float3(rayDirAndTmax.x, rayDirAndTmax.y, rayDirAndTmax.z));
    hrec.rayDistance     = rayDirAndTmax.w;
}
static __forceinline__ __device__ bool TraceOccluded(OptixTraversableHandle tlasHandle,   float3  rayOrigin,   float3  rayDirection, float rayTmin, float rayTmax)
{
    unsigned int p0 = 0;
    optixTrace(tlasHandle, rayOrigin, rayDirection, rayTmin, rayTmax, 0.0f, OptixVisibilityMask(255), OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT, RAY_TYPE_OCCLUDED, RAY_TYPE_COUNT, RAY_TYPE_OCCLUDED,p0);
    return p0;
}
#endif
#endif