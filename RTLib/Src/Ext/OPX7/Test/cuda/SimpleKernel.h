#ifndef  RTLIB_EXT_OPX7_TEST_SIMPLE_KERNEL_H
#define  RTLIB_EXT_OPX7_TEST_SIMPLE_KERNEL_H
#include <cuda_runtime.h>
#include <optix.h>
#include <RTLib/Ext/CUDA/Math/Math.h>
#include <RTLib/Ext/CUDA/Math/Random.h>
#include <RTLib/Ext/CUDA/Math/Matrix.h>
#include <RTLib/Ext/CUDA/Math/VectorFunction.h>
#include <RTLib/Ext/CUDA/Math/Hash.h>
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
struct LightRecord
{
    float3 position;
    float3 direction;
    float3 normal;
    float3 emission;
    float  distance;
    float  invPdf;
};
struct MeshLight
{
    using Matrix4x4 = RTLib::Ext::CUDA::Math::Matrix4x4;

    float3*             vertices;
    float3*             normals;
    float2*             texCrds;
    uint3*              indices;
    unsigned int        indCount;
    float3              emission;
    Matrix4x4           transform;
    cudaTextureObject_t emissionTex;

    RTLIB_INLINE RTLIB_HOST_DEVICE auto GetFaceNormalWithNonNormalized(unsigned int triIdx)const noexcept-> float3
    {
        namespace rtlib = RTLib::Ext::CUDA::Math;
        auto v0xyzw = transform * make_float4(vertices[indices[triIdx].x], 1.0f);
        auto v1xyzw = transform * make_float4(vertices[indices[triIdx].y], 1.0f);
        auto v2xyzw = transform * make_float4(vertices[indices[triIdx].z], 1.0f);
        auto v0     = make_float3(v0xyzw.x, v0xyzw.y, v0xyzw.z) / v0xyzw.w;
        auto v1     = make_float3(v1xyzw.x, v1xyzw.y, v1xyzw.z) / v1xyzw.w;
        auto v2     = make_float3(v2xyzw.x, v2xyzw.y, v2xyzw.z) / v2xyzw.w;
        return rtlib::cross(v1 - v0, v2 - v0);
    }
    RTLIB_INLINE RTLIB_HOST_DEVICE auto GetVertex(unsigned int triIdx, float3 bary)const noexcept-> float3
    {
        namespace rtlib = RTLib::Ext::CUDA::Math;
        auto v0xyzw = transform * make_float4(vertices[indices[triIdx].x], 1.0f);
        auto v1xyzw = transform * make_float4(vertices[indices[triIdx].y], 1.0f);
        auto v2xyzw = transform * make_float4(vertices[indices[triIdx].z], 1.0f);
        auto v0 = make_float3(v0xyzw.x, v0xyzw.y, v0xyzw.z) / v0xyzw.w;
        auto v1 = make_float3(v1xyzw.x, v1xyzw.y, v1xyzw.z) / v1xyzw.w;
        auto v2 = make_float3(v2xyzw.x, v2xyzw.y, v2xyzw.z) / v2xyzw.w;
        auto p = bary.x * v0 + bary.y * v1 + bary.z * v2;
        return p;
    }
    RTLIB_INLINE RTLIB_HOST_DEVICE auto GetTexCrd(unsigned int triIdx, float3 bary)const noexcept-> float2
    {
        namespace rtlib = RTLib::Ext::CUDA::Math;
        auto t0 = texCrds[indices[triIdx].x];
        auto t1 = texCrds[indices[triIdx].y];
        auto t2 = texCrds[indices[triIdx].z];
        auto t = bary.x * t0 + bary.y * t1 + bary.z * t2;
        return t;
    }

#ifdef __CUDACC__
    template<typename RNG>
    RTLIB_INLINE RTLIB_DEVICE auto Sample(const float3& p_in, RNG& rng)const noexcept->LightRecord
    {
        auto triIdx = rng.next() % indCount;
        //normal
        auto nf   = GetFaceNormalWithNonNormalized(triIdx);
        auto bary = rtlib::random_in_unit_triangle(make_float3(1.0f, 0.0f, 0.0f), make_float3(0.0f, 1.0f, 0.0f), make_float3(0.0f, 0.0f, 1.0f), rng);
        auto p    = GetVertex(triIdx, bary);
        auto t    = GetTexCrd(triIdx, bary);
        auto d    = p - p_in;
        LightRecord lRec = {};
        lRec.position  = p;
        lRec.direction = rtlib::normalize(d);
        lRec.distance  = rtlib::length(d);
        lRec.emission  = GetEmission(t);
        lRec.normal    = rtlib::normalize(nf);
        lRec.invPdf    = rtlib::length(nf) / 2.0f * static_cast<float>(indCount);
        return lRec;
    };
    RTLIB_INLINE RTLIB_DEVICE auto GetEmission(float2 uv)const noexcept -> float3 {
        auto emitTC = tex2D<float4>(this->emissionTex, uv.x, uv.y);
        auto emitBC = this->emission;
        auto emitColor = emitBC * make_float3(float(emitTC.x), float(emitTC.y), float(emitTC.z));
        return emitColor;
    }
#endif
};
struct MeshLightList
{
    unsigned int count;
    MeshLight*   data;

#ifdef __CUDACC__
    template<typename RNG>
    RTLIB_INLINE RTLIB_DEVICE auto Sample(const float3& p_in, RNG& rng)const noexcept->LightRecord
    {
        auto triIdx  = rng.next() % count;
        auto lRec    = data[triIdx].Sample(p_in, rng);
        lRec.invPdf *= static_cast<float>(count);
        return lRec;
    };
#endif
};

template<typename T>
struct RegularGrid2
{
    float2 aabbOffset;
    float2 aabbSize;
    uint2  bounds;
    T* data;
    RTLIB_INLINE RTLIB_HOST_DEVICE auto Sample(const float2 p)const noexcept -> const T&
    {
        auto fLen = (p - aabbOffset) / aabbSize;
        auto iLen = rtlib::clamp(make_uint2(
            static_cast<unsigned int>(fLen.x),
            static_cast<unsigned int>(fLen.y)
        ), make_uint2(0), bounds - make_uint2(1));
        return data[bounds.x * iLen.y + iLen.x];

    }
    RTLIB_INLINE RTLIB_HOST_DEVICE auto Sample(const float2 p)       noexcept ->      T&
    {
        auto fLen = (p - aabbOffset) / aabbSize;
        auto iLen = rtlib::clamp(make_uint2(
            static_cast<unsigned int>(fLen.x),
            static_cast<unsigned int>(fLen.y)
        ), make_uint2(0), bounds - make_uint2(1));
        return data[bounds.x * iLen.y + iLen.x];

    }
};

template<typename T>
struct RegularGrid3
{
    float3 aabbOffset;
    float3 aabbSize  ;
    uint3  bounds    ;
    T*     data      ;
    RTLIB_INLINE RTLIB_HOST_DEVICE auto Find(const float3 p)const noexcept -> const T&
    {
        auto fLen = (p - aabbOffset) / aabbSize;
        auto iLen = rtlib::clamp(make_uint3(
            static_cast<unsigned int>(bounds.x * fLen.x),
            static_cast<unsigned int>(bounds.y * fLen.y),
            static_cast<unsigned int>(bounds.z * fLen.z)
        ),make_uint3(0),bounds-make_uint3(1));
        return data[bounds.x * bounds.y * iLen.z + bounds.x * iLen.y + iLen.x];

    }
    RTLIB_INLINE RTLIB_HOST_DEVICE auto Find(const float3 p)       noexcept ->      T&
    {
        auto fLen = (p - aabbOffset) / aabbSize;
        auto iLen = rtlib::clamp(make_uint3(
            static_cast<unsigned int>(bounds.x * fLen.x),
            static_cast<unsigned int>(bounds.y * fLen.y),
            static_cast<unsigned int>(bounds.z * fLen.z)
        ), make_uint3(0), bounds - make_uint3(1));
        return data[bounds.x * bounds.y * iLen.z + bounds.x * iLen.y + iLen.x];

    }
};
template<typename T>
struct HashGrid3
{
    float3 aabbOffset;
    float3 aabbSize;
    uint3  bounds;
    unsigned int size;
    T* data;
    RTLIB_INLINE RTLIB_HOST_DEVICE auto Find(const float3 p)const noexcept -> const T&
    {
        auto fLen = (p - aabbOffset) / aabbSize;
        auto iLen = rtlib::clamp(make_uint3(
            static_cast<unsigned int>(bounds.x * fLen.x),
            static_cast<unsigned int>(bounds.y * fLen.y),
            static_cast<unsigned int>(bounds.z * fLen.z)
        ), make_uint3(0), bounds - make_uint3(1));
        return data[GetCellIndex(iLen)];

    }
    RTLIB_INLINE RTLIB_HOST_DEVICE auto Find(const float3 p)       noexcept ->      T&
    {
        auto fLen = (p - aabbOffset) / aabbSize;
        auto iLen = rtlib::clamp(make_uint3(
            static_cast<unsigned int>(bounds.x * fLen.x),
            static_cast<unsigned int>(bounds.y * fLen.y),
            static_cast<unsigned int>(bounds.z * fLen.z)
        ), make_uint3(0), bounds - make_uint3(1));
        return data[GetCellIndex(iLen)];

    }
private:
    RTLIB_INLINE RTLIB_HOST_DEVICE auto GetCellIndex(const uint3 idx)const noexcept -> unsigned int
    {
        namespace rtlib = RTLib::Ext::CUDA::Math;
        //unsigned long long baseIndex = bounds.x * bounds.y * iLen.z + bounds.x * iLen.y + iLen.x;
        //return rtlib::hash6432shift(baseIndex) % size;
        return rtlib::pcg1d(bounds.x * bounds.y * bounds.z +rtlib::pcg1d(idx.z + rtlib::pcg1d(idx.y + rtlib::pcg1d(idx.x))) + size) % size;
    }
    RTLIB_INLINE RTLIB_HOST_DEVICE auto GetCellIndex1(const uint3 idx)const noexcept -> unsigned int
    {
        namespace rtlib = RTLib::Ext::CUDA::Math;
        unsigned long long baseIndex = bounds.x * bounds.y * iLen.z + bounds.x * iLen.y + iLen.x;
        return rtlib::hash6432shift(baseIndex) % size;
    }
};

enum   ParamFlag
{
    PARAM_FLAG_NONE= 0,
    PARAM_FLAG_NEE = 1,
};
struct Params {
    unsigned int*           seedBuffer;
    float3*                accumBuffer;
    uchar4*                frameBuffer;
    unsigned int           width;
    unsigned int           height;
    unsigned int           maxDepth;
    unsigned int           samplesForLaunch;
    unsigned int           samplesForAccum;
    unsigned int           flags;
    OptixTraversableHandle gasHandle;
    MeshLightList          lights;
    HashGrid3<float4>      grid;
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
enum   HitgroupType
{
    HIT_GROUP_TYPE_DIFFUSE   = 0,
    HIT_GROUP_TYPE_PHONG     = 1,
    HIT_GROUP_TYPE_GLASS     = 2,
    HIT_GROUP_TYPE_NEE_LIGHT = 3,
    HIT_GROUP_TYPE_DEF_LIGHT = 4,
};
struct HitgroupData {
    HitgroupType        type;
    float               refIndex;
    float               shinness;
    float3              diffuse;
    float3              specular;
    float3              emission;
    float3*             vertices;
    float2*             texCrds;
    uint3*              indices;
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