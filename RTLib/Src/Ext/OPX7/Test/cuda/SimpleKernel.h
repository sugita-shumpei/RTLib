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
#include <RTLib/Ext/OPX7/Utils/OPX7UtilsPathGuiding.h>
#include <RTLib/Ext/OPX7/Utils/OPX7UtilsMorton.h>
#include <RTLib/Ext/OPX7/Utils/OPX7UtilsGrid.h>
#include <PathGuidingConfig.h>
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

    float3* vertices;
    float3* normals;
    float2* texCrds;
    uint3* indices;
    unsigned int        indCount;
    float3              emission;
    Matrix4x4           transform;
    cudaTextureObject_t emissionTex;

    RTLIB_INLINE RTLIB_HOST_DEVICE auto GetFaceNormalWithNonNormalized(unsigned int triIdx)const noexcept-> float3
    {
        auto v0xyzw = transform * make_float4(vertices[indices[triIdx].x], 1.0f);
        auto v1xyzw = transform * make_float4(vertices[indices[triIdx].y], 1.0f);
        auto v2xyzw = transform * make_float4(vertices[indices[triIdx].z], 1.0f);
        auto v0 = make_float3(v0xyzw.x, v0xyzw.y, v0xyzw.z) / v0xyzw.w;
        auto v1 = make_float3(v1xyzw.x, v1xyzw.y, v1xyzw.z) / v1xyzw.w;
        auto v2 = make_float3(v2xyzw.x, v2xyzw.y, v2xyzw.z) / v2xyzw.w;
        return RTLib::Ext::CUDA::Math::cross(v1 - v0, v2 - v0);
    }
    RTLIB_INLINE RTLIB_HOST_DEVICE auto GetVertex(unsigned int triIdx, float3 bary)const noexcept-> float3
    {
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
        auto nf = GetFaceNormalWithNonNormalized(triIdx);
        auto bary = RTLib::Ext::CUDA::Math::random_in_unit_triangle(make_float3(1.0f, 0.0f, 0.0f), make_float3(0.0f, 1.0f, 0.0f), make_float3(0.0f, 0.0f, 1.0f), rng);
        auto p = GetVertex(triIdx, bary);
        auto t = GetTexCrd(triIdx, bary);
        auto d = p - p_in;
        LightRecord lRec = {};
        lRec.position = p;
        lRec.direction = RTLib::Ext::CUDA::Math::normalize(d);
        lRec.distance = RTLib::Ext::CUDA::Math::length(d);
        lRec.emission = GetEmission(t);
        lRec.normal = RTLib::Ext::CUDA::Math::normalize(nf);
        lRec.invPdf = RTLib::Ext::CUDA::Math::length(nf) / 2.0f * static_cast<float>(indCount);
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
    MeshLight* data;

#ifdef __CUDACC__
    template<typename RNG>
    RTLIB_INLINE RTLIB_DEVICE auto Sample(const float3& p_in, RNG& rng)const noexcept->LightRecord
    {
        auto triIdx = rng.next() % count;
        auto lRec = data[triIdx].Sample(p_in, rng);
        lRec.invPdf *= static_cast<float>(count);
        return lRec;
    };
#endif
};
/**/
using MortonQTree        = RTLib::Ext::OPX7::Utils::MortonQuadTreeT<4>;
using MortonQTreeWrapper = RTLib::Ext::OPX7::Utils::MortonQuadTreeWrapperT<4>;
using MortonTraceVertex  = RTLib::Ext::OPX7::Utils::MortonTraceVertexT<4>;
using HashGrid3          = RTLib::Ext::OPX7::Utils::HashGrid3;
using DoubleBufferedHashGrid3 = RTLib::Ext::OPX7::Utils::DoubleBufferedHashGrid3;
template<typename T>
struct Reservoir
{
    float        w = 0.0f;
    float        w_sum = 0.0f;
    unsigned int m = 0;
    T            y = {};
    RTLIB_INLINE RTLIB_HOST_DEVICE bool Update(T x_i, float w_i, float rnd01)
    {
        w = 0.0f;
        w_sum += w_i;
        ++m;
        if ((w_i / w_sum) >= rnd01)
        {
            y = x_i;
            return true;
        }
        return false;
    }
};
struct ReservoirState
{
    float targetDensity;
};

enum   ParamFlag
{
    PARAM_FLAG_NONE     = 0,
    PARAM_FLAG_NEE      = 1,
    PARAM_FLAG_RIS      = 2,
    PARAM_FLAG_USE_GRID = 4,
    PARAM_FLAG_USE_TREE = 8,
    PARAM_FLAG_BUILD    = 16,
    PARAM_FLAG_FINAL    = 32,
};
enum   DebugFrameType
{
    DEBUG_FRAME_TYPE_NORMAL = 0,
    DEBUG_FRAME_TYPE_DEPTH = 1,
    DEBUG_FRAME_TYPE_DIFFUSE = 2,
    DEBUG_FRAME_TYPE_SPECULAR = 3,
    DEBUG_FRAME_TYPE_EMISSION = 4,
    DEBUG_FRAME_TYPE_SHINNESS = 5,
    DEBUG_FRAME_TYPE_REFR_INDEX = 6,
    DEBUG_FRAME_TYPE_GRID_VALUE = 7,
    DEBUG_FRAME_TYPE_COUNT = 8,
};
struct Params {
    unsigned int*                 seedBuffer;
    float3*                      accumBuffer;
    uchar4*                      frameBuffer;
    float4*                diffuseGridBuffer;
    MortonQTreeWrapper            mortonTree;
    unsigned int                       width;
    unsigned int                      height;
    unsigned int                    maxDepth;
    unsigned int            samplesForLaunch;
    unsigned int             samplesForAccum;
    unsigned int                       flags;
    unsigned int              debugFrameType;
    OptixTraversableHandle         gasHandle;
    MeshLightList                     lights;
    DoubleBufferedHashGrid3             grid;
    rtlib::test::STree                  tree;
};
struct RayGenData {
    float3 u, v, w;
    float3 eye;
    RTLIB_INLINE RTLIB_HOST_DEVICE auto GetRayOrigin()const->float3 { return eye; }
    RTLIB_INLINE RTLIB_HOST_DEVICE auto GetRayDirection(float2 barycentrics)const->float3
    {
        return RTLib::Ext::CUDA::Math::normalize(barycentrics.x * u + barycentrics.y * v + w);
    }
};
struct MissData {
    float4  bgColor;
};
enum   HitgroupType
{
    HIT_GROUP_TYPE_DIFFUSE = 0,
    HIT_GROUP_TYPE_PHONG = 1,
    HIT_GROUP_TYPE_GLASS = 2,
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
    float3*             normals;
    float2*             texCrds;
    uint3*              indices;
    cudaTextureObject_t diffuseTex;
    cudaTextureObject_t specularTex;
    cudaTextureObject_t emissionTex;
#ifdef __CUDACC__
    RTLIB_INLINE RTLIB_DEVICE auto GetSphereNormal(float3 position, unsigned int primIdx)const noexcept -> float3 {
        return RTLib::Ext::CUDA::Math::normalize(optixTransformPointFromObjectToWorldSpace(vertices[primIdx]) - position);
    }
    RTLIB_INLINE RTLIB_DEVICE auto GetTriangleFNormal(float2 barycentrics, unsigned int primIdx)const noexcept -> float3
    {
        auto p0 = vertices[indices[primIdx].x];
        auto p1 = vertices[indices[primIdx].y];
        auto p2 = vertices[indices[primIdx].z];
        return RTLib::Ext::CUDA::Math::normalize(optixTransformNormalFromObjectToWorldSpace(RTLib::Ext::CUDA::Math::cross(p1 - p0, p2 - p0)));
    }
    RTLIB_INLINE RTLIB_DEVICE auto GetTriangleVNormal(float2 barycentrics, unsigned int primIdx)const noexcept -> float3
    {
        if (normals) {
            auto vn0 = normals[indices[primIdx].x];
            auto vn1 = normals[indices[primIdx].y];
            auto vn2 = normals[indices[primIdx].z];
            auto vn = (1.0f - barycentrics.x - barycentrics.y) * vn0 + barycentrics.x * vn1 + barycentrics.y * vn2;
            return RTLib::Ext::CUDA::Math::normalize(optixTransformNormalFromObjectToWorldSpace(vn));
        }
        else {
            return GetTriangleFNormal(barycentrics, primIdx);
        }
    }
    RTLIB_INLINE RTLIB_DEVICE auto GetTexCrd(float2 barycentrics, unsigned int primIdx)const noexcept -> float2
    {
        auto t0 = texCrds[indices[primIdx].x];
        auto t1 = texCrds[indices[primIdx].y];
        auto t2 = texCrds[indices[primIdx].z];
        return (1.0f - barycentrics.x - barycentrics.y) * t0 + barycentrics.x * t1 + barycentrics.y * t2;
    }
    RTLIB_INLINE RTLIB_DEVICE auto SampleDiffuse(float2 uv)const noexcept -> float3
    {
        auto diffuseCol = tex2D<float4>(diffuseTex, uv.x, uv.y);
        return make_float3(diffuseCol.x, diffuseCol.y, diffuseCol.z) * diffuse;
    }
    RTLIB_INLINE RTLIB_DEVICE auto SampleSpecular(float2 uv)const noexcept -> float3
    {
        auto specularCol = tex2D<float4>(specularTex, uv.x, uv.y);
        return make_float3(specularCol.x, specularCol.y, specularCol.z) * specular;
    }
    RTLIB_INLINE RTLIB_DEVICE auto SampleEmission(float2 uv)const noexcept -> float3
    {
        auto emissionCol = tex2D<float4>(emissionTex, uv.x, uv.y);
        return make_float3(emissionCol.x, emissionCol.y, emissionCol.z) * emission;
    }
#endif
};
enum   RayType
{
    RAY_TYPE_RADIANCE = 0,
    RAY_TYPE_OCCLUDED = 1,
    RAY_TYPE_COUNT
};
enum   HitRecordFlag
{
    HIT_RECORD_FLAG_NONE = 0,
    HIT_RECORD_FLAG_MISS = 1,
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
        return reinterpret_cast<BasicHitRecord<UserData>*>(
            RTLib::Ext::CUDA::Math::to_combine(
                optixGetPayload_6(), optixGetPayload_7()
            )
            );
    }
    static __forceinline__ __device__ void SetGlobalRayOrigin(float3 rayOrg)
    {
        optixSetPayload_0(__float_as_uint(rayOrg.x));
        optixSetPayload_1(__float_as_uint(rayOrg.y));
        optixSetPayload_2(__float_as_uint(rayOrg.z));
    }
    static __forceinline__ __device__ auto  AsRayOrigin(unsigned int p0, unsigned int p1, unsigned int p2)->float3
    {
        return make_float3(
            __uint_as_float(p0),
            __uint_as_float(p1),
            __uint_as_float(p2)
        );
    }
    static __forceinline__ __device__ void SetGlobalRayDirAndTmax(float4 rayDirAndTmax)
    {
        optixSetPayload_3(__float_as_uint(rayDirAndTmax.x));
        optixSetPayload_4(__float_as_uint(rayDirAndTmax.y));
        float sign = (rayDirAndTmax.z > 0.0f);
        optixSetPayload_5(__float_as_uint(sign * rayDirAndTmax.w));
    }
    static __forceinline__ __device__ auto  AsRayDirAndTmax(unsigned int p3, unsigned int p4, unsigned int p5)->float4 {
        float rayDirX = __uint_as_float(p3);
        float rayDirY = __uint_as_float(p4);
        float aRayDirZ = sqrtf(RTLib::Ext::CUDA::Math::max(1.0f - rayDirX * rayDirX - rayDirY * rayDirY, 0.0f));
        float sRayTmax = __uint_as_float(p5);
        float rayDirZ = (sRayTmax > 0.0f) ? aRayDirZ : -aRayDirZ;
        return make_float4(rayDirX, rayDirY, rayDirZ, fabsf(sRayTmax));
    }
#endif
};
struct BasicHitRecordDefaultUserData {};
using  HitRecord = BasicHitRecord<BasicHitRecordDefaultUserData>;
#ifdef __CUDACC__

template<typename UserData>
static __forceinline__ __device__ void TraceRadiance(OptixTraversableHandle tlasHandle, float3  rayOrigin, float3  rayDirection, float rayTmin, float rayTmax, BasicHitRecord<UserData>& hrec)
{
    unsigned int p0, p1, p2, p3, p4, p5, p6, p7;
    unsigned long long iptr = reinterpret_cast<unsigned long long>(&hrec);
    p6 = RTLib::Ext::CUDA::Math::to_upper(iptr);
    p7 = RTLib::Ext::CUDA::Math::to_lower(iptr);

    optixTrace(tlasHandle, rayOrigin, rayDirection, rayTmin, rayTmax, 0.0f, OptixVisibilityMask(255), OPTIX_RAY_FLAG_NONE, RAY_TYPE_RADIANCE, RAY_TYPE_COUNT, RAY_TYPE_RADIANCE, p0, p1, p2, p3, p4, p5, p6, p7);
    hrec.rayOrigin = BasicHitRecord<UserData>::AsRayOrigin(p0, p1, p2);
    float4 rayDirAndTmax = BasicHitRecord<UserData>::AsRayDirAndTmax(p3, p4, p5);
    hrec.rayDirection = RTLib::Ext::CUDA::Math::normalize(make_float3(rayDirAndTmax.x, rayDirAndTmax.y, rayDirAndTmax.z));
    hrec.rayDistance = rayDirAndTmax.w;
}
static __forceinline__ __device__ bool TraceOccluded(OptixTraversableHandle tlasHandle, float3  rayOrigin, float3  rayDirection, float rayTmin, float rayTmax)
{
    unsigned int p0 = 0;
    optixTrace(tlasHandle, rayOrigin, rayDirection, rayTmin, rayTmax, 0.0f, OptixVisibilityMask(255), OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT, RAY_TYPE_OCCLUDED, RAY_TYPE_COUNT, RAY_TYPE_OCCLUDED, p0);
    return p0;
}

#endif
#endif