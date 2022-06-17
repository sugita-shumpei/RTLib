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
enum RayType {
    RAY_TYPE_RADIANCE = 0,
    RAY_TYPE_OCCLUSION,
    RAY_TYPE_COUNT,
};
struct ParallelLight {
    float3   corner;
    float3   v1, v2;
    float3   normal;
    float3 emission;
};
struct Params {
    uchar4* image;
    unsigned int           width;
    unsigned int           height;
    OptixTraversableHandle gasHandle;
};
struct RayGenData {
    float3 u, v, w;
    float3 eye;
};
struct MissData {
    float4  bgColor;
};
struct HitgroupData {
    float3*             vertices;
    uint3*              indices;
    float3              diffuse;
    float3              specular;
    float3              emission;
    float               shinness;
    cudaTextureObject_t diffuseTex;
    cudaTextureObject_t specularTex;
    cudaTextureObject_t emissionTex;
};
struct RadiancePRD {
    float3       emitted;
    float3       radiance;
    float3       attenuation;
    unsigned int seed;
    int          countEmitted;
    int          done;
    int          pad;
};
#endif