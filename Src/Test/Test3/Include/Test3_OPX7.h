#ifndef TEST3_OPX7__H
#define TEST3_OPX7__H

#include <OptiXToolkit/ShaderUtil/vec_math.h>
#include <OptiXToolkit/ShaderUtil/Preprocessor.h>
#include <OptiXToolkit/ShaderUtil/color.h>
#include <cuda_runtime.h>
#include <optix.h>

#ifndef __CUDACC__
#include <cstring>
#endif

struct Params {
	OptixTraversableHandle tlas;
	uchar4*                framebuffer;
	unsigned int*          seedbuffer;
	unsigned int           width;
	unsigned int           height;
	unsigned int           samples;
	float3                 bgColor;
	float3                 camEye;
	float3                 camU;
	float3                 camV;
	float3                 camW;

#ifdef __CUDACC__
	__forceinline__ __device__ auto generate_ray(float2 d)  noexcept -> float3
	{
		using namespace otk;

		return normalize(camU * d.x + camV * d.y + camW - camEye);
	}
#endif

};


OTK_INLINE OTK_DEVICE unsigned int xorshift32(unsigned int& seed)
{
	unsigned int x = seed;
	x ^= (x << 13);
	x ^= (x >> 17);
	x ^= (x <<  5);
	return seed = x;
}

OTK_INLINE OTK_DEVICE float xorshift32_f32_01(unsigned int& seed)
{
	unsigned int uv = xorshift32(seed) >> static_cast<unsigned int>(9) | static_cast<unsigned int>(0x3f800000);
#ifdef __CUDACC__
	return __uint_as_float(uv) - 1.0f;
#else
	float res = 0.0f;
	std::memcpy(&res, &uv, sizeof(float));
	return res;
#endif
}

#ifdef __CUDACC__
OTK_INLINE OTK_DEVICE float4 get_sphere_data(float tMax)
{
	auto primitiveIndex = optixGetPrimitiveIndex();
	auto gas = optixGetGASTraversableHandle();
	auto sbtGasIndex = optixGetSbtGASIndex();

	float4 sphereData[1];
	optixGetSphereData(
		gas,
		primitiveIndex,
		sbtGasIndex,
		tMax,
		sphereData
	);

	return sphereData[0];
}
#endif

#endif
