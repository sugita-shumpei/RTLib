#ifndef TEST4_OPX7__H
#define TEST4_OPX7__H

#include <OptiXToolkit/ShaderUtil/vec_math.h>
#include <OptiXToolkit/ShaderUtil/SelfIntersectionAvoidance.h>
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
	unsigned int           depth;
	float3                 bgColor;
	float3                 camEye;
	float3                 camU;
	float3                 camV;
	float3                 camW;

#ifdef __CUDACC__
	__forceinline__ __device__ auto generate_ray(float2 d)  noexcept -> float3
	{
		using namespace otk;

		return normalize(camU * d.x + camV * d.y + camW);
	}
#endif

};

struct HitgroupData {
	float4 diffuse ;
	float4 emission;
};

struct float3x3 {
	float3 row0;
	float3 row1;
	float3 row2;
};

OTK_INLINE OTK_DEVICE unsigned int xorshift32(unsigned int& seed)
{
	unsigned int x = seed;
	x ^= (x << 13);
	x ^= (x >> 17);
	x ^= (x <<  5);
	return seed = x;
}

OTK_INLINE OTK_DEVICE float  xorshift32_f32_01(unsigned int& seed)
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

OTK_INLINE OTK_DEVICE float2 xorshift32_f32x2_01(unsigned int& seed)
{
	return make_float2(
		xorshift32_f32_01(seed),
		xorshift32_f32_01(seed)
	);
}

OTK_INLINE OTK_DEVICE float3 xorshift32_f32x3_01(unsigned int& seed)
{
	return make_float3(
		xorshift32_f32_01(seed),
		xorshift32_f32_01(seed),
		xorshift32_f32_01(seed)
	);
}

OTK_INLINE OTK_DEVICE float3 xorshift32_cosine_distribution(unsigned int& seed)
{
	using namespace otk;
	float sint = sqrtf(xorshift32_f32_01(seed));
	float cost = sqrtf(fmaxf(1.0f - sint, 0.0f));
	float p    = 2.0f * M_PIf * xorshift32_f32_01(seed);
	float cosp = cosf(p);
	float sinp = sinf(p);
	return make_float3(sint * cosp, sint * sinp, cost);
}


#ifdef __CUDACC__
OTK_INLINE OTK_DEVICE float3x3 get_triangle_data(float time)
{
	auto primitiveIndex = optixGetPrimitiveIndex();
	auto gas = optixGetGASTraversableHandle();
	auto sbtGasIndex = optixGetSbtGASIndex();

	float3 vertices[3];
	optixGetTriangleVertexData(
		gas, 
		primitiveIndex,
		sbtGasIndex,
		time, 
		vertices
	);

	float3x3 res = { vertices[0],vertices[1],vertices[2] };
	return res;
}
OTK_INLINE OTK_DEVICE float4 get_sphere_data(float time)
{
	auto primitiveIndex = optixGetPrimitiveIndex();
	auto gas = optixGetGASTraversableHandle();
	auto sbtGasIndex = optixGetSbtGASIndex();

	float4 sphereData[1];
	optixGetSphereData(
		gas,
		primitiveIndex,
		sbtGasIndex,
		time,
		sphereData
	);
	return sphereData[0];
}

OTK_INLINE OTK_DEVICE void pack_float3(
	float3 v, 
	unsigned int& p0, 
	unsigned int& p1, 
	unsigned int& p2)
{
	p0 = __float_as_uint(v.x);
	p1 = __float_as_uint(v.y);
	p2 = __float_as_uint(v.z);
}

OTK_INLINE OTK_DEVICE auto unpack_float3(
	unsigned int p0,
	unsigned int p1,
	unsigned int p2) -> float3
{
	return make_float3(
		__uint_as_float(p0),
		__uint_as_float(p1),
		__uint_as_float(p2)
	);
}

#endif

#endif
