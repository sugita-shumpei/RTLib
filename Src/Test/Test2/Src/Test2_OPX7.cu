#define __CUDACC__
#include <Test2_OPX7.h>
extern "C" {
	__constant__ Params params;
}

__forceinline__ __device__ auto generate_ray(float2 d)  noexcept -> float3
{
	using namespace otk;
	return normalize(params.camU * d.x + params.camV * d.y + params.camW );
}

extern "C" __global__ void __raygen__Test2()
{
	using namespace otk;
	uint3 idx = optixGetLaunchIndex();
	uint3 dim = optixGetLaunchDimensions();

	auto d = 2.0f * make_float2(
		static_cast<float>(idx.x) / static_cast<float>(dim.x),
		static_cast<float>(idx.y) / static_cast<float>(dim.y)
	) - 1.0f;

	auto rayOrigin = params.camEye;
	auto rayDirection = generate_ray(d);

	unsigned int p0, p1, p2;
	optixTrace(
		params.tlas,
		rayOrigin,
		rayDirection,
		0.0f,
		1e16f,
		0.0f,
		OptixVisibilityMask(255),
		OPTIX_RAY_FLAG_NONE,
		0,
		1,
		0,
		p0, p1, p2);

	float3 result = make_float3(0.0f, 0.0f, 0.0f);
	result.x = __int_as_float(p0);
	result.y = __int_as_float(p1);
	result.z = __int_as_float(p2);

	params.framebuffer[params.width * idx.y + idx.x] = make_color(result);
}

extern "C" __global__ void __miss__Test2()
{
	auto color = params.bgColor;
	
	optixSetPayload_0(__float_as_int(color.x));
	optixSetPayload_1(__float_as_int(color.y));
	optixSetPayload_2(__float_as_int(color.z));
}

extern "C" __global__ void __closesthit__Test2()
{
	using namespace otk;
	auto primitiveIndex = optixGetPrimitiveIndex();
	auto gas = optixGetGASTraversableHandle();
	auto sbtGasIndex = optixGetSbtGASIndex();
	float4 sphereData[1];
	optixGetSphereData(
		gas, 
		primitiveIndex,
		sbtGasIndex,
		0.0f,
		sphereData
	);
	auto p = optixGetWorldRayOrigin() + optixGetRayTmax()*optixGetWorldRayDirection();
	auto c = 0.5f*normalize(p - make_float3(sphereData[0].x, sphereData[0].y, sphereData[0].z)) + make_float3(0.5f, 0.5f, 0.5f);

	optixSetPayload_0(__float_as_int(c.x));
	optixSetPayload_1(__float_as_int(c.y));
	optixSetPayload_2(__float_as_int(c.z));
}