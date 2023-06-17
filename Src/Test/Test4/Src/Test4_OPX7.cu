#define __CUDACC__
#include <Test4_OPX7.h>
extern "C" {
	__constant__ Params params;
}

extern "C" __global__ void __raygen__Test4()
{
	using namespace otk;
	uint3 idx = optixGetLaunchIndex();
	uint3 dim = optixGetLaunchDimensions();
	unsigned int pixelIdx = params.width * idx.y + idx.x;

	constexpr int numSamples = 10;
	float3 result = make_float3(0.0f, 0.0f, 0.0f);

	unsigned int seed = params.seedbuffer[pixelIdx];

	for (unsigned int i = 0; i < params.samples; ++i)
	{
		auto d = 2.0f * make_float2(
			static_cast<float>(idx.x+ xorshift32_f32_01(seed)) / static_cast<float>(params.width ),
			static_cast<float>(idx.y+ xorshift32_f32_01(seed)) / static_cast<float>(params.height)
		) - 1.0f;

		auto rayOrigin    = params.camEye;
		auto rayDirection = params.generate_ray(d);

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

		result.x += __int_as_float(p0);
		result.y += __int_as_float(p1);
		result.z += __int_as_float(p2);
	}

	result /= static_cast<float>(params.samples);

	params.seedbuffer[pixelIdx]  = seed;
	params.framebuffer[pixelIdx] = make_color(result);
}

extern "C" __global__ void __miss__Test4()
{
	auto color = params.bgColor;

	optixSetPayload_0(__float_as_int(color.x));
	optixSetPayload_1(__float_as_int(color.y));
	optixSetPayload_2(__float_as_int(color.z));

}

extern "C" __global__ void __closesthit__Test4()
{
	auto hgData = reinterpret_cast<HitgroupData*>(optixGetSbtDataPointer());

	using namespace otk;
	auto vertices= get_triangle_data(0.0f);
	auto worldPos= optixGetWorldRayOrigin() + optixGetRayTmax() * optixGetWorldRayDirection();

	auto worldV0 = optixTransformPointFromWorldToObjectSpace(vertices.row0);
	auto worldV1 = optixTransformPointFromWorldToObjectSpace(vertices.row1);
	auto worldV2 = optixTransformPointFromWorldToObjectSpace(vertices.row2);
	auto worldVN = otk::normalize(otk::cross(worldV1 - worldV0, worldV2 - worldV0));

	auto c = hgData->diffuse;

	optixSetPayload_0(__float_as_int(c.x));
	optixSetPayload_1(__float_as_int(c.y));
	optixSetPayload_2(__float_as_int(c.z));
}