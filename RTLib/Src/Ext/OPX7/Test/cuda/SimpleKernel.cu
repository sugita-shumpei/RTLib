#define __CUDACC__
#include "SimpleKernel.h"
extern "C" {
	__constant__ SimpleKernelParams params;
}
extern "C" __global__ void __raygen__simple_kernel()
{
	auto launch_dim  = optixGetLaunchDimensions();
	auto launch_idx  = optixGetLaunchIndex();
	auto frame_index = launch_idx.y * launch_dim.x + launch_idx.x;
	auto texCoords   = make_float2(
		static_cast<float>(launch_idx.x) / static_cast<float>(launch_dim.x),
		static_cast<float>(launch_idx.y) / static_cast<float>(launch_dim.y)
	);
	float3 result = make_float3(
		texCoords.x,
		texCoords.y,
		1.0f - (texCoords.x + texCoords.y) / 2.0f
	);
	params.frameBufferForCompute[ frame_index] = result;
	params.frameBufferForGraphics[frame_index] = make_uchar4(
		static_cast<unsigned char>(result.x * 255),
		static_cast<unsigned char>(result.y * 255),
		static_cast<unsigned char>(result.z * 255),255
	);
	params.accumBuffer[frame_index] = result;
}
extern "C" __global__ void __miss__simple_kernel()
{

}
extern "C" __global__ void __closesthit__simple_kernel()
{

}