#define __CUDACC__
#include "SimpleKernel.h"
extern "C" {
	__constant__ SimpleKernelParams params;
}
extern "C" __global__ void __raygen__simple_kernel()
{
	auto launch_dim = optixGetLaunchDimensions();
	auto launch_idx = optixGetLaunchIndex();
	auto frame_index = launch_idx.y * launch_dim.x + launch_idx.x;
	float3 result = make_float3(1.0f);
	params.frameBufferForCompute[ frame_index] = make_float3(1.0f);
	params.frameBufferForGraphics[frame_index] = make_uchar4(255);
	params.accumBuffer[frame_index] = result;
}
extern "C" __global__ void __miss__simple_kernel()
{

}
extern "C" __global__ void __closesthit__simple_kernel()
{

}