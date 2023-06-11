#include <Test0_OPX7.h>
extern "C" {
	__constant__ Params params;
}
extern "C" __global__ void __raygen__Test0()
{
	uint3 launchIdx = optixGetLaunchIndex();
	uint3 launchDim = optixGetLaunchDimensions();
	params.framebuffer[params.width * launchIdx.y + launchIdx.x] = params.clearColor;

}
extern "C" __global__ void __miss__Test0()
{

}