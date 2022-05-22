#ifndef  RTLIB_EXT_OPX7_TEST_SIMPLE_KERNEL_H
#define  RTLIB_EXT_OPX7_TEST_SIMPLE_KERNEL_H
#include <RTLib/Ext/CUDA/Math/VectorFunction.h>
#include <optix.h>
enum   SimpleKernelRayType
{
	SimpleKernelRayTypeTrace,
	SimpleKernelRayTypeOcclude,
};
struct SimpleKernelParams
{
	OptixTraversableHandle tlas;
	uchar4* frameBufferForGraphics;
	float3* frameBufferForCompute;
	float3* accumBuffer;
	unsigned int fbWidth;
	unsigned int fbHeight;
	unsigned int samplePerLaunch;
	unsigned int sampleForAccum ;
};
struct SimpleKernelSBTRaygenData
{
	float3 cameraEye;
	float3 cameraU;
	float3 cameraV;
	float3 cameraW;
};
struct SimpleKernelSBTMissData
{
	uchar4 missColor;
};
struct SimpleKernelSBTHitgroupData
{

};
#endif