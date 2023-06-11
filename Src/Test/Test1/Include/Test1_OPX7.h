#ifndef TEST1_OPX7__H
#define TEST1_OPX7__H
#include <OptiXToolkit/ShaderUtil/vec_math.h>
#include <OptiXToolkit/ShaderUtil/Preprocessor.h>
#include <OptiXToolkit/ShaderUtil/color.h>
#include <cuda_runtime.h>
#include <optix.h>

struct Params {
	OptixTraversableHandle tlas;
	unsigned int width;
	unsigned int height;
	uchar4* framebuffer;
	float3 bgColor;
	float3 camEye;
	float3 camU;
	float3 camV;
	float3 camW;
};

#endif
