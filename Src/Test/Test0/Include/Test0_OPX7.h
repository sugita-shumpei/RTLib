#ifndef TEST0_OPX7__H
#define TEST0_OPX7__H
#include <OptiXToolkit/ShaderUtil/vec_math.h>
#include <OptiXToolkit/ShaderUtil/Preprocessor.h>
#include <cuda_runtime.h>
#include <optix.h>
struct Params {
	OptixTraversableHandle tlas;
	unsigned int width ;
	unsigned int height;
	uchar4* framebuffer;
	uchar4 clearColor;
};
#endif
