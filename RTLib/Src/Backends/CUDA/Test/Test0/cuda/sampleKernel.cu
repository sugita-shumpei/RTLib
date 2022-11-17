#define __CUDACC__
#include <RTLib/Backends/CUDA/Math/Random.h>
#include <RTLib/Backends/CUDA/Math/VectorFunctions.h>
namespace rtlib = RTLib::Backends::Cuda::Math;
extern "C" __global__ void randomKernel(unsigned int* seedBuffer, uchar4* outBuffer, int width, int height){
   int i = blockIdx.x * blockDim.x + threadIdx.x;
   int j = blockIdx.y * blockDim.y + threadIdx.y;
   if(i<width&&j<height){
       unsigned int seed = seedBuffer[j*width+i];
       auto rng = rtlib::Xorshift32(seed);
       auto col = rtlib::random_float3(make_float3(0.0f), make_float3(1.0f),rng);
       outBuffer[j*width+i] = make_uchar4(col.x*255,col.y*255,col.z*255,255);
       seedBuffer[j*width+i] = rng.m_seed;
   }
}