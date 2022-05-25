#define __CUDACC__
#include <RTLib/Ext/CUDA/Math/Random.h>
#include <RTLib/Ext/CUDA/Math/VectorFunction.h>
namespace rtlib = RTLib::Ext::CUDA::Math;
extern "C" __global__ void randomKernel(unsigned int* seedBuffer, uchar4* outBuffer, int width, int height){
   int i = blockIdx.x * blockDim.x + threadIdx.x;
   int j = blockIdx.y * blockDim.y + threadIdx.y;
   if(i<width&&j<height){
       unsigned int seed = seedBuffer[j*width+i];
       auto rng = rtlib::Xorshift32(seed);
       auto col = rtlib::random_float1(0.0f,1.0f,rng);
       outBuffer[j*width+i] = make_uchar4(col*255,col*255,col*255,255);
       seedBuffer[j*width+i] = rng.m_seed;
   }
}