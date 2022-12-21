#define __CUDACC__
#include <RTLib/Ext/CUDA/Math/Random.h>
#include <RTLib/Ext/CUDA/Math/VectorFunction.h>
namespace rtlib = RTLib::Ext::CUDA::Math;
extern "C" __global__ void rgbKernel(uchar4* inBuffer,uchar4* outBuffer, int width, int height){
   int i = blockIdx.x * blockDim.x + threadIdx.x;
   int j = blockIdx.y * blockDim.y + threadIdx.y;
   if(i<width&&j<height){
       outBuffer[j*width+i] = rtlib::srgb_to_rgba(inBuffer[j*width+i]);
   }
}
extern "C" __global__ void blurKernel(uchar4* inBuffer,uchar4* outBuffer, int width, int height){
   int i = blockIdx.x * blockDim.x + threadIdx.x;
   int j = blockIdx.y * blockDim.y + threadIdx.y;
   if(i<width&&j<height){
       unsigned long long seed = static_cast<unsigned long long>(j)*width+i;
       auto rng      = rtlib::Xorshift128(seed);
       auto random_v = rtlib::random_float2(-5.0f,5.0f,rng);
       auto new_i    = rtlib::clamp((int)(i+random_v.x),0,width-1);
       auto new_j    = rtlib::clamp((int)(j+random_v.y),0,height-1);
       outBuffer[j*width+i] = inBuffer[new_j*width+new_i];
   }
};
//1024/32
extern "C" __global__ void naiveScanKernel_ScanPerThreads(
    const unsigned int* inBuffer ,
    unsigned int*       outBuffer,
    unsigned int        stride,
    unsigned int        offset,
    unsigned int        numElem)
{
    //1024 * 2 /32 = 64
    extern __shared__ unsigned int temp[];
    //blockDim.x = 1024/32
    unsigned int  off   = blockIdx.x * blockDim.x;
    unsigned int  thid  = threadIdx.x; 
    int pout = 0, pin   = 1;
    temp[thid]           = inBuffer[(off + thid) * stride + offset];
    temp[numElem + thid] = inBuffer[(off + thid) * stride + offset];
    __syncthreads();
    for (unsigned int o = 1; o <= numElem; o<<=1) {
        pout = 1 - pout; // swap double buffer indices
        pin  = 1 - pout;
        if (thid >= o)
            //temp[pin * numElem]
            temp[pout * numElem + thid]  = temp[pin * numElem + thid] +temp[pin * numElem + thid - o];
        else
            temp[pout * numElem + thid]  = temp[pin * numElem + thid];
        __syncthreads();
    }
    outBuffer[off + thid] = temp[pout * numElem + thid];
}
extern "C" __global__ void naiveScanKernel_AddPerThreads(
    const unsigned int* srcBuffer,
    unsigned int*       dstBuffer,
    unsigned int        numBlock ) {
    unsigned int dstIdx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int srcIdx =(dstIdx)/numBlock;
    if (srcIdx > 0) {
        dstBuffer[dstIdx] += srcBuffer[srcIdx - 1];
    }
}
extern "C" __global__ void downSweepScanKernel(unsigned int numElem, const unsigned int* countBuffer, unsigned int* offsetBuffer)
{

}