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
extern "C" __global__ void naiveScanKernel_Scan(
    const unsigned int* inBuffer ,
    unsigned int*       outBuffer,
    unsigned int        stride,
    unsigned int        offset,
    unsigned int        numElem,
    unsigned int        maxRangeInBuffer)
{
    //1024 * 2 /32 = 64
    extern __shared__ unsigned int temp[];
    //blockDim.x = 1024/32
    unsigned int  off   = blockIdx.x * blockDim.x;
    unsigned int  thid  = threadIdx.x; 
    unsigned int  idx_per_elem = off + thid;
    unsigned int  idx = idx_per_elem * stride + offset;
    int pout = 0, pin   = 1;
    if (idx < maxRangeInBuffer)
    {

        temp[thid]           = inBuffer[idx];
        temp[numElem + thid] = inBuffer[idx];
    }
    else {
        temp[thid]           = 0;
        temp[numElem + thid] = 0;
    }
    __syncthreads();
    for (unsigned int o = 1; o <= numElem; o<<=1) {
        pout = 1 - pout; // swap double buffer indices
        pin  = 1 - pout;
        if (thid >= o) {
            //temp[pin * numElem]
            temp[pout * numElem + thid] = temp[pin * numElem + thid] + temp[pin * numElem + thid - o];
        }
        else {
            temp[pout * numElem + thid] = temp[pin * numElem + thid];
        }
        __syncthreads();
    }
    outBuffer[idx_per_elem] = temp[pout * numElem + thid];
}
extern "C" __global__ void naiveScanKernel_Add(
    const unsigned int* srcBuffer,
    unsigned int*       dstBuffer,
    unsigned int        numBlock ,
    unsigned int        maxRangeDstBuffer) {
    unsigned int dstIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (dstIdx >= maxRangeDstBuffer) {
        return;
    }
    unsigned int srcIdx =(dstIdx)/numBlock;
    if (srcIdx > 0) {
        dstBuffer[dstIdx] += srcBuffer[srcIdx - 1];
    }
}
extern "C" __global__ void downSweepScanKernel_Scan(
    const unsigned int* inBuffer,
    unsigned int*       outBuffer,
    unsigned int        stride,
    unsigned int        offset,
    unsigned int        numElem,
    unsigned int        maxRangeInBuffer)
{

    //1024 * 2 /32 = 64
    extern __shared__ unsigned int temp[];
    //blockDim.x = 1024/32
    unsigned int  off  = blockIdx.x * blockDim.x;
    unsigned int  thid = threadIdx.x;
    unsigned int  idx_per_elem = off + 2 * thid;
    unsigned int  idx = idx_per_elem * stride + offset;
    unsigned int  idx_off = 1;
    int pout = 0, pin = 1;
    temp[2 * thid + 0] = ((idx + 0) < maxRangeInBuffer) ? inBuffer[idx + 0] : 0;
    temp[2 * thid + 1] = ((idx + 1) < maxRangeInBuffer) ? inBuffer[idx + 1] : 0;
    for (int d = numElem >> 1; d > 0; d >> 1) {
        __syncthreads();
        if (thid < d) {
            int ai = idx_off * (2 * thid + 1) - 1;
            int bi = idx_off * (2 * thid + 2) - 1;
            temp[bi] += temp[ai];

        }
        idx_off *= 2;
    }
    if (thid == 0)
    {
        temp[numElem - 1] = 0;
    }
    for (int d = 1; d < numElem;d*=2)
    {
        idx_off >>= 1;
        __syncthreads();
        if (thid < d) {
            int ai = idx_off * (2 * thid + 1) - 1;
            int bi = idx_off * (2 * thid + 2) - 1;
            unsigned int t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }
    __syncthreads();
    if ((idx + 0) < maxRangeInBuffer) {
        outBuffer[idx + 0] = temp[2 * thid + 1];
        if (idx == numElem - 1) {
            outBuffer[idx + 0] += inBuffer[numElem - 1];
        }
    }
    if ((idx + 1) < maxRangeInBuffer) {
        outBuffer[idx + 1] = temp[2 * thid + 2];
        if ((idx+1) == numElem - 1) {
            outBuffer[idx + 1] += inBuffer[numElem - 1];
        }
    }
}