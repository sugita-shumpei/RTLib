#include <cuda.h>
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