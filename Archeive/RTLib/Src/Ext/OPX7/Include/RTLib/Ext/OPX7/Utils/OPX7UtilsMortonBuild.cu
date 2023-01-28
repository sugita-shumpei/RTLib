#include <cuda.h>
extern "C" __global__ void mortonBuildKernel(float* weightBuilding, unsigned int level, unsigned int nodesPerElement, unsigned int numNodes) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numNodes){
        float* weightBuildingNodeStart  = weightBuilding + idx * nodesPerElement;
        for (int i = 0;i<level;++i){
            unsigned int srcOffset = (__powf(4.0f,level  -i)-1)/3;
            unsigned int dstOffset = (__powf(4.0f,level-1-i)-1)/3;
            for (unsigned int code=0;code<__powf(4.0f,level-i);++code)
            {
                weightBuildingNodeStart[dstOffset+(code>>2)]+= weightBuildingNodeStart[srcOffset+code];
            }
        }
        
    }
}
extern "C" __global__ void mortonClearKernel(float* weightBuilding, unsigned int numWeightBuilding) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numWeightBuilding) {
        weightBuilding[idx] = 0.0f;
    }
}