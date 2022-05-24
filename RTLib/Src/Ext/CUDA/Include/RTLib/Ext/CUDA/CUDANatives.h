#ifndef RTLIB_EXT_CUDA_CUDA_NATIVES_H
#define RTLIB_EXT_CUDA_CUDA_NATIVES_H
#include <cuda.h>
namespace RTLib
{
    namespace Ext
    {
        namespace CUDA
        {
            class  CUDAContext;
            class  CUDABuffer;
            class  CUDABufferView;
            class  CUDAModule;
            class  CUDAFunction;
            class  CUDAImage;
            class  CUDAStream;
            class  CUDATexture;
            class  CUDANatives;
            
            struct CUDANatives{
                static auto GetCUcontext(CUDAContext* context)->CUcontext;
                static auto GetCUdevice(CUDAContext* context)->CUdevice;
                static auto GetCUdeviceptr(CUDABuffer* buffer)->CUdeviceptr;
                static auto GetCUdeviceptr(const CUDABufferView& bufferView)->CUdeviceptr;
                static auto GetCUmodule(CUDAModule* module)->CUmodule;
                static auto GetCUfunction(CUDAFunction* function)->CUfunction;
                static auto GetCUarray(CUDAImage* image)->CUarray;
                static auto GetCUarrayWithLevel(CUDAImage* image, unsigned int level)->CUarray;
                static auto GetCUmipmappedArray(CUDAImage* image)->CUmipmappedArray;
                static auto GetCUstream(CUDAStream* stream)->CUstream;
                static auto GetCUtexObject(CUDATexture* texture)->CUtexObject;
            };
        }
    }
}
#endif