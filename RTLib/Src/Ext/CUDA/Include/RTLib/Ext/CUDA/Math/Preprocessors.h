#ifndef RTLIB_EXT_CUDA_MATH_PREPROCESSORS_H
#define RTLIB_EXT_CUDA_MATH_PREPROCESSORS_H
#ifdef __CUDACC__
#define RTLIB_DEVICE __device__
#define RTLIB_HOST_DEVICE __host__ __device__
#define RTLIB_INLINE __forceinline__
#else
#define RTLIB_DEVICE 
#define RTLIB_HOST_DEVICE 
#define RTLIB_INLINE inline
#endif
#endif