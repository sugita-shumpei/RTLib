#ifndef RTLIB_CORE_PREPROCESSOR__H
#define RTLIB_CORE_PREPROCESSOR__H

#ifndef __CUDACC__
#ifdef WIN32

#ifdef RTLIB_DLL_EXPORT 
#define RTLIB_DLL __declspec(dllexport)
#else
#define RTLIB_DLL __declspec(dllimport)
#endif

#endif
#define RTLIB_DEVICE 
#define RTLIB_HOST 
#define RTLIB_HOST_DEVICE 
#define RTLIB_GLOBAL 
#define RTLIB_INLINE inline
#else
#define RTLIB_DLL 
#define RTLIB_DEVICE __device__
#define RTLIB_HOST __host__
#define RTLIB_HOST_DEVICE __host__ __device__
#define RTLIB_GLOBAL __global__
#define RTLIB_INLINE __forceinline__
#endif

#endif
