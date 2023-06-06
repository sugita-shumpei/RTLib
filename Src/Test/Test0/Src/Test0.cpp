#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <cuda.h>
#include <optix.h>
#include <optix_stubs.h>
#include <optix_function_table_definition.h>
#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>
#include <nlohmann/json.hpp>
#include <uuid.h>
#include <iostream>
#include <array>
#include <vector>
#include <fstream>
#include <utility>
#include <unordered_map>
#include <unordered_set>
#include <cstdio>

#ifndef NDEBUG
#define CU_CHECK(EXPR) \
	do { \
		CUresult res = EXPR; \
		if (res != CUresult::CUDA_SUCCESS){ \
			std::printf("%s(%d): CUDA Driver Api Call Failed: "#EXPR"\n", __FILE__,__LINE__); \
			abort(); \
		} \
	}while(false);
#else
#define CU_CHECK(EXPR) EXPR
#endif

#ifndef NDEBUG
#define OPX7_CHECK(EXPR) \
	do { \
		OptixResult res = EXPR; \
		if (res != OptixResult::OPTIX_SUCCESS){ \
			std::printf("%s(%d): Optix7 Api Call Failed: "#EXPR"\n", __FILE__,__LINE__); \
			abort(); \
		} \
	}while(false);
#else
#define OPX7_CHECK(EXPR) EXPR
#endif

struct OptixApp
{
	void init_cuda_driver() {
		CU_CHECK(cuInit(0));
		CU_CHECK(cuDeviceGet(&cu_device, 0));
		CU_CHECK(cuCtxCreate(&cu_context, 0, cu_device));
	}
	void free_cuda_driver() {
		CU_CHECK(cuCtxDestroy(cu_context));
		cu_context = 0;
		cu_device = 0;
	}

	void init_opx7_context() {
		OPX7_CHECK(optixInit());
		OptixDeviceContextOptions options = {};
#ifndef NDEBUG
		options.validationMode = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL;
		options.logCallbackFunction = opx7_log_callback;
		options.logCallbackLevel = 4;
#else
		options.validationMode = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_OFF;
		options.logCallbackLevel = 0;
#endif
		OPX7_CHECK(optixDeviceContextCreate(cu_context, &options, &opx7_context));
	}
	void free_opx7_context() {
		OPX7_CHECK(optixDeviceContextDestroy(opx7_context));
	}

	static void opx7_log_callback(unsigned int level, const char* tag, const char* message, void* cbdata)
	{
		constexpr const char* level2Str[] = { "Disable","Fatal","Error","Warning","Print" };
		printf("[%s][%s]: %s\n", level2Str[level], tag, message);
	}

	CUdevice  cu_device;
	CUcontext cu_context;

	OptixDeviceContext opx7_context;
};
int main() {
	OptixApp app;
	app.init_cuda_driver();
	app.init_opx7_context();
	{

	}
	app.free_opx7_context();
	app.free_cuda_driver();
	return 0;
}
