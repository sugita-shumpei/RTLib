#include <TestOptiX.h>
#include <optix_function_table_definition.h>

static void DebugLogCallback(unsigned int level, const char* tag, const char* message, void* cbdata) {
	std::cout << "[" << level << "][" << tag << "]: " << message << std::endl;
}

RTLib::Test::TestOptiXApplication::TestOptiXApplication() noexcept
{
	m_CtxOpx7 = nullptr;
	m_CtxCuda = nullptr;
	m_DevCuda = 0;
}

RTLib::Test::TestOptiXApplication::~TestOptiXApplication() noexcept
{
	m_CtxOpx7 = nullptr;
	m_CtxCuda = nullptr;
	m_DevCuda = 0;
}

void RTLib::Test::TestOptiXApplication::Init()
{
	TEST_MACRO_CUDA_CHECK(cudaFree(0));
	int numDeviCnt = 0;
	TEST_MACRO_CUDA_CHECK(cudaGetDeviceCount(&numDeviCnt));
	if (numDeviCnt == 0) {
		throw std::runtime_error("Failed To Get CUDA Device!");
	}
	TEST_MACRO_OPTIX_CHECK(optixInit());
	TEST_MACRO_CUDA_CHECK(cudaGetDevice(&m_DevCuda));
	TEST_MACRO_CU_CHECK(cuCtxCreate(&m_CtxCuda, 0, m_DevCuda));
	{
		auto options                = OptixDeviceContextOptions{};
		options.validationMode      = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL;
		options.logCallbackLevel    = 4;
		options.logCallbackFunction = DebugLogCallback;

		TEST_MACRO_OPTIX_CHECK(optixDeviceContextCreate(m_CtxCuda, &options, &m_CtxOpx7));
	}
}

void RTLib::Test::TestOptiXApplication::Main()
{
}

void RTLib::Test::TestOptiXApplication::Free() noexcept
{
	if (m_CtxOpx7) {
		optixDeviceContextDestroy(m_CtxOpx7);
		m_CtxOpx7 = nullptr;
	}
	if (m_CtxCuda) {
		TEST_MACRO_CU_CHECK(cuCtxDestroy(m_CtxCuda));
		m_CtxCuda = nullptr;
	}
	m_DevCuda = 0;
	TEST_MACRO_CUDA_CHECK(cudaFree(0));
}
