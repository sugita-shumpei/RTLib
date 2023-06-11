#include <Test1_Context.h>
#include <optix_stubs.h>
#include <optix_function_table.h>
#include <optix_function_table_definition.h>

Test1::Context::Context()
{

}

Test1::Context::~Context()
{

}

void Test1::Context::init()
{
    this->init_cuda_driver();
    this->init_opx7_device_context();
}

void Test1::Context::free()
{
    this->free_opx7_device_context();
    this->free_cuda_driver();
}

auto Test1::Context::get_cuda_context() const noexcept -> CUcontext
{
    return this->m_CudaContext;
}

auto Test1::Context::get_cuda_device() const noexcept -> CUdevice
{
    return this->m_CudaDevice;
}

auto Test1::Context::get_opx7_device_context() const noexcept -> OptixDeviceContext
{
    return this->m_Opx7DeviceContext;
}

void Test1::Context::init_cuda_driver()
{
    OTK_ERROR_CHECK(cuInit(0));
    OTK_ERROR_CHECK(cuDeviceGet(&m_CudaDevice , 0));
    OTK_ERROR_CHECK(cuCtxCreate(&m_CudaContext, 0, m_CudaDevice));
}

void Test1::Context::free_cuda_driver()
{
    if (!m_CudaContext) { return; }
    OTK_ERROR_CHECK(cuCtxDestroy(m_CudaContext));
    m_CudaContext = 0;
    m_CudaDevice = 0;
}

void Test1::Context::init_opx7_device_context()
{
    OTK_ERROR_CHECK(optixInit());
    OptixDeviceContextOptions options = {};
#ifndef NDEBUG
    options.validationMode = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL;
    options.logCallbackFunction = opx7_log_callback;
    options.logCallbackLevel = 4;
#else
    options.validationMode = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_OFF;
    options.logCallbackLevel = 0;
#endif
    OTK_ERROR_CHECK(optixDeviceContextCreate(m_CudaContext, &options, &m_Opx7DeviceContext));
}

void Test1::Context::free_opx7_device_context()
{
    if (!m_Opx7DeviceContext) { return; }
    OTK_ERROR_CHECK(optixDeviceContextDestroy(m_Opx7DeviceContext));
    m_Opx7DeviceContext = nullptr;
}
