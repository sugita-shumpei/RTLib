#ifndef TEST_TEST4_MAIN__H
#define TEST_TEST4_MAIN__H
#include <Test4_OPX7.h>
#include <TestLib/AccelerationStructure.h>
#include <TestLib/Context.h>
#include <TestLib/PipelineGroup.h>
#include <TestLib/Module.h>
#include <TestLib/ShaderBindingTable.h>
#include <TestLib/Pipeline.h>
#include <TestLib/Camera.h>
#include <TestLib/OGL4Renderer.h>
#include <TestLib/CUGL.h>
#include <TestLib/CornelBox.h>
#include <OptiXToolkit/OptiXMemory/Builders.h>
#include <RTLib-Test-Test4-OPX7-ptx-generated.h>
#include <RTLib-Test-Test4-OPX7-optixir-generated.h>
#include <fstream>
#include <memory>
#include <random>
#include <GLFW/glfw3.h>
#include <optix_stack_size.h>
#include <cudaGL.h>
#include <glm/glm.hpp>
#include <glm/gtx/string_cast.hpp>
namespace Test4
{
	auto create_glfw_window(int width, int height, const char* title) -> GLFWwindow*
	{
		glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_API);
		glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
		glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
		glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
		glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GLFW_TRUE);
		return glfwCreateWindow(width, height, title, nullptr, nullptr);
	}
	auto init_pipeline_group(TestLib::Context* ctx) -> std::unique_ptr<TestLib::PipelineGroup>
	{
		auto pipelineGroupOptions = OptixPipelineCompileOptions{};
		{
			pipelineGroupOptions.pipelineLaunchParamsVariableName = "params";
			pipelineGroupOptions.numAttributeValues = 3;
			pipelineGroupOptions.numPayloadValues = 3;
			pipelineGroupOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
			pipelineGroupOptions.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_SPHERE;
			pipelineGroupOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
		}
		auto pipelineGroup = std::make_unique<TestLib::PipelineGroup>(ctx, pipelineGroupOptions);
		{
			auto moduleCompileOptions = TestLib::ModuleOptions{};
			moduleCompileOptions.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
#ifndef NDEBUG
			moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_DEFAULT;
			moduleCompileOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
#else
			moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL;
			moduleCompileOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_3;
#endif
			// Module
			if (!pipelineGroup->load_module(
				"Test4",
				moduleCompileOptions,
				Test4_OPX7_optixir_text(),
				Test4_OPX7_optixir_size)) {
				throw std::runtime_error("Failed To Load Module: Test4");
			}

			OptixBuiltinISOptions builtinISOptions = {};
			builtinISOptions.builtinISModuleType = OPTIX_PRIMITIVE_TYPE_SPHERE;

			if (!pipelineGroup->load_builtin_is_module(
				"BuiltinIs.Sphere", moduleCompileOptions, builtinISOptions)
				) {
				throw std::runtime_error("BuiltinIs.Sphere");
			}
			// ProgramGroup:Raygen
			if (!pipelineGroup->load_program_group_rg(
				/*Key*/"Test4",
				/*Module*/"Test4",
				/*Entry*/"Test4")
				) {
				throw std::runtime_error("Failed To Load Program Group RayGen: Test4");
			}
			// ProgramGroup:Miss
			if (!pipelineGroup->load_program_group_ms(
				/*Key*/"Test4",
				/*Module*/"Test4",
				/*Entry*/"Test4")) {
				throw std::runtime_error("Failed To Load Program Group Miss: Test4");
			}
			if (!pipelineGroup->load_program_group_hg(
				/*Key*/"Test4",
				/*ModuleCh*/"Test4", /*EntryCh*/"Test4",
				/*ModuleAh*/"", /*EntryAh*/"",
				/*ModuleIs*/"BuiltinIs.Sphere", /*EntryIs*/"")
				) {
				throw std::runtime_error("Failed To Load Program Group Miss: Test4");
			}
			// Pipeline
			if (!pipelineGroup->load_pipeline(
				/*Key*/   "Test4",
				/*Raygen*/"Test4",
				/*Miss*/{ "Test4" },
				/*Hitg*/{ "Test4" },
				/*CCallable*/{},
				/*DCallable*/{},

#ifndef NDEBUG
				OPTIX_COMPILE_DEBUG_LEVEL_DEFAULT,
#else
				OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL,
#endif
				1)) {
				throw std::runtime_error("Failed To Load Pipeline: Test4");
			}
			auto pipeline = pipelineGroup->get_pipeline("Test4");
		}
		return pipelineGroup;
	}
	auto init_shader_binding_table(TestLib::PipelineGroup* pipelineGroup, CUstream stream = nullptr) -> std::unique_ptr<TestLib::ShaderBindingTable>
	{
		auto shaderBindingTable = std::make_unique<TestLib::ShaderBindingTable>();
		shaderBindingTable->raygen = std::make_shared<TestLib::TypeShaderRecord<otk::EmptyData>>(1);
		shaderBindingTable->raygen->pack_header(pipelineGroup->get_program_group_rg("Test4"));
		shaderBindingTable->raygen->copy_to_device_async(stream);

		shaderBindingTable->miss = std::make_shared<TestLib::TypeShaderRecord<otk::EmptyData>>(1);
		shaderBindingTable->miss->pack_header(pipelineGroup->get_program_group_ms("Test4"));
		shaderBindingTable->miss->copy_to_device_async(stream);

		shaderBindingTable->hitgroup = std::make_shared<TestLib::TypeShaderRecord<otk::EmptyData>>(1);
		shaderBindingTable->hitgroup->pack_header(pipelineGroup->get_program_group_hg("Test4"));
		shaderBindingTable->hitgroup->copy_to_device_async(stream);

		return shaderBindingTable;
	}
	void init_seed_buffer(otk::SyncVector<unsigned int>* seedBuffer, CUstream stream = nullptr)
	{
		std::random_device rd;
		std::mt19937 mt(rd());
		std::uniform_int_distribution<unsigned int> uni;
		std::generate(std::begin(*seedBuffer), std::end(*seedBuffer), [&uni, &mt]() { return uni(mt); });

		seedBuffer->copyToDeviceAsync(stream);
	}
	//auto init_camera() -> std::unique_ptr<TestLib::PerspectiveCamera>
	//{
	//	return std::make_unique<TestLib::PerspectiveCamera>("camera",45.0f,1.0f,1e-5f,1e+5f);
	//}
}
#endif
