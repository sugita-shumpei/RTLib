#ifndef TEST_TEST1_MAIN__H
#define TEST_TEST1_MAIN__H
#include <Test1_OPX7.h>
#include <Test1_Context.h>
#include <Test1_PipelineGroup.h>
#include <Test1_Module.h>
#include <Test1_ShaderBindingTable.h>
#include <Test1_Pipeline.h>
#include <Test1_OGL4Renderer.h>
#include <Test1_Camera.h>
#include <OptiXToolkit/OptiXMemory/Builders.h>
#include <RTLib-Test-Test1-OPX7-ptx-generated.h>
#include <RTLib-Test-Test1-OPX7-optixir-generated.h>
#include <fstream>
#include <memory>
#include <GLFW/glfw3.h>
#include <optix_stack_size.h>
#include <cudaGL.h>
#include <glm/glm.hpp>
#include <glm/gtx/string_cast.hpp>
namespace Test1
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
	auto init_pipeline_group(Test1::Context* ctx) ->std::unique_ptr<Test1::PipelineGroup>
	{
		auto pipelineGroupOptions = OptixPipelineCompileOptions{};
		{
			pipelineGroupOptions.pipelineLaunchParamsVariableName = "params";
			pipelineGroupOptions.numAttributeValues = 3;
			pipelineGroupOptions.numPayloadValues = 3;
			pipelineGroupOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
			pipelineGroupOptions.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;
		}
		auto pipelineGroup = std::make_unique<Test1::PipelineGroup>(ctx, pipelineGroupOptions);
		{
			auto module_compile_options = Test1::ModuleOptions{};
			module_compile_options.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
			// Module
			if (!pipelineGroup->load_module("Test1", module_compile_options, Test1_OPX7_ptx_text(), Test1_OPX7_ptx_size)) {
				throw std::runtime_error("Failed To Load Module: Test1");
			}
			// ProgramGroup:Raygen
			if (!pipelineGroup->load_program_group_rg(/*Key*/"Test1", /*Module*/"Test1", /*Entry*/"Test1")) {
				throw std::runtime_error("Failed To Load Program Group RayGen: Test1");
			}
			// ProgramGroup:Miss
			if (!pipelineGroup->load_program_group_ms(/*Key*/"Test1", /*Module*/"Test1", /*Entry*/"Test1")) {
				throw std::runtime_error("Failed To Load Program Group Miss: Test1");
			}
			if (!pipelineGroup->load_program_group_hg(/*Key*/"Test1", /*ModuleCh*/"Test1", /*EntryCh*/"Test1", /*ModuleAh*/"", /*EntryAh*/"", /*ModuleIs*/"", /*EntryIs*/"")) {
				throw std::runtime_error("Failed To Load Program Group Miss: Test1");
			}
			// Pipeline
			if (!pipelineGroup->load_pipeline(/*Key*/"Test1",/*Raygen*/"Test1", /*Miss*/{ "Test1" },  /*Hitg*/{ "Test1" },  /*CCallable*/{},  /*DCallable*/{}, OPTIX_COMPILE_DEBUG_LEVEL_DEFAULT, 1)) {
				throw std::runtime_error("Failed To Load Pipeline: Test1");
			}
		}
		auto pipeline = pipelineGroup->get_pipeline("Test1");
		auto programGroups = pipeline->get_program_groups();
		auto opx7Pipeline = pipeline->get_opx7_pipeline();

		OptixStackSizes stack_sizes = {};
		for (auto  program_group : programGroups)
		{
			OTK_ERROR_CHECK(optixUtilAccumulateStackSizes(program_group->get_opx7_program_group(), &stack_sizes));
		}

		uint32_t direct_callable_stack_size_from_traversal;
		uint32_t direct_callable_stack_size_from_state;
		uint32_t continuation_stack_size;

		OTK_ERROR_CHECK(optixUtilComputeStackSizes(&stack_sizes, 1,
			0,  // maxCCDepth
			0,  // maxDCDEpth
			&direct_callable_stack_size_from_traversal,
			&direct_callable_stack_size_from_state, &continuation_stack_size));

		OTK_ERROR_CHECK(optixPipelineSetStackSize(opx7Pipeline, direct_callable_stack_size_from_traversal, direct_callable_stack_size_from_state, continuation_stack_size, 2));

		return pipelineGroup;
	}
	auto init_shader_binding_table(Test1::PipelineGroup* pipelineGroup) -> std::unique_ptr<Test1::ShaderBindingTable>
	{
		auto shaderBindingTable = std::make_unique<Test1::ShaderBindingTable>();
		shaderBindingTable->raygen = std::make_shared<Test1::TypeShaderRecord<otk::EmptyData>>(1);
		shaderBindingTable->raygen->pack_header(pipelineGroup->get_program_group_rg("Test1"));
		shaderBindingTable->raygen->copy_to_device_async(nullptr);

		shaderBindingTable->miss = std::make_shared<Test1::TypeShaderRecord<otk::EmptyData>>(1);
		shaderBindingTable->miss->pack_header(pipelineGroup->get_program_group_ms("Test1"));
		shaderBindingTable->miss->copy_to_device_async(nullptr);

		shaderBindingTable->hitgroup = std::make_shared<Test1::TypeShaderRecord<otk::EmptyData>>(1);
		shaderBindingTable->hitgroup->pack_header(pipelineGroup->get_program_group_hg("Test1"));
		shaderBindingTable->hitgroup->copy_to_device_async(nullptr);

		return shaderBindingTable;
	}
	//auto init_camera() -> std::unique_ptr<Test1::PerspectiveCamera>
	//{
	//	return std::make_unique<Test1::PerspectiveCamera>("camera",45.0f,1.0f,1e-5f,1e+5f);
	//}
}
#endif
