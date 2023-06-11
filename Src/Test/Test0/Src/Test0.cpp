#include <Test0_OPX7.h>
#include <Test0_OPX7_ptx_generated.h>
// optix-toolkit
#include <OptiXToolkit/Error/ErrorCheck.h>
#include <OptixToolkit/Error/cuErrorCheck.h>
#include <OptixToolkit/Error/cudaErrorCheck.h>
#include <OptixToolkit/Error/optixErrorCheck.h>
#include <OptixToolkit/Memory/DeviceBuffer.h>
#include <OptiXToolkit/Memory/BinnedSuballocator.h>
#include <OptiXToolkit/OptiXMemory/Builders.h>
#include <OptiXToolkit/OptiXMemory/SyncRecord.h>
// glad
#include <glad/gl.h>
// glfw
#include <GLFW/glfw3.h>
// optix
#include <optix_stubs.h>
#include <optix_function_table_definition.h>
// cuda
#include <cuda_gl_interop.h>
// glm
#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtc/quaternion.hpp>
// assimp
#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>
// json
#include <nlohmann/json.hpp>
// uuid
#include <uuid.h>
// c++17
#include <iostream>
#include <array>
#include <vector>
#include <fstream>
#include <utility>
#include <unordered_map>
#include <unordered_set>
#include <string>
#include <cstdio>

struct OptixApp
{
	void init_cuda_driver() {
		OTK_ERROR_CHECK(cuInit(0));
		OTK_ERROR_CHECK(cuDeviceGet(&cu_device, 0));
		OTK_ERROR_CHECK(cuCtxCreate(&cu_context, 0, cu_device));
	}
	void free_cuda_driver() {
		OTK_ERROR_CHECK(cuCtxDestroy(cu_context));
		cu_context = 0;
		cu_device = 0;
	}

	void init_opx7_context() {
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
		OTK_ERROR_CHECK(optixDeviceContextCreate(cu_context, &options, &opx7_context));
	}
	void free_opx7_context() {
		OTK_ERROR_CHECK(optixDeviceContextDestroy(opx7_context));
		opx7_context = nullptr;
	}

	void init_opx7_module() {
		opx7_module_compile_options = {};
		opx7_module_compile_options.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
		opx7_pipeline_compile_options = {};
		opx7_pipeline_compile_options.pipelineLaunchParamsVariableName = "params";
		opx7_pipeline_compile_options.numAttributeValues = 3;
		opx7_pipeline_compile_options.numPayloadValues = 3;
		char log[1024];
		size_t logSize = sizeof(log);
		OTK_ERROR_CHECK(optixModuleCreateFromPTX(opx7_context, &opx7_module_compile_options, &opx7_pipeline_compile_options, Test0_OPX7_ptx_text(), Test0_OPX7_ptx_size, log, &logSize, &opx7_module));
		if (logSize != sizeof(log)) {
			std::cout << log << std::endl;
		}
	}
	void free_opx7_module() {
		OTK_ERROR_CHECK(optixModuleDestroy(opx7_module));
		opx7_module = nullptr;
	}

	void init_opx7_program_groups() {
		OptixProgramGroupDesc program_group_descs[2] = { };
		auto program_group_desc_rg = otk::ProgramGroupDescBuilder(program_group_descs, opx7_module)
			.raygen("__raygen__Test0")
			.miss("__miss__Test0");
		char log[1024];
		size_t logSize = sizeof(log);
		OptixProgramGroup programGroups[2] = {};
		OTK_ERROR_CHECK(optixProgramGroupCreate(opx7_context, &program_group_descs[0], 1, &opx7_program_group_options, log, &logSize, &programGroups[0]));
		if (logSize != sizeof(log)) {
			std::cout << log << std::endl;
		}

		OTK_ERROR_CHECK(optixProgramGroupCreate(opx7_context, &program_group_descs[1], 1, &opx7_program_group_options, log, &logSize, &programGroups[1]));
		if (logSize != sizeof(log)) {
			std::cout << log << std::endl;
		}
		opx7_program_group_rg = programGroups[0] ;
		opx7_program_group_ms = programGroups[1] ;
	}
	void free_opx7_program_groups() {
		OTK_ERROR_CHECK(optixProgramGroupDestroy(opx7_program_group_rg));
		OTK_ERROR_CHECK(optixProgramGroupDestroy(opx7_program_group_ms));
		opx7_program_group_rg = nullptr;
		opx7_program_group_ms = nullptr;
	}

	void init_opx7_pipeline() 
	{

		opx7_pipeline_link_options = {};
		opx7_pipeline_link_options.maxTraceDepth = 1;
#ifndef NDEBUG
		opx7_pipeline_link_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_DEFAULT;
#else
		opx7_pipeline_link_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;
#endif
		OptixProgramGroup programGroups[2] = { opx7_program_group_rg,opx7_program_group_ms };
		char log[1024];
		size_t logSize = sizeof(log);
		OTK_ERROR_CHECK(optixPipelineCreate(opx7_context, &opx7_pipeline_compile_options, &opx7_pipeline_link_options, programGroups, 2, log, &logSize, &opx7_pipeline));
		if (logSize != sizeof(log)) {
			std::cout << log << std::endl;
		}
		
	}
	void free_opx7_pipeline()
	{
		OTK_ERROR_CHECK(optixPipelineDestroy(opx7_pipeline));
		opx7_pipeline = nullptr;
	}

	void init_shader_record()
	{

		raygenRecord = std::make_unique<otk::SyncRecord<otk::EmptyData>>(1);
		raygenRecord->packHeader(opx7_program_group_rg);
		raygenRecord->copyToDevice();

		missRecord = std::make_unique<otk::SyncRecord<otk::EmptyData>>(1);
		missRecord->packHeader(opx7_program_group_ms);
		missRecord->copyToDevice();
	
		shader_table = {};
		shader_table.raygenRecord = *raygenRecord;
		shader_table.missRecordBase = *missRecord;
		shader_table.missRecordCount = 1;
		
		shader_table.missRecordStrideInBytes = sizeof(otk::EmptyRecord);
		
	}
	void free_shader_record()
	{
		raygenRecord.reset();
		missRecord.reset();
	}

	void init_launch_params()
	{
		params = std::make_unique<otk::SyncVector<Params>>(1);
		auto& paramsData = params->at(0);
		paramsData.tlas = 0;
		paramsData.width = width;
		paramsData.height = height;
		paramsData.framebuffer = framebuffer->typedDevicePtr();
		paramsData.clearColor = make_uchar4(255,0,255,255);
		params->copyToDevice();
	}
	void free_launch_params()
	{
		params.reset();
	}

	void get_launch_params(Params& params_) const{
		params_ = params->at(0);
	}
	void set_launch_params(const Params& params_) {
		width  = params_.width ;
		height = params_.height;
		params->at(0) = params_;
		params->copyToDevice() ;
	}

	void init_frame_buffer()
	{
		
		framebuffer = std::make_unique<otk::SyncVector<uchar4>>(width * height);
		framebuffer->copyToDevice();
	}
	void free_frame_buffer()
	{
		framebuffer.reset();
	}

	void launch() {
		OTK_ERROR_CHECK(optixLaunch(opx7_pipeline, nullptr,reinterpret_cast<CUdeviceptr>(params->devicePtr()), sizeof(Params), &shader_table, width,height, 1));
		OTK_ERROR_CHECK(cuStreamSynchronize(nullptr));
		OTK_ERROR_CHECK(cuMemcpyDtoH(&framebuffer->at(0), reinterpret_cast<CUdeviceptr>(params->at(0).framebuffer), width * height * sizeof(uchar4)));
	}

	static void opx7_log_callback(unsigned int level, const char* tag, const char* message, void* cbdata)
	{
		constexpr const char* level2Str[] = { "Disable","Fatal","Error","Warning","Print" };
		printf("[%s][%s]: %s\n", level2Str[level], tag, message);
	}

	unsigned int width = 800;
	unsigned int height = 600;
	CUdevice  cu_device;
	CUcontext cu_context;
	CUgraphicsResource cu_graphics_resource_framebuffer;

	OptixDeviceContext opx7_context;
	OptixModule opx7_module;
	OptixModuleCompileOptions opx7_module_compile_options = {};
	OptixProgramGroup opx7_program_group_rg;
	OptixProgramGroup opx7_program_group_ms;
	OptixProgramGroupOptions opx7_program_group_options = {};
	OptixPipeline opx7_pipeline;
	OptixPipelineCompileOptions opx7_pipeline_compile_options = {};
	OptixPipelineLinkOptions opx7_pipeline_link_options = {};
	std::unique_ptr<otk::SyncRecord<otk::EmptyData>> raygenRecord = {};
	std::unique_ptr<otk::SyncRecord<otk::EmptyData>> missRecord = {};
	std::unique_ptr<otk::SyncVector<Params>> params = {};
	std::unique_ptr<otk::SyncVector<uchar4>> framebuffer = {};
	OptixShaderBindingTable shader_table;
};
struct OGL4Renderer {
	OGL4Renderer(unsigned int width_, unsigned int height_, std::string title_)
		:width{width_},height{height_},title{title_},window{nullptr}
	{
	}
	~OGL4Renderer()
	{}

	void init()
	{
		glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_API);
		glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
		glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
		glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
		glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GLFW_TRUE);
		window = glfwCreateWindow(width, height, title.c_str(), nullptr, nullptr);
		auto prvWindow = glfwGetCurrentContext();
		glfwMakeContextCurrent(window);

		gl = std::make_unique<GladGLContext>();
		if (!gladLoadGLContext(gl.get(), (GLADloadfunc)glfwGetProcAddress)) {
			throw std::runtime_error("Failed To Init Context!");
		}

		GLuint bff[3];
		gl->GenBuffers(3, bff);
		vbo = bff[0]; ibo = bff[1]; pbo = bff[2];
		{
			gl->BindBuffer(GL_ARRAY_BUFFER, vbo);
			gl->BufferData(GL_ARRAY_BUFFER, sizeof(vertices) , vertices, GL_STATIC_DRAW);
			gl->BindBuffer(GL_ARRAY_BUFFER, 0);

			gl->BindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);
			gl->BufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices) , indices, GL_STATIC_DRAW);
			gl->BindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

			gl->BindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
			gl->BufferData(GL_PIXEL_UNPACK_BUFFER, width*height*sizeof(uchar4), nullptr, GL_STREAM_DRAW);
			gl->BindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
		}

		gl->GenTextures(1, &tex);
		{
			gl->BindTexture(GL_TEXTURE_2D, tex);
			gl->TexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
			gl->TexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
			gl->TexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
			gl->TexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
			gl->TexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
		}

		gl->GenVertexArrays(1, &vao);
		{
			gl->BindVertexArray(vao);
			{
				gl->BindBuffer(GL_ARRAY_BUFFER, vbo);
				
				gl->EnableVertexAttribArray(0);
				gl->VertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(glm::vec4), static_cast<float*>(nullptr) + 0);

				gl->EnableVertexAttribArray(1);
				gl->VertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, sizeof(glm::vec4), static_cast<float*>(nullptr) + 2);

				gl->BindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);

				gl->BindVertexArray(0);
				gl->BindBuffer(GL_ARRAY_BUFFER, 0);
				gl->BindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
			}
		}

		prg = new_program();
		texLoc = gl->GetUniformLocation(prg,"tex");

		glfwMakeContextCurrent(prvWindow);
	}
	void free() {
		auto prvWindow = glfwGetCurrentContext();
		glfwMakeContextCurrent(window);

		gl->DeleteProgram(prg);
		prg = 0; 
		texLoc = 0;

		gl->DeleteVertexArrays(1, &vao);
		vao = 0;

		gl->DeleteTextures(1, &tex);
		tex = 0;

		GLuint bff[3] = { vbo,ibo, pbo };
		gl->DeleteBuffers(3, bff);
		vbo = 0; ibo = 0; pbo = 0;

		gl.reset();

		glfwMakeContextCurrent(prvWindow);
		glfwDestroyWindow(window);
		window = nullptr;
	}

	void update() {
		glfwPollEvents();
	}
	void render() {
		
		auto prvWindow = glfwGetCurrentContext();
		glfwMakeContextCurrent(window);
		
		gl->Clear(GL_COLOR_BUFFER_BIT);
		
		gl->ClearColor(0, 0, 0, 0);

		gl->UseProgram(prg);

		gl->ActiveTexture(GL_TEXTURE0 + 0);
		gl->BindTexture(GL_TEXTURE_2D, tex);

		gl->BindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
		gl->TexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
		gl->BindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

		gl->Uniform1i(texLoc, 0);

		//std::vector<uchar4> pixels(width * height, make_uchar4(255, 255, 255, 255));

		gl->BindVertexArray(vao);
		gl->DrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, nullptr);
		glfwSwapBuffers(window);

		glfwMakeContextCurrent(prvWindow);
		// gladGLContext->DrawElements();
	}

	bool close() {
		return glfwWindowShouldClose(window);
	}

	auto get_pbo() const -> GLuint { return pbo; }

	void set_current() {
		glfwMakeContextCurrent(window);
	}
private:
	GLuint new_program() {
		GLint len;
		std::vector<char> log;

		GLuint prog = gl->CreateProgram();
		GLuint vso = gl->CreateShader(GL_VERTEX_SHADER);
		GLuint fso = gl->CreateShader(GL_FRAGMENT_SHADER);
		{
			const char* vsSourcePtr = vsSource;
			gl->ShaderSource(vso, 1, &vsSourcePtr, nullptr);
			gl->CompileShader(vso);
			GLint res;
			gl->GetShaderiv(vso, GL_COMPILE_STATUS, &res);
			gl->GetShaderiv(vso, GL_INFO_LOG_LENGTH, &len);

			log.resize(len + 1);
			gl->GetShaderInfoLog(vso, len, &len, log.data());
			log.resize(len);

			if (len > 0) {
				std::cout << log.data() << std::endl;
			}

			if (res != GL_TRUE)
			{
				throw std::runtime_error("Failed To Compile Vert Shader!");
			}
		}

		{
			const char* fsSourcePtr = fsSource;
			gl->ShaderSource(fso, 1, &fsSourcePtr, nullptr);
			gl->CompileShader(fso);
			GLint res;
			gl->GetShaderiv(fso, GL_COMPILE_STATUS, &res);
			gl->GetShaderiv(fso, GL_INFO_LOG_LENGTH, &len);

			log.resize(len + 1);
			gl->GetShaderInfoLog(fso, len, &len, log.data());
			log.resize(len);

			if (len > 0) {
				std::cout << log.data() << std::endl;
			}

			if (res != GL_TRUE)
			{
				throw std::runtime_error("Failed To Compile Frag Shader!");
			}
		}

		{
			gl->AttachShader(prog, vso);
			gl->AttachShader(prog, fso);

			gl->LinkProgram(prog);

			gl->DetachShader(prog, vso);
			gl->DetachShader(prog, fso);

			gl->DeleteShader(vso);
			gl->DeleteShader(fso);

			GLint res;
			gl->GetProgramiv(prog, GL_LINK_STATUS, &res);
			gl->GetProgramiv(prog, GL_INFO_LOG_LENGTH, &len);

			log.resize(len + 1);
			gl->GetProgramInfoLog(prog, len, &len, log.data());
			log.resize(len);

			if (len > 0) {
				std::cout << log.data() << std::endl;
			}

			if (res != GL_TRUE)
			{
				gl->DeleteProgram(prog);
				throw std::runtime_error("Failed To Link Program!");
			}
			return prog;
		}
	}

	GLFWwindow* window ;
	unsigned int width ;
	unsigned int height;
	std::string  title ;
	std::unique_ptr<GladGLContext> gl = {};
	
	inline static constexpr char vsSource[] =
		"#version 460 core\n"
		"layout(location = 0) in vec2 position;\n"
		"layout(location = 1) in vec2 texCoord;\n"
		"out vec2 fragCoord;\n"
		"void main()\n"
		"{\n"		
		"	gl_Position = vec4(position,1.0,1.0); \n"
		"	fragCoord = texCoord; \n"
		"}\n";

	inline static constexpr char fsSource[] =
		"#version 460 core\n"
		"in vec2 fragCoord;\n"
		"layout(location = 0) out vec4 fragColor;\n"
		"uniform sampler2D tex;\n"
		"void main()\n"
		"{\n"
		"	fragColor = texture(tex,fragCoord); \n"
		"}\n";

	static constexpr glm::vec4 vertices[4] = 
	{
		{-1.0f,-1.0f,0.0f,0.0f},
		{-1.0f,+1.0f,0.0f,1.0f},
		{+1.0f,+1.0f,1.0f,1.0f},
		{+1.0f,-1.0f,1.0f,0.0f}
	};
	static constexpr uint32_t indices[6] =
	{
		0,1,2,
		2,3,0
	};

	GLuint prg = 0;
	GLuint vbo = 0;
	GLuint ibo = 0;
	GLuint pbo = 0;
	GLuint vao = 0;
	GLuint tex = 0;

	GLint texLoc = 0;
};
int main() {
	glfwInit();

	auto app = OptixApp();
	auto renderer = OGL4Renderer(app.width, app.height, "title");
	auto graphics_resource = static_cast<cudaGraphicsResource*>(nullptr);
	try {
		app.init_cuda_driver();
		app.init_opx7_context();
		app.init_opx7_module();
		app.init_opx7_program_groups();
		app.init_opx7_pipeline();
		app.init_frame_buffer();
		app.init_shader_record();
		app.init_launch_params();

		renderer.init();
		{
			renderer.set_current();
			OTK_ERROR_CHECK(cudaGraphicsGLRegisterBuffer(&graphics_resource, renderer.get_pbo(), cudaGraphicsMapFlagsNone));

			while (!renderer.close())
			{

				void* framebufferPtr = nullptr;
				size_t framebufferSize = 0;

				OTK_ERROR_CHECK(cudaGraphicsMapResources(1, &graphics_resource, nullptr));
				OTK_ERROR_CHECK(cudaGraphicsResourceGetMappedPointer(&framebufferPtr, &framebufferSize, graphics_resource));
				
				{
					Params params = {};

					app.get_launch_params(params);
					params.framebuffer = (uchar4*)framebufferPtr;
					app.set_launch_params(params);

					app.launch();

				}

				OTK_ERROR_CHECK(cudaGraphicsUnmapResources(1, &graphics_resource, nullptr));

				renderer.render();
				renderer.update();
			}

			OTK_ERROR_CHECK(cudaGraphicsUnregisterResource(graphics_resource));
		}
		renderer.free();

		app.free_launch_params();
		app.free_shader_record();
		app.free_frame_buffer();
		app.free_opx7_pipeline();
		app.free_opx7_program_groups();
		app.free_opx7_module();
		app.free_opx7_context();
		app.free_cuda_driver();
	}
	catch (std::runtime_error& err) {
		std::cout << err.what() << std::endl;
	}

	glfwTerminate();
	return 0;
}
