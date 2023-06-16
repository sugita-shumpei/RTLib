#include <Test4_Main.h>
int main()
{
	auto context = std::make_unique<TestLib::Context>();
	context->init();

	auto stream = CUstream(nullptr);
	OTK_ERROR_CHECK(cuStreamCreate(&stream, 0));

	auto pipelineGroup      = Test4::init_pipeline_group(context.get());
	auto shaderBindingTable = Test4::init_shader_binding_table(pipelineGroup.get());

	auto vertexBuffer = std::make_unique<otk::SyncVector<float3>>(2);
	{
		vertexBuffer->at(0) = make_float3(0.0f, 0.5f, -2.0f);
		vertexBuffer->at(1) = make_float3(0.0f,-100.0f, -2.0f);
		vertexBuffer->copyToDeviceAsync(stream);
	}

	auto radiusBuffer = std::make_unique<otk::SyncVector<float>>(2);
	{
		radiusBuffer->at(0) = 0.5f;
		radiusBuffer->at(1) = 100.0f;
		radiusBuffer->copyToDeviceAsync(stream);
	}

	auto tempBuffer = std::make_unique<otk::DeviceBuffer>(1024);

	auto blas = std::make_unique<TestLib::AccelerationStructure>(context.get());
	{
		OptixAccelBuildOptions options = {};
		options.operation  = OPTIX_BUILD_OPERATION_BUILD;
		options.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE | OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS;
		blas->set_options(options);

		std::vector<OptixBuildInput> buildInputs(1);

		CUdeviceptr vertexBuffers[1] = { reinterpret_cast<CUdeviceptr>(vertexBuffer->devicePtr()) };
		CUdeviceptr radiusBuffers[1] = { reinterpret_cast<CUdeviceptr>(radiusBuffer->devicePtr()) };
		unsigned int flags[1] = { OPTIX_GEOMETRY_FLAG_NONE };

		auto buildInputBuilder = otk::BuildInputBuilder(buildInputs.data(), buildInputs.size()).spheres(vertexBuffers,2,radiusBuffers,flags,1);

		blas->set_build_inputs(buildInputs);
		blas->build_async(stream, tempBuffer.get());
	}

	auto tlas = std::make_unique<TestLib::AccelerationStructure>(context.get());
	auto instBuffer = std::make_unique<otk::SyncVector<OptixInstance>>(1);
	{
		float transforms[12] = {
			1.0f,0.0f,0.0f,0.0f,
			0.0f,1.0f,0.0f,0.0f,
			0.0f,0.0f,1.0f,0.0f
		};

		instBuffer->at(0).traversableHandle = blas->get_opx7_traversable_handle();
		instBuffer->at(0).instanceId = 0;
		instBuffer->at(0).sbtOffset = 0;
		instBuffer->at(0).visibilityMask = OptixVisibilityMask(255);
		instBuffer->at(0).flags = OPTIX_INSTANCE_FLAG_NONE;

		std::memcpy(instBuffer->at(0).transform, transforms, sizeof(transforms));
		instBuffer->copyToDeviceAsync(stream);

		OptixAccelBuildOptions options = {};
		options.operation  = OPTIX_BUILD_OPERATION_BUILD;
		options.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE | OPTIX_BUILD_FLAG_ALLOW_COMPACTION;

		tlas->set_options(options);

		std::vector<OptixBuildInput> buildInputs(1);
		auto buildInputBuilder = otk::BuildInputBuilder(buildInputs.data(),buildInputs.size()).instanceArray(static_cast<CUdeviceptr>(*instBuffer), instBuffer->size());

		tlas->set_build_inputs(buildInputs);
		tlas->build_async(stream, tempBuffer.get());
	}

	auto pipeline = pipelineGroup->get_pipeline("Test4");
	pipeline->set_max_traversable_graph_depth(2);
	pipeline->compute_stack_sizes(0, 0);
	pipeline->update();

	unsigned int width = 800;
	unsigned int height = 600;

	glfwInit();
	auto window = Test4::create_glfw_window(width, height, "title");

	int fbWidth; int fbHeight;
	glfwGetFramebufferSize(window, &fbWidth, &fbHeight);

	auto camera = TestLib::Camera(
		make_float3(0.0f, 0.0f,  0.0f),
		make_float3(0.0f, 0.0f, -1.0f),
		make_float3(0.0f, 1.0f,  0.0f),
		(float)fbWidth / (float)fbHeight,
		60.0f,
		1.0e-3f
	);

	auto frameBuffer = std::make_unique<otk::DeviceBuffer>(fbWidth * fbHeight * sizeof(uchar4));
	auto seedBuffer  = std::make_unique<otk::SyncVector<unsigned int>>(fbWidth * fbHeight);
	Test4::init_seed_buffer(seedBuffer.get());

	auto paramsBuffer = std::make_unique<otk::SyncVector<Params>>(1);

	{
		auto renderer = TestLib::OGL4Renderer(window);

		renderer.init();
		renderer.set_current();
		
		auto pboCUGL = std::make_unique<TestLib::BufferCUGL>(renderer.get_pbo(), CU_GRAPHICS_MAP_RESOURCE_FLAGS_WRITE_DISCARD);
		bool isResized = false;


		CUevent begEvent, endEvent;
		OTK_ERROR_CHECK(cuEventCreate(&begEvent, 0));
		OTK_ERROR_CHECK(cuEventCreate(&endEvent, 0));
		while (!glfwWindowShouldClose(window)) {
			{
				if (isResized) {
					camera.set_aspect(static_cast<float>(fbWidth) / static_cast<float>(fbHeight));
					
					pboCUGL.reset();
					pboCUGL = std::make_unique<TestLib::BufferCUGL>(renderer.get_pbo(), CU_GRAPHICS_MAP_RESOURCE_FLAGS_WRITE_DISCARD);
					
					frameBuffer->resize(fbWidth* fbHeight * sizeof(uchar4));

					seedBuffer->resize(fbWidth * fbHeight);
					Test4::init_seed_buffer(seedBuffer.get());
				}
				size_t framebufferSize = 0;
				{
					auto [u, v, w]     = camera.get_uvw();
					auto& params       = paramsBuffer->at(0);
					params.tlas        = tlas->get_opx7_traversable_handle();
					params.width       = fbWidth;
					params.height      = fbHeight;
					params.samples     = 10;
					params.framebuffer = reinterpret_cast<uchar4*>(pboCUGL->map(stream, framebufferSize));
					params.seedbuffer  = seedBuffer->typedDevicePtr();
					params.bgColor     = make_float3(0, 0, 1);
					params.camEye      = camera.get_eye();
					params.camU        = u;
					params.camV        = v;
					params.camW        = w;

					paramsBuffer->copyToDeviceAsync(stream);
				}
				OTK_ERROR_CHECK(cuEventRecord(begEvent, stream));
				pipeline->launch(stream, reinterpret_cast<CUdeviceptr>(paramsBuffer->devicePtr()), sizeof(Params), shaderBindingTable.get(), fbWidth, fbHeight, 1);
				OTK_ERROR_CHECK(cuEventRecord(endEvent, stream));
				OTK_ERROR_CHECK(cuEventSynchronize(endEvent));
				float milliseconds = 0;
				OTK_ERROR_CHECK(cuEventElapsedTime(&milliseconds, begEvent, endEvent));
				std::string durationStr = "duration: " + std::to_string(milliseconds) + "ms";
				glfwSetWindowTitle(window, durationStr.c_str());
			}
			OTK_ERROR_CHECK(cuStreamSynchronize(stream));
			pboCUGL->unmap(stream);

			renderer.render();

			glfwSwapBuffers(window);
			glfwPollEvents();
			glfwGetFramebufferSize(window, &fbWidth, &fbHeight);

			if (renderer.resize(fbWidth, fbHeight)) {
				isResized = true;
			}
			else {
				isResized = false;
			}
		}

		OTK_ERROR_CHECK(cuEventDestroy(begEvent));
		OTK_ERROR_CHECK(cuEventDestroy(endEvent));

		renderer.free();
	}


	glfwDestroyWindow(window);
	glfwTerminate();

	vertexBuffer.reset();
	radiusBuffer.reset();

	blas.reset();
	tlas.reset();

	instBuffer.reset();

	tempBuffer.reset();

	paramsBuffer.reset();
	frameBuffer.reset();
	seedBuffer.reset();

	pipelineGroup.reset();
	shaderBindingTable.reset();

	OTK_ERROR_CHECK(cuStreamDestroy(stream));
	context->free();

	return 0;
}