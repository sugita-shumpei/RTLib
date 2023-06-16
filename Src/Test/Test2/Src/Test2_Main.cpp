#include <Test2_Main.h>
int main()
{
	auto context = std::make_unique<TestLib::Context>();
	context->init();

	auto pipelineGroup      = Test2::init_pipeline_group(context.get());
	auto shaderBindingTable = Test2::init_shader_binding_table(pipelineGroup.get());
	auto vertexBuffer       = std::make_unique<otk::SyncVector<float3>>(1);
	{
		vertexBuffer->at(0) = make_float3(0.0f, 0.0f, -2.0f);
		vertexBuffer->copyToDevice();
	}

	auto radiusBuffer = std::make_unique<otk::SyncVector<float>>(1);
	{
		radiusBuffer->at(0) = 0.5f;
		radiusBuffer->copyToDevice();
	}

	auto tempBuffer = std::make_unique<otk::DeviceBuffer>(1024);

	auto blas = OptixTraversableHandle();
	auto blasBuffer = std::make_unique<otk::DeviceBuffer>();
	{
		OptixAccelBuildOptions options = {};
		options.operation = OPTIX_BUILD_OPERATION_BUILD;
		options.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE | OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS;

		OptixBuildInput buildInputs[1] = {};
		CUdeviceptr vertexBuffers[1] = { reinterpret_cast<CUdeviceptr>(vertexBuffer->devicePtr()) };
		CUdeviceptr radiusBuffers[1] = { reinterpret_cast<CUdeviceptr>(radiusBuffer->devicePtr()) };
		unsigned int flags[1] = { OPTIX_GEOMETRY_FLAG_NONE };

		buildInputs[0].type                            = OPTIX_BUILD_INPUT_TYPE_SPHERES;
		buildInputs[0].sphereArray.vertexBuffers       = vertexBuffers;
		buildInputs[0].sphereArray.vertexStrideInBytes = 0;
		buildInputs[0].sphereArray.numVertices         = 1;
		buildInputs[0].sphereArray.radiusBuffers       = radiusBuffers;
		buildInputs[0].sphereArray.radiusStrideInBytes = 0;
		buildInputs[0].sphereArray.numSbtRecords       = 1;
		buildInputs[0].sphereArray.singleRadius        = false;
		buildInputs[0].sphereArray.flags               = flags;

		Test2::build_acceleration_structure(context.get(), nullptr, { buildInputs[0] }, options, tempBuffer.get(), *blasBuffer.get(), blas);
	}

	auto tlas = OptixTraversableHandle();
	auto instBuffer = std::make_unique<otk::SyncVector<OptixInstance>>(1);
	auto tlasBuffer = std::make_unique<otk::DeviceBuffer>();
	{
		float transforms[12] = {
			1.0f,0.0f,0.0f,0.0f,
			0.0f,1.0f,0.0f,0.0f,
			0.0f,0.0f,1.0f,0.0f
		};

		instBuffer->at(0).traversableHandle = blas;
		instBuffer->at(0).instanceId = 0;
		instBuffer->at(0).sbtOffset = 0;
		instBuffer->at(0).visibilityMask = OptixVisibilityMask(255);
		instBuffer->at(0).flags = OPTIX_INSTANCE_FLAG_NONE;
		std::memcpy(instBuffer->at(0).transform, transforms, sizeof(transforms));

		instBuffer->copyToDevice();

		OptixAccelBuildOptions options = {};
		options.operation = OPTIX_BUILD_OPERATION_BUILD;
		options.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE | OPTIX_BUILD_FLAG_ALLOW_COMPACTION;

		OptixBuildInput buildInputs[1] = {};
		otk::BuildInputBuilder buildInputBuilder(buildInputs);
		buildInputBuilder.instanceArray(static_cast<CUdeviceptr>(*instBuffer), instBuffer->size());

		Test2::build_acceleration_structure(context.get(), nullptr, { buildInputs[0] }, options, tempBuffer.get(), *tlasBuffer.get(), tlas);
	}

	auto pipeline = pipelineGroup->get_pipeline("Test2");
	pipeline->set_max_traversable_graph_depth(2);
	pipeline->compute_stack_sizes(0, 0);
	pipeline->update();

	unsigned int width  = 800;
	unsigned int height = 600;

	glfwInit();
	auto window = Test2::create_glfw_window(width, height, "title");

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
	auto paramsBuffer = std::make_unique<otk::SyncVector<Params>>(1);

	{
		auto renderer = TestLib::OGL4Renderer(window);

		renderer.init();
		renderer.set_current();

		auto pboCUGL = std::make_unique<TestLib::BufferCUGL>(renderer.get_pbo(), CU_GRAPHICS_MAP_RESOURCE_FLAGS_WRITE_DISCARD);

		while (!glfwWindowShouldClose(window)) {
			{
				size_t framebufferSize = 0;
				{
					auto [u, v, w] = camera.get_uvw();
					auto& params = paramsBuffer->at(0);
					params.tlas = tlas;
					params.width = fbWidth;
					params.height = fbHeight;
					params.framebuffer = reinterpret_cast<uchar4*>(pboCUGL->map(nullptr, framebufferSize));
					params.bgColor = make_float3(0, 0, 1);
					params.camEye = camera.get_eye();
					params.camU = u;
					params.camV = v;
					params.camW = w;

					paramsBuffer->copyToDeviceAsync(nullptr);
				}
				pipeline->launch(nullptr, reinterpret_cast<CUdeviceptr>(paramsBuffer->devicePtr()), sizeof(Params), shaderBindingTable.get(), fbWidth, fbHeight, 1);
				OTK_ERROR_CHECK(cuStreamSynchronize(nullptr));
			}
			pboCUGL->unmap(nullptr);

			renderer.render();

			glfwSwapBuffers(window);
			glfwPollEvents();
			glfwGetFramebufferSize(window, &fbWidth, &fbHeight);

			renderer.resize(fbWidth, fbHeight);
		}

		renderer.free();
	}

	glfwDestroyWindow(window);
	glfwTerminate();

	vertexBuffer.reset();
	radiusBuffer.reset();

	blasBuffer.reset();
	tlasBuffer.reset();
	instBuffer.reset();

	tempBuffer.reset();

	paramsBuffer.reset();
	frameBuffer.reset();

	pipelineGroup.reset();
	shaderBindingTable.reset();

	context->free();

	return 0;
}