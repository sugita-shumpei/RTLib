#include <Test1_Main.h>
int main()
{
	auto context = std::make_unique<Test1::Context>();
	context->init();

	auto pipelineGroup      = Test1::init_pipeline_group(context.get());
	auto shaderBindingTable = Test1::init_shader_binding_table(pipelineGroup.get());

	auto vertexBuffer = std::make_unique<otk::SyncVector<float3>>(3);
	{
		vertexBuffer->at(0) = make_float3( 0.5f, -0.5f, -1.0f);
		vertexBuffer->at(1) = make_float3(-0.5f, -0.5f, -1.0f);
		vertexBuffer->at(2) = make_float3( 0.0f, +0.5f, -1.0f);

		vertexBuffer->copyToDevice();
	}

	auto tempBuffer = std::make_unique<otk::DeviceBuffer>(1024);

	auto blas       = OptixTraversableHandle();
	auto blasBuffer = std::make_unique<otk::DeviceBuffer>();
	{
		OptixAccelBuildOptions options = {};
		options.operation  = OPTIX_BUILD_OPERATION_BUILD;
		options.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE | OPTIX_BUILD_FLAG_ALLOW_DISABLE_OPACITY_MICROMAPS;

		OptixBuildInput buildInputs[1] = {};
		CUdeviceptr vertexBuffers[1] = { reinterpret_cast<CUdeviceptr>(vertexBuffer->devicePtr()) };
		buildInputs[0].type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
		buildInputs[0].triangleArray.vertexBuffers = vertexBuffers;

		unsigned int flags[1] = { OPTIX_GEOMETRY_FLAG_NONE };
		buildInputs[0].triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
		buildInputs[0].triangleArray.numVertices = 3;
		buildInputs[0].triangleArray.vertexStrideInBytes = 0;
		buildInputs[0].triangleArray.flags = flags;
		buildInputs[0].triangleArray.numSbtRecords = 1;

		OptixAccelBufferSizes bufferSizes = {};
		OTK_ERROR_CHECK(optixAccelComputeMemoryUsage(
			context->get_opx7_device_context(), &options, buildInputs, 1, &bufferSizes
		));

		blasBuffer->allocate(bufferSizes.outputSizeInBytes);
		if (tempBuffer->size() < bufferSizes.tempSizeInBytes) {
			tempBuffer->resize(bufferSizes.tempSizeInBytes);
		}

		OTK_ERROR_CHECK(optixAccelBuild(
			context->get_opx7_device_context(),
			nullptr,
			&options,
			buildInputs, 1,
			reinterpret_cast<CUdeviceptr>(tempBuffer->devicePtr()),
			tempBuffer->size(),
			reinterpret_cast<CUdeviceptr>(blasBuffer->devicePtr()),
			blasBuffer->size(),
			&blas, nullptr, 0
		));


	}

	auto tlas       = OptixTraversableHandle();
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
		options.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE ;

		OptixBuildInput buildInputs[1] = {};
		otk::BuildInputBuilder buildInputBuilder(buildInputs);
		buildInputBuilder.instanceArray(static_cast<CUdeviceptr>(*instBuffer), instBuffer->size());

		OptixAccelBufferSizes bufferSizes = {};
		OTK_ERROR_CHECK(optixAccelComputeMemoryUsage(
			context->get_opx7_device_context(), &options, buildInputs, 1, &bufferSizes
		));

		tlasBuffer->allocate(bufferSizes.outputSizeInBytes);
		if (tempBuffer->size() < bufferSizes.tempSizeInBytes) {
			tempBuffer->resize(bufferSizes.tempSizeInBytes);
		}

		OTK_ERROR_CHECK(optixAccelBuild(
			context->get_opx7_device_context(),
			nullptr,
			&options,
			buildInputs, 1,
			reinterpret_cast<CUdeviceptr>(tempBuffer->devicePtr()),
			tempBuffer->size(),
			reinterpret_cast<CUdeviceptr>(tlasBuffer->devicePtr()),
			tlasBuffer->size(),
			&tlas, nullptr, 0
		));
	}

	unsigned int width  = 800; 
	unsigned int height = 600;
	
	glfwInit();
	auto window = Test1::create_glfw_window(width, height, "title");

	int fbWidth; int fbHeight;
	glfwGetFramebufferSize(window, &fbWidth, &fbHeight);

	auto camera = Test1::Camera(make_float3(0.0f, 0.0f, 0.0f), make_float3(0.0f, 0.0f, -1.0f), make_float3(0.0f, 1.0f, 0.0f), (float)fbWidth / (float)fbHeight, 60.0f, 1.0f);

	auto frameBuffer  = std::make_unique<otk::DeviceBuffer>(fbWidth * fbHeight * sizeof(uchar4)); 
	auto paramsBuffer = std::make_unique<otk::SyncVector<Params>>(1);
	{
		auto [u, v, w]     = camera.get_uvw();
		auto& params       = paramsBuffer->at(0);
		params.tlas        = tlas;
		params.width       = fbWidth ;
		params.height      = fbHeight;
		params.framebuffer = (uchar4*)frameBuffer->devicePtr();
		params.bgColor     = make_float3(0, 0, 1);
		params.camEye      = camera.get_eye();
		params.camU        = u;
		params.camV        = v;
		params.camW        = w;

		paramsBuffer->copyToDeviceAsync(nullptr);
	}

	{
		
		{
			auto renderer = Test1::OGL4Renderer(window);

			renderer.init();
			renderer.set_current();

			CUgraphicsResource graphicsResource = nullptr;
			OTK_ERROR_CHECK(cuGraphicsGLRegisterBuffer(&graphicsResource, renderer.get_pbo(), CU_GRAPHICS_MAP_RESOURCE_FLAGS_WRITE_DISCARD));

			while (!glfwWindowShouldClose(window)) {
				CUdeviceptr framebufferPtr; size_t framebufferSize;
	
				OTK_ERROR_CHECK(cuGraphicsMapResources(1, &graphicsResource, nullptr));
				OTK_ERROR_CHECK(cuGraphicsResourceGetMappedPointer(&framebufferPtr, &framebufferSize, graphicsResource));
				{
					{
						auto& params       = paramsBuffer->at(0);
						params.width       = fbWidth;
						params.height      = fbHeight;
						params.framebuffer = reinterpret_cast<uchar4*>(framebufferPtr);
						paramsBuffer->copyToDeviceAsync(nullptr);
					}

					auto pipeline = pipelineGroup->get_pipeline("Test1");
					pipeline->launch(nullptr, reinterpret_cast<CUdeviceptr>(paramsBuffer->devicePtr()), sizeof(Params), shaderBindingTable.get(), fbWidth, fbHeight, 1);
					OTK_ERROR_CHECK(cuStreamSynchronize(nullptr));
				}
				OTK_ERROR_CHECK(cuGraphicsUnmapResources(1, &graphicsResource, nullptr));

				renderer.render();

				glfwSwapBuffers(window);
				glfwPollEvents(); 
				glfwGetFramebufferSize(window, &fbWidth, &fbHeight);

				renderer.resize(fbWidth, fbHeight);
			}

			renderer.free();

			OTK_ERROR_CHECK(cuGraphicsUnregisterResource(graphicsResource));
		}
	}

	glfwDestroyWindow(window);
	glfwTerminate();

	vertexBuffer.reset();

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