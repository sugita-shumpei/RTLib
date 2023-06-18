#include <Test4_Main.h>
int main()
{
	auto context = std::make_unique<TestLib::Context>();
	context->init();

	auto stream = CUstream(nullptr);
	OTK_ERROR_CHECK(cuStreamCreate(&stream, 0));

	auto pipelineGroup = Test4::init_pipeline_group(context.get());

	TestLib::CornelBox cornelbox;

	cornelbox.add_Backwall();
	cornelbox.add_Ceiling();
	cornelbox.add_Floor();
	cornelbox.add_Leftwall();
	cornelbox.add_Rightwall();
	cornelbox.add_Shortbox();
	cornelbox.add_Tallbox();
	cornelbox.add_Light();

	auto vertexBuffer     = std::make_unique<otk::DeviceBuffer>(sizeof(float3) * cornelbox.vertices.size());
	auto vertexBufferView = TestLib::BufferView(*vertexBuffer, sizeof(float3));
	vertexBufferView.copy_to_device_async(stream, cornelbox.vertices.data());

	auto indexBuffer = std::make_unique<otk::DeviceBuffer>(sizeof(uint3) * cornelbox.indices.size());
	auto indexBufferView = TestLib::BufferView(*indexBuffer, sizeof(uint3));
	indexBufferView.copy_to_device_async(stream, cornelbox.indices.data());

	auto tempBuffer = std::make_unique<otk::DeviceBuffer>(1024);


	auto blas = std::make_unique<TestLib::AccelerationStructure>(context.get());
	{
		OptixAccelBuildOptions options = {};
		options.operation  = OPTIX_BUILD_OPERATION_BUILD;
		options.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE | OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS;
		blas->set_options(options);

		OptixBuildInput buildInput = {};

		CUdeviceptr vertexBuffers[1] = {};
		unsigned int flags[1] = { OPTIX_GEOMETRY_FLAG_DISABLE_TRIANGLE_FACE_CULLING };

		buildInput.type                               = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
		buildInput.triangleArray.vertexBuffers        = vertexBuffers;
		buildInput.triangleArray.vertexFormat         = OPTIX_VERTEX_FORMAT_FLOAT3;
		buildInput.triangleArray.vertexStrideInBytes  = vertexBufferView.strideInBytes;
		buildInput.triangleArray.indexFormat          = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
		buildInput.triangleArray.indexStrideInBytes   = indexBufferView.strideInBytes;
		buildInput.triangleArray.opacityMicromap      = {};
		buildInput.triangleArray.preTransform         = 0;
		buildInput.triangleArray.transformFormat      = OPTIX_TRANSFORM_FORMAT_NONE;
		buildInput.triangleArray.primitiveIndexOffset = 0;
		buildInput.triangleArray.flags                = flags;
		buildInput.triangleArray.numSbtRecords        = 1;

		blas->set_num_build_inputs(cornelbox.groupNames.size());
		{
			size_t i = 0;
			for (auto& name : cornelbox.groupNames) {
				vertexBuffers[0]                          = vertexBufferView.devicePtr + cornelbox.verticesMap[name].x * vertexBufferView.strideInBytes;
				buildInput.triangleArray.numVertices      = cornelbox.verticesMap[name].y;
				buildInput.triangleArray.indexBuffer      = indexBufferView.devicePtr + cornelbox.indicesMap[name].x * indexBufferView.strideInBytes;
				buildInput.triangleArray.numIndexTriplets = cornelbox.indicesMap[name].y;
				blas->set_build_input(i, buildInput);

				++i;
			}
		}
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

	auto shaderBindingTable = std::make_unique<TestLib::ShaderBindingTable>();
	{
		shaderBindingTable->raygen = std::make_shared<TestLib::TypeShaderRecord<otk::EmptyData>>(1);
		shaderBindingTable->raygen->pack_header(pipelineGroup->get_program_group_rg("Test4"));
		shaderBindingTable->raygen->copy_to_device_async(stream);

		shaderBindingTable->miss = std::make_shared<TestLib::TypeShaderRecord<otk::EmptyData>>(1);
		shaderBindingTable->miss->pack_header(pipelineGroup->get_program_group_ms("Test4"));
		shaderBindingTable->miss->copy_to_device_async(stream);

		auto hitgroupSbt = std::make_shared<TestLib::TypeShaderRecord<HitgroupData>>(blas->get_num_build_inputs());
		hitgroupSbt->pack_header(pipelineGroup->get_program_group_hg("Test4"));

		for (size_t i = 0; i < cornelbox.groupNames.size(); ++i) {
			hitgroupSbt->data[i].diffuse  = make_float4(cornelbox.diffuses[i].x, cornelbox.diffuses[i].y, cornelbox.diffuses[i].z,1.0f);
			hitgroupSbt->data[i].emission = make_float4(cornelbox.emissions[i].x, cornelbox.emissions[i].y, cornelbox.emissions[i].z, 1.0f);
		}
		shaderBindingTable->hitgroup = hitgroupSbt;
		shaderBindingTable->hitgroup->copy_to_device_async(stream);
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
		make_float3(0.0f, 1.0f,  3.0f),
		make_float3(0.0f, 1.0f,  2.0f),
		make_float3(0.0f, 1.0f,  0.0f),
		(float)fbWidth / (float)fbHeight,
		60.0f,
		1.0f
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
					params.samples     = 100;
					params.depth       = 3;
					params.framebuffer = reinterpret_cast<uchar4*>(pboCUGL->map(stream, framebufferSize));
					params.seedbuffer  = seedBuffer->typedDevicePtr();
					params.bgColor     = make_float3(0, 0, 0);
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
			{
				auto eye   = camera.get_eye();
				auto front = camera.get_front();
				auto right = camera.get_right();
				auto up    = camera.get_up();
				if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
				{
					std::cout << eye.x << "," << eye.y << "," << eye.z << std::endl;
					using namespace otk;
					camera.set_eye(eye+0.1f * front);
				}
				if ((glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)||
					(glfwGetKey(window, GLFW_KEY_LEFT)==GLFW_PRESS))
				{
					std::cout << eye.x << "," << eye.y << "," << eye.z << std::endl;
					using namespace otk;
					camera.set_eye(eye - 0.1f * right);

				}
				if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
				{
					std::cout << eye.x << "," << eye.y << "," << eye.z << std::endl;

					using namespace otk;
					camera.set_eye(eye - 0.1f * front);
				}
				if ((glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) ||
					(glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS))
				{
					std::cout << eye.x << "," << eye.y << "," << eye.z << std::endl;
					using namespace otk;
					camera.set_eye(eye + 0.1f * right);
				}
				if (glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS)
				{
					std::cout << eye.x << "," << eye.y << "," << eye.z << std::endl;

					using namespace otk;
					camera.set_eye(eye + 0.1f * up);
				}
				if (glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS)
				{
					std::cout << eye.x << "," << eye.y << "," << eye.z << std::endl;

					using namespace otk;
					camera.set_eye(eye - 0.1f * up);
				}
			}
		}
		OTK_ERROR_CHECK(cuEventDestroy(begEvent));
		OTK_ERROR_CHECK(cuEventDestroy(endEvent));

		renderer.free();
	}


	glfwDestroyWindow(window);
	glfwTerminate();

	vertexBuffer.reset();
	indexBuffer.reset();

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