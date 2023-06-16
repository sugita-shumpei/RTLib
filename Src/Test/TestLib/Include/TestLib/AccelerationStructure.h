#ifndef TEST_TESTLIB_ACCELERATION_STRUCTURE__H
#define TEST_TESTLIB_ACCELERATION_STRUCTURE__H
#include <TestLib/Context.h>
#include <OptiXToolkit/Memory/DeviceBuffer.h>
#include <optix.h>
#include <cuda.h>
#include <vector>
#include <memory>
namespace TestLib
{
	struct AccelerationStructure
	{
		AccelerationStructure(TestLib::Context* context);
		virtual ~AccelerationStructure();

		auto get_options() const noexcept -> const OptixAccelBuildOptions&;
		void set_options(const OptixAccelBuildOptions& options) noexcept;

		auto get_build_inputs() const noexcept -> const std::vector<OptixBuildInput>&;
		void set_build_inputs(const std::vector<OptixBuildInput>& buildInputs)noexcept;

		auto get_num_build_inputs() const noexcept -> size_t;
		void set_num_build_inputs(size_t numBuildInputs);

		auto get_build_input(size_t idx) const noexcept -> const OptixBuildInput&;
		void set_build_input(size_t idx, const OptixBuildInput& buildInput)noexcept;

		auto get_opx7_traversable_handle()const noexcept -> OptixTraversableHandle;

		void build(otk::DeviceBuffer* tempBuffer = nullptr);

		void build_async(CUstream stream, otk::DeviceBuffer* tempBuffer = nullptr);

	private:
		struct Impl;
		std::unique_ptr<Impl> m_Impl;
	};
}
#endif
