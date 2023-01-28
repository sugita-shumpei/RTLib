#ifndef RTLIB_EXT_OPX7_OPX7_NATIVES_H
#define RTLIB_EXT_OPX7_OPX7_NATIVES_H
#include <RTLib/Ext/OPX7/OPX7Common.h>
#include <memory>
namespace RTLib {
	namespace Ext
	{
        namespace CUDA {
            class CUDABuffer;
        }
		namespace OPX7
		{
			class OPX7Context;
			class OPX7Module;
			class OPX7ProgramGroup;
			class OPX7Pipeline;
			class OPX7ShaderTable;
			struct OPX7Natives
			{
				static auto GetOptixDeviceContext(OPX7Context* context)->OptixDeviceContext;
				static auto GetOptixModule(OPX7Module* module)->OptixModule;
				static auto GetOptixProgramGroup(OPX7ProgramGroup* programGroup)->OptixProgramGroup;
				static auto GetOptixPipeline(OPX7Pipeline* module)->OptixPipeline;
                struct AccelBuildOutput {
                    std::unique_ptr<CUDA::CUDABuffer> buffer;
                    OptixTraversableHandle            handle;
                };
				static auto BuildAccelerationStructure(RTLib::Ext::OPX7::OPX7Context* context, const OptixAccelBuildOptions& accelBuildOptions, const std::vector<OptixBuildInput>& buildInputs)->AccelBuildOutput;
			};
		}
	}
}
#endif
