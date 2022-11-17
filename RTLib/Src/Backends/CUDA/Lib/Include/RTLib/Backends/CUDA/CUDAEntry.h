#ifndef RTLIB_BACKENDS_CUDA_ENTRY_H
#define RTLIB_BACKENDS_CUDA_ENTRY_H
#include <memory>
#include <vector>
namespace RTLib
{
	namespace Backends {
		namespace Cuda {
			enum class DimensionType {
				eUnknown = 0,

				e1D,
				e2D,
				e3D,
				eCubemap,

				eLayered1D,
				eLayered2D,
				eLayeredCubemap
			};
			enum ContextCreateFlags {
				ContextCreateScheduleAuto = 0,
				ContextCreateScheduleSpin = 1,
				ContextCreateScheduleYield = 2,
				ContextCreateScheduleBlockingSync = 4,
				ContextCreateScheduleMask =
				ContextCreateScheduleSpin | ContextCreateScheduleYield | ContextCreateScheduleBlockingSync,
				ContextCreateMapHost = 8,
				ContextCreateLmemResizeToMax = 16,
				ContextCreateMask =
				ContextCreateScheduleMask | ContextCreateMapHost | ContextCreateLmemResizeToMax,
				ContextCreateDefault = ContextCreateScheduleAuto,
			};
			enum StreamCreateFlags {
				StreamCreateDefault     = 0,
				StreamCreateNonBlocking = 1,
			};
			enum class ArrayFormat {
				eUnknown = 0,
				
				eUInt8 ,
				eUInt16 ,
				eUInt32,

				eSInt8,
				eSInt16,
				eSInt32,

				eFloat16,
				eFloat32,

				eNV12,

				eUnormInt8X1,
				eUnormInt8X2,
				eUnormInt8X4,
				eUnormInt16X1,
				eUnormInt16X2,
				eUnormInt16X4,

				eSnormInt8X1,
				eSnormInt8X2,
				eSnormInt8X4,
				eSnormInt16X1,
				eSnormInt16X2,
				eSnormInt16X4,
			};
			struct Memory1DCopy {
				size_t srcOffsetInBytes;
				size_t dstOffsetInBytes;
				size_t sizeInBytes;
			};
			struct Memory2DCopy {
				unsigned int srcXOffsetInBytes = 0;
				unsigned int srcYOffset        = 0;
				unsigned int srcPitchInBytes   = 0;

				unsigned int dstXOffsetInBytes = 0;
				unsigned int dstYOffset        = 0;
				unsigned int dstPitchInBytes   = 0;

				unsigned int widthInBytes      = 0;
				unsigned int height            = 0;
			};
			enum class TextureResourceType
			{
				eArray = 0,
				eMipmappedArray,
				eLinearMemory1D,
				eLinearMemory2D
			};
			enum class AddressMode {
				eWarp  = 0,
				eClamp = 1,
				eMirror= 2,
				eBorder= 3,
			};
			enum class FilterMode {
				ePoint = 0,
				eLinear= 1,
			};
			enum class TextureReadMode
			{
				eElementType     = 0,
				eNormalizedFloat = 1,
			};
			struct TextureDesc {
				AddressMode  addressMode[3]              = {};
				float        borderColor[4]               = {};
				bool         disableTrilinearOptimization = false;
				FilterMode   filterMode                   = FilterMode::ePoint;
				unsigned int maxAnisotropy                = 0; // [1,16[
				float        maxMipmapLevelClamp  = 0.0f;
				float        minMipmapLevelClamp  = 0.0f;
				FilterMode   mipmapFilterMode     = FilterMode::ePoint;
				float        mipmapFilterBias     = 0.0f;
				bool         normalizedCoords = false;
				TextureReadMode readMode = TextureReadMode::eElementType;
				bool         sRGB = false;
			};
			enum class JitOption {
				eMaxRegisters,
				eThreadsPerBlock,
				eWallTime,
				eInfoLogBuffer,
				eInfoLogBufferSizeBytes,
				eErrorLogBuffer,
				eErrorLogBufferSizeBytes,
				eOptimizationLevel,
				eTargetFromContext,
				eTarget,
				eFallbackStrategy,
				eGenerateDebugInfo,
				eLogVerbose,
				eGenerateLineInfo,
				eCacheMode,
				eFastCompile,
				eGlobalSymbolNames,
				eGlobalSymbolAddresses,
				eGlobalSymbolCount,
				eLTO,
				eFTZ,
				ePrecDiv,
				ePrecSqrt,
				eFma,
				eReferencedKernelNames,
				eReferencedKernelCount,
				eReferencedVariableNames,
				eReferencedVariableCount,
				eOptimizeUnusedDeviceVariables,
			};
			struct KernelLaunchDesc {
				unsigned int gridDimX;
				unsigned int gridDimY;
				unsigned int gridDimZ;
				unsigned int blockDimX;
				unsigned int blockDimY;
				unsigned int blockDimZ;
				unsigned int sharedMemBytes;
				std::vector<void*> params;
			};
			struct GlobalAddressDesc {
				void* pointer;
				size_t bytes ;
			};
			class Device;
			class Entry {
			private:
				Entry() noexcept;
			public:
				static auto Handle() noexcept -> Entry&;
				Entry(const Entry&) = delete;
				Entry(Entry&&) = delete;
				Entry& operator=(const Entry&) = delete;
				Entry& operator=(Entry&&) = delete;
				~Entry() noexcept;

				auto EnumerateDevices()const noexcept -> const std::vector<Device>&;
			private:
				struct Impl;
				std::unique_ptr<Impl> m_Impl;
			};
		}
	}
}
#endif
