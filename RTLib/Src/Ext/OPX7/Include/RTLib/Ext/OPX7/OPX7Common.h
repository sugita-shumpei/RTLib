#ifndef RTLIB_EXT_OPX7_OPX7_COMMON_H
#define RTLIB_EXT_OPX7_OPX7_COMMON_H
#include <RTLib/Core/Common.h>
#include <RTLib/Ext/CUDA/CUDABuffer.h>
#include <optix_types.h>
#include <optional>
#include <iostream>
#include <vector>
namespace RTLib
{
	namespace Ext
	{
		namespace OPX7
		{
			enum class OPX7ContextValidationLogLevel : unsigned int
			{
				eDisable = 0,
				eFatal,
				eError,
				eWarn,
				ePrint,
			};
			enum class OPX7ContextValidationMode
			{
				eOFF = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_OFF,
				eALL = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL,
			};
			inline constexpr char *ToString(OPX7ContextValidationLogLevel level)
			{
				switch (level)
				{
				case RTLib::Ext::OPX7::OPX7ContextValidationLogLevel::eDisable:
					return "Disable";
					break;
				case RTLib::Ext::OPX7::OPX7ContextValidationLogLevel::eFatal:
					return "Fatal";
					break;
				case RTLib::Ext::OPX7::OPX7ContextValidationLogLevel::eError:
					return "Error";
					break;
				case RTLib::Ext::OPX7::OPX7ContextValidationLogLevel::eWarn:
					return "Warn";
					break;
				case RTLib::Ext::OPX7::OPX7ContextValidationLogLevel::ePrint:
					return "Print";
					break;
				default:
					return "Unknown";
					break;
				}
			}
			static void DefaultLogCallback(unsigned int level, const char *tag, const char *message, void *cbData)
			{
				std::string levelStr = "";
				if (level < 5)
				{
					levelStr = ToString(static_cast<OPX7ContextValidationLogLevel>(level));
				}
				std::cerr << "[" << levelStr << "][" << tag << "]: " << message << std::endl;
			}
			struct OPX7ContextCreateDesc
			{
				OptixLogCallback pfnLogCallback = DefaultLogCallback;
				void *pCallbackData = nullptr;
				OPX7ContextValidationLogLevel level = OPX7ContextValidationLogLevel::eDisable;
				OPX7ContextValidationMode validationMode = OPX7ContextValidationMode::eOFF;
			};
			enum class OPX7CompileOptimizationLevel
			{
				eDefault = OPTIX_COMPILE_OPTIMIZATION_DEFAULT,
				eLevel0 = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0,
				eLevel1 = OPTIX_COMPILE_OPTIMIZATION_LEVEL_1,
				eLevel2 = OPTIX_COMPILE_OPTIMIZATION_LEVEL_2,
				eLevel3 = OPTIX_COMPILE_OPTIMIZATION_LEVEL_3,
			};
			enum class OPX7CompileDebugLevel
			{
				eDefault = OPTIX_COMPILE_DEBUG_LEVEL_DEFAULT,
				eNone = OPTIX_COMPILE_DEBUG_LEVEL_NONE,
				eMinimal = OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL,
				eModerate = OPTIX_COMPILE_DEBUG_LEVEL_MODERATE,
				eFull = OPTIX_COMPILE_DEBUG_LEVEL_FULL,
			};
			using OPX7ModuleCompileBoundValueEntry = OptixModuleCompileBoundValueEntry;
			using OPX7ModuleCompileBoundValueEntries = std::vector<OPX7ModuleCompileBoundValueEntry>;
			enum OptixPayloadSemantics
			{
				OptixPayloadSemanticsTraceCallerNone = OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_NONE,
				OptixPayloadSemanticsTraceCallerRead = OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ,
				OptixPayloadSemanticsTraceCallerWrite = OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_WRITE,
				OptixPayloadSemanticsTraceCallerReadWrite = OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ_WRITE,

				OptixPayloadSemanticsMSNone = OPTIX_PAYLOAD_SEMANTICS_MS_NONE,
				OptixPayloadSemanticsMSRead = OPTIX_PAYLOAD_SEMANTICS_MS_READ,
				OptixPayloadSemanticsMSWrite = OPTIX_PAYLOAD_SEMANTICS_MS_WRITE,
				OptixPayloadSemanticsMSReadWrite = OPTIX_PAYLOAD_SEMANTICS_MS_READ_WRITE,

				OptixPayloadSemanticsCHNone = OPTIX_PAYLOAD_SEMANTICS_CH_NONE,
				OptixPayloadSemanticsCHRead = OPTIX_PAYLOAD_SEMANTICS_CH_READ,
				OptixPayloadSemanticsCHWrite = OPTIX_PAYLOAD_SEMANTICS_CH_WRITE,
				OptixPayloadSemanticsCHReadWrite = OPTIX_PAYLOAD_SEMANTICS_CH_READ_WRITE,

				OptixPayloadSemanticsAHNone = OPTIX_PAYLOAD_SEMANTICS_AH_NONE,
				OptixPayloadSemanticsAHRead = OPTIX_PAYLOAD_SEMANTICS_AH_READ,
				OptixPayloadSemanticsAHWrite = OPTIX_PAYLOAD_SEMANTICS_AH_WRITE,
				OptixPayloadSemanticsAHReadWrite = OPTIX_PAYLOAD_SEMANTICS_AH_READ_WRITE,

				OptixPayloadSemanticsISNone = OPTIX_PAYLOAD_SEMANTICS_IS_NONE,
				OptixPayloadSemanticsISRead = OPTIX_PAYLOAD_SEMANTICS_IS_READ,
				OptixPayloadSemanticsISWrite = OPTIX_PAYLOAD_SEMANTICS_IS_WRITE,
				OptixPayloadSemanticsISReadWrite = OPTIX_PAYLOAD_SEMANTICS_IS_READ_WRITE,

			};
			using OPX7PayloadType = OptixPayloadType;
			using OPX7PayloadTypes = std::vector<OPX7PayloadType>;
			struct OPX7ModuleCompileOptions
			{
				int maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
				OPX7CompileOptimizationLevel optLevel = OPX7CompileOptimizationLevel::eDefault;
				OPX7CompileDebugLevel debugLevel = OPX7CompileDebugLevel::eDefault;
				OPX7ModuleCompileBoundValueEntries boundValueEntries = {};
				OPX7PayloadTypes payloadTypes = {};
			};
			enum OPX7PrimitiveTypeFlagBits
			{
				OPX7PrimitiveTypeFlagsCustom = OPTIX_PRIMITIVE_TYPE_FLAGS_CUSTOM,
				OPX7PrimitiveTypeFlagsRoundQuadraticBspline = OPTIX_PRIMITIVE_TYPE_FLAGS_ROUND_QUADRATIC_BSPLINE,
				OPX7PrimitiveTypeFlagsRoundCubicBspline = OPTIX_PRIMITIVE_TYPE_FLAGS_ROUND_CUBIC_BSPLINE,
				OPX7PrimitiveTypeFlagsRoundLinear = OPTIX_PRIMITIVE_TYPE_FLAGS_ROUND_LINEAR,
				OPX7PrimitiveTypeFlagsRoundCatmullRom = OPTIX_PRIMITIVE_TYPE_FLAGS_ROUND_CATMULLROM,
				OPX7PrimitiveTypeFlagsTriangle = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE,
			};
			using OPX7PrimitiveTypeFlags = unsigned int;
			enum OPX7TraversableGraphFlagBits
			{
				OPX7TraversableGraphFlagsAllowAny = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY,
				OPX7TraversableGraphFlagsAllowSingleGAS = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS,
				OPX7TraversableGraphFlagsAllowSingleLevelInstancing = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING,
			};
			using OPX7TraversableGraphFlags = unsigned int;
			enum OPX7ExceptionFlagBits
			{
				OPX7ExceptionFlagsNone = OPTIX_EXCEPTION_FLAG_NONE,
				OPX7ExceptionFlagsStackOverflow = OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW,
				OPX7ExceptionFlagsTraceDepth = OPTIX_EXCEPTION_FLAG_TRACE_DEPTH,
				OPX7ExceptionFlagsUser = OPTIX_EXCEPTION_FLAG_USER,
				OPX7ExceptionFlagsDebug = OPTIX_EXCEPTION_FLAG_DEBUG,
			};
			using OPX7ExceptionFlags = unsigned int;
			struct OPX7PipelineCompileOptions
			{
				bool usesMotionBlur = false;
				OPX7TraversableGraphFlags traversableGraphFlags = 0;
				int numPayloadValues = 0;
				int numAttributeValues = 0;
				const char *launchParamsVariableNames = nullptr;
				OPX7ExceptionFlags exceptionFlags = 0;
				OPX7PrimitiveTypeFlags usesPrimitiveTypeFlags = 0;
			};
			struct OPX7ModuleCreateDesc
			{
				OPX7ModuleCompileOptions moduleOptions = {};
				OPX7PipelineCompileOptions pipelineOptions = {};
				std::vector<char> ptxBinary = {};
			};
			enum class OPX7ModuleCompileState
			{
				eNotStarted = OPTIX_MODULE_COMPILE_STATE_NOT_STARTED,
				eStarted = OPTIX_MODULE_COMPILE_STATE_STARTED,
				eImpendingFailure = OPTIX_MODULE_COMPILE_STATE_IMPENDING_FAILURE,
				eFailed = OPTIX_MODULE_COMPILE_STATE_FAILED,
				eCompleted = OPTIX_MODULE_COMPILE_STATE_COMPLETED,
			};
			class OPX7Module;
			struct OPX7ProgramGroupSingleModule
			{

				OPX7Module *module = nullptr;
				const char *entryFunctionName = nullptr;
			};
			struct OPX7ProgramGroupCallables
			{
				OPX7ProgramGroupSingleModule continuation;
				OPX7ProgramGroupSingleModule direct;
			};
			struct OPX7ProgramGroupHitgroup
			{
				OPX7ProgramGroupSingleModule intersect;
				OPX7ProgramGroupSingleModule closesthit;
				OPX7ProgramGroupSingleModule anyhit;
			};
			enum class OPX7ProgramGroupKind
			{
				eRayGen = OPTIX_PROGRAM_GROUP_KIND_RAYGEN,
				eMiss = OPTIX_PROGRAM_GROUP_KIND_MISS,
				eHitgroup = OPTIX_PROGRAM_GROUP_KIND_HITGROUP,
				eCallables = OPTIX_PROGRAM_GROUP_KIND_CALLABLES,
				eException = OPTIX_PROGRAM_GROUP_KIND_EXCEPTION,
			};
			enum OPX7ProgramGroupFlagBits
			{
				OPX7ProgramGroupFlagsNone = OPTIX_PROGRAM_GROUP_FLAGS_NONE
			};
			using OPX7ProgramGroupFlags = unsigned int;
			struct OPX7ProgramGroupOptions
			{
				std::optional<OPX7PayloadType> payloadType = std::nullopt;
			};
			struct OPX7ProgramGroupCreateDesc
			{
				OPX7ProgramGroupCreateDesc() noexcept
				{
					kind = OPX7ProgramGroupKind::eRayGen;
					flags = OPX7ProgramGroupFlagsNone;
					raygen = {};
				}
				static auto Raygen(const OPX7ProgramGroupSingleModule& raygen, OPX7ProgramGroupFlags flags = OPX7ProgramGroupFlagBits::OPX7ProgramGroupFlagsNone)noexcept -> OPX7ProgramGroupCreateDesc {
					auto desc = OPX7ProgramGroupCreateDesc{};
					desc.kind = OPX7ProgramGroupKind::eRayGen;
					desc.raygen = raygen;
					desc.flags = flags;
					return desc;
				}
				static auto Miss(const OPX7ProgramGroupSingleModule& miss, OPX7ProgramGroupFlags flags = OPX7ProgramGroupFlagBits::OPX7ProgramGroupFlagsNone)noexcept -> OPX7ProgramGroupCreateDesc {
					auto desc = OPX7ProgramGroupCreateDesc{};
					desc.kind = OPX7ProgramGroupKind::eMiss;
					desc.miss = miss;
					desc.flags = flags;
					return desc;
				}
				static auto Exception(const OPX7ProgramGroupSingleModule& exception, OPX7ProgramGroupFlags flags = OPX7ProgramGroupFlagBits::OPX7ProgramGroupFlagsNone)noexcept -> OPX7ProgramGroupCreateDesc {
					auto desc = OPX7ProgramGroupCreateDesc{};
					desc.kind = OPX7ProgramGroupKind::eException;
					desc.exception = exception;
					desc.flags = flags;
					return desc;
				}
				static auto Callables(const OPX7ProgramGroupSingleModule& direct = {}, const OPX7ProgramGroupSingleModule& continuation = {},
					OPX7ProgramGroupFlags flags = OPX7ProgramGroupFlagBits::OPX7ProgramGroupFlagsNone)noexcept -> OPX7ProgramGroupCreateDesc {
					auto desc = OPX7ProgramGroupCreateDesc{};
					desc.kind = OPX7ProgramGroupKind::eCallables;
					desc.callables.continuation = continuation;
					desc.callables.direct = direct;
					desc.flags = flags;
					return desc;
				}
				static auto Hitgroup(const OPX7ProgramGroupSingleModule& closesthit = {}, const OPX7ProgramGroupSingleModule& anyhit = {}, const OPX7ProgramGroupSingleModule& intersect = {},
					OPX7ProgramGroupFlags flags = OPX7ProgramGroupFlagBits::OPX7ProgramGroupFlagsNone)noexcept -> OPX7ProgramGroupCreateDesc {
					auto desc = OPX7ProgramGroupCreateDesc{};
					desc.kind = OPX7ProgramGroupKind::eHitgroup;
					desc.hitgroup.closesthit = closesthit;
					desc.hitgroup.anyhit = anyhit;
					desc.hitgroup.intersect = intersect;
					desc.flags = flags;
					return desc;
				}
				OPX7ProgramGroupKind kind;
				OPX7ProgramGroupFlags flags;
				union
				{
					OPX7ProgramGroupSingleModule raygen;
					OPX7ProgramGroupSingleModule miss;
					OPX7ProgramGroupSingleModule exception;
					OPX7ProgramGroupCallables callables;
					OPX7ProgramGroupHitgroup hitgroup;
				};
			};
			struct OPX7PipelineLinkOptions
			{
				unsigned int maxTraceDepth = 0;
				OPX7CompileDebugLevel debugLevel = OPX7CompileDebugLevel::eDefault;
			};
			class OPX7ProgramGroup;
			struct OPX7PipelineCreateDesc
			{
				OPX7PipelineCompileOptions compileOptions = {};
				OPX7PipelineLinkOptions linkOptions = {};
				std::vector<OPX7ProgramGroup *> programGroups = {};
			};
			struct OPX7ShaderTableCreateDesc
			{
				/*RAYGEN*/
				unsigned int raygenRecordSizeInBytes = 0;
				/*EXCEPTION*/
				unsigned int exceptionRecordSizeInBytes = 0;
				/*MISS*/
				unsigned int missRecordStrideInBytes = 0;
				unsigned int missRecordCount = 0;
				/*HITGROUP*/
				unsigned int hitgroupRecordStrideInBytes = 0;
				unsigned int hitgroupRecordCount = 0;
				/*CALLABLES*/
				unsigned int callablesRecordStrideInBytes = 0;
				unsigned int callablesRecordCount = 0;
			};
			enum class OPX7VertexFormat : uint64_t
			{
				eNone = 0,
				eFloat32x3 = static_cast<uint64_t>(Core::SizedTypeFlagBits::eFloat32X3),
				eFloat32x2 = static_cast<uint64_t>(Core::SizedTypeFlagBits::eFloat32X2),
				eFloat16x3 = static_cast<uint64_t>(Core::SizedTypeFlagBits::eFloat16X3),
				eFloat16x2 = static_cast<uint64_t>(Core::SizedTypeFlagBits::eFloat16X2),
				eSnorm16x3 = static_cast<uint64_t>(Core::SizedTypeFlagBits::eInt16X3),
				eSnorm16x2 = static_cast<uint64_t>(Core::SizedTypeFlagBits::eInt16X2),
			};
			enum class OPX7TriIdxFormat : uint64_t
			{
				eNone = 0,
				eUInt16x3 = static_cast<uint64_t>(Core::SizedTypeFlagBits::eUInt16X3),
				eUInt32x3 = static_cast<uint64_t>(Core::SizedTypeFlagBits::eUInt32X3)
			};
			enum class OPX7SbtOffsetFormat : uint64_t
			{
				eNone = 0,
				eUInt8 = static_cast<uint64_t>(Core::SizedTypeFlagBits::eUInt8),
				eUInt16 = static_cast<uint64_t>(Core::SizedTypeFlagBits::eUInt16),
				eUInt32 = static_cast<uint64_t>(Core::SizedTypeFlagBits::eUInt32),
			};
			enum       OPX7GeometryFlagBits
			{
				OPX7GeometryFlagsNone = OPTIX_GEOMETRY_FLAG_NONE,
				OPX7GeometryFlagsDisableAnyHit = OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT,
				OPX7GeometryFlagsRequireSingleAnyHit = OPTIX_GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL,
			};
			using      OPX7GeometryFlags = unsigned int;
			enum class OPX7TransformFormat
			{
				eNone = OPTIX_TRANSFORM_FORMAT_NONE,
				eMatrixFloat32X12 = OPTIX_TRANSFORM_FORMAT_MATRIX_FLOAT12,
			};
			enum class OPX7BuildInputType
			{
				eTriangles        = OPTIX_BUILD_INPUT_TYPE_TRIANGLES,
				eCustomPrimitives = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES,
				eInstances        = OPTIX_BUILD_INPUT_TYPE_INSTANCES,
				eInstancePointers = OPTIX_BUILD_INPUT_TYPE_INSTANCE_POINTERS,
				eCurves           = OPTIX_BUILD_INPUT_TYPE_CURVES,
			};

		}
	}
}
#endif
