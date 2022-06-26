#include <RTLib/Ext/OPX7/OPX7AccelerationStructure.h>
#include <RTLib/Ext/CUDA/CUDANatives.h>
auto RTLib::Ext::OPX7::OPX7GeometryTriangle::GetOptixBuildInputWithHolder() const noexcept -> std::tuple<OptixBuildInput, std::unique_ptr<Holder>> 
{
	struct OPX7GeometryTriangleHolder:public Holder
	{
		virtual ~OPX7GeometryTriangleHolder()noexcept {}
		CUdeviceptr    vertexBufferGpuAddress = 0;
		std::vector<OPX7::OPX7GeometryFlags> flags = {};
	};

	auto holder = new OPX7GeometryTriangleHolder();
	holder->vertexBufferGpuAddress = CUDA::CUDANatives::GetCUdeviceptr(GetVertexView().view);
	holder->flags = m_GeometryFlags;

	OptixBuildInput buildInput;
	buildInput.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

	buildInput.triangleArray.vertexBuffers = &holder->vertexBufferGpuAddress;
	buildInput.triangleArray.vertexStrideInBytes = GetVertexView().stride;
	buildInput.triangleArray.numVertices = GetVertexView().GetNumVertices();
	auto vbFormat = GetVertexView().format;
	switch (vbFormat)
	{
	case RTLib::Ext::OPX7::OPX7VertexFormat::eNone:
		buildInput.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_NONE;
		break;
	case RTLib::Ext::OPX7::OPX7VertexFormat::eFloat32x3:
		buildInput.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
		break;
	case RTLib::Ext::OPX7::OPX7VertexFormat::eFloat32x2:
		buildInput.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT2;
		break;
	case RTLib::Ext::OPX7::OPX7VertexFormat::eFloat16x3:
		buildInput.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_HALF3;
		break;
	case RTLib::Ext::OPX7::OPX7VertexFormat::eFloat16x2:
		buildInput.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_HALF2;
		break;
	case RTLib::Ext::OPX7::OPX7VertexFormat::eSnorm16x3:
		buildInput.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_SNORM16_3;
		break;
	case RTLib::Ext::OPX7::OPX7VertexFormat::eSnorm16x2:
		buildInput.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_SNORM16_2;
		break;
	default:
		break;
	}

	buildInput.triangleArray.indexBuffer = CUDA::CUDANatives::GetCUdeviceptr(GetTriIdxView().view);
	buildInput.triangleArray.indexStrideInBytes = GetTriIdxView().stride;
	buildInput.triangleArray.numIndexTriplets = GetTriIdxView().GetNumIndices();
	auto ibFormat = GetTriIdxView().format;
	switch (ibFormat)
	{
	case RTLib::Ext::OPX7::OPX7TriIdxFormat::eNone:
		buildInput.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_NONE;
		break;
	case RTLib::Ext::OPX7::OPX7TriIdxFormat::eUInt16x3:
		buildInput.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_SHORT3;
		break;
	case RTLib::Ext::OPX7::OPX7TriIdxFormat::eUInt32x3:
		buildInput.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
		break;
	default:
		break;
	}

	buildInput.triangleArray.sbtIndexOffsetBuffer = CUDA::CUDANatives::GetCUdeviceptr(GetSbtOffsetView().view);
	buildInput.triangleArray.sbtIndexOffsetStrideInBytes = GetSbtOffsetView().stride;
	buildInput.triangleArray.numSbtRecords = GetSbtOffsetView().numRecords;
	auto sbFormat = GetSbtOffsetView().format;
	switch (sbFormat)
	{
	case RTLib::Ext::OPX7::OPX7SbtOffsetFormat::eNone:
		buildInput.triangleArray.sbtIndexOffsetSizeInBytes = 0;
		break;
	case RTLib::Ext::OPX7::OPX7SbtOffsetFormat::eUInt8:
		buildInput.triangleArray.sbtIndexOffsetSizeInBytes = 1;
		break;
	case RTLib::Ext::OPX7::OPX7SbtOffsetFormat::eUInt16:
		buildInput.triangleArray.sbtIndexOffsetSizeInBytes = 2;
		break;
	case RTLib::Ext::OPX7::OPX7SbtOffsetFormat::eUInt32:
		buildInput.triangleArray.sbtIndexOffsetSizeInBytes = 4;
		break;
	default:
		break;
	}

	buildInput.triangleArray.preTransform = CUDA::CUDANatives::GetCUdeviceptr(GetTransformView().view);
	auto tbFormat = GetTransformView().format;
	switch (tbFormat)
	{
	case RTLib::Ext::OPX7::OPX7TransformFormat::eNone:
		buildInput.triangleArray.transformFormat = OPTIX_TRANSFORM_FORMAT_NONE;
		break;
	case RTLib::Ext::OPX7::OPX7TransformFormat::eMatrixFloat32X12:
		buildInput.triangleArray.transformFormat = OPTIX_TRANSFORM_FORMAT_MATRIX_FLOAT12;
		break;
	default:
		break;
	}

	buildInput.triangleArray.primitiveIndexOffset = GetPrimIdxOffset();
	buildInput.triangleArray.flags = holder->flags.data();

	return { buildInput,std::unique_ptr<Holder>(holder) };
}

auto RTLib::Ext::OPX7::OPX7GeometrySphere  ::GetOptixBuildInputWithHolder() const noexcept -> std::tuple<OptixBuildInput, std::unique_ptr<Holder>> 
{
	struct OPX7GeometrySphereHolder :public Holder
	{
		virtual ~OPX7GeometrySphereHolder()noexcept {}
		CUdeviceptr    vertexBufferGpuAddress = 0;
		CUdeviceptr    radiusBufferGpuAddress = 0;
		std::vector<OPX7::OPX7GeometryFlags> flags = {};
	};
	auto holder = new OPX7GeometrySphereHolder();
	holder->vertexBufferGpuAddress = CUDA::CUDANatives::GetCUdeviceptr(GetVertexView().view);
	holder->radiusBufferGpuAddress = CUDA::CUDANatives::GetCUdeviceptr(GetRadiusView().view);
	holder->flags = m_GeometryFlags;

	OptixBuildInput buildInput;
	buildInput.type = OPTIX_BUILD_INPUT_TYPE_SPHERES;

	buildInput.sphereArray.vertexBuffers = &holder->vertexBufferGpuAddress;
	buildInput.sphereArray.vertexStrideInBytes = GetVertexView().stride;
	buildInput.sphereArray.numVertices = GetVertexView().GetNumVertices();

	buildInput.sphereArray.radiusBuffers = &holder->radiusBufferGpuAddress;
	buildInput.sphereArray.radiusStrideInBytes = GetRadiusView().stride;
	buildInput.sphereArray.singleRadius = GetRadiusView().singleRadius;

	buildInput.sphereArray.sbtIndexOffsetBuffer      = CUDA::CUDANatives::GetCUdeviceptr(GetSbtOffsetView().view);
	buildInput.sphereArray.sbtIndexOffsetStrideInBytes = GetSbtOffsetView().stride;
	auto sbFormat = GetSbtOffsetView().format;
	switch (sbFormat)
	{
	case RTLib::Ext::OPX7::OPX7SbtOffsetFormat::eNone:
		buildInput.sphereArray.sbtIndexOffsetSizeInBytes = 0;
		break;
	case RTLib::Ext::OPX7::OPX7SbtOffsetFormat::eUInt8:
		buildInput.sphereArray.sbtIndexOffsetSizeInBytes = 1;
		break;
	case RTLib::Ext::OPX7::OPX7SbtOffsetFormat::eUInt16:
		buildInput.sphereArray.sbtIndexOffsetSizeInBytes = 2;
		break;
	case RTLib::Ext::OPX7::OPX7SbtOffsetFormat::eUInt32:
		buildInput.sphereArray.sbtIndexOffsetSizeInBytes = 4;
		break;
	default:
		break;
	}
	buildInput.sphereArray.numSbtRecords        = GetSbtOffsetView().numRecords;

	buildInput.sphereArray.primitiveIndexOffset = GetPrimIdxOffset();
	buildInput.sphereArray.flags                = holder->flags.data();

	return { buildInput,std::unique_ptr<Holder>(holder) };
}

auto RTLib::Ext::OPX7::OPX7GeometryCustom  ::GetOptixBuildInputWithHolder() const noexcept -> std::tuple<OptixBuildInput, std::unique_ptr<Holder>> 
{
	struct OPX7GeometryCustomHolder :public Holder
	{
		virtual ~OPX7GeometryCustomHolder()noexcept {}
		CUdeviceptr      aabbBufferGpuAddress = 0;
		std::vector<OPX7::OPX7GeometryFlags> flags = {};
	};
	auto holder = new OPX7GeometryCustomHolder();
	holder->aabbBufferGpuAddress = CUDA::CUDANatives::GetCUdeviceptr(GetAabbView().view);
	holder->flags = m_GeometryFlags;

	OptixBuildInput buildInput;
	buildInput.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
	buildInput.customPrimitiveArray.aabbBuffers   = &holder->aabbBufferGpuAddress;
	buildInput.customPrimitiveArray.numPrimitives = GetAabbView().GetNumPrimitives();
	buildInput.customPrimitiveArray.strideInBytes = GetAabbView().stride;

	buildInput.customPrimitiveArray.sbtIndexOffsetBuffer = CUDA::CUDANatives::GetCUdeviceptr(GetSbtOffsetView().view);
	buildInput.customPrimitiveArray.sbtIndexOffsetStrideInBytes = GetSbtOffsetView().stride;
	auto sbFormat = GetSbtOffsetView().format;
	switch (sbFormat)
	{
	case RTLib::Ext::OPX7::OPX7SbtOffsetFormat::eNone:
		buildInput.customPrimitiveArray.sbtIndexOffsetSizeInBytes = 0;
		break;
	case RTLib::Ext::OPX7::OPX7SbtOffsetFormat::eUInt8:
		buildInput.customPrimitiveArray.sbtIndexOffsetSizeInBytes = 1;
		break;
	case RTLib::Ext::OPX7::OPX7SbtOffsetFormat::eUInt16:
		buildInput.customPrimitiveArray.sbtIndexOffsetSizeInBytes = 2;
		break;
	case RTLib::Ext::OPX7::OPX7SbtOffsetFormat::eUInt32:
		buildInput.customPrimitiveArray.sbtIndexOffsetSizeInBytes = 4;
		break;
	default:
		break;
	}
	buildInput.customPrimitiveArray.numSbtRecords = GetSbtOffsetView().numRecords;

	buildInput.customPrimitiveArray.primitiveIndexOffset = GetPrimIdxOffset();
	buildInput.customPrimitiveArray.flags = holder->flags.data();

	return { buildInput,std::unique_ptr<Holder>(holder) };
}

void RTLib::Ext::OPX7::OPX7GeometryAccelerationStructure::Build()
{
}

void RTLib::Ext::OPX7::OPX7GeometryAccelerationStructure::Update()
{
}
