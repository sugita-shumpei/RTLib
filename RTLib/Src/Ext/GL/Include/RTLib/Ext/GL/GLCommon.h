#ifndef RTLIB_EXT_GL_GL_COMMON_H
#define RTLIB_EXT_GL_GL_COMMON_H
#include <RTLib/Core/Common.h>
#include <RTLib/Ext/GL/GLTypeFormat.h>
#include <half.h>
#include <glad/glad.h>
#include <cstdint>

namespace RTLib
{
	namespace Ext
	{
		namespace GL
		{
			using GLExtent2D = Core::Extent2D;
			using GLExtent3D = Core::Extent3D;

			using GLOffset2D = Core::Offset2D;
			using GLOffset3D = Core::Offset3D;

			using GLImageSubresourceLayers = Core::ImageSubresourceLayers;
			using GLBufferCopy       = Core::BufferCopy;
			using GLBufferImageCopy  = Core::BufferImageCopy;
			using GLMemoryBufferCopy = Core::MemoryBufferCopy;
			using GLBufferMemoryCopy = Core::BufferMemoryCopy;
			using GLImageMemoryCopy  = Core::ImageMemoryCopy;
			enum  GLBufferUsageFlagBits : unsigned int
			{
				GLBufferUsageUnknown           = 0,
				GLBufferUsageVertex            = 1 << 0,
				GLBufferUsageAtomicCounter     = 1 << 1,
				GLBufferUsageDispatchIndirect  = 1 << 2,
				GLBufferUsageDrawIndirect      = 1 << 3,
				GLBufferUsageIndex             = 1 << 4,
				GLBufferUsageQuery             = 1 << 5,
				GLBufferUsageStorage           = 1 << 6,
				GLBufferUsageTexture           = 1 << 7,
				GLBufferUsageTransformFeedBack = 1 << 8,
				GLBufferUsageUniform           = 1 << 9,
				GLBufferUsageImageCopySrc	   = 1 << 10,
				GLBufferUsageImageCopyDst      = 1 << 11,
				GLBufferUsageGenericCopySrc    = 1 << 12,
				GLBufferUsageGenericCopyDst    = 1 << 13,
			}; 
			using GLBufferUsageFlags = unsigned int;
			inline constexpr auto GetGLBufferUsageCount(GLBufferUsageFlags  usageFlags)->unsigned int {
				unsigned int count = 0;
				if (usageFlags == GLBufferUsageGenericCopySrc) { return 1; }
				if (usageFlags == GLBufferUsageGenericCopyDst) { return 1; }
				if ((usageFlags & GLBufferUsageVertex)            == GLBufferUsageVertex)            { ++count; }
				if ((usageFlags & GLBufferUsageIndex)             == GLBufferUsageIndex)             { ++count; }
				if ((usageFlags & GLBufferUsageUniform)           == GLBufferUsageUniform)           { ++count; }
				if ((usageFlags & GLBufferUsageStorage)           == GLBufferUsageStorage)           { ++count; }
				if ((usageFlags & GLBufferUsageImageCopySrc)      == GLBufferUsageImageCopySrc)      { ++count; }
				if ((usageFlags & GLBufferUsageImageCopyDst)      == GLBufferUsageImageCopyDst)      { ++count; }
				if ((usageFlags & GLBufferUsageDrawIndirect)      == GLBufferUsageDrawIndirect)      { ++count; }
				if ((usageFlags & GLBufferUsageDispatchIndirect)  == GLBufferUsageDispatchIndirect)  { ++count; }
				if ((usageFlags & GLBufferUsageTransformFeedBack) == GLBufferUsageTransformFeedBack) { ++count; }
				if ((usageFlags & GLBufferUsageTexture)           == GLBufferUsageTexture)           { ++count; }
				if ((usageFlags & GLBufferUsageQuery)             == GLBufferUsageQuery)             { ++count; }
				if ((usageFlags & GLBufferUsageAtomicCounter)     == GLBufferUsageAtomicCounter)     { ++count; }
				return count;
			}
			inline constexpr auto GetGLBufferMainUsage (GLBufferUsageFlags  usageFlags)->GLBufferUsageFlagBits
			{
				if ((usageFlags & GLBufferUsageVertex)) {
					return GLBufferUsageVertex;
				}
				if ((usageFlags & GLBufferUsageIndex)) {
					return GLBufferUsageIndex;
				}
				if ((usageFlags & GLBufferUsageUniform)) {
					return GLBufferUsageUniform;
				}
				if ((usageFlags & GLBufferUsageStorage)) {
					return GLBufferUsageStorage;
				}
				if ((usageFlags & GLBufferUsageDrawIndirect)) {
					return GLBufferUsageDrawIndirect;
				}
				if ((usageFlags & GLBufferUsageDispatchIndirect)) {
					return GLBufferUsageDispatchIndirect;
				}
				if ((usageFlags & GLBufferUsageTransformFeedBack)) {
					return GLBufferUsageTransformFeedBack;
				}
				if ((usageFlags & GLBufferUsageTexture)) {
					return GLBufferUsageTexture;
				}
				if ((usageFlags & GLBufferUsageQuery)) {
					return GLBufferUsageQuery;
				}
				if ((usageFlags & GLBufferUsageAtomicCounter)) {
					return GLBufferUsageAtomicCounter;
				}
				if ((usageFlags & GLBufferUsageImageCopySrc)) {
					return GLBufferUsageImageCopySrc;
				}
				if ((usageFlags & GLBufferUsageImageCopyDst)) {
					return GLBufferUsageImageCopyDst;
				}
				if ((usageFlags & GLBufferUsageGenericCopySrc)){
					return GLBufferUsageGenericCopySrc;
				}
				if ((usageFlags & GLBufferUsageGenericCopyDst)) {
					return GLBufferUsageGenericCopyDst;
				}
				return GLBufferUsageUnknown;
			}
			enum  GLShaderStageFlagBits {
				GLShaderStageVertex        = Core::ShaderStageVertex,
				GLShaderStageGeometry      = Core::ShaderStageGeometry,
				GLShaderStageTessControl     = Core::ShaderStageTessControl,
				GLShaderStageTessEvaluation  = Core::ShaderStageTessEvalauation,
				GLShaderStageFragment        = Core::ShaderStageFragment,
				GLShaderStageCompute         = Core::ShaderStageCompute,
			};
			using GLShaderStageFlags = unsigned int;
			//GL_(*)_DRAW	   (COPY: Cpu -> Gpu, Use: Cpu)   Upload Buffer
			//GL_(*)_READ      (COPY: Gpu -> Cpu, Use: Cpu) Readback Buffer
			//GL_(*)_COPY      (COPY: Gpu -> Gpu, Use: Gpu)
			//GL_DYNAMIC_STORAGE_BIT: (Enable BufferSubData)
			//GL_MAP_READ_BIT  (Map:  ReadOnly(Cpu->Gpu))
			//GL_MAP_WRITE_BIT (Map: WriteOnly(Gpu->Cpu))
			//GL_MAP_PERSISTENT_BIT: MAP中でもGPU側で読み出し、書き込み処理を実行可能, CPU側のポインターはMAP中常に有効
			//GL_MAP_COHERENT_BIT  : MAP結果が即座に反映、次のGLコマンド以降に反映される.
			//GL_CLIENT_STORAGE_BIT:  (CPU PAGED なメモリに確保)
			//
			//VK_DEVICE_LOCAL
			//VK_HOST_VISIBLE =GL_CLIENT_STORAGE_BIT|GL_MAP_READ_BIT|GL_MAP_WRITE_BIT
			//VK_HOST_COHERENT=GL_MAP_COHERENT_BIT  
			enum  GLMemoryPropertyFlagBits :unsigned int
			{
				GLMemoryPropertyDefault     = 0,
				GLMemoryPropertyHostWrite   = 1<<0,
				GLMemoryPropertyHostRead    = 1<<1,
				GLMemoryPropertyHostCoherent= 1<<2,
				GLMemoryPropertyHostPaged   = 1<<3,
			};
			using GLMemoryPropertyFlags = unsigned int;

			struct GLBufferCreateDesc
			{
				size_t                size    = 0;
				GLBufferUsageFlags    usage   = GLBufferUsageGenericCopySrc;
				GLMemoryPropertyFlags access  = GLMemoryPropertyHostPaged;
				const void*           pData   = nullptr;
			};

			enum class GLImageType
			{
				e1D,
				e2D,
				e3D
			};
			enum class GLDrawMode
			{
				ePoints,
				eLineStrip,
				eLineLoop,
				eLines,
				eLineStripAdjacency,
				eLinesAdjacency,
				eTriangleStrip,
				eTriangleFan,
				eTriangles,
				eTriangleStripAdjacency,
				eTrianglesAdjacency
			};
			struct GLTextureCreateDesc
			{
				GLImageType imageType   = GLImageType::e1D;
				GLFormat    format      = GLFormat::eRGBA8;
				GLExtent3D  extent      = GLExtent3D();
				uint32_t    mipLevels   = 1;
				uint32_t    arrayLayers = 1;
			};
			enum  GLClearBufferFlagBits
			{
				GLClearBufferFlagsColor   = 1,
				GLClearBufferFlagsDepth   = 2,
				GLClearBufferFlagsStencil = 4,
			};
			using GLClearBufferFlags = unsigned int;
		}
	}
}
#endif
