#ifndef RTLIB_EXT_GL_INTERNALS_GL_BUFFER_BINDING_POINT_H
#define RTLIB_EXT_GL_INTERNALS_GL_BUFFER_BINDING_POINT_H
#include <RTLib/Ext/GL/GLContext.h>
#include <GLBindingPointCommon.h>
#include <unordered_map>
#include <utility>
#include <optional>
namespace RTLib
{
	namespace Ext
	{
		namespace GL
		{
			namespace Internals
			{
				struct GLBufferBindingRange
				{
					GLsizei   size = 0;
					GLintptr  offset = 0;
				};

				class  GLBufferBindingPoint;

				class  GLBufferBindable
				{
				private:
					struct BindingRanges
					{
						size_t                                         capacity = 0;
						std::unordered_map<GLuint, GLBufferBindingRange> data     = {};
					};
					using OpBindingTarget = std::optional<GLenum>;
					using OpBindingRanges = std::unordered_map<GLenum, BindingRanges>;
					friend class BufferBindngPoint;
				public:
					~GLBufferBindable()noexcept;
				private:
					GLBufferBindable(GLuint resId, GLBufferBindingPoint* bufferBP);

					auto GetBufferBP()const noexcept      -> GLBufferBindingPoint*;
					auto GetResId()const noexcept         -> GLuint;
					auto GetBindingTarget()const noexcept -> OpBindingTarget;
					auto GetBindingRanges()const noexcept -> const OpBindingRanges&;
				private:		
					GLBufferBindingPoint*   m_BufferBP = nullptr;
					GLuint                m_ResId    = 0;
					OpBindingTarget       m_Target   = std::nullopt;
					OpBindingRanges       m_Ranges   = {};
				};
				struct GLBufferTargetBindingHandle
				{
					GLBufferBindable* bindable = nullptr;
					GLBindingState     state = GLBindingState::eUnbinded;
				};
				struct GLBufferRangeBindingHandle
				{
					GLBufferBindable* bindable = nullptr;
					GLBindingState     state = GLBindingState::eUnbinded;
					GLBufferBindingRange range = {};
				};

				class  GLBufferBindingPoint
				{
				private:
					friend class RTLib::Ext::GL::GLContext;
					using BufferRangeBindingHandles = std::vector<GLBufferRangeBindingHandle>;
				public:
					 GLBufferBindingPoint()noexcept{}
					~GLBufferBindingPoint()noexcept{}

					bool    TakeTargetBinding(GLenum target, GLBufferBindable* bindable);
					void ReleaseTargetBinding(GLenum target);
					bool   ResetTargetBinding(GLenum target);

					bool    TakeRangeBinding(GLenum target, GLuint index, GLBufferBindable* bindable, const GLBufferBindingRange& range = {});
					void ReleaseRangeBinding(GLenum target, GLuint index);
					bool   ResetRangeBinding(GLenum target, GLuint index);

					bool IsBindedTarget(GLenum target)const noexcept;
					bool IsBindedIndex( GLenum target, GLuint index);
					bool IsBindedRange (GLenum target, GLuint index, const GLBufferBindingRange& range = {});

					bool IsValidTarget( GLenum target)const noexcept;
					bool IsValidRangeTarget(GLenum target)const noexcept;

					auto EnumerateValidTargets()const noexcept      ->  std::vector<GLenum>;
					auto EnumerateValidTargetRanges()const noexcept ->  std::vector<std::pair<GLenum,size_t>>;
					auto EnumerateRangeCapacity(GLenum target)const noexcept -> size_t;
				private:
					void AddValidTarget(GLenum target)noexcept;
					void AddValidTargetRange(GLenum target, size_t maxRangeCount)noexcept;
				private:
					std::unordered_map<GLenum, GLBufferTargetBindingHandle> m_TargetHandles = {};
					std::unordered_map<GLenum, BufferRangeBindingHandles> m_RangesHandles = {};
				};
			}
			
		}
	}
}
#endif
