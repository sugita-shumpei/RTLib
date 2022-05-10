#include "ImplGLFramebuffer.h"
namespace RTLib
{
	namespace Ext
	{
		namespace GL
		{
			namespace Internal
			{
				class ImplGLFramebufferBase : public ImplGLBindableBase
				{
				public:
					friend class ImplGLBindable;

				public:
					virtual ~ImplGLFramebufferBase() noexcept {}

				protected:
					virtual bool Create() noexcept override
					{
						GLuint resId;
						glGenFramebuffers(1, &resId);
						if (resId == 0)
						{
							return false;
						}
						SetResId(resId);
						return true;
					}
					virtual void Destroy() noexcept
					{
						GLuint resId = GetResId();
						glDeleteFramebuffers(1, &resId);
						SetResId(0);
					}
					virtual void Bind(GLenum target)
					{
						GLuint resId = GetResId();
						if (resId > 0)
						{
							glBindFramebuffer(target, resId);
						}
					}
					virtual void Unbind(GLenum target)
					{
#ifndef NDEBUG
						glBindFramebuffer(target, 0);
#endif
					}
				};
			}
		}

	}

}

auto RTLib::Ext::GL::Internal::ImplGLFramebuffer::New(ImplGLResourceTable* table, ImplGLBindingPoint* bPoint) -> ImplGLFramebuffer* {
	if (!table || !bPoint) {
		return nullptr;
	}
	auto buffer = new ImplGLFramebuffer(table, bPoint);
	if (buffer) {
		buffer->InitBase<ImplGLFramebufferBase>();
		auto res = buffer->Create();
		if (!res) {
			delete buffer;
			return nullptr;
		}
	}
	return buffer;
}

RTLib::Ext::GL::Internal::ImplGLFramebuffer::~ImplGLFramebuffer() noexcept {}

bool RTLib::Ext::GL::Internal::ImplGLFramebuffer::Bind(GLenum target) {
	return ImplGLBindable::Bind(target);
}

bool RTLib::Ext::GL::Internal::ImplGLFramebuffer::AttachColorTexture(GLuint idx, ImplGLTexture *texture, GLint level, GLint layer)
{
	if (!IsBinded() || !texture)
	{
		return false;
	}
	if (!texture->IsAllocated())
	{
		return false;
	}
	switch (texture->GetTxTarget())
	{
	case GL_TEXTURE_1D:
		if (layer != 0)
		{
			return false;
		}
		glFramebufferTexture1D(*GetBindedTarget(), GL_COLOR_ATTACHMENT0 + idx, GL_TEXTURE_1D, texture->GetResId(), level);
		return true;
	case GL_TEXTURE_2D:
		if (layer != 0)
		{
			return false;
		}
		glFramebufferTexture2D(*GetBindedTarget(), GL_COLOR_ATTACHMENT0 + idx, GL_TEXTURE_2D, texture->GetResId(), level);
		return true;
	case GL_TEXTURE_RECTANGLE:
		if (layer != 0 || level != 0)
		{
			return false;
		}
		glFramebufferTexture2D(*GetBindedTarget(), GL_COLOR_ATTACHMENT0 + idx, GL_TEXTURE_3D, texture->GetResId(), level);
		return true;
	case GL_TEXTURE_2D_MULTISAMPLE:
		if (layer != 0 || level != 0)
		{
			return false;
		}
		glFramebufferTexture2D(*GetBindedTarget(), GL_COLOR_ATTACHMENT0 + idx, GL_TEXTURE_3D, texture->GetResId(), level);
		return true;
	case GL_TEXTURE_3D:
		glFramebufferTexture3D(*GetBindedTarget(), GL_COLOR_ATTACHMENT0 + idx, GL_TEXTURE_3D, texture->GetResId(), level, layer);
		return true;
	default:
		return false;
	}
	return false;
}

bool RTLib::Ext::GL::Internal::ImplGLFramebuffer::AttachDepthTexture(ImplGLTexture *texture, GLint level, GLint layer)
{
	if (!IsBinded() || !texture)
	{
		return false;
	}
	if (!texture->IsAllocated())
	{
		return false;
	}
	switch (texture->GetTxTarget())
	{
	case GL_TEXTURE_1D:
		if (layer != 0)
		{
			return false;
		}
		glFramebufferTexture1D(*GetBindedTarget(), GL_DEPTH_ATTACHMENT, GL_TEXTURE_1D, texture->GetResId(), level);
		return true;
	case GL_TEXTURE_2D:
		if (layer != 0)
		{
			return false;
		}
		glFramebufferTexture2D(*GetBindedTarget(), GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, texture->GetResId(), level);
		return true;
	case GL_TEXTURE_RECTANGLE:
		if (layer != 0 || level != 0)
		{
			return false;
		}
		glFramebufferTexture2D(*GetBindedTarget(), GL_DEPTH_ATTACHMENT, GL_TEXTURE_3D, texture->GetResId(), level);
		return true;
	case GL_TEXTURE_2D_MULTISAMPLE:
		if (layer != 0 || level != 0)
		{
			return false;
		}
		glFramebufferTexture2D(*GetBindedTarget(), GL_DEPTH_ATTACHMENT, GL_TEXTURE_3D, texture->GetResId(), level);
		return true;
	case GL_TEXTURE_3D:
		glFramebufferTexture3D(*GetBindedTarget(), GL_DEPTH_ATTACHMENT, GL_TEXTURE_3D, texture->GetResId(), level, layer);
		return true;
	default:
		return false;
	}
	return false;
}

bool RTLib::Ext::GL::Internal::ImplGLFramebuffer::AttachStencilTexture(ImplGLTexture *texture, GLint level, GLint layer)
{
	if (!IsBinded() || !texture)
	{
		return false;
	}
	if (!texture->IsAllocated())
	{
		return false;
	}
	switch (texture->GetTxTarget())
	{
	case GL_TEXTURE_1D:
		if (layer != 0)
		{
			return false;
		}
		glFramebufferTexture1D(*GetBindedTarget(), GL_STENCIL_ATTACHMENT, GL_TEXTURE_1D, texture->GetResId(), level);
		return true;
	case GL_TEXTURE_2D:
		if (layer != 0)
		{
			return false;
		}
		glFramebufferTexture2D(*GetBindedTarget(), GL_STENCIL_ATTACHMENT, GL_TEXTURE_2D, texture->GetResId(), level);
		return true;
	case GL_TEXTURE_RECTANGLE:
		if (layer != 0 || level != 0)
		{
			return false;
		}
		glFramebufferTexture2D(*GetBindedTarget(), GL_STENCIL_ATTACHMENT, GL_TEXTURE_3D, texture->GetResId(), level);
		return true;
	case GL_TEXTURE_2D_MULTISAMPLE:
		if (layer != 0 || level != 0)
		{
			return false;
		}
		glFramebufferTexture2D(*GetBindedTarget(), GL_STENCIL_ATTACHMENT, GL_TEXTURE_3D, texture->GetResId(), level);
		return true;
	case GL_TEXTURE_3D:
		glFramebufferTexture3D(*GetBindedTarget(), GL_STENCIL_ATTACHMENT, GL_TEXTURE_3D, texture->GetResId(), level, layer);
		return true;
	default:
		return false;
	}
	return false;
}

bool RTLib::Ext::GL::Internal::ImplGLFramebuffer::AttachDepthStencilTexture(ImplGLTexture *texture, GLint level, GLint layer)
{
	if (!IsBinded() || !texture)
	{
		return false;
	}
	if (!texture->IsAllocated())
	{
		return false;
	}
	switch (texture->GetTxTarget())
	{
	case GL_TEXTURE_1D:
		if (layer != 0)
		{
			return false;
		}
		glFramebufferTexture1D(*GetBindedTarget(), GL_DEPTH_STENCIL_ATTACHMENT, GL_TEXTURE_1D, texture->GetResId(), level);
		return true;
	case GL_TEXTURE_2D:
		if (layer != 0)
		{
			return false;
		}
		glFramebufferTexture2D(*GetBindedTarget(), GL_DEPTH_STENCIL_ATTACHMENT, GL_TEXTURE_2D, texture->GetResId(), level);
		return true;
	case GL_TEXTURE_RECTANGLE:
		if (layer != 0 || level != 0)
		{
			return false;
		}
		glFramebufferTexture2D(*GetBindedTarget(), GL_DEPTH_STENCIL_ATTACHMENT, GL_TEXTURE_3D, texture->GetResId(), level);
		return true;
	case GL_TEXTURE_2D_MULTISAMPLE:
		if (layer != 0 || level != 0)
		{
			return false;
		}
		glFramebufferTexture2D(*GetBindedTarget(), GL_DEPTH_STENCIL_ATTACHMENT, GL_TEXTURE_3D, texture->GetResId(), level);
		return true;
	case GL_TEXTURE_3D:
		glFramebufferTexture3D(*GetBindedTarget(), GL_DEPTH_STENCIL_ATTACHMENT, GL_TEXTURE_3D, texture->GetResId(), level, layer);
		return true;
	default:
		return false;
	}
	return false;
}

bool RTLib::Ext::GL::Internal::ImplGLFramebuffer::AttachColorRenderbuffer(GLuint idx, GLint level)
{
	return false;
}

bool RTLib::Ext::GL::Internal::ImplGLFramebuffer::AttachDepthRenderbuffer(GLint level)
{
	return false;
}

bool RTLib::Ext::GL::Internal::ImplGLFramebuffer::AttachStencilRenderbuffer(GLint level)
{
	return false;
}

bool RTLib::Ext::GL::Internal::ImplGLFramebuffer::AttachDepthStencilRenderbuffer(GLint level)
{
	return false;
}

bool RTLib::Ext::GL::Internal::ImplGLFramebuffer::IsCompleted() const noexcept
{
	auto bPoint = GetBindingPoint();
	if (!bPoint || !IsBinded())
	{
		return false;
	}
	return glCheckFramebufferStatus(*GetBindedTarget()) == GL_FRAMEBUFFER_COMPLETE;
}

RTLib::Ext::GL::Internal::ImplGLFramebuffer::ImplGLFramebuffer(ImplGLResourceTable* table, ImplGLBindingPoint* bPoint) noexcept :ImplGLBindable(table, bPoint) {}
