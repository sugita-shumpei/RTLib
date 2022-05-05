#include "ImplGLFramebuffer.h"

bool RTLib::Ext::GL::Internal::ImplGLFramebuffer::AttachColorTexture(GLuint idx, ImplGLTexture* texture, GLint level, GLint layer)
{
	if (!IsBinded()|| !texture) {
		return false;
	}
	if (!texture->IsAllocated()) {
		return false;
	}
	switch (texture->GetTxTarget()) {
	case GL_TEXTURE_1D:
		if (layer != 0) {
			return false;
		}
		glFramebufferTexture1D(*GetTarget(), GL_COLOR_ATTACHMENT0 + idx, GL_TEXTURE_1D, texture->GetResId(), level);
		return true;
	case GL_TEXTURE_2D:
		if (layer != 0) {
			return false;
		}
		glFramebufferTexture2D(*GetTarget(), GL_COLOR_ATTACHMENT0 + idx, GL_TEXTURE_2D, texture->GetResId(), level);
		return true;
	case GL_TEXTURE_RECTANGLE:
		if (layer != 0 || level != 0) {
			return false;
		}
		glFramebufferTexture2D(*GetTarget(), GL_COLOR_ATTACHMENT0 + idx, GL_TEXTURE_3D, texture->GetResId(), level);
		return true;
	case GL_TEXTURE_2D_MULTISAMPLE:
		if (layer != 0 || level != 0) {
			return false;
		}
		glFramebufferTexture2D(*GetTarget(), GL_COLOR_ATTACHMENT0 + idx, GL_TEXTURE_3D, texture->GetResId(), level);
		return true;
	case GL_TEXTURE_3D:
		glFramebufferTexture3D(*GetTarget(), GL_COLOR_ATTACHMENT0 + idx, GL_TEXTURE_3D, texture->GetResId(), level, layer);
		return true;
	default:
		return false;
	}
	return false;
}

bool RTLib::Ext::GL::Internal::ImplGLFramebuffer::AttachDepthTexture(ImplGLTexture* texture, GLint level, GLint layer)
{
	if (!IsBinded() || !texture) {
		return false;
	}
	if (!texture->IsAllocated()) {
		return false;
	}
	switch (texture->GetTxTarget()) {
	case GL_TEXTURE_1D:
		if (layer != 0) {
			return false;
		}
		glFramebufferTexture1D(*GetTarget(), GL_DEPTH_ATTACHMENT, GL_TEXTURE_1D, texture->GetResId(), level);
		return true;
	case GL_TEXTURE_2D:
		if (layer != 0) {
			return false;
		}
		glFramebufferTexture2D(*GetTarget(), GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, texture->GetResId(), level);
		return true;
	case GL_TEXTURE_RECTANGLE:
		if (layer != 0 || level != 0) {
			return false;
		}
		glFramebufferTexture2D(*GetTarget(), GL_DEPTH_ATTACHMENT, GL_TEXTURE_3D, texture->GetResId(), level);
		return true;
	case GL_TEXTURE_2D_MULTISAMPLE:
		if (layer != 0 || level != 0) {
			return false;
		}
		glFramebufferTexture2D(*GetTarget(), GL_DEPTH_ATTACHMENT, GL_TEXTURE_3D, texture->GetResId(), level);
		return true;
	case GL_TEXTURE_3D:
		glFramebufferTexture3D(*GetTarget(), GL_DEPTH_ATTACHMENT, GL_TEXTURE_3D, texture->GetResId(), level, layer);
		return true;
	default:
		return false;
	}
	return false;
}

bool RTLib::Ext::GL::Internal::ImplGLFramebuffer::AttachStencilTexture(ImplGLTexture* texture, GLint level, GLint layer)
{
	if (!IsBinded() || !texture) {
		return false;
	}
	if (!texture->IsAllocated()) {
		return false;
	}
	switch (texture->GetTxTarget()) {
	case GL_TEXTURE_1D:
		if (layer != 0) {
			return false;
		}
		glFramebufferTexture1D(*GetTarget(), GL_STENCIL_ATTACHMENT, GL_TEXTURE_1D, texture->GetResId(), level);
		return true;
	case GL_TEXTURE_2D:
		if (layer != 0) {
			return false;
		}
		glFramebufferTexture2D(*GetTarget(), GL_STENCIL_ATTACHMENT, GL_TEXTURE_2D, texture->GetResId(), level);
		return true;
	case GL_TEXTURE_RECTANGLE:
		if (layer != 0 || level != 0) {
			return false;
		}
		glFramebufferTexture2D(*GetTarget(), GL_STENCIL_ATTACHMENT, GL_TEXTURE_3D, texture->GetResId(), level);
		return true;
	case GL_TEXTURE_2D_MULTISAMPLE:
		if (layer != 0 || level != 0) {
			return false;
		}
		glFramebufferTexture2D(*GetTarget(), GL_STENCIL_ATTACHMENT, GL_TEXTURE_3D, texture->GetResId(), level);
		return true;
	case GL_TEXTURE_3D:
		glFramebufferTexture3D(*GetTarget(), GL_STENCIL_ATTACHMENT, GL_TEXTURE_3D, texture->GetResId(), level, layer);
		return true;
	default:
		return false;
	}
	return false;
}

bool RTLib::Ext::GL::Internal::ImplGLFramebuffer::AttachDepthStencilTexture(ImplGLTexture* texture, GLint level, GLint layer)
{
	if (!IsBinded() || !texture) {
		return false;
	}
	if (!texture->IsAllocated()) {
		return false;
	}
	switch (texture->GetTxTarget()) {
	case GL_TEXTURE_1D:
		if (layer != 0) {
			return false;
		}
		glFramebufferTexture1D(*GetTarget(), GL_DEPTH_STENCIL_ATTACHMENT, GL_TEXTURE_1D, texture->GetResId(), level);
		return true;
	case GL_TEXTURE_2D:
		if (layer != 0) {
			return false;
		}
		glFramebufferTexture2D(*GetTarget(), GL_DEPTH_STENCIL_ATTACHMENT, GL_TEXTURE_2D, texture->GetResId(), level);
		return true;
	case GL_TEXTURE_RECTANGLE:
		if (layer != 0 || level != 0) {
			return false;
		}
		glFramebufferTexture2D(*GetTarget(), GL_DEPTH_STENCIL_ATTACHMENT, GL_TEXTURE_3D, texture->GetResId(), level);
		return true;
	case GL_TEXTURE_2D_MULTISAMPLE:
		if (layer != 0 || level != 0) {
			return false;
		}
		glFramebufferTexture2D(*GetTarget(), GL_DEPTH_STENCIL_ATTACHMENT, GL_TEXTURE_3D, texture->GetResId(), level);
		return true;
	case GL_TEXTURE_3D:
		glFramebufferTexture3D(*GetTarget(), GL_DEPTH_STENCIL_ATTACHMENT, GL_TEXTURE_3D, texture->GetResId(), level, layer);
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

bool RTLib::Ext::GL::Internal::ImplGLFramebuffer::IsCompleted() const noexcept {
	auto bPoint = GetBindingPoint();
	if (!bPoint || !IsBinded()) {
		return false;
	}
	return glCheckFramebufferStatus(*GetTarget()) == GL_FRAMEBUFFER_COMPLETE;
}
