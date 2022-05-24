#include <RTLib/Ext/CUGL/CUGLContext.h>

RTLib::Ext::CUGL::CUGLContext::~CUGLContext() noexcept
{
}

bool RTLib::Ext::CUGL::CUGLContext::Initialize()
{
	return (!m_CtxCUDA)||(!m_CtxGL);
}

void RTLib::Ext::CUGL::CUGLContext::Terminate()
{
	m_CtxCUDA = nullptr;
	m_CtxGL   = nullptr;
}
