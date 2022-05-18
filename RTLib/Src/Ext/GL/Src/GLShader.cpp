#include <RTLib/Ext/GL/GLShader.h>
#include "GLTypeConversions.h"
RTLib::Ext::GL::GLShader::~GLShader() noexcept {}

//ResetSourceGLSL

bool RTLib::Ext::GL::GLShader::ResetSourceGLSL(const std::vector<char>& source) {

	GLuint obj = GetResId();
	if (!obj || !m_PreAttachableState) { return false; }
	if (m_PreAttachableState->ownBinary || m_PreAttachableState->ownSource) {
		return false;
	}
	GLint length = source.size();
	const GLchar* pSource = source.data();
	glShaderSource(
		obj, 1, &pSource, &length
	);
	m_PreAttachableState->ownSource = true;
	return true;
}

bool RTLib::Ext::GL::GLShader::Compile() {
	GLuint obj = GetResId();
	if (!obj || !m_PreAttachableState) { return false; }
	if (!m_PreAttachableState->ownSource) {
		return false;
	}
	glCompileShader(obj);
	GLint  res;
	glGetShaderiv(obj, GL_COMPILE_STATUS, &res);
	if (res == GL_TRUE) {
		m_PreAttachableState = std::nullopt;
	}
	return res == GL_TRUE;
}

bool RTLib::Ext::GL::GLShader::Compile(std::string& logData) {
	logData.clear();
	GLuint obj = GetResId();
	if (!obj || !m_PreAttachableState) { return false; }
	if (!m_PreAttachableState->ownSource) {
		return false;
	}
	glCompileShader(obj);
	GLint  res;
	glGetShaderiv(obj, GL_COMPILE_STATUS, &res);
	GLint logLength = 0;
	glGetShaderiv(obj, GL_INFO_LOG_LENGTH, &logLength);
	logData.resize(logLength + 1);
	glGetShaderInfoLog(obj, static_cast<GLsizei>(logLength), nullptr, logData.data());
	logData.resize(logLength);
	if (res == GL_TRUE) {
		m_PreAttachableState = std::nullopt;
	}
	return res == GL_TRUE;
}

bool RTLib::Ext::GL::GLShader::IsAttachable() const noexcept { return !m_PreAttachableState; }


//ResetBinarySpirv

bool RTLib::Ext::GL::GLShader::ResetBinarySPV(const std::vector<uint32_t>& spirvData) {

	GLuint obj = GetResId();
	if (!obj || !m_SpvSupported || !m_PreAttachableState) { return false; }

	if (m_PreAttachableState->ownBinary || m_PreAttachableState->ownSource) { return false; }
	glShaderBinary(
		1, &obj, GL_SHADER_BINARY_FORMAT_SPIR_V, spirvData.data(), spirvData.size() * sizeof(uint32_t)
	);
	m_PreAttachableState->ownBinary = true;
	return true;
}

bool RTLib::Ext::GL::GLShader::Specialize(const GLchar* pEntryPoint​, GLuint numSpecializationConstants​, const GLuint* pConstantIndex​, const GLuint* pConstantValue​)
{
	GLuint obj = GetResId();
	if (!obj || !m_SpvSupported || !m_PreAttachableState) { return false; }
	if (!m_PreAttachableState->ownBinary) { return false; }
	glSpecializeShader(obj, pEntryPoint​, numSpecializationConstants​, pConstantIndex​, pConstantValue​);
	m_PreAttachableState = std::nullopt;
	return true;
}

auto RTLib::Ext::GL::GLShader::GetShaderType() const noexcept -> GLenum {
	return GetGLShaderStagesShaderType(m_ShaderStage);
}

RTLib::Ext::GL::GLShader::GLShader(GLContext* context, GLShaderStageFlagBits shaderStage, bool isSpirvSupported) noexcept :m_SpvSupported{ isSpirvSupported }, m_ShaderStage{ shaderStage } {
}

auto RTLib::Ext::GL::GLShader::GetResId() const noexcept -> GLuint
{
	return m_ResId;
}

void RTLib::Ext::GL::GLShader::Destroy()
{
	if (!m_ResId) { return; }
	glDeleteShader(m_ResId);
	m_ResId = 0;
}

auto RTLib::Ext::GL::GLShader::New(GLContext* context, GLShaderStageFlagBits shaderStage, bool isSpirvSupport) -> GLShader*
{
	auto shader = new GLShader(context, shaderStage, isSpirvSupport);
	GLuint resId = glCreateShader(shader->GetShaderType());
	shader->m_ResId = resId;
	return shader;
}
