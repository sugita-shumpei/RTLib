#include "ImplGLShader.h"
namespace RTLib {
	namespace Ext {
		namespace GL {
			namespace Internal {
				class ImplGLShaderBase : public ImplGLResourceBase {
				public:
					ImplGLShaderBase(GLenum shaderType) :ImplGLResourceBase(), m_ShaderType{ shaderType }{}
					virtual ~ImplGLShaderBase()noexcept {}

					auto GetShaderType()const noexcept -> GLenum { return m_ShaderType; }
				protected:
					virtual bool  Create()noexcept override {
						GLuint resId = glCreateShader(m_ShaderType);
						if (resId == 0) {
							return false;
						}
						SetResId(resId);
						return true;
					}
					virtual void Destroy()noexcept override {
						glDeleteShader(GetResId());
						SetResId(0);
					}
				private:
					GLenum m_ShaderType;
				};
			}
		}
	}

}
RTLib::Ext::GL::Internal::ImplGLShader::~ImplGLShader() noexcept {}

//ResetSourceGLSL

bool RTLib::Ext::GL::Internal::ImplGLShader::ResetSourceGLSL(const std::vector<char>& source) {

	GLuint obj = GetResId();
	if (!obj || !m_PreAttachableState) { return false; }
	if (m_PreAttachableState->ownBinary || m_PreAttachableState->ownSource) {
		return false;
	}
	const GLchar* pSource = source.data();
	glShaderSource(
		obj, static_cast<GLsizei>(source.size()), &pSource, nullptr
	);
	m_PreAttachableState->ownSource = true;
	return true;
}

bool RTLib::Ext::GL::Internal::ImplGLShader::Compile() {
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

bool RTLib::Ext::GL::Internal::ImplGLShader::Compile(std::string& logData) {
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

bool RTLib::Ext::GL::Internal::ImplGLShader::IsAttachable() const noexcept { return !m_PreAttachableState; }


//ResetBinarySpirv

bool RTLib::Ext::GL::Internal::ImplGLShader::ResetBinarySPV(const std::vector<uint32_t>& spirvData) {

	GLuint obj = GetResId();
	if (!obj || !m_SpvSupported ||!m_PreAttachableState) { return false; }

	if (m_PreAttachableState->ownBinary || m_PreAttachableState->ownSource) { return false; }
	glShaderBinary(
		1, &obj, GL_SHADER_BINARY_FORMAT_SPIR_V, spirvData.data(),spirvData.size() * sizeof(uint32_t)
	);
	m_PreAttachableState->ownBinary = true;
	return true;
}

bool RTLib::Ext::GL::Internal::ImplGLShader::Specialize(const GLchar* pEntryPoint​, GLuint numSpecializationConstants​, const GLuint* pConstantIndex​, const GLuint* pConstantValue​)
{
	GLuint obj = GetResId();
	if (!obj || !m_SpvSupported || !m_PreAttachableState) { return false; }
	if (!m_PreAttachableState->ownBinary) { return false; }
	glSpecializeShader(obj, pEntryPoint​, numSpecializationConstants​, pConstantIndex​, pConstantValue​);
	m_PreAttachableState = std::nullopt;
	return true;
}

auto RTLib::Ext::GL::Internal::ImplGLShader::GetShaderType() const noexcept -> GLenum {
	auto base = GetBase();
	if (base) {
		return static_cast<const ImplGLShaderBase*>(base)->GetShaderType();
	}
	else {
		return GL_VERTEX_SHADER;
	}
}

RTLib::Ext::GL::Internal::ImplGLShader::ImplGLShader(ImplGLResourceTable* table, bool isSpirvSupported) noexcept :ImplGLResource(table), m_SpvSupported(isSpirvSupported) {}

RTLib::Ext::GL::Internal::ImplGLVertexShader::~ImplGLVertexShader() noexcept
{
}

RTLib::Ext::GL::Internal::ImplGLVertexShader::ImplGLVertexShader(ImplGLResourceTable* table, bool isSpirvSupported) noexcept:ImplGLShader(table,isSpirvSupported)
{
}

auto RTLib::Ext::GL::Internal::ImplGLVertexShader::New(ImplGLResourceTable* table,  bool isSpirvSupported) -> ImplGLVertexShader* {
	if (!table) {
		return nullptr;
	}
	auto shader = new ImplGLVertexShader(table, isSpirvSupported);
	if (shader) {
		shader->InitBase<ImplGLShaderBase>(GL_VERTEX_SHADER);
		auto res = shader->Create();
		if (!res) {
			delete shader;
			return nullptr;
		}
	}
	return shader;
}

auto RTLib::Ext::GL::Internal::ImplGLFragmentShader::New(ImplGLResourceTable* table, bool isSpirvSupported) -> ImplGLFragmentShader*
{
	if (!table) {
		return nullptr;
	}
	auto shader = new ImplGLFragmentShader(table, isSpirvSupported);
	if (shader) {
		shader->InitBase<ImplGLShaderBase>(GL_FRAGMENT_SHADER);
		auto res = shader->Create();
		if (!res) {
			delete shader;
			return nullptr;
		}
	}
	return shader;
}

RTLib::Ext::GL::Internal::ImplGLFragmentShader::~ImplGLFragmentShader() noexcept
{
}

RTLib::Ext::GL::Internal::ImplGLFragmentShader::ImplGLFragmentShader(ImplGLResourceTable* table, bool isSpirvSupported) noexcept :ImplGLShader(table, isSpirvSupported)
{
}

auto RTLib::Ext::GL::Internal::ImplGLGeometryShader::New(ImplGLResourceTable* table, bool isSpirvSupported) -> ImplGLGeometryShader*
{
	if (!table) {
		return nullptr;
	}
	auto shader = new ImplGLGeometryShader(table, isSpirvSupported);
	if (shader) {
		shader->InitBase<ImplGLShaderBase>(GL_GEOMETRY_SHADER);
		auto res = shader->Create();
		if (!res) {
			delete shader;
			return nullptr;
		}
	}
	return shader;
}

RTLib::Ext::GL::Internal::ImplGLGeometryShader::~ImplGLGeometryShader() noexcept
{
}

RTLib::Ext::GL::Internal::ImplGLGeometryShader::ImplGLGeometryShader(ImplGLResourceTable* table, bool isSpirvSupported) noexcept :ImplGLShader(table, isSpirvSupported)
{
}

auto RTLib::Ext::GL::Internal::ImplGLTesselationEvaluationShader::New(ImplGLResourceTable* table, bool isSpirvSupported) -> ImplGLTesselationEvaluationShader*
{
	if (!table) {
		return nullptr;
	}
	auto shader = new ImplGLTesselationEvaluationShader(table, isSpirvSupported);
	if (shader) {
		shader->InitBase<ImplGLShaderBase>(GL_TESS_EVALUATION_SHADER);
		auto res = shader->Create();
		if (!res) {
			delete shader;
			return nullptr;
		}
	}
	return shader;
}

RTLib::Ext::GL::Internal::ImplGLTesselationEvaluationShader::~ImplGLTesselationEvaluationShader() noexcept
{
}

RTLib::Ext::GL::Internal::ImplGLTesselationEvaluationShader::ImplGLTesselationEvaluationShader(ImplGLResourceTable* table, bool isSpirvSupported) noexcept :ImplGLShader(table, isSpirvSupported)
{
}

auto RTLib::Ext::GL::Internal::ImplGLTesselationControlShader::New(ImplGLResourceTable* table, bool isSpirvSupported) -> ImplGLTesselationControlShader*
{
	if (!table) {
		return nullptr;
	}
	auto shader = new ImplGLTesselationControlShader(table, isSpirvSupported);
	if (shader) {
		shader->InitBase<ImplGLShaderBase>(GL_TESS_CONTROL_SHADER);
		auto res = shader->Create();
		if (!res) {
			delete shader;
			return nullptr;
		}
	}
	return shader;
}

RTLib::Ext::GL::Internal::ImplGLTesselationControlShader::~ImplGLTesselationControlShader() noexcept
{
}

RTLib::Ext::GL::Internal::ImplGLTesselationControlShader::ImplGLTesselationControlShader(ImplGLResourceTable* table, bool isSpirvSupported) noexcept :ImplGLShader(table, isSpirvSupported)
{
}

auto RTLib::Ext::GL::Internal::ImplGLComputeShader::New(ImplGLResourceTable* table, bool isSpirvSupported) -> ImplGLComputeShader*
{
	if (!table) {
		return nullptr;
	}
	auto shader = new ImplGLComputeShader(table, isSpirvSupported);
	if (shader) {
		shader->InitBase<ImplGLShaderBase>(GL_COMPUTE_SHADER);
		auto res = shader->Create();
		if (!res) {
			delete shader;
			return nullptr;
		}
	}
	return shader;
}

RTLib::Ext::GL::Internal::ImplGLComputeShader::~ImplGLComputeShader() noexcept
{
}

RTLib::Ext::GL::Internal::ImplGLComputeShader::ImplGLComputeShader(ImplGLResourceTable* table, bool isSpirvSupported) noexcept :ImplGLShader(table, isSpirvSupported)
{
}
