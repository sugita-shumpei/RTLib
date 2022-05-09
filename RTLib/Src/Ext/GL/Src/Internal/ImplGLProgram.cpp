#include "ImplGLProgram.h"
#include "ImplGLShader.h"
namespace RTLib {
	namespace Ext {
		namespace GL {
			namespace Internal {
				class ImplGLProgramBase : public ImplGLResourceBase {
				public:
					virtual ~ImplGLProgramBase()noexcept {}
				protected:
					virtual bool  Create()noexcept override {
						GLuint resId = glCreateProgram();
						if (resId == 0) {
							return false;
						}
						SetResId(resId);
						return true;
					}
					virtual void Destroy()noexcept override {
						glDeleteProgram(GetResId());
						SetResId(0);
					}
				};
				auto ImplGLGraphicsProgram::New(ImplGLResourceTable* table) -> ImplGLGraphicsProgram*
				{
					if (!table) {
						return nullptr;
					}
					auto shader = new ImplGLGraphicsProgram(table);
					if (shader) {
						shader->InitBase<ImplGLProgramBase>();
						auto res = shader->Create();
						if (!res) {
							delete shader;
							return nullptr;
						}
						shader->AddShaderType(GL_VERTEX_SHADER, true);
						shader->AddShaderType(GL_FRAGMENT_SHADER, true);
						shader->AddShaderType(GL_GEOMETRY_SHADER);
						shader->AddShaderType(GL_TESS_CONTROL_SHADER);
						shader->AddShaderType(GL_TESS_EVALUATION_SHADER);
					}
					return shader;
				}
				ImplGLGraphicsProgram::~ImplGLGraphicsProgram() noexcept {}
				ImplGLGraphicsProgram::ImplGLGraphicsProgram(ImplGLResourceTable* table) noexcept :ImplGLProgram(table) {}
				auto ImplGLComputeProgram::New(ImplGLResourceTable* table) -> ImplGLComputeProgram*
				{
					if (!table) {
						return nullptr;
					}
					auto shader = new ImplGLComputeProgram(table);
					if (shader) {
						shader->InitBase<ImplGLProgramBase>();
						auto res = shader->Create();
						if (!res) {
							delete shader;
							return nullptr;
						}
						shader->AddShaderType(GL_COMPUTE_SHADER, true);
					}
					return shader;
				}
				ImplGLComputeProgram::~ImplGLComputeProgram() noexcept {}
				ImplGLComputeProgram::ImplGLComputeProgram(ImplGLResourceTable* table) noexcept :ImplGLProgram(table) {}
			}
		}
	}
}
RTLib::Ext::GL::Internal::ImplGLProgram::~ImplGLProgram() noexcept {}
bool RTLib::Ext::GL::Internal::ImplGLProgram::AttachShader(ImplGLShader* shader) {
	auto resId = GetResId();
	if (!resId || !shader) {
		return false;
	}
	if (!shader->IsAttachable()) {
		return false;
	}

	GLenum shaderType = shader->GetShaderType();
	if (!IsAttachable(shaderType)) {
		return false;
	}
	m_PrelinkState->attachedStages[shaderType].isEnabled = true;
	glAttachShader(resId, shader->GetResId());
	return true;
}

bool RTLib::Ext::GL::Internal::ImplGLProgram::Link() {
	if (!IsLinkable()) { return false; }
	auto resId = GetResId();
	glLinkProgram(resId);
	GLint res;
	glGetProgramiv(resId, GL_LINK_STATUS, &res);
	if (res == GL_TRUE) {
		m_PrelinkState = std::nullopt;
	}
	return res == GL_TRUE;
}

bool RTLib::Ext::GL::Internal::ImplGLProgram::Link(std::string& logData) {
	logData.clear();
	if (!IsLinkable()) { return false; }
	auto resId = GetResId();
	glLinkProgram(resId);
	GLint res;
	glGetProgramiv(resId, GL_LINK_STATUS, &res);
	if (res == GL_TRUE) {
		m_PrelinkState = std::nullopt;
	}
	auto len = GLint(0);
	glGetProgramiv(resId, GL_INFO_LOG_LENGTH, &len);
	logData.resize(len + 1);
	glGetProgramInfoLog(resId, len, nullptr, logData.data());
	logData.resize(len);
	return res;
}

bool RTLib::Ext::GL::Internal::ImplGLProgram::IsAvailable() const noexcept {
	return !m_PrelinkState;
}

bool RTLib::Ext::GL::Internal::ImplGLProgram::IsLinkable() const noexcept {
	if (!GetResId() || IsAvailable()) { return false; }
	for (auto& [target, state] : m_PrelinkState->attachedStages) {
		if (state.isRequired && !state.isEnabled) {
			return false;
		}
	}
	return true;
}

bool RTLib::Ext::GL::Internal::ImplGLProgram::IsAttachable(GLenum shaderType) const noexcept {

	if (!m_PrelinkState) { return false; }
	if (m_PrelinkState->attachedStages.count(shaderType) == 0) {
		return false;
	}
	if (m_PrelinkState->attachedStages.at(shaderType).isEnabled) {
		return false;
	}
	return true;
}

RTLib::Ext::GL::Internal::ImplGLProgram::ImplGLProgram(ImplGLResourceTable* table) noexcept :ImplGLResource(table) {}

void RTLib::Ext::GL::Internal::ImplGLProgram::AddShaderType(GLenum shaderType, bool isRequired) noexcept {
	if (!m_PrelinkState) {
		return;
	}
	m_PrelinkState->attachedStages[shaderType] = { isRequired,false };
}
