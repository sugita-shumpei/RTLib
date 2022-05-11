#include "ImplGLProgram.h"
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

	GLenum programType = shader->GetShaderType();
	if (!IsAttachable(programType)) {
		return false;
	}
	m_AttachedStages[programType].isEnabled = true;
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
		m_IsLinked = true;
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
		m_IsLinked = true;
	}
	auto len = GLint(0);
	glGetProgramiv(resId, GL_INFO_LOG_LENGTH, &len);
	logData.resize(len + 1);
	glGetProgramInfoLog(resId, len, nullptr, logData.data());
	logData.resize(len);
	return res == GL_TRUE;
}

bool RTLib::Ext::GL::Internal::ImplGLProgram::IsLinked() const noexcept {
	return m_IsLinked;
}

bool RTLib::Ext::GL::Internal::ImplGLProgram::IsLinkable() const noexcept {
	if (!GetResId() || IsLinked()) { return false; }
	for (auto& [target, state] : m_AttachedStages) {
		if (state.isRequired && !state.isEnabled) {
			return false;
		}
	}
	return true;
}

bool RTLib::Ext::GL::Internal::ImplGLProgram::IsAttachable(GLenum programType) const noexcept {

	if (IsLinked()) { return false; }
	if (m_AttachedStages.count(programType) == 0) {
		return false;
	}
	if (m_AttachedStages.at(programType).isEnabled) {
		return false;
	}
	return true;
}

bool RTLib::Ext::GL::Internal::ImplGLProgram::Enable()
{
	auto resId = GetResId();
	if (!resId || !IsLinked() || !m_ProgramSlot) {
		return false;
	}
	if (m_IsEnabled) { return true; }
	if (!m_ProgramSlot->Register(this)) {
		return false;
	}
	glUseProgram(resId);
	m_IsEnabled = true;
	return true;
}

void RTLib::Ext::GL::Internal::ImplGLProgram::Disable()
{
	auto resId = GetResId();
	if (!resId || !IsLinked() || !m_ProgramSlot || !m_IsEnabled) {
		return;
	}
	m_ProgramSlot->Unregister();
	glUseProgram(0);
	m_IsEnabled = false;
}

bool RTLib::Ext::GL::Internal::ImplGLProgram::IsEnabled() const noexcept { return m_IsEnabled; }

bool RTLib::Ext::GL::Internal::ImplGLProgram::HasShaderType(GLenum shaderType) const noexcept
{
	if (m_AttachedStages.count(shaderType) == 0) {
		return false;
	}
	return m_AttachedStages.at(shaderType).isEnabled;
}

auto RTLib::Ext::GL::Internal::ImplGLProgram::GetShaderStages() const noexcept -> GLbitfield
{
	GLbitfield stages = 0;
	if (HasShaderType(GL_VERTEX_SHADER_BIT)) {
		stages |= GL_VERTEX_SHADER_BIT;
	}
	if (HasShaderType(GL_FRAGMENT_SHADER_BIT)) {
		stages |= GL_FRAGMENT_SHADER_BIT;
	}
	if (HasShaderType(GL_GEOMETRY_SHADER_BIT)) {
		stages |= GL_GEOMETRY_SHADER_BIT;
	}	
	if (HasShaderType(GL_TESS_CONTROL_SHADER_BIT)) {
		stages |= GL_TESS_CONTROL_SHADER_BIT;
	}
	if (HasShaderType(GL_TESS_EVALUATION_SHADER_BIT)) {
		stages |= GL_TESS_EVALUATION_SHADER_BIT;
	}
	if (HasShaderType(GL_COMPUTE_SHADER_BIT)) {
		stages |= GL_COMPUTE_SHADER_BIT;
	}
	return stages;
}

	 RTLib::Ext::GL::Internal::ImplGLProgram::ImplGLProgram(ImplGLResourceTable* table, ImplGLProgramSlot* slot) noexcept :ImplGLResource(table), m_ProgramSlot{slot} {}

void RTLib::Ext::GL::Internal::ImplGLProgram::AddShaderType(GLenum shaderType, bool isRequired) noexcept {
	if (IsLinked()) {
		return;
	}
	m_AttachedStages[shaderType] = { isRequired,false };
}


auto RTLib::Ext::GL::Internal::ImplGLProgram::GetUniformBlockIndex(const char* name)->GLuint
{
    auto resId = GetResId();
    if (resId == 0||!name ||!IsLinked()){
        return GL_INVALID_INDEX;
    }
    return glGetUniformBlockIndex(resId,name);
}
bool RTLib::Ext::GL::Internal::ImplGLProgram::SetUniformBlockBinding(GLuint blockIndex, GLuint bindingIndex)
{
    auto resId = GetResId();
    if (resId == 0 ||!IsLinked() || blockIndex==GL_INVALID_INDEX){
        return false;
    }
    glUniformBlockBinding(resId,blockIndex, bindingIndex);
    return true;
}

auto RTLib::Ext::GL::Internal::ImplGLGraphicsProgram::New(RTLib::Ext::GL::Internal::ImplGLResourceTable* table, ImplGLProgramSlot* slot) -> ImplGLGraphicsProgram*
{
	if (!table) {
		return nullptr;
	}
	auto program = new ImplGLGraphicsProgram(table, slot);
	if (program) {
		program->InitBase<ImplGLProgramBase>();
		auto res = program->Create();
		if (!res) {
			delete program;
			return nullptr;
		}
		program->AddShaderType(GL_VERTEX_SHADER, true);
		program->AddShaderType(GL_FRAGMENT_SHADER, true);
		program->AddShaderType(GL_GEOMETRY_SHADER);
		program->AddShaderType(GL_TESS_CONTROL_SHADER);
		program->AddShaderType(GL_TESS_EVALUATION_SHADER);
	}
	return program;
}
RTLib::Ext::GL::Internal::ImplGLGraphicsProgram::~ImplGLGraphicsProgram() noexcept {}
RTLib::Ext::GL::Internal::ImplGLGraphicsProgram::ImplGLGraphicsProgram(ImplGLResourceTable* table, ImplGLProgramSlot* slot) noexcept :ImplGLProgram(table,slot) {}
auto RTLib::Ext::GL::Internal::ImplGLComputeProgram::New(ImplGLResourceTable* table, ImplGLProgramSlot* slot) -> ImplGLComputeProgram*
{
	if (!table) {
		return nullptr;
	}
	auto program = new ImplGLComputeProgram(table, slot);
	if (program) {
		program->InitBase<ImplGLProgramBase>();
		auto res = program->Create();
		if (!res) {
			delete program;
			return nullptr;
		}
		program->AddShaderType(GL_COMPUTE_SHADER, true);
	}
	return program;
}
RTLib::Ext::GL::Internal::ImplGLComputeProgram::~ImplGLComputeProgram() noexcept {}
bool RTLib::Ext::GL::Internal::ImplGLComputeProgram::Launch(GLuint numGroupX, GLuint numGroupY, GLuint numGroupZ)
{
	bool isEnabled = IsEnabled();
	if (!isEnabled) {
		if (!Enable()) {
			return false;
		}
	}
	glDispatchCompute(numGroupX, numGroupY, numGroupZ);
	if (!isEnabled) {
		Disable();
	}
	return true;
}
RTLib::Ext::GL::Internal::ImplGLComputeProgram::ImplGLComputeProgram(ImplGLResourceTable* table, ImplGLProgramSlot* slot) noexcept :ImplGLProgram(table, slot) {}

auto RTLib::Ext::GL::Internal::ImplGLSeparateProgram::New(ImplGLResourceTable* table, ImplGLProgramSlot* slot) -> ImplGLSeparateProgram*
{
	if (!table) {
		return nullptr;
	}
	auto program = new ImplGLSeparateProgram(table, slot);
	if (program) {
		program->InitBase<ImplGLProgramBase>();
		auto res = program->Create();
		if (!res) {
			delete program;
			return nullptr;
		}
		program->SetSeparatable(true);
		program->AddShaderType(GL_VERTEX_SHADER);
		program->AddShaderType(GL_FRAGMENT_SHADER);
		program->AddShaderType(GL_GEOMETRY_SHADER);
		program->AddShaderType(GL_TESS_CONTROL_SHADER);
		program->AddShaderType(GL_TESS_EVALUATION_SHADER);
		program->AddShaderType(GL_TESS_CONTROL_SHADER);

	}
	return program;
}

RTLib::Ext::GL::Internal::ImplGLSeparateProgram::~ImplGLSeparateProgram() noexcept
{
}

RTLib::Ext::GL::Internal::ImplGLSeparateProgram::ImplGLSeparateProgram(ImplGLResourceTable* table, ImplGLProgramSlot* slot) noexcept:ImplGLProgram{table,slot}
{
}

void RTLib::Ext::GL::Internal::ImplGLSeparateProgram::SetSeparatable(bool value)
{
	auto resId = GetResId();
	if (!resId || IsLinked()) {
		return;
	}
	glProgramParameteri(resId, GL_PROGRAM_SEPARABLE, value ? GL_TRUE : GL_FALSE);
}

bool RTLib::Ext::GL::Internal::ImplGLProgramSlot::IsActive() const noexcept
{
	return m_UsedProgram != nullptr;
}

bool RTLib::Ext::GL::Internal::ImplGLProgramSlot::Register(ImplGLProgram* program)
{
	if (IsActive()) { return false; }
	m_UsedProgram = program;
	return true;
}
void RTLib::Ext::GL::Internal::ImplGLProgramSlot::Unregister()
{
	m_UsedProgram = nullptr;
}
