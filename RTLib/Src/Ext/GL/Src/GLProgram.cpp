#include <RTLib/Ext/GL/GLProgram.h>
#include <RTLib/Ext/GL/GLShader.h>
#include <RTLib/Ext/GL/GLContext.h>
RTLib::Ext::GL::GLProgram::~GLProgram() noexcept {}
bool RTLib::Ext::GL::GLProgram::AttachShader(GLShader* shader) {
	auto resId = GetResId();
	if (!resId || !shader) {
		return false;
	}
	if (!shader->IsAttachable()) {
		return false;
	}

	GLShaderStageFlagBits programType = shader->GetShaderStage();
	if (!IsAttachable(programType)) {
		return false;
	}
	m_AttachedStages[programType].isEnabled = true;
	glAttachShader(resId, shader->GetResId());
	return true;
}

bool RTLib::Ext::GL::GLProgram::Link() {
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

bool RTLib::Ext::GL::GLProgram::Link(std::string& logData) {
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

bool RTLib::Ext::GL::GLProgram::IsLinked() const noexcept {
	return m_IsLinked;
}

bool RTLib::Ext::GL::GLProgram::IsLinkable() const noexcept {
	if (!GetResId() || IsLinked()) { return false; }
	for (auto& [target, state] : m_AttachedStages) {
		if (state.isRequired && !state.isEnabled) {
			return false;
		}
	}
	return true;
}

bool RTLib::Ext::GL::GLProgram::IsAttachable(GLShaderStageFlagBits programType) const noexcept {

	if (IsLinked()) { return false; }
	if (m_AttachedStages.count(programType) == 0) {
		return false;
	}
	if (m_AttachedStages.at(programType).isEnabled) {
		return false;
	}
	return true;
}

bool RTLib::Ext::GL::GLProgram::HasShaderType(GLShaderStageFlagBits shaderType) const noexcept
{
	if (m_AttachedStages.count(shaderType) == 0) {
		return false;
	}
	return m_AttachedStages.at(shaderType).isEnabled;
}

auto RTLib::Ext::GL::GLProgram::GetShaderStages() const noexcept -> GLShaderStageFlags
{
	GLbitfield stages = 0;
	if (HasShaderType(GLShaderStageVertex)) {
		stages |= GLShaderStageVertex;
	}
	if (HasShaderType(GLShaderStageGeometry)) {
		stages |= GLShaderStageGeometry;
	}
	if (HasShaderType(GLShaderStageTessControl)) {
		stages |= GLShaderStageTessControl;
	}
	if (HasShaderType(GLShaderStageTessEvaluation)) {
		stages |= GLShaderStageTessEvaluation;
	}
	if (HasShaderType(GLShaderStageFragment)) {
		stages |= GLShaderStageFragment;
	}
	if (HasShaderType(GLShaderStageCompute)) {
		stages |= GLShaderStageCompute;
	}
	return stages;
}

auto RTLib::Ext::GL::GLProgram::GetUniformLocation(const char* name) -> GLint
{
	if (!IsLinked()) {
		return -1;
	}
	return glGetUniformLocation(m_ResId, name);
}

RTLib::Ext::GL::GLProgram::GLProgram(GLContext* context) noexcept :m_Context{ context } {}

void RTLib::Ext::GL::GLProgram::AddShaderType(GLShaderStageFlagBits shaderType, bool isRequired) noexcept {
	if (IsLinked()) {
		return;
	}
	m_AttachedStages[shaderType] = { isRequired,false };
}


auto RTLib::Ext::GL::GLProgram::GetUniformBlockIndex(const char* name)->GLuint
{
	auto resId = GetResId();
	if (resId == 0 || !name || !IsLinked()) {
		return GL_INVALID_INDEX;
	}
	return glGetUniformBlockIndex(resId, name);
}
bool RTLib::Ext::GL::GLProgram::SetUniformBlockBinding(GLuint blockIndex, GLuint bindingIndex)
{
	auto resId = GetResId();
	if (resId == 0 || !IsLinked() || blockIndex == GL_INVALID_INDEX) {
		return false;
	}
	glUniformBlockBinding(resId, blockIndex, bindingIndex);
	return true;
}

bool RTLib::Ext::GL::GLProgram::SetUniformImageUnit(GLint location, GLuint imageUnit)
{
	auto resId = GetResId();
	if (resId == 0 || !IsLinked() || imageUnit == GL_INVALID_INDEX) {
		return false;
	}
	if (m_Context->SupportVersion(4, 1)) {
		if (m_ImageUnits.count(location) > 0) {
			if (m_ImageUnits.at(location) == imageUnit) {
				return true;
			}
		}
		glProgramUniform1i(resId, location, imageUnit);
		SetImageUnit(location, imageUnit);
	}
	else {
		m_Context->SetProgram(this);
		m_Context->SetUniformImageUnit(location, imageUnit);
	}
	return true;
}

void RTLib::Ext::GL::GLProgram::SetSeparetable(bool isEnabled)
{
	if (!m_Context->SupportVersion(4, 1)) {
		std::cerr << "Error: Separatable Program Must Support OpenGL 4.1\n";
		return;
	}
	if (m_IsSeparatable != isEnabled) {
		glProgramParameteri(m_ResId, GL_PROGRAM_SEPARABLE, isEnabled);
		m_IsSeparatable = isEnabled;
	}
} 

bool RTLib::Ext::GL::GLProgram::IsSeparatable() const noexcept
{
	return m_IsSeparatable;
}

auto RTLib::Ext::GL::GLProgram::New(GLContext* context) -> GLProgram*
{
	if (!context) { return nullptr; }
	auto program = new GLProgram(context);
	program->m_ResId = glCreateProgram();
	program->m_AttachedStages[GLShaderStageVertex        ] = {};
	program->m_AttachedStages[GLShaderStageGeometry      ] = {};
	program->m_AttachedStages[GLShaderStageTessControl   ] = {};
	program->m_AttachedStages[GLShaderStageTessEvaluation] = {};
	program->m_AttachedStages[GLShaderStageFragment      ] = {};
	program->m_AttachedStages[GLShaderStageCompute       ] = {};
	return program;
}

void RTLib::Ext::GL::GLProgram::Destroy()
{
	if (m_ResId) { glDeleteProgram(m_ResId); m_ResId = 0; }
}

auto RTLib::Ext::GL::GLProgram::GetResId() const noexcept -> GLuint
{
	return m_ResId;
}

void RTLib::Ext::GL::GLProgram::SetImageUnit(GLint location, GLint imageUnit) noexcept
{
	m_ImageUnits[location] = imageUnit;
}
