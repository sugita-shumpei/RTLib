#include <RTLib/Ext/GL/GLProgramPipeline.h>
#include <RTLib/Ext/GL/GLProgram.h>
#include <RTLib/Ext/GL/GLContext.h>
#include <RTLib/Ext/GL/GLUtility.h>
struct RTLib::Ext::GL::GLProgramPipeline::Impl
{
	Impl(GLContext* ctx)noexcept {
		context = ctx;
	}
	GLContext* context = nullptr;
	std::unordered_map<GLShaderStageFlagBits, GLProgram*> stagePrograms = {
		{GLShaderStageVertex        ,nullptr},
		{GLShaderStageGeometry      ,nullptr},
		{GLShaderStageTessControl   ,nullptr},
		{GLShaderStageTessEvaluation,nullptr},
		{GLShaderStageFragment      ,nullptr},
		{GLShaderStageCompute       ,nullptr},
	};
	GLuint resId = 0;
};

auto RTLib::Ext::GL::GLProgramPipeline::New(GLContext* context) -> GLProgramPipeline*
{
	auto pipeline = new GLProgramPipeline(context);
	GLuint resId;
	glGenProgramPipelines(1, &resId);
	pipeline->m_Impl->resId = resId;
	return pipeline;
}

RTLib::Ext::GL::GLProgramPipeline::~GLProgramPipeline() noexcept
{
}

void RTLib::Ext::GL::GLProgramPipeline::Destroy()
{
	if (!m_Impl) { return; }
	glDeleteVertexArrays(1, &m_Impl->resId);
	m_Impl->resId = 0;
}

auto RTLib::Ext::GL::GLProgramPipeline::GetProgram(GLShaderStageFlagBits stage) const noexcept -> const GLProgram*
{
	return m_Impl->stagePrograms.at(stage);
}

auto RTLib::Ext::GL::GLProgramPipeline::GetProgram(GLShaderStageFlagBits stage) noexcept -> GLProgram*
{
	return m_Impl->stagePrograms.at(stage);
}

void RTLib::Ext::GL::GLProgramPipeline::SetProgram(GLShaderStageFlags stages, GLProgram* program)
{
	GLbitfield mask = 0;
	if (stages & GLShaderStageVertex) {
		if (m_Impl->stagePrograms.at(GLShaderStageVertex) == program) {
			stages |= ~GLShaderStageVertex;
		}
		else {
			mask |= GL_VERTEX_SHADER_BIT;
		}
	}
	if (stages & GLShaderStageGeometry) {
		if (m_Impl->stagePrograms.at(GLShaderStageGeometry) == program) {
			stages |= ~GLShaderStageGeometry;
		}
		else {
			mask |= GL_GEOMETRY_SHADER_BIT;
		}
	}
	if (stages & GLShaderStageTessControl) {
		if (m_Impl->stagePrograms.at(GLShaderStageTessControl) == program) {
			stages |= ~GLShaderStageTessControl;
		}
		else {
			mask |= GL_TESS_CONTROL_SHADER_BIT;
		}
	}
	if (stages & GLShaderStageTessEvaluation) {
		if (m_Impl->stagePrograms.at(GLShaderStageTessEvaluation) == program) {
			stages |= ~GLShaderStageTessEvaluation;
		}
		else {
			mask |= GL_TESS_EVALUATION_SHADER_BIT;
		}
	}
	if (stages & GLShaderStageFragment) {
		if (m_Impl->stagePrograms.at(GLShaderStageFragment) == program) {
			stages |= ~GLShaderStageFragment;
		}
		else {
			mask |= GL_FRAGMENT_SHADER_BIT;
		}
	}
	if (stages & GLShaderStageCompute) {
		if (m_Impl->stagePrograms.at(GLShaderStageCompute) == program) {
			stages |= ~GLShaderStageCompute;
		}
		else {
			mask |= GL_COMPUTE_SHADER_BIT;
		}
	}
	if (mask != 0) {
		glUseProgramStages(m_Impl->resId, mask, program->GetResId());
	}
}

RTLib::Ext::GL::GLProgramPipeline::GLProgramPipeline(GLContext* context) noexcept:m_Impl(new Impl(context))
{
}

auto RTLib::Ext::GL::GLProgramPipeline::GetResId() const noexcept -> GLuint
{
	return m_Impl->resId;
}
