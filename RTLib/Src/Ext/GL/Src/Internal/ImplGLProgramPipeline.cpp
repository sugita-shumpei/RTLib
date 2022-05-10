#include "ImplGLProgramPipeline.h"
#include "ImplGLProgram.h"
#include "ImplGLUtils.h"
namespace RTLib {
	namespace Ext {
		namespace GL {
			namespace Internal {
				class ImplGLProgramPipelineBase : public ImplGLBindableBase {
				public:
					ImplGLProgramPipelineBase(ImplGLProgramSlot* slot) :m_ProgramSlot(slot) {}
					virtual ~ImplGLProgramPipelineBase()noexcept {}
					auto GetProgramSlot()noexcept -> ImplGLProgramSlot* { return m_ProgramSlot; }
				protected:
					virtual bool      Create()noexcept override {
						GLuint resId;
						glGenProgramPipelines(1, &resId);
						if (resId == 0) {
							return false;
						}
						SetResId(resId);
						return true;
					}
					virtual void     Destroy()noexcept {
						GLuint resId = GetResId();
						glDeleteProgramPipelines(1, &resId);
						SetResId(0);
						m_ProgramSlot = nullptr;
					}
					virtual void   Bind(GLenum target) {
						GLuint resId = GetResId();
						if (resId > 0) {
#ifndef NDEBUG
							std::cout << "BIND " << ToString(target) << ": " << GetName() << std::endl;
#endif
							glBindProgramPipeline(resId);
						}
					}
					virtual void Unbind(GLenum target) {
#ifndef NDEBUG
						std::cout << "UNBIND " << ToString(target) << ": " << GetName() << std::endl;
						glBindProgramPipeline(0);
#endif
					}
				private:
					ImplGLProgramSlot* m_ProgramSlot = nullptr;
				};
			}

		}
	}
}

RTLib::Ext::GL::Internal::ImplGLProgramPipeline::~ImplGLProgramPipeline() noexcept {}

bool RTLib::Ext::GL::Internal::ImplGLProgramPipeline::Bind()  {
	auto programSlot = GetProgramSlot();
	if (!programSlot) { 
		return false; 
	}
	if ( programSlot->IsActive()) { return false; }
	return ImplGLBindable::Bind(GL_PROGRAM_PIPELINE);
}

bool RTLib::Ext::GL::Internal::ImplGLProgramPipeline::Attach(GLbitfield shaderStages, ImplGLProgram* program)
{
	auto resId = GetResId();
	if (!resId || !program) {
		return false;
	}
	if (!program->IsLinked()) {
		return false;
	}
	{
		GLbitfield erasedStages = shaderStages;
		for (auto& [stage, AttachState] : m_AttachStates) {
			//ShaderStages>stage
			erasedStages &= ~stage;
		}
		if (erasedStages) { return false; }
	}
	glUseProgramStages(resId, shaderStages, program->GetResId());
	for (auto& [stage, AttachState] : m_AttachStates) {
		//ShaderStages>stage
		if ((stage & shaderStages)== stage) {
			AttachState.program = program;
		}
	}
	return true;
}

auto RTLib::Ext::GL::Internal::ImplGLProgramPipeline::Activate(GLbitfield shaderStage)->ImplGLProgram*
{
	auto pipelineResId = GetResId();
	if (!pipelineResId) { return nullptr; }
	auto program = GetAttachedProgram(shaderStage);
	if (!program) { return nullptr; }
	if (m_ActiveProgram){ 
		return (m_ActiveProgram == program)? program: nullptr;
	}
	auto programResId = program->GetResId();
	if (GetProgramSlot()->Register(program)) {
		glActiveShaderProgram(pipelineResId, programResId);
		m_ActiveProgram = program;
		return program;
	}
	return nullptr;
}

void RTLib::Ext::GL::Internal::ImplGLProgramPipeline::Deactivate()
{
	auto resId = GetResId();
	if (!resId || !m_ActiveProgram) {
		return; 
	}
	glUseProgram(0);
	GetProgramSlot()->Unregister();
	m_ActiveProgram = nullptr;
}

bool RTLib::Ext::GL::Internal::ImplGLProgramPipeline::HasAttachedProgram(GLbitfield shaderStage) const noexcept
{
	return !GetAttachedProgram(shaderStage);
}

bool RTLib::Ext::GL::Internal::ImplGLProgramPipeline::HasActiveProgram() const noexcept
{
	return !GetActiveProgram();
}

auto RTLib::Ext::GL::Internal::ImplGLProgramPipeline::GetActiveProgram() noexcept -> ImplGLProgram*
{
	return m_ActiveProgram;
}

RTLib::Ext::GL::Internal::ImplGLProgramPipeline::ImplGLProgramPipeline(ImplGLResourceTable* table, ImplGLBindingPoint* bPoint) noexcept :ImplGLBindable(table, bPoint){}

void RTLib::Ext::GL::Internal::ImplGLProgramPipeline::AddShaderStage(GLbitfield shaderStage, bool isRequired) noexcept
{
	m_AttachStates[shaderStage] = { isRequired , nullptr };
}

auto RTLib::Ext::GL::Internal::ImplGLProgramPipeline::GetAttachedProgram(GLbitfield shaderStage) noexcept -> ImplGLProgram*
{
	for (const auto& [stage, attachState] : m_AttachStates) {
		//ShaderStages>stage
		if (shaderStage == stage) {
			return attachState.program;
		}
	}
	return nullptr;
}

auto RTLib::Ext::GL::Internal::ImplGLProgramPipeline::GetAttachedProgram(GLbitfield shaderStage) const noexcept -> const ImplGLProgram*
{
	for (const auto& [stage, attachState] : m_AttachStates) {
		//ShaderStages>stage
		if (shaderStage == stage) {
			return attachState.program;
		}
	}
	return nullptr;
}

auto RTLib::Ext::GL::Internal::ImplGLProgramPipeline::GetActiveProgram() const noexcept -> const ImplGLProgram*
{
	return m_ActiveProgram;
}

auto RTLib::Ext::GL::Internal::ImplGLProgramPipeline::GetProgramSlot() noexcept -> ImplGLProgramSlot*
{
	auto base = GetBase();
	return base ? static_cast<ImplGLProgramPipelineBase*>(base)->GetProgramSlot(): nullptr;
}

auto RTLib::Ext::GL::Internal::ImplGLGraphicsProgramPipeline::New(ImplGLResourceTable* table, ImplGLBindingPoint* bPoint, ImplGLProgramSlot* programSlot) -> ImplGLGraphicsProgramPipeline* {
	if (!table || !bPoint||!programSlot) {
		return nullptr;
	}
	auto programPipeline = new ImplGLGraphicsProgramPipeline(table, bPoint);
	if (programPipeline) {
		programPipeline->InitBase<ImplGLProgramPipelineBase>(programSlot);
		auto res = programPipeline->Create();
		if (!res) {
			delete programPipeline;
			return nullptr;
		}
		programPipeline->AddShaderStage(GL_VERTEX_SHADER_BIT  , true);
		programPipeline->AddShaderStage(GL_FRAGMENT_SHADER_BIT, true);
		programPipeline->AddShaderStage(GL_GEOMETRY_SHADER_BIT);
		programPipeline->AddShaderStage(GL_TESS_CONTROL_SHADER_BIT);
		programPipeline->AddShaderStage(GL_TESS_EVALUATION_SHADER_BIT);
	}
	return programPipeline;
}

	 RTLib::Ext::GL::Internal::ImplGLGraphicsProgramPipeline::~ImplGLGraphicsProgramPipeline() noexcept
{
}

     RTLib::Ext::GL::Internal::ImplGLGraphicsProgramPipeline::ImplGLGraphicsProgramPipeline(ImplGLResourceTable* table, ImplGLBindingPoint* bPoint) noexcept:ImplGLProgramPipeline(table, bPoint)
{
}

auto RTLib::Ext::GL::Internal::ImplGLComputeProgramPipeline::New(ImplGLResourceTable* table, ImplGLBindingPoint* bPoint, ImplGLProgramSlot* programSlot) -> ImplGLComputeProgramPipeline* {
	if (!table || !bPoint || !programSlot) {
		return nullptr;
	}
	auto programPipeline = new ImplGLComputeProgramPipeline(table, bPoint);
	if (programPipeline) {
		programPipeline->InitBase<ImplGLProgramPipelineBase>(programSlot);
		auto res = programPipeline->Create();
		if (!res) {
			delete programPipeline;
			return nullptr;
		}
		programPipeline->AddShaderStage(GL_COMPUTE_SHADER_BIT, true);
	}
	return programPipeline;
}

RTLib::Ext::GL::Internal::ImplGLComputeProgramPipeline::~ImplGLComputeProgramPipeline() noexcept
{
}

RTLib::Ext::GL::Internal::ImplGLComputeProgramPipeline::ImplGLComputeProgramPipeline(ImplGLResourceTable* table, ImplGLBindingPoint* bPoint) noexcept:ImplGLProgramPipeline(table, bPoint)
{
}
