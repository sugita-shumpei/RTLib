#ifndef RTLIB_EXT_GL_INTERNAL_IMPL_GL_PROGRAM_PIPELINE_H
#define RTLIB_EXT_GL_INTERNAL_IMPL_GL_PROGRAM_PIPELINE_H
#include "ImplGLBindable.h"
namespace RTLib
{
    namespace Ext
    {
        namespace GL
        {
            namespace Internal
            {
                class ImplGLProgramSlot;
                class ImplGLProgram;
                class ImplGLProgramPipeline : public ImplGLBindable
                {
                public:
                    virtual ~ImplGLProgramPipeline() noexcept;

                    bool Bind();

                    bool     Attach(GLbitfield shaderStages, ImplGLProgram* program);
                    bool HasAttachedProgram(GLbitfield shaderStage) const noexcept;

                    auto   Activate(GLbitfield shaderStage)->ImplGLProgram*;
                    void Deactivate();

                    bool HasActiveProgram()const noexcept;
                    auto GetActiveProgram()noexcept -> ImplGLProgram*;
                    
                    auto GetAttachedProgram(GLbitfield shaderStage)      noexcept ->      ImplGLProgram*;
                    auto GetAttachedProgram(GLbitfield shaderStage)const noexcept ->const ImplGLProgram*;
                protected:
                    ImplGLProgramPipeline(ImplGLResourceTable *table, ImplGLBindingPoint *bPoint) noexcept;
                    void AddShaderStage(GLbitfield shaderStage, bool isRequired = false)noexcept;
                    auto GetActiveProgram()const noexcept ->const ImplGLProgram*;
                    auto GetProgramSlot()noexcept -> ImplGLProgramSlot*;
                private:
                    struct  AttachState {
                        bool           isRequired = false;
                        ImplGLProgram*    program = nullptr;
                    };
                private:
                    std::unordered_map<GLbitfield, AttachState> m_AttachStates  = {};
                    ImplGLProgram*                              m_ActiveProgram = nullptr;
                };
                class ImplGLGraphicsProgramPipeline : public ImplGLProgramPipeline
                {
                public:
                    static auto New(ImplGLResourceTable* table, ImplGLBindingPoint* bPoint, ImplGLProgramSlot* slot)->ImplGLGraphicsProgramPipeline*;
                    virtual ~ImplGLGraphicsProgramPipeline() noexcept;
                protected:
                    ImplGLGraphicsProgramPipeline(ImplGLResourceTable* table, ImplGLBindingPoint* bPoint) noexcept;
                };
                class ImplGLComputeProgramPipeline : public ImplGLProgramPipeline
                {
                public:
                    static auto New(ImplGLResourceTable* table, ImplGLBindingPoint* bPoint, ImplGLProgramSlot* slot)->ImplGLComputeProgramPipeline*;
                    virtual ~ImplGLComputeProgramPipeline() noexcept;
                protected:
                    ImplGLComputeProgramPipeline(ImplGLResourceTable* table, ImplGLBindingPoint* bPoint) noexcept;
                };
            }

        }
    }
}
#endif
