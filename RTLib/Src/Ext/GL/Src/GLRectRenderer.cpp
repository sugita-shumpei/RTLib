#include <RTLib/Ext/GL/GLRectRenderer.h>
#include <RTLib/Ext/GL/GLVertexArray.h>
#include <RTLib/Ext/GL/GLBuffer.h>
#include <RTLib/Ext/GL/GLTexture.h>
#include <RTLib/Ext/GL/GLProgram.h>
#include <RTLib/Ext/GL/GLShader.h>
#include <string>
#include <memory>
static inline constexpr char screenVSSource[] =
R"(#version 330 core
#extension GL_ARB_separate_shader_objects : enable
layout(location = 0) in vec3  vertexInPosition;
layout(location = 0) out vec2 vertexOutTexCoord;
out gl_PerVertex {
    vec4 gl_Position;
};
void main(){
    gl_Position = vec4(vertexInPosition, 1.0);
    vertexOutTexCoord = vec2(1.0f+vertexInPosition.x,1.0f-vertexInPosition.y)/2.0f;
})";
static inline constexpr char screenFSSource[] =
R"(#version 330 core
#extension GL_ARB_separate_shader_objects : enable
layout(location = 0) in vec2 vertexOutTexCoord;
layout(location = 0) out vec4 fragColor;
uniform sampler2D smp;
void main() {
	fragColor = texture2D(smp, vertexOutTexCoord);
})";
struct  RTLib::Ext::GL::GLRectRenderer::Impl
{
	GLContext* context = nullptr;
	std::unique_ptr<GLProgram> program = nullptr;
	std::unique_ptr<GLVertexArray> vao = nullptr;
	std::unique_ptr<GLBuffer> vbo = nullptr;
	std::unique_ptr<GLBuffer> ibo = nullptr;
	GLint texLoc = -1;
};
void RTLib::Ext::GL::GLRectRenderer::Destroy() noexcept
{
	if (!m_Impl) { return; }
	m_Impl->program.reset();
	m_Impl->vao->Destroy();
	m_Impl->vao.reset();
	m_Impl->vbo->Destroy();
	m_Impl->vbo.reset();
	m_Impl->ibo->Destroy();
	m_Impl->ibo.reset();
	m_Impl->texLoc = -1;
}
void RTLib::Ext::GL::GLRectRenderer::DrawTexture(GLTexture* texture)
{
	m_Impl->context->SetProgram(m_Impl->program.get());
	m_Impl->context->SetTexture(0, texture);
	m_Impl->program->SetUniformImageUnit(m_Impl->texLoc, 0);
	m_Impl->context->SetVertexArray(m_Impl->vao.get());
	m_Impl->context->DrawElements(GLDrawMode::eTriangles, GLIndexFormat::eUInt32, 6, 0);
}
RTLib::Ext::GL::GLRectRenderer::GLRectRenderer(GLContext* ctx)noexcept : m_Impl{ new Impl() } {
	m_Impl->context = ctx;
}

auto RTLib::Ext::GL::GLRectRenderer::New(GLContext* ctx) -> GLRectRenderer*
{
	if (!ctx) { return nullptr; }
	auto renderer = new GLRectRenderer(ctx);
	constexpr float    rectVertices[12] = {
		-1.0f,-1.0f,0.0f,
		 1.0f,-1.0f,0.0f,
		 1.0f, 1.0f,0.0f,
		-1.0f, 1.0f,0.0f
	};
	constexpr uint32_t rectIndices[6] = {
		0,1,2,2,3,0
	};
	{
		auto bufDesc = GLBufferCreateDesc();
		bufDesc.usage = GLBufferUsageVertex;
		bufDesc.access = GLMemoryPropertyDefault;
		bufDesc.size = sizeof(rectVertices);
		bufDesc.pData = rectVertices;
		renderer->m_Impl->vbo = std::unique_ptr<GLBuffer>(ctx->CreateBuffer(bufDesc));
		renderer->m_Impl->vbo->SetName("RectRenderer.VBO");
	}
	{
		auto bufDesc = GLBufferCreateDesc();
		bufDesc.usage = GLBufferUsageIndex;
		bufDesc.access = GLMemoryPropertyDefault;
		bufDesc.size = sizeof(rectIndices);
		bufDesc.pData = rectIndices;
		renderer->m_Impl->ibo = std::unique_ptr<GLBuffer>(ctx->CreateBuffer(bufDesc));
		renderer->m_Impl->ibo->SetName("RectRenderer.IBO");
	}
	{
		renderer->m_Impl->vao = std::unique_ptr<GLVertexArray>(ctx->CreateVertexArray());
		renderer->m_Impl->vao->SetName("RectRenderer.VAO");
		renderer->m_Impl->vao->SetVertexAttribBinding(0, 0);
		renderer->m_Impl->vao->SetVertexAttribFormat(0, GLVertexFormat::eFloat32x3, false, 0);
		renderer->m_Impl->vao->SetVertexBuffer(0, renderer->m_Impl->vbo.get(), sizeof(float) * 3, 0);
		renderer->m_Impl->vao->SetIndexBuffer(renderer->m_Impl->ibo.get());
		renderer->m_Impl->vao->Enable();
	}
	renderer->m_Impl->program = std::unique_ptr<GLProgram>(ctx->CreateProgram());
	{
		std::string log;

		auto screenVS = std::unique_ptr<GLShader>(ctx->CreateShader(GLShaderStageVertex));
		screenVS->ResetSourceGLSL(std::vector<char>(std::begin(screenVSSource), std::end(screenVSSource)));
		if (!screenVS->Compile(log)) {
			std::cerr << log << std::endl;
		}

		auto screenFS = std::unique_ptr<GLShader>(ctx->CreateShader(GLShaderStageFragment));
		screenFS->ResetSourceGLSL(std::vector<char>(std::begin(screenFSSource), std::end(screenFSSource)));
		if (!screenFS->Compile(log)) {
			std::cerr << log << std::endl;
		}

		renderer->m_Impl->program->AttachShader(screenVS.get());
		renderer->m_Impl->program->AttachShader(screenFS.get());

		if (!renderer->m_Impl->program->Link(log)) {
			std::cerr << log << std::endl;
		}

	}
	renderer->m_Impl->texLoc  = renderer->m_Impl->program->GetUniformLocation("smp");
	renderer->m_Impl->program->SetUniformImageUnit(renderer->m_Impl->texLoc, 0);
	return renderer;
}

RTLib::Ext::GL::GLRectRenderer::~GLRectRenderer() noexcept {}
