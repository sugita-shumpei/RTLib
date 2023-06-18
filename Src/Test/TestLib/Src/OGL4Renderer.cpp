#include <TestLib/OGL4Renderer.h>
#include <glm/vec4.hpp>
#include <condition_variable>
#include <unordered_map>
#include <stdexcept>
#include <vector>
#include <iostream>
static constexpr glm::vec4 vertices[4] =
{
	{-1.0f,-1.0f,0.0f,0.0f},
	{-1.0f,+1.0f,0.0f,1.0f},
	{+1.0f,+1.0f,1.0f,1.0f},
	{+1.0f,-1.0f,1.0f,0.0f}
};
static constexpr uint32_t indices[6] =
{
	0,1,2,
	2,3,0
};
inline static constexpr char vsSource[] =
"#version 460 core\n"
"layout(location = 0) in vec2 position;\n"
"layout(location = 1) in vec2 texCoord;\n"
"out vec2 fragCoord;\n"
"void main()\n"
"{\n"
"	gl_Position = vec4(position,1.0,1.0); \n"
"	fragCoord = texCoord; \n"
"}\n";

inline static constexpr char fsSource[] =
"#version 460 core\n"
"in vec2 fragCoord;\n"
"layout(location = 0) out vec4 fragColor;\n"
"uniform sampler2D tex;\n"
"void main()\n"
"{\n"
"	fragColor = texture(tex,fragCoord); \n"
"}\n";

TestLib::OGL4Renderer:: OGL4Renderer(GLFWwindow* window)
	:m_Window{window},m_GladContext{std::make_unique<GladGLContext>()}
{
	glfwGetFramebufferSize(m_Window, &m_FbWidth, &m_FbHeight);
	glfwMakeContextCurrent(m_Window);
	if (!gladLoadGLContext(m_GladContext.get(), (GLADloadfunc)glfwGetProcAddress)) {
		throw std::runtime_error("Failed To Init Renderer!");
	}
}

TestLib::OGL4Renderer::~OGL4Renderer()
{
	m_GladContext.reset();
}

void TestLib::OGL4Renderer::init()
{
	glfwMakeContextCurrent(m_Window);
	init_vbo();
	init_ibo();
	init_pbo();
	init_vao();
	init_tex();
	init_prg();
}

void TestLib::OGL4Renderer::free()
{
	glfwMakeContextCurrent(m_Window);
	free_prg();
	free_tex();
	free_vao();
	free_pbo();
	free_ibo();
	free_vbo();
}

void TestLib::OGL4Renderer::render()
{
	glfwMakeContextCurrent(m_Window);
	m_GladContext->Clear(GL_COLOR_BUFFER_BIT);

	m_GladContext->ClearColor(0, 0, 0, 0);
	m_GladContext->Viewport(0, 0, m_FbWidth, m_FbHeight);

	m_GladContext->UseProgram(m_Prg);

	m_GladContext->ActiveTexture(GL_TEXTURE0 + 0);
	m_GladContext->BindTexture(GL_TEXTURE_2D, m_Tex);

	m_GladContext->BindBuffer(GL_PIXEL_UNPACK_BUFFER, m_Pbo);
	m_GladContext->TexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, m_FbWidth, m_FbHeight, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
	m_GladContext->BindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

	m_GladContext->Uniform1i(m_TexLoc, 0);

	//std::vector<uchar4> pixels(width * height, make_uchar4(255, 255, 255, 255));

	m_GladContext->BindVertexArray(m_Vao);
	m_GladContext->DrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, nullptr);
}

auto TestLib::OGL4Renderer::get_window() const noexcept -> GLFWwindow* { return m_Window; }

auto TestLib::OGL4Renderer::get_pbo() const noexcept -> GLuint { return m_Pbo; }

void TestLib::OGL4Renderer::set_current() { glfwMakeContextCurrent(m_Window); }

void TestLib::OGL4Renderer::init_vbo() 
{
	m_GladContext->GenBuffers(1, &m_Vbo);
	m_GladContext->BindBuffer(GL_ARRAY_BUFFER, m_Vbo);
	m_GladContext->BufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
}

void TestLib::OGL4Renderer::free_vbo() 
{
	m_GladContext->DeleteBuffers(1, &m_Vbo);
	m_Vbo = 0;
}

void TestLib::OGL4Renderer::init_ibo() 
{
	m_GladContext->GenBuffers(1, &m_Ibo);
	m_GladContext->BindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_Ibo);
	m_GladContext->BufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

}
void TestLib::OGL4Renderer::free_ibo() 
{
	m_GladContext->DeleteBuffers(1, &m_Ibo);
	m_Ibo = 0;
}

void TestLib::OGL4Renderer::init_vao() 
{
	m_GladContext->GenVertexArrays(1, &m_Vao);
	m_GladContext->BindVertexArray(m_Vao);

	m_GladContext->BindBuffer(GL_ARRAY_BUFFER, m_Vbo);

	m_GladContext->EnableVertexAttribArray(0);
	m_GladContext->VertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(glm::vec4), static_cast<float*>(nullptr) + 0);

	m_GladContext->EnableVertexAttribArray(1);
	m_GladContext->VertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, sizeof(glm::vec4), static_cast<float*>(nullptr) + 2);

	m_GladContext->BindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_Ibo);

	m_GladContext->BindVertexArray(0);
	m_GladContext->BindBuffer(GL_ARRAY_BUFFER, 0);
	m_GladContext->BindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

}
void TestLib::OGL4Renderer::free_vao() 
{
	m_GladContext->DeleteVertexArrays(1, &m_Vao);
	m_Vao = 0;
}

void TestLib::OGL4Renderer::init_pbo() 
{
	m_GladContext->GenBuffers(1, &m_Pbo);
	m_GladContext->BindBuffer(GL_PIXEL_UNPACK_BUFFER, m_Pbo);
	m_GladContext->BufferData(GL_PIXEL_UNPACK_BUFFER, m_FbWidth*m_FbHeight*4, nullptr, GL_STREAM_DRAW);
	m_GladContext->BindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
}
void TestLib::OGL4Renderer::free_pbo() 
{
	m_GladContext->DeleteBuffers(1, &m_Pbo);
	m_Pbo = 0;
}

void TestLib::OGL4Renderer::init_tex() 
{
	m_GladContext->GenTextures(1, &m_Tex);
	m_GladContext->BindTexture(GL_TEXTURE_2D, m_Tex);
	m_GladContext->TexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, m_FbWidth, m_FbHeight, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
	m_GladContext->TexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	m_GladContext->TexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	m_GladContext->TexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	m_GladContext->TexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
}
void TestLib::OGL4Renderer::free_tex() 
{
	m_GladContext->DeleteTextures(1, &m_Tex);
	m_Tex = 0;
}

void TestLib::OGL4Renderer::init_prg() 
{
	m_Prg = m_GladContext->CreateProgram();
	GLint len;
	std::vector<char> log;
	GLuint vso = m_GladContext->CreateShader(GL_VERTEX_SHADER);
	GLuint fso = m_GladContext->CreateShader(GL_FRAGMENT_SHADER);
	{
		const char* vsSourcePtr = vsSource;
		m_GladContext->ShaderSource(vso, 1, &vsSourcePtr, nullptr);
		m_GladContext->CompileShader(vso);
		GLint res;
		m_GladContext->GetShaderiv(vso, GL_COMPILE_STATUS, &res);
		m_GladContext->GetShaderiv(vso, GL_INFO_LOG_LENGTH, &len);

		log.resize(len + 1);
		m_GladContext->GetShaderInfoLog(vso, len, &len, log.data());
		log.resize(len);

		if (len > 0) {
			std::cout << log.data() << std::endl;
		}

		if (res != GL_TRUE)
		{
			throw std::runtime_error("Failed To Compile Vert Shader!");
		}
	}
	{
		const char* fsSourcePtr = fsSource;
		m_GladContext->ShaderSource(fso, 1, &fsSourcePtr, nullptr);
		m_GladContext->CompileShader(fso);
		GLint res;
		m_GladContext->GetShaderiv(fso, GL_COMPILE_STATUS, &res);
		m_GladContext->GetShaderiv(fso, GL_INFO_LOG_LENGTH, &len);

		log.resize(len + 1);
		m_GladContext->GetShaderInfoLog(fso, len, &len, log.data());
		log.resize(len);

		if (len > 0) {
			std::cout << log.data() << std::endl;
		}

		if (res != GL_TRUE)
		{
			throw std::runtime_error("Failed To Compile Frag Shader!");
		}
	}
	{
		m_GladContext->AttachShader(m_Prg, vso);
		m_GladContext->AttachShader(m_Prg, fso);

		m_GladContext->LinkProgram(m_Prg);

		m_GladContext->DetachShader(m_Prg, vso);
		m_GladContext->DetachShader(m_Prg, fso);

		m_GladContext->DeleteShader(vso);
		m_GladContext->DeleteShader(fso);

		GLint res;
		m_GladContext->GetProgramiv(m_Prg, GL_LINK_STATUS, &res);
		m_GladContext->GetProgramiv(m_Prg, GL_INFO_LOG_LENGTH, &len);

		log.resize(len + 1);
		m_GladContext->GetProgramInfoLog(m_Prg, len, &len, log.data());
		log.resize(len);

		if (len > 0) {
			std::cout << log.data() << std::endl;
		}

		if (res != GL_TRUE)
		{
			m_GladContext->DeleteProgram(m_Prg);
			throw std::runtime_error("Failed To Link Program!");
		}
	}
	m_TexLoc = m_GladContext->GetUniformLocation(m_Prg,"tex");
}
void TestLib::OGL4Renderer::free_prg() 
{
	m_GladContext->DeleteProgram(m_Prg);
	m_Prg = 0;
	m_TexLoc = 0;
}

bool TestLib::OGL4Renderer::resize(int fb_width, int fb_height)
{
	if ((m_FbWidth != fb_width) || (m_FbHeight != fb_height))
	{
		m_FbWidth = fb_width; m_FbHeight = fb_height;

		m_GladContext->DeleteBuffers( 1, &m_Pbo);
		init_pbo();

		m_GladContext->DeleteTextures(1, &m_Tex);
		init_tex();

		return true;
	}
	return false;
}