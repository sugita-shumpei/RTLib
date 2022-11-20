#include <glad/gl.h>
#include <RTLib/Backends/GLFW/GLFWEntry.h>
#include <RTLib/Window/Window.h>
#include <RTLib/Inputs/Mouse.h>
#include <RTLib/Inputs/Keyboard.h>
#include <RTLib/Inputs/Cursor.h>
#include <RTLib/Camera/Camera.h>
#include <vector>
#include <iomanip>
#include <memory>
#include <cassert>
#include <iostream>
#include <array>
#include <unordered_map>
#include <chrono>
static inline constexpr float vertices[4 * 4] = {
	-1.0f,-1.0f,0.0f,1.0f,
	 1.0f,-1.0f,0.0f,1.0f,
	 1.0f, 1.0f,0.0f,1.0f,
	-1.0f, 1.0f,0.0f,1.0f
};

static inline constexpr uint32_t indices[6] = {
	0,1,2,
	2,3,0
};

struct Uniforms
{
	float model[16];
	float view [16];
	float proj [16];
};

static inline constexpr char vsSources[] = 
R"(#version 460 core
layout(location = 0) in vec4 position;
layout(std140,binding = 0) uniform Uniforms{
	mat4 model;
	mat4 view;
	mat4 proj;
} uniforms;
out vec2 texCoord;
void main(){
	gl_Position = uniforms.proj * uniforms.view * uniforms.model * position;
	texCoord    = (position.xy+vec2(1.0f))/2.0f;
}
)";

static inline constexpr char fsSources[] =
R"(#version 460 core
in vec2 texCoord;
layout(location = 0) out vec4 fragColor;
void main(){
	fragColor = vec4(texCoord.x,texCoord.y, 1.0f-dot(texCoord,vec2(1.0f))/2.0f,1.0f);
}
)";


int main(int argc, const char** argv)
{
	auto& entry = RTLib::Backends::Glfw::Entry::Handle();
	auto  window = std::unique_ptr<RTLib::Window::Window>();
	{
		auto desc = RTLib::Window::WindowDesc();
		desc.width = 800;
		desc.height = 600;
		desc.title = "Test";
		desc.isResizable = true;
		desc.graphics.flags = RTLib::Window::GraphicsFlagsOpenGL;
		desc.graphics.openGL.required = true;
		desc.graphics.openGL.isES = false;
		desc.graphics.openGL.isCoreProfile = true;
		desc.graphics.openGL.isForwardCompat = true;
		window = entry.CreateWindowUnique(desc);
	}

	entry.SetCurrentWindow(window.get());

	auto gladContext46 = std::make_unique<GladGLContext>();
	gladLoadGLContext(gladContext46.get(), entry.GetProcAddress);

	constexpr size_t kBufIndexMain= 0;
	constexpr size_t kPrgIndexCnt = 1;

	GLuint glPrgResources[kPrgIndexCnt] = {};
	auto& mainProgram = glPrgResources[kBufIndexMain];
	{
		mainProgram = gladContext46->CreateProgram();

		GLuint glShdVert = gladContext46->CreateShader(GL_VERTEX_SHADER)  ;
		GLuint glShdFrag = gladContext46->CreateShader(GL_FRAGMENT_SHADER);
		const char* vsHead = vsSources;
		gladContext46->ShaderSource(glShdVert, 1, &vsHead, nullptr);
		gladContext46->CompileShader(glShdVert);
		{
			std::vector<char> infoLog = {};
			int res = 0;
			gladContext46->GetShaderiv(glShdVert, GL_COMPILE_STATUS, &res);
			int len = 0;
			gladContext46->GetShaderiv(glShdVert, GL_INFO_LOG_LENGTH, &len);
			infoLog.resize(len+1);
			gladContext46->GetShaderInfoLog(glShdVert, infoLog.size(), &len, infoLog.data());
			infoLog.resize(len+1);
			if (len > 0) {
				std::cout << infoLog.data() << std::endl;
			}

			assert(res == GL_TRUE);
			gladContext46->AttachShader(mainProgram, glShdVert);
		}

		const char* fsHead = fsSources;
		gladContext46->ShaderSource(glShdFrag, 1, &fsHead, nullptr);
		gladContext46->CompileShader(glShdFrag);
		{
			std::vector<char> infoLog = {};
			int res = 0;
			gladContext46->GetShaderiv(glShdFrag, GL_COMPILE_STATUS, &res);
			int len = 0;
			gladContext46->GetShaderiv(glShdFrag, GL_INFO_LOG_LENGTH, &len);
			infoLog.resize(len+1);
			gladContext46->GetShaderInfoLog(glShdFrag, infoLog.size(), &len, infoLog.data());
			infoLog.resize(len+1);
			if (len > 0) {
				std::cout << infoLog.data() << std::endl;
			}

			assert(res == GL_TRUE);
			gladContext46->AttachShader(mainProgram, glShdFrag);

		}
		gladContext46->LinkProgram(mainProgram);
		{
			std::vector<char> infoLog = {};
			int res = 0;
			gladContext46->GetProgramiv(mainProgram, GL_LINK_STATUS, &res);
			int len = 0;
			gladContext46->GetProgramiv(mainProgram, GL_INFO_LOG_LENGTH, &len);
			infoLog.resize(len + 1);
			gladContext46->GetProgramInfoLog(mainProgram, infoLog.size(), &len, infoLog.data());
			infoLog.resize(len + 1);
			if (len > 0) {
				std::cout << infoLog.data() << std::endl;
			}

			assert(res == GL_TRUE);
		}
		gladContext46->DeleteShader(glShdVert);
		gladContext46->DeleteShader(glShdFrag);
	}

	constexpr size_t kBufIndexVbo = 0;
	constexpr size_t kBufIndexIbo = 1;
	constexpr size_t kBufIndexUbo = 2;
	constexpr size_t kBufIndexCnt = 3;

	GLuint glBufResources[kBufIndexCnt] = {};
	auto& vbo = glBufResources[kBufIndexVbo];
	auto& ibo = glBufResources[kBufIndexIbo];
	auto& ubo = glBufResources[kBufIndexUbo];

	gladContext46->GenBuffers(kBufIndexCnt, glBufResources);
	{
		gladContext46->BindBuffer(GL_ARRAY_BUFFER, vbo);
		gladContext46->BindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);
		gladContext46->BindBuffer(GL_UNIFORM_BUFFER, ubo);
	}
	gladContext46->NamedBufferStorage(vbo, sizeof(vertices), vertices, 0);
	gladContext46->NamedBufferStorage(ibo, sizeof(indices) , indices , 0);
	gladContext46->NamedBufferData(ubo, sizeof(Uniforms) , nullptr , GL_STATIC_DRAW);

	constexpr size_t kVaoIndexMain = 0;
	constexpr size_t kVaoIndexCnt  = 1;

	GLuint glVaoResources[kVaoIndexCnt] = {};
	auto& mainVao = glVaoResources[kVaoIndexMain];
	gladContext46->CreateVertexArrays(kVaoIndexCnt, glVaoResources);
	{
		gladContext46->EnableVertexArrayAttrib(mainVao, 0);
		gladContext46->VertexArrayAttribFormat(mainVao, 0, 4, GL_FLOAT, GL_FALSE, 0);
		gladContext46->VertexArrayAttribBinding(mainVao, 0, 0);
		gladContext46->VertexArrayVertexBuffer(mainVao, 0, vbo, 0, sizeof(float) * 4);
		gladContext46->VertexArrayElementBuffer(mainVao, ibo);
	}

    auto cameraController = RTLib::CameraController();
	cameraController.SetMovementSpeed(10.0f);
	cameraController.SetZoom(45.0f);
    {
		auto camera = cameraController.GetCamera((float)window->GetSize()[0] / (float)window->GetSize()[1]);
		auto uniforms = Uniforms();
		auto model = RTLib::CameraUtils::IdentityMatrix<float, 4>();
		auto view = camera.GetLookAtMatrixRH();
		auto proj = camera.GetPerspectiveMatrixLH(0.1f, 10.0f);
		std::memcpy(uniforms.model, &model, sizeof(model));
		std::memcpy(uniforms.view, &view, sizeof(view));
		std::memcpy(uniforms.proj, &proj, sizeof(proj));
		gladContext46->NamedBufferSubData(ubo, 0, sizeof(uniforms), &uniforms);
    }

	bool isWindow46Closed = false;
	bool isWindow33Closed = false;
	bool isSizeChanged    = false;
    bool isUpdated        = false;

	window->SetUserPointer(&isSizeChanged);
	window->SetSizeCallback([](RTLib::Window::Window* window, int width, int height) {
		std::cout << "[" << width << "," << height << "]\n";
		bool* isSizeChanged = (bool*)window->GetUserPointer();
		*isSizeChanged = true;
	});
	
	entry.GetWindowKeyboard(window.get())->SetCallback([](RTLib::Inputs::KeyCode code, unsigned int state, void* pUserData) {
		if (code == RTLib::Inputs::KeyCode::eW) {
			if (state & RTLib::Inputs::KeyStatePressed) {
				std::cout << "Window Pressed "; 
				if (state & RTLib::Inputs::KeyStateUpdated) {
					std::cout << " And Updated ";
				}
				std::cout << std::endl;
				
			}
			if (state & RTLib::Inputs::KeyStateReleased) {
				std::cout << "Window Released" << std::endl;
				if (state & RTLib::Inputs::KeyStateUpdated) {
					std::cout << " And Updated ";
				}
				std::cout << std::endl;
			}
		}
	});

	window->Show();
	float deltaTime = 0.0f;
	while (true) {
		auto beg = std::chrono::system_clock::now();
		if (!isWindow46Closed) {
			entry.SetCurrentWindow(window.get());
			if (isUpdated||isSizeChanged) {
				auto camera = cameraController.GetCamera((float)window->GetSize()[0] / (float)window->GetSize()[1]);
				{
					auto uniforms             = Uniforms();
					auto model = RTLib::CameraUtils::IdentityMatrix<float,4>();
					auto view  = camera.GetLookAtMatrixRH();
					auto proj  = camera.GetPerspectiveMatrixLH(0.1f, 10.0f);
					std::memcpy(uniforms.model, &model, sizeof(model));
					std::memcpy(uniforms.view, &view, sizeof(view));
					std::memcpy(uniforms.proj, &proj, sizeof(proj));
					gladContext46->NamedBufferSubData(ubo, 0, sizeof(uniforms), &uniforms);
				}
			}
			if (isSizeChanged) {
				gladContext46->Viewport(0, 0, window->GetFramebufferSize()[0], window->GetFramebufferSize()[1]);
			}
			gladContext46->ClearColor(1.0f, 0.0f, 0.0f, 1.0f);
			gladContext46->Clear(GL_COLOR_BUFFER_BIT);
			gladContext46->UseProgram(mainProgram);
			gladContext46->BindVertexArray(mainVao);
            gladContext46->BindBufferBase(GL_UNIFORM_BUFFER, 0, ubo);
			gladContext46->DrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT,nullptr);
			window->SwapBuffers();
		}
		isUpdated = false;
		isSizeChanged = false;
		entry.PollEvents();

		if (!isWindow46Closed) {
			isWindow46Closed = window->ShouldClose();
			if(isWindow46Closed) window->Hide();
		}
		if ( isWindow46Closed) {
			break;
		}
		auto end = std::chrono::system_clock::now();
		deltaTime = std::chrono::duration_cast<std::chrono::milliseconds>(end - beg).count() * 0.001f;

		if (entry.GetWindowKeyboard(window.get())->GetKey(RTLib::Inputs::KeyCode::eA)    & RTLib::Inputs::KeyStatePressed) {
			std::cout << cameraController.GetPosition()[0] << "-" << cameraController.GetPosition()[1] << "-" << cameraController.GetPosition()[2] << std::endl;
			cameraController.ProcessKeyboard(RTLib::CameraMovement::eLeft, deltaTime); isUpdated = true;
		}
		if (entry.GetWindowKeyboard(window.get())->GetKey(RTLib::Inputs::KeyCode::eD)    & RTLib::Inputs::KeyStatePressed) {
			std::cout << cameraController.GetPosition()[0] << "-" << cameraController.GetPosition()[1] << "-" << cameraController.GetPosition()[2] << std::endl;
			cameraController.ProcessKeyboard(RTLib::CameraMovement::eRight, deltaTime); isUpdated = true;
		}
		if (entry.GetWindowKeyboard(window.get())->GetKey(RTLib::Inputs::KeyCode::eW)    & RTLib::Inputs::KeyStatePressed) {
			std::cout << cameraController.GetPosition()[0] << "-" << cameraController.GetPosition()[1] << "-" << cameraController.GetPosition()[2] << std::endl;
			cameraController.ProcessKeyboard(RTLib::CameraMovement::eForward, deltaTime); isUpdated = true;
		}
		if (entry.GetWindowKeyboard(window.get())->GetKey(RTLib::Inputs::KeyCode::eS)    & RTLib::Inputs::KeyStatePressed) {
			std::cout << cameraController.GetPosition()[0] << "-" << cameraController.GetPosition()[1] << "-" << cameraController.GetPosition()[2] << std::endl;
			cameraController.ProcessKeyboard(RTLib::CameraMovement::eBackward, deltaTime); isUpdated = true;
		}
		if (entry.GetWindowKeyboard(window.get())->GetKey(RTLib::Inputs::KeyCode::eUp)   & RTLib::Inputs::KeyStatePressed) {
			std::cout << cameraController.GetPosition()[0] << "-" << cameraController.GetPosition()[1] << "-" << cameraController.GetPosition()[2] << std::endl;
			cameraController.ProcessKeyboard(RTLib::CameraMovement::eUp, deltaTime); isUpdated = true;
		}
		if (entry.GetWindowKeyboard(window.get())->GetKey(RTLib::Inputs::KeyCode::eDown) & RTLib::Inputs::KeyStatePressed) {
			std::cout << cameraController.GetPosition()[0] << "-" << cameraController.GetPosition()[1] << "-" << cameraController.GetPosition()[2] << std::endl;
			cameraController.ProcessKeyboard(RTLib::CameraMovement::eDown, deltaTime); isUpdated = true;
		}
	}

	gladContext46->DeleteVertexArrays(kVaoIndexCnt, glVaoResources);
	gladContext46->DeleteBuffers(kBufIndexCnt, glBufResources);
	gladContext46->DeleteProgram(mainProgram);

	gladContext46.reset();
	window.reset();
	return 0;
}