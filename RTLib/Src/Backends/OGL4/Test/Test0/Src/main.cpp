#include <glad/gl.h>
#include <GLFW/glfw3.h>
#include <vector>
#include <memory>
#include <cassert>
#include <iostream>
#include <array>
#include <cmath>
#include <initializer_list>
#include <string>
#include <cstdlib>
#include <vector>
#include <tuple>
#define RTLIB_UTILS_MATH_CONSTANTS_PI 3.14159265
#define RTLIB_UTILS_CAMERA_MACRO_GETTER(TYPE,NAME, VARIABLE) \
    auto Get##NAME()const noexcept -> decltype(TYPE::VARIABLE) { return VARIABLE; }; \
    template<typename T> \
    auto Get##NAME##As()const noexcept -> T { static_assert(sizeof(T)==sizeof(TYPE::VARIABLE));\
        T res = T(); \
        auto v = Get##NAME(); std::memcpy(&res,&v, sizeof(v)); \
        return res;  \
    }
#define RTLIB_UTILS_CAMERA_MACRO_SETTER(TYPE, NAME, VARIABLE) \
    void Set##NAME(const decltype(TYPE::VARIABLE)& v) noexcept { VARIABLE = v; }; \
    template<typename T> \
    void Set##NAME##As(const T& v) noexcept { \
        static_assert(sizeof(T) == sizeof(VARIABLE)); \
        std::memcpy(&VARIABLE, &v, sizeof(v)); \
    }
#define RTLIB_UTILS_CAMERA_MACRO_GETTER_AND_SETTER(TYPE, NAME,VARIABLE) \
    RTLIB_UTILS_CAMERA_MACRO_GETTER(TYPE, NAME,VARIABLE); \
    RTLIB_UTILS_CAMERA_MACRO_SETTER(TYPE, NAME,VARIABLE)

namespace RTLib
{
    
    namespace Utils
    {
        template<typename T, size_t N>
        inline constexpr auto Add(const std::array<T, N>& a, const std::array<T, N>& b)->std::array<T, N>
        {
            std::array<T, N> c;
            for (int i = 0; i < N; ++i) {
                c[i] = a[i] + b[i];
            }
            return c;
        }
        template<typename T, size_t N>
        inline constexpr auto Sub(const std::array<T, N>& a, const std::array<T, N>& b)->std::array<T, N>
        {
            std::array<T, N> c;
            for (int i = 0; i < N; ++i) {
                c[i] = a[i] - b[i];
            }
            return c;
        }
        template<typename T, size_t N>
        inline constexpr auto Div(const std::array<T, N>& a, const T& b)->std::array<T, N>
        {
            std::array<T, N> c;
            for (int i = 0; i < N; ++i) {
                c[i] = a[i] / b;
            }
            return c;
        }
        template<typename T, size_t N>
        inline constexpr auto Div(const std::array<T, N>& a, const std::array<T, N>& b)->std::array<T, N>
        {
            std::array<T, N> c;
            for (int i = 0; i < N; ++i) {
                c[i] = a[i] / b[i];
            }
            return c;
        }
        template<typename T, size_t N>
        inline constexpr auto Mul(const std::array<T, N>& a, const std::array<T, N>& b)->std::array<T, N>
        {
            std::array<T, N> c;
            for (int i = 0; i < N; ++i) {
                c[i] = a[i] * b[i];
            }
            return c;
        }
        template<typename T, size_t N>
        inline constexpr auto Mul(const std::array<T, N>& a, const T& b)->std::array<T, N>
        {
            std::array<T, N> c;
            for (int i = 0; i < N; ++i) {
                c[i] = a[i] * b;
            }
            return c;
        }
        template<typename T, size_t N>
        inline constexpr auto LenSqr(const std::array<T, N>& a) -> T {
            T res = 0;

            for (int i = 0; i < N; ++i) {
                res += a[i] * a[i];
            }
            return res;
        }
        template<typename T, size_t N>
        inline constexpr auto Len(const std::array<T, N>& a) -> T {
            return std::sqrt(LenSqr(a));
        }
        template<typename T, size_t N>
        inline constexpr auto Normalize(const std::array<T, N>& a) -> std::array<T, N> {
            return Mul(a,std::sqrt(LenSqr(a)));
        }
        template<typename T>
        inline constexpr auto Cross(const std::array<T, 3>& a, const std::array<T, 3>& b)->std::array<T, 3>{
            return std::array<T, 3>{
                a[1] * b[2] - a[2] * b[1],
                a[2] * b[0] - a[0] * b[2],
                a[0] * b[1] - a[1] * b[0]
            };
        }
        class Camera
        {
            std::array<float, 3> m_Eye;
            std::array<float, 3> m_LookAt;
            std::array<float, 3> m_Vup;
            float m_FovY;
            float m_Aspect;

        public:
            Camera() noexcept : m_Eye{}, m_LookAt{}, m_Vup{}, m_FovY{}, m_Aspect{} {}
            Camera(
                const std::array<float, 3>& eye,
                const std::array<float, 3>& lookAt,
                const std::array<float, 3>& vup,
                const float fovY,
                const float aspect) noexcept
                : m_Eye{ eye },
                m_LookAt{ lookAt },
                m_Vup{ vup },
                m_FovY{ fovY },
                m_Aspect{ aspect }
            {}
            //Direction
            inline std::array<float, 3> GetDirection() const noexcept
            {
                auto lens = RTLib::Utils::Sub(m_LookAt, m_Eye);
                auto len  = RTLib::Utils::Len(lens);
                return RTLib::Utils::Div(lens, std::array<float, 3>{len, len, len});
            }
            inline void SetDirection(const std::array<float, 3>& direction) noexcept
            {
                auto lens  = RTLib::Utils::Sub(m_LookAt, m_Eye);
                auto d_len = RTLib::Utils::Len(direction);
                auto l_len = RTLib::Utils::Len(lens);
                m_Eye = RTLib::Utils::Sub(m_Eye, RTLib::Utils::Mul(direction, std::array<float, 3>{l_len / d_len, l_len / d_len, l_len / d_len}));
            }
            //Get And Set
            RTLIB_UTILS_CAMERA_MACRO_GETTER_AND_SETTER(Camera, Eye, m_Eye);
            RTLIB_UTILS_CAMERA_MACRO_GETTER_AND_SETTER(Camera, LookAt, m_LookAt);
            RTLIB_UTILS_CAMERA_MACRO_GETTER_AND_SETTER(Camera, Vup, m_Vup);
            RTLIB_UTILS_CAMERA_MACRO_GETTER_AND_SETTER(Camera, FovY, m_FovY);
            RTLIB_UTILS_CAMERA_MACRO_GETTER_AND_SETTER(Camera, Aspect, m_Aspect);
            //getUVW
            void GetUVW(std::array<float, 3>& u, std::array<float, 3>& v, std::array<float, 3>& w) const noexcept {
                w = Sub(m_LookAt, m_Eye);
                //front
                //u = rtlib::normalize(rtlib::cross(w,m_Vup));
                u = Normalize(Cross(m_Vup, w));
                v = Normalize(Cross(w, u));
                auto vlen = Len(w) * std::tanf(RTLIB_UTILS_MATH_CONSTANTS_PI * m_FovY / 360.0f);
                auto ulen = vlen * m_Aspect;
                u[0] *= ulen;
                u[1] *= ulen;
                u[2] *= ulen;
                v[0] *= vlen;
                v[1] *= vlen;
                v[2] *= vlen;
            }
            std::tuple<std::array<float, 3>, std::array<float, 3>, std::array<float, 3>> GetUVW() const noexcept {
                std::tuple<std::array<float, 3>, std::array<float, 3>, std::array<float, 3>> uvw;
                this->GetUVW(std::get<0>(uvw), std::get<1>(uvw), std::get<2>(uvw));
                return uvw;
            }
        };
        enum class CameraMovement : uint8_t
        {
            eForward = 0,
            eBackward = 1,
            eLeft = 2,
            eRight = 3,
            eUp = 4,
            eDown = 5,
        };
        struct     CameraController
        {
            inline static constexpr float defaultYaw = -90.0f;
            inline static constexpr float defaultPitch = 0.0f;
            inline static constexpr float defaultSpeed = 1.0f;
            inline static constexpr float defaultSensitivity = 0.025f;
            inline static constexpr float defaultZoom = 45.0f;

        private:
            std::array<float, 3> m_Position;
            std::array<float, 3> m_Front;
            std::array<float, 3> m_Up;
            std::array<float, 3> m_Right;
            float m_Yaw;
            float m_Pitch;
            float m_MovementSpeed;
            float m_MouseSensitivity;
            float m_Zoom;

        public:
            CameraController(
                const std::array<float, 3>& position = std::array<float, 3>{0.0f, 0.0f, 0.0f},
                const std::array<float, 3>& up = std::array<float, 3>{0.0f, 1.0f, 0.0f},
                float yaw = defaultYaw,
                float pitch = defaultPitch,
                float sensitivity = defaultSensitivity,
                float movementSpeed = defaultSpeed,
                float zoom = defaultZoom) noexcept : m_Position{ position },
                m_Up{ up },
                m_Yaw{ yaw },
                m_Pitch{ pitch },
                m_MouseSensitivity{ sensitivity },
                m_MovementSpeed{ movementSpeed },
                m_Zoom{ zoom }
            {
                UpdateCameraVectors();
            }
            void SetCamera(const Camera& camera) noexcept
            {
                m_Position = camera.GetEye();
                m_Front = Sub(camera.GetLookAt(), m_Position);
                m_Up = camera.GetVup();
                m_Right = Normalize(Cross(m_Up, m_Front));
            }
            auto GetCamera(float fovY, float aspect) const noexcept -> Camera
            {
                return Camera(m_Position, Add(m_Position, m_Front), m_Up, fovY, aspect);
            }
            auto GetCamera(float aspect)const noexcept->Camera {
                return GetCamera(m_Zoom, aspect);
            }
            void ProcessKeyboard(CameraMovement mode, float deltaTime) noexcept
            {
                float velocity = m_MovementSpeed * deltaTime;
                if (mode == CameraMovement::eForward)
                {
                    m_Position[0] += m_Front[0] * velocity;
                    m_Position[1] += m_Front[1] * velocity;
                    m_Position[2] += m_Front[2] * velocity;
                }
                if (mode == CameraMovement::eBackward)
                {
                    m_Position[0] -= m_Front[0] * velocity;
                    m_Position[1] -= m_Front[1] * velocity;
                    m_Position[2] -= m_Front[2] * velocity;

                }
                if (mode == CameraMovement::eLeft)
                {
                    m_Position[0] += m_Right[0] * velocity;
                    m_Position[1] += m_Right[1] * velocity;
                    m_Position[2] += m_Right[2] * velocity;
                }
                if (mode == CameraMovement::eRight)
                {
                    m_Position[0] -= m_Right[0] * velocity;
                    m_Position[1] -= m_Right[1] * velocity;
                    m_Position[2] -= m_Right[2] * velocity;
                }
                if (mode == CameraMovement::eUp)
                {
                    m_Position[0] -= m_Up[0] * velocity;
                    m_Position[1] -= m_Up[1] * velocity;
                    m_Position[2] -= m_Up[2] * velocity;
                }
                if (mode == CameraMovement::eDown)
                {
                    m_Position[0] += m_Up[0] * velocity;
                    m_Position[1] += m_Up[1] * velocity;
                    m_Position[2] += m_Up[2] * velocity;
                }

            }
            void ProcessMouseMovement(float xoffset, float yoffset, bool constrainPitch = true) noexcept
            {
                xoffset *= m_MouseSensitivity;
                yoffset *= m_MouseSensitivity;
                m_Yaw -= xoffset;
                m_Pitch += yoffset;
                if (constrainPitch)
                {
                    if (m_Pitch > 89.0f)
                    {
                        m_Pitch = 89.0f;
                    }
                    if (m_Pitch < -89.0f)
                    {
                        m_Pitch = -89.0f;
                    }
                }
                UpdateCameraVectors();
            }
            void ProcessMouseScroll(float yoffset) noexcept
            {
                float next_zoom = m_Zoom - yoffset;
                if (next_zoom >= 1.0f && next_zoom <= 45.0f)
                    m_Zoom = next_zoom;
                if (next_zoom <= 1.0f)
                    m_Zoom = 1.0f;
                if (next_zoom >= 45.0f)
                    m_Zoom = 45.0f;
            }

            RTLIB_UTILS_CAMERA_MACRO_GETTER_AND_SETTER(CameraController, Position, m_Position);
            RTLIB_UTILS_CAMERA_MACRO_GETTER_AND_SETTER(CameraController, Up, m_Up);
            RTLIB_UTILS_CAMERA_MACRO_GETTER_AND_SETTER(CameraController, Yaw, m_Yaw);
            RTLIB_UTILS_CAMERA_MACRO_GETTER_AND_SETTER(CameraController, Pitch, m_Pitch);
            RTLIB_UTILS_CAMERA_MACRO_GETTER_AND_SETTER(CameraController, MouseSensitivity, m_MouseSensitivity);
            RTLIB_UTILS_CAMERA_MACRO_GETTER_AND_SETTER(CameraController, MovementSpeed, m_MovementSpeed);
            RTLIB_UTILS_CAMERA_MACRO_GETTER_AND_SETTER(CameraController, Zoom, m_Zoom);
        private:
            // Calculates the front vector from the Camera's (updated) Euler Angles
            void UpdateCameraVectors() noexcept
            {
                // Calculate the new Front vector
                std::array<float, 3> front = {};
                float yaw   = RTLIB_UTILS_MATH_CONSTANTS_PI * (m_Yaw) / 180.0f;
                float pitch = RTLIB_UTILS_MATH_CONSTANTS_PI * (m_Pitch) / 180.0f;
                front[0] = cos(yaw) * cos(pitch);
                front[1] = sin(pitch);
                front[2] = sin(yaw) * cos(pitch);
                m_Front = Normalize(front);
                m_Right = Normalize(Cross(m_Up, m_Front));
            }
        };

    }
}
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

struct Uniform
{
	float model[16];
	float view [16];
	float proj [16];
};

static inline constexpr char vsSources[] = 
R"(#version 460 core
layout(location = 0) in vec4 position;
layout(std140) uniform Uniforms{
	mat4 model;
	mat4 view;
	mat4 proj;
};
out vec2 texCoord;
void main(){
	gl_Position = position;
	texCoord = (position.xy+vec2(1.0f))/2.0f;
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
	glfwInit();
	glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_API);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GLFW_TRUE);
	glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
	auto glfwWindow46 = std::unique_ptr<GLFWwindow,void(*)(GLFWwindow*)>(glfwCreateWindow(800, 600, "version46", nullptr, nullptr),glfwDestroyWindow);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	
	glfwMakeContextCurrent(glfwWindow46.get());
	auto gladContext46 = std::make_unique<GladGLContext>();
	gladLoadGLContext(gladContext46.get(), glfwGetProcAddress);

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
	gladContext46->NamedBufferData(vbo, sizeof(vertices), vertices, GL_STATIC_DRAW);
	gladContext46->NamedBufferData(ibo, sizeof(indices) , indices , GL_STATIC_DRAW);
	gladContext46->NamedBufferData(ubo, sizeof(Uniform) , nullptr , GL_STATIC_DRAW);

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

    auto camera = RTLib::Utils::CameraController();
    camera.GetCamera(8.0f/6.0f).

	bool isWindow46Closed = false;
	bool isWindow33Closed = false;
	glfwShowWindow(glfwWindow46.get());

	while (true) {
		if (!isWindow46Closed) {
			glfwMakeContextCurrent(glfwWindow46.get());
			gladContext46->ClearColor(1.0f, 0.0f, 0.0f, 1.0f);
			gladContext46->Clear(GL_COLOR_BUFFER_BIT);
			gladContext46->UseProgram(mainProgram);
			gladContext46->BindVertexArray(mainVao);
			gladContext46->DrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT,nullptr);
			glfwSwapBuffers(glfwWindow46.get());
		}
		glfwPollEvents();
		if (!isWindow46Closed) {
			isWindow46Closed = glfwWindowShouldClose(glfwWindow46.get()) == GLFW_TRUE;
		}
		if (isWindow46Closed) {
			glfwHideWindow(glfwWindow46.get());
		}
		if (isWindow46Closed) {
			break;
		}
	}

	gladContext46->DeleteVertexArrays(kVaoIndexCnt, glVaoResources);
	gladContext46->DeleteBuffers(kBufIndexCnt, glBufResources);
	gladContext46->DeleteProgram(mainProgram);

	gladContext46.reset();
	glfwWindow46.reset();
	glfwTerminate();
	return 0;
}