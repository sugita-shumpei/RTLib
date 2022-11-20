#ifndef RTLIB_BACKENDS_GLFW_INTERNALS_GLFW_INTERNALS_H
#define RTLIB_BACKENDS_GLFW_INTERNALS_GLFW_INTERNALS_H
#include <GLFW/glfw3.h>
namespace RTLib
{
	namespace Inputs {
		enum class KeyCode: unsigned int;
	}
	namespace Backends {
		namespace Glfw
		{
			namespace Internals
			{
				auto GetGlfwKeyCode(RTLib::Inputs::KeyCode) -> int;
				auto GetInptKeyCode(int keyCode)->RTLib::Inputs::KeyCode;
			}
		}
	}
}
#endif