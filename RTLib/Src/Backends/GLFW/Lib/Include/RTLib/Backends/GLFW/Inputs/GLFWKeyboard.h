#ifndef RTLIB_BACKENDS_GLFW_INPUTS_GLFW_KEYBOARD_H
#define RTLIB_BACKENDS_GLFW_INPUTS_GLFW_KEYBOARD_H
#include <RTLib/Inputs/Keyboard.h>
#include <memory>
namespace RTLib
{
	namespace Backends
	{
		namespace Glfw
		{
			class Entry;
			namespace Window {
				class Window;
			}
			namespace Inputs
			{
				class Keyboard : public RTLib::Inputs::Keyboard
				{
					friend class RTLib::Backends::Glfw::Entry;
					friend class RTLib::Backends::Glfw::Window::Window;
				public:
					Keyboard() noexcept:RTLib::Inputs::Keyboard() {}
					virtual ~Keyboard() noexcept {}
				};
			}
		}
	}
}
#endif