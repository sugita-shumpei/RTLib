#ifndef RTLIB_BACKENDS_GLFW_GLFW_ENTRY_H
#define RTLIB_BACKENDS_GLFW_GLFW_ENTRY_H
#include <memory>
namespace RTLib
{
	namespace Window {
		struct WindowDesc;
		class  Window;
	}
	namespace Inputs {
		class Keyboard;
	}
	namespace Backends
	{
		namespace Glfw
		{
			using PfnGlProc         = void(*)(void);
			using PfnGetProcAddress = auto(*)(const char*)->PfnGlProc;
			class Entry
			{
				Entry() noexcept;
			public:
				static auto Handle() -> Entry&
				{
					static auto entry = Entry();
					return entry;
				}
				Entry(const Entry&) = delete;
				Entry& operator=(const Entry&) = delete;
				Entry(Entry&&) = delete;
				Entry& operator=(Entry&&) = delete;
				~Entry() noexcept;

				void PollEvents() noexcept;
				void WaitEvents() noexcept;

				auto CreateWindow(const RTLib::Window::WindowDesc& desc)const->RTLib::Window::Window*;
				auto CreateWindowUnique(const RTLib::Window::WindowDesc& desc)const->std::unique_ptr<RTLib::Window::Window>;

				auto GetCurrentWindow()const noexcept  -> RTLib::Window::Window*;
				void SetCurrentWindow(RTLib::Window::Window*  window)noexcept;

				auto GetGlobalKeyboard()const noexcept -> RTLib::Inputs::Keyboard*;
				auto GetWindowKeyboard(RTLib::Window::Window* window)const noexcept -> RTLib::Inputs::Keyboard*;

				PfnGetProcAddress GetProcAddress;
			private:
				struct Impl;
				std::unique_ptr<Impl> m_Impl;
			};
		}
	}
}
#endif