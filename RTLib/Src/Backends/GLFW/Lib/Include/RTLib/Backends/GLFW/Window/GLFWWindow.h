#ifndef RTLIB_BACKENDS_GLFW_WINDOW_GLFW_WINDOW_H
#define RTLIB_BACKENDS_GLFW_WINDOW_GLFW_WINDOW_H
#include <RTLib/Window/Window.h>
#include <memory>
namespace RTLib
{
    namespace Inputs {
        class Keyboard;
    }
	namespace Backends
	{
		namespace Glfw
		{
			class Entry;
			namespace Window
			{
				class Window : public RTLib::Window::Window
				{
                    friend class RTLib::Backends::Glfw::Entry;
				public:
                    Window(const RTLib::Window::WindowDesc& desc) noexcept;
                    virtual ~Window() noexcept;

                    virtual auto GetPosition() const noexcept ->std::array<int, 2> override;
                    virtual void SetPosition(const std::array<int, 2>& position) noexcept override;

                    virtual auto GetSize()const noexcept -> std::array<int, 2> override;
                    virtual void SetSize(const std::array<int, 2>& size)noexcept override;

                    virtual auto GetFramebufferSize()const noexcept -> std::array<int, 2> override;

                    virtual auto GetTitle()const noexcept -> std::string override;
                    virtual void SetTitle(const std::string& title)noexcept override;

                    virtual void Maximize() noexcept override;
                    virtual void Restore() noexcept override;

                    virtual void Focus() noexcept override;

                    virtual bool  ShouldClose() const noexcept override;
                    virtual void RequestClose() noexcept override;

                    virtual bool IsResizable() const noexcept override;
                    virtual void SetResizable(bool isResizable)noexcept override;

                    virtual bool IsVisible() const noexcept override;
                    virtual void Show() noexcept override;
                    virtual void Hide() noexcept override;

                    virtual void SwapBuffers() noexcept override;
                    virtual void SetCurrent() noexcept override;
                    virtual auto GetHandle()const noexcept -> void* override;
                private:
                    auto Internal_GetKeyboard()const noexcept -> RTLib::Inputs::Keyboard*;
                private:
                    struct Impl;
                    std::unique_ptr<Impl> m_Impl;
				};
			}
		}
	}
}
#endif