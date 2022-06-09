#ifndef RTLIB_EXT_GLFW_GLFW_WINDOW_H
#define RTLIB_EXT_GLFW_GLFW_WINDOW_H
#include <RTLib/Core/Window.h>
#include <RTLib/Ext/GLFW/UuidDefinitions.h>
struct GLFWwindow;
namespace RTLib
{
	namespace Ext
	{
		namespace GLFW
		{
			class GLFWWindow : public RTLib::Core::Window
			{
			public:
				RTLIB_CORE_TYPE_OBJECT_DECLARE_DERIVED_METHOD(Ext::GLFW::GLFWWindow, Core::Window, RTLIB_TYPE_UUID_RTLIB_EXT_GLFW_GLFW_WINDOW);
				virtual ~GLFWWindow()noexcept;

				virtual void Destroy()noexcept override;
				virtual bool Resize(int width, int height) override;
				virtual void Hide() override;
				virtual void Show() override;
				virtual bool ShouldClose() override;
				virtual void SetTitle(const char* title)   override;
				virtual auto GetTitle()const noexcept -> std::string override;
				virtual auto GetSize()->Core::Extent2D override;
				virtual auto GetFramebufferSize()->Core::Extent2D override;
				virtual void SetSizeCallback(Core::PfnWindowSizeCallback callback) override;
				virtual void SetUserPointer(void* pCustomData) override;
				virtual auto GetUserPointer()const -> void* { return nullptr; }
				virtual void SetResizable(bool resizable) override;
				virtual void SetVisibility(bool visible) override;
				auto GetGLFWwindow()->GLFWwindow*;
			protected:
				GLFWWindow(GLFWwindow* glfwNativeHandle)noexcept;
				virtual void Initialize();
			private:
				struct Impl;
				std::unique_ptr<Impl> m_Impl;
			};
		}
	}
}
#endif
