#ifndef RTLIB_CORE_WINDOW_H
#define RTLIB_CORE_WINDOW_H
#include <RTLib/Core/Common.h>
#include <RTLib/Core/BaseObject.h>
#include <RTLib/Core/UuidDefinitions.h>
#include <string>
namespace RTLib
{
	namespace Core {
		class Window;
		using PfnCursorPosCallback  = void(*)(Window* window, double  x, double   y);
		using PfnWindowSizeCallback = void(*)(Window* window, int width, int height);
		class Window :public BaseObject
		{
		public:
			RTLIB_CORE_TYPE_OBJECT_DECLARE_DERIVED_METHOD(Core::Window, Core::BaseObject, RTLIB_TYPE_UUID_RTLIB_CORE_WINDOW);
			virtual ~Window() noexcept{}
			
			virtual void Destroy()noexcept = 0;
			virtual bool Resize(int width, int height) = 0;
			virtual void Hide() = 0;
			virtual void Show() = 0;
			virtual bool ShouldClose() = 0;
			virtual void SetTitle(const char* title)   = 0;
			virtual auto GetTitle()const noexcept -> std::string = 0;
			virtual auto GetSize()->Core::Extent2D = 0;
			virtual auto GetFramebufferSize()->Core::Extent2D = 0;
			virtual void SetSizeCallback(PfnWindowSizeCallback callback) = 0;
			virtual void SetCursorPosCallback(PfnCursorPosCallback callback) = 0;
			virtual void SetUserPointer(void* pCustomData) = 0;
			virtual auto GetUserPointer()const -> void* { return nullptr; }
			virtual void SetResizable(bool resizable) = 0;
			virtual void SetVisibility(bool visible)     = 0;
			

			template<typename T>
			auto GetTypeUserPointer()const -> const T* { return reinterpret_cast<const T*>(GetUserPointer()); }
			template<typename T>
			auto GetTypeUserPointer() -> T* { return reinterpret_cast<T*>(GetUserPointer()); }
		};
	}
}
#endif
