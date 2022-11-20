#ifndef RTLIB_WINDOW_WINDOW_H
#define RTLIB_WINDOW_WINDOW_H
#include <string>
#include <array>
namespace RTLib
{
    namespace Window
    {
        enum GraphicsFlags: unsigned int
        {
            GraphicsFlagsUnknown= 0,
            GraphicsFlagsOpenGL = 1,
            GraphicsFlagsVulkan = 2,
        };
        struct WindowDesc {
            static inline constexpr int defaultVal = 0;
            int width = 0;
            int height = 0;
            int xpos = defaultVal;
            int ypos = defaultVal;
            std::string title = "";
            void* pUserData = nullptr;
            bool isResizable = false;
            bool isVisible = false;
            struct GraphicsDesc
            {
                unsigned int flags = GraphicsFlagsUnknown;
                struct GraphicsOpenGLDesc {
                    bool required = false;
                    bool isCoreProfile = false;
                    bool isES = false;
                    bool isForwardCompat = false;
                    int  majorVersion = defaultVal;
                    int  minorVersion = defaultVal;
                } openGL = {};
                struct GraphicsVulkanDesc {
                    bool required = false;
                } vulkan;
            } graphics = {};
            auto GraphicsOpenGL()noexcept -> GraphicsDesc::GraphicsOpenGLDesc& {
                if (!(graphics.flags& GraphicsFlagsOpenGL)) {
                    graphics.flags |= GraphicsFlagsOpenGL;
                    graphics.openGL = GraphicsDesc::GraphicsOpenGLDesc();
                }
                return graphics.openGL;
            }
            auto GraphicsVulkan()noexcept -> GraphicsDesc::GraphicsVulkanDesc& {
                if (!(graphics.flags & GraphicsFlagsVulkan)) {
                    graphics.flags |= GraphicsFlagsVulkan;
                    graphics.vulkan = GraphicsDesc::GraphicsVulkanDesc();
                }
                return graphics.vulkan;
            }
        };
        class Window
        {
        public:
            using SingleStateCallback = void(*)(Window* window, bool) ;
            using PositionCallback    = void(*)(Window* window, int   x,int   y);
            using SizeCallback        = void(*)(Window* window, int wid,int hei);
            using CloseCallback       = void(*)(Window* window);
        public:
            Window() noexcept {}
            virtual ~Window() noexcept {}

            virtual auto GetPosition() const noexcept ->std::array<int,2> = 0;
            virtual void SetPosition(const std::array<int,2>& position) noexcept = 0; 

            virtual auto GetSize()const noexcept -> std::array<int,2>   = 0;
            virtual void SetSize(const std::array<int,2>& size)noexcept = 0;

            virtual auto GetFramebufferSize()const noexcept -> std::array<int,2> = 0;

            virtual auto GetTitle()const noexcept -> std::string = 0;
            virtual void SetTitle(const std::string& title)noexcept = 0;

            virtual void Maximize() noexcept = 0;
            virtual void Restore () noexcept = 0;

            virtual void Focus() noexcept = 0;

            virtual bool  ShouldClose() const noexcept = 0;
            virtual void RequestClose() noexcept = 0;

            virtual bool IsResizable() const noexcept = 0;
            virtual void SetResizable(bool isResizable)noexcept = 0;

            virtual bool IsVisible() const noexcept = 0;
            virtual void Show() noexcept = 0;
            virtual void Hide() noexcept = 0;

            virtual void SwapBuffers() noexcept = 0;
            virtual auto GetHandle()const noexcept -> void* = 0;

            void SetPositionCallback(PositionCallback callback) noexcept{
                m_PositionCallback = callback;
            }
            auto GetPositionCallback() const noexcept -> SizeCallback{
                return m_PositionCallback;
            }

            void SetSizeCallback(SizeCallback callback)noexcept{
                m_SizeCallback = callback;
            }
            auto GetSizeCallback()const noexcept -> SizeCallback{
                return m_SizeCallback;
            }

            void SetFramebufferSizeCallback(SizeCallback callback)noexcept{
                m_FramebufferSizeCallback = callback;
            }
            auto GetFramebufferSizeCallback()const noexcept -> SizeCallback{
                return m_FramebufferSizeCallback;
            }
            
            void SetCloseCallback(CloseCallback callback) noexcept{ m_CloseCallback = callback;}
            auto GetCloseCallback() const noexcept -> CloseCallback { return m_CloseCallback;  }

            void SetMaximizeCallback(SingleStateCallback callback) noexcept {
                m_MaximizeCallback = callback;
            }
            auto GetMaximizeCallback()const noexcept -> SingleStateCallback {
                return m_MaximizeCallback;
            }

            void SetFocusCallback(SingleStateCallback callback) noexcept {
                m_FocusCallback = callback;
            }
            auto GetFocusCallback()const noexcept -> SingleStateCallback {
                return m_FocusCallback;
            }

            void SetUserPointer(void* pUserData)noexcept{
                m_PUserData = pUserData;
            }
            auto GetUserPointer()const noexcept -> void*{
                return m_PUserData;
            }
        private:
            static void DefaultSingleStateCallback(Window*,bool){}
            static void DefaultPositionCallback(Window*,int,int){}
            static void DefaultSizeCallback(Window*,int,int){}
            static void DefaultCloseCallback(Window*){}
        private:
            void* m_PUserData = nullptr;
            PositionCallback m_PositionCallback    = DefaultPositionCallback;
            SizeCallback m_SizeCallback            = DefaultSizeCallback;
            SizeCallback m_FramebufferSizeCallback = DefaultSizeCallback;
            CloseCallback m_CloseCallback          = DefaultCloseCallback;
            SingleStateCallback m_MaximizeCallback = DefaultSingleStateCallback;
            SingleStateCallback m_FocusCallback    = DefaultSingleStateCallback;
        };
    }
}
#endif
