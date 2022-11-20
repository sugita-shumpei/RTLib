#include <RTLib/Backends/GLFW/Window/GLFWWindow.h>
#include <RTLib/Backends/GLFW/Inputs/GLFWKeyboard.h>
#include <RTLib/Inputs/Keyboard.h>
#include <RTLib/Backends/GLFW/GLFWEntry.h>
#include <cassert>
#include "Internals/GLFWInternals.h"
struct RTLib::Backends::Glfw::Window::Window::Impl
{
    Impl(const RTLib::Window::WindowDesc& desc, void* handle) :keyboards{ new RTLib::Backends::Glfw::Inputs::Keyboard() } {
        graphicsFlags = desc.graphics.flags;
        if (graphicsFlags & RTLib::Window::GraphicsFlagsVulkan) {
            if (desc.graphics.vulkan.required) {
                assert(glfwVulkanSupported() == GLFW_TRUE);
            }
        }
        bool isFindValidVersion = false;
        if (graphicsFlags& RTLib::Window::GraphicsFlagsOpenGL) {
            glfwWindowHint(GLFW_CLIENT_API, desc.graphics.openGL.isES ? GLFW_OPENGL_ES_API : GLFW_OPENGL_API);
            glfwWindowHint(GLFW_OPENGL_PROFILE, desc.graphics.openGL.isCoreProfile ? GLFW_OPENGL_CORE_PROFILE : GLFW_OPENGL_COMPAT_PROFILE);
            glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, desc.graphics.openGL.isForwardCompat ? GLFW_TRUE : GLFW_FALSE);
            if (desc.graphics.openGL.majorVersion != desc.defaultVal) {
                glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, desc.graphics.openGL.majorVersion);
                glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, desc.graphics.openGL.minorVersion);
                isFindValidVersion = false;
            }
            else {
                isFindValidVersion = true;
            }
        }
        else {
            glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        }
        glfwWindowHint(GLFW_RESIZABLE,desc.isResizable?GLFW_TRUE:GLFW_FALSE);
        bool isFixPosition = false;
        if ((desc.xpos != desc.defaultVal) || desc.ypos != desc.defaultVal) {
            isFixPosition = true;
        }
        glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
        if (isFindValidVersion) {
            if (!desc.graphics.openGL.isES) {
                std::vector<std::pair<int, int>> glVersions = {
                    {4,6},{4,5},{4,3},{4,2},{4,1},{4,0},
                    {3,3},{3,2},{3,1},{3,0},
                    {2,1},{2,0},
                    {1,5},{1,3},{1,2},{1,1},{1,0},
                };
                for (auto& [major, minor] : glVersions) {
                    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, major);
                    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, minor);
                    window = glfwCreateWindow(desc.width, desc.height, desc.title.c_str(), nullptr, nullptr);
                    if (window) {
                        break;
                    }
                }
                if (desc.graphics.openGL.required) {
                    assert(window);
                }
                if (!window) {
                    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
                    window = glfwCreateWindow(desc.width, desc.height, desc.title.c_str(), nullptr, nullptr);
                    assert(window);

                }
            }
        }
        else {
            window = glfwCreateWindow(desc.width, desc.height, desc.title.c_str(), nullptr, nullptr);
            if (desc.graphics.openGL.required) {
                assert(window);
            }
            if (!window) {
                glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
                window = glfwCreateWindow(desc.width, desc.height, desc.title.c_str(), nullptr, nullptr);
                assert(window);
            }
        }
        if (isFixPosition) {
            glfwSetWindowPos(window, desc.xpos, desc.ypos);
        }
        glfwDefaultWindowHints();
        position[0] = desc.xpos;
        position[1] = desc.ypos;
        size[0] = desc.width;
        size[1] = desc.height;
        {
            int fbWid, fbHei;
            glfwGetFramebufferSize(window, &fbWid, &fbHei);
            framebufferSize[0] = fbWid;
            framebufferSize[1] = fbHei;
        }
        title = desc.title;
        isVisible = desc.isVisible;
        isResizable = desc.isResizable;
        glfwSetWindowSizeCallback(window, Internals_GlfwWindowSizeCallback);
        glfwSetWindowFocusCallback(window, Internals_GlfwWindowFocusCallback);
        glfwSetWindowMaximizeCallback(window, Internals_GlfwWindowMaximizeCallback);
        glfwSetWindowCloseCallback(window, Internals_GlfwWindowCloseCallback);
        glfwSetKeyCallback(window, Internals_GlfwKeyCallback);
        glfwSetWindowUserPointer(window, handle);
        if (isVisible) {
            glfwShowWindow(window);
        }
    }
    ~Impl() noexcept {
        glfwDestroyWindow(window);
    }
    static void Internals_GlfwWindowPositionCallback(GLFWwindow* window, int x, int y) {
        if (!window) {
            return;
        }
        auto handle = static_cast<RTLib::Backends::Glfw::Window::Window*>(glfwGetWindowUserPointer(window));
        if (!handle) {
            return;
        }
        if ((handle->m_Impl->position[0] != x) || (handle->m_Impl->position[1] != y)) {
            handle->GetPositionCallback()(handle, x, y);
            handle->m_Impl->position[0] = x;
            handle->m_Impl->position[1] = y;
        }
    }
    static void Internals_GlfwWindowSizeCallback(GLFWwindow* window, int width, int height) {
        if (!window) {
            return;
        }
        auto handle = static_cast<RTLib::Backends::Glfw::Window::Window*>(glfwGetWindowUserPointer(window));
        if (!handle) {
            return;
        }
        if ((handle->m_Impl->size[0] != width) || (handle->m_Impl->size[1] != height)){
            handle->GetSizeCallback()(handle, width, height);
            int fbWidth, fbHeight;
            glfwGetFramebufferSize(window, &fbWidth, &fbHeight);
            handle->GetFramebufferSizeCallback()(handle, fbWidth, fbHeight);
            handle->m_Impl->size[0] = width;
            handle->m_Impl->size[1] = height;
            handle->m_Impl->framebufferSize[0] = fbWidth;
            handle->m_Impl->framebufferSize[1] = fbHeight;
        }
    }
    static void Internals_GlfwWindowCloseCallback(GLFWwindow* window) {
        if (!window) {
            return;
        }
        auto handle = static_cast<RTLib::Backends::Glfw::Window::Window*>(glfwGetWindowUserPointer(window));
        if (!handle) {
            return;
        }
        if (handle->m_Impl->isClosed) { return; }
        handle->GetCloseCallback()(handle);
        handle->m_Impl->isClosed = true;
    }
    static void Internals_GlfwWindowMaximizeCallback(GLFWwindow* window, int isMaximized) {
        if (!window) {
            return;
        }
        auto handle = static_cast<RTLib::Backends::Glfw::Window::Window*>(glfwGetWindowUserPointer(window));
        if (!handle) {
            return;
        }
        handle->GetMaximizeCallback()(handle,isMaximized==GLFW_TRUE);
    }
    static void Internals_GlfwWindowFocusCallback(GLFWwindow* window, int isFocused) {
        if (!window) {
            return;
        }
        auto handle = static_cast<RTLib::Backends::Glfw::Window::Window*>(glfwGetWindowUserPointer(window));
        if (!handle) {
            return;
        }
        handle->GetFocusCallback()(handle, isFocused == GLFW_TRUE);
    }
    static void Internals_GlfwKeyCallback(GLFWwindow* window, int key, int scancode, int action, int mods)
    {
        if (!window) {
            return;
        }
        auto handle = static_cast<RTLib::Backends::Glfw::Window::Window*>(glfwGetWindowUserPointer(window));
        if (!handle) {
            return;
        }
        auto tmpKey = Internals::GetInptKeyCode(key);
        UpdateKeyboard(static_cast<RTLib::Backends::Glfw::Inputs::Keyboard*>(Entry::Handle().GetGlobalKeyboard()),tmpKey,action);
        UpdateKeyboard(handle->m_Impl->keyboards.get(), tmpKey, action);
    }
    static void UpdateKeyboard(Inputs::Keyboard* keyboard, RTLib::Inputs::KeyCode keyCode, int action) {
        unsigned int prvKeyState = 0;
        unsigned int curKeyState = 0;
        if (keyboard->Internal_KeyStates().count(keyCode) > 0) {
            prvKeyState = keyboard->Internal_KeyStates().at(keyCode);
        }
        if (prvKeyState & RTLib::Inputs::KeyStatePressed) {
            if (action == GLFW_RELEASE)
            {
                curKeyState = RTLib::Inputs::KeyStateReleased | RTLib::Inputs::KeyStateUpdated;
            }
            else {
                curKeyState = RTLib::Inputs::KeyStatePressed;
            }
        }
        else {
            if (action == GLFW_PRESS)
            {
                curKeyState = RTLib::Inputs::KeyStatePressed | RTLib::Inputs::KeyStateUpdated;
            }
            else {
                curKeyState = RTLib::Inputs::KeyStateReleased;
            }
        }
        keyboard->GetCallback()(keyCode, curKeyState, keyboard->GetUserPointer());
        keyboard->Internal_KeyStates()[keyCode] = curKeyState;
    }
    GLFWwindow* window = nullptr;
    std::unique_ptr<Inputs::Keyboard> keyboards = nullptr;
    std::array<int, 2> position = {};
    std::array<int, 2> size = {};
    std::array<int, 2> framebufferSize = {};
    std::string title = "";
    unsigned int graphicsFlags = RTLib::Window::GraphicsFlagsUnknown;
    bool isClosed    = false;
    bool isVisible   = false;
    bool isResizable = false;
};
auto RTLib::Backends::Glfw::Window::Window::GetPosition() const noexcept ->std::array<int, 2> {
    return m_Impl->position;
}
void RTLib::Backends::Glfw::Window::Window::SetPosition(const std::array<int, 2>& position) noexcept {
    if (position == m_Impl->position)
    {
        return;
    }
    auto& handle = Entry::Handle();
    glfwSetWindowPos(m_Impl->window, position[0], position[1]);
    GetPositionCallback()(this, position[0], position[1]);
    m_Impl->position = position;
}
auto RTLib::Backends::Glfw::Window::Window::GetSize()const noexcept -> std::array<int, 2> {
    return m_Impl->size;
}
void RTLib::Backends::Glfw::Window::Window::SetSize(const std::array<int, 2>& size)noexcept {
    if (size == m_Impl->size)
    {
        return;
    }
    auto& handle = Entry::Handle();
    glfwSetWindowSize(m_Impl->window, size[0], size[1]);
    GetSizeCallback()(this,  size[0], size[1]);
    int fbWid, fbHei;
    glfwGetFramebufferSize(m_Impl->window, &fbWid, &fbHei);
    GetFramebufferSizeCallback()(this, fbWid, fbHei);
    m_Impl->size = size;
    m_Impl->framebufferSize[0] = fbWid;
    m_Impl->framebufferSize[1] = fbHei;
}
auto RTLib::Backends::Glfw::Window::Window::GetFramebufferSize()const noexcept -> std::array<int, 2> {
    return m_Impl->framebufferSize;
}
auto RTLib::Backends::Glfw::Window::Window::GetTitle()const noexcept -> std::string {
    return m_Impl->title;
}
void RTLib::Backends::Glfw::Window::Window::SetTitle(const std::string& title)noexcept {
    if (m_Impl->title == title) {
        return;
    }
    glfwSetWindowTitle(m_Impl->window, title.c_str());
    m_Impl->title = title;
}
void RTLib::Backends::Glfw::Window::Window::Maximize() noexcept {
    auto& handle = Entry::Handle();
    glfwMaximizeWindow(m_Impl->window);
    GetMaximizeCallback()(this,  true);
}
void RTLib::Backends::Glfw::Window::Window::Restore() noexcept {
    auto& handle = Entry::Handle();
    glfwRestoreWindow(m_Impl->window);
    GetMaximizeCallback()(this, false);
}
void RTLib::Backends::Glfw::Window::Window::Focus() noexcept {
    auto& handle = Entry::Handle();
    glfwFocusWindow(m_Impl->window);
    GetFocusCallback()(this,  true);
}
bool  RTLib::Backends::Glfw::Window::Window::ShouldClose() const noexcept {
    return m_Impl->isClosed;
}
void RTLib::Backends::Glfw::Window::Window::RequestClose() noexcept {
    if (m_Impl->isClosed) {
        return;
    }
    auto& handle = Entry::Handle();
    glfwSetWindowShouldClose(m_Impl->window, GLFW_TRUE);
    m_Impl->isClosed = true;
    return;
}
bool RTLib::Backends::Glfw::Window::Window::IsResizable() const noexcept {
    return m_Impl->isResizable;
}
void RTLib::Backends::Glfw::Window::Window::SetResizable(bool isResizable)noexcept {
    if (m_Impl->isResizable && !isResizable) {
        auto& handle = Entry::Handle();
        glfwSetWindowAttrib(m_Impl->window,GLFW_RESIZABLE, GLFW_FALSE);
    }
    if (!m_Impl->isResizable && isResizable) {
        auto& handle = Entry::Handle();
        glfwSetWindowAttrib(m_Impl->window, GLFW_RESIZABLE, GLFW_TRUE);
    }

}
bool RTLib::Backends::Glfw::Window::Window::IsVisible() const noexcept {
    return m_Impl->isVisible;
}
void RTLib::Backends::Glfw::Window::Window::Show() noexcept {
    if (m_Impl->isVisible) {
        return;
    }
    auto& handle = Entry::Handle();
    glfwShowWindow(m_Impl->window);
    m_Impl->isVisible = true;
}
void RTLib::Backends::Glfw::Window::Window::Hide() noexcept {
    if (!m_Impl->isVisible) {
        return;
    }
    auto& handle = Entry::Handle();
    glfwHideWindow(m_Impl->window);
    m_Impl->isVisible = false;
}

void RTLib::Backends::Glfw::Window::Window::SwapBuffers() noexcept
{
    auto& handle = Entry::Handle();
    glfwSwapBuffers(m_Impl->window);
}
auto RTLib::Backends::Glfw::Window::Window::GetHandle()const noexcept -> void* 
{
    return m_Impl->window;
}
RTLib::Backends::Glfw::Window::Window::Window(const RTLib::Window::WindowDesc& desc) noexcept
    : RTLib::Window::Window(),
      m_Impl(new Impl(desc,this))
{
}
RTLib::Backends::Glfw::Window::Window::~Window() noexcept
{
    m_Impl.reset();
}

auto RTLib::Backends::Glfw::Window::Window::Internal_GetKeyboard()const noexcept -> RTLib::Inputs::Keyboard*
{
    return m_Impl->keyboards.get();
}