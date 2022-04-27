#ifndef TEST_TEST_GLFW_IMGUI_GLAD_H
#define TEST_TEST_GLFW_IMGUI_GLAD_H
#include <glad/glad.h>
#include <TestGLFW.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <stdexcept>
#include <iostream>
namespace RTLib {
    namespace Test {
        class TestGLFWImGuiGLADAppInitDelegate :public TestGLFWAppInitDelegate {
        public:
            TestGLFWImGuiGLADAppInitDelegate(TestLib::TestApplication* parent, int width, int height, const char* title)noexcept :TestGLFWAppInitDelegate(parent,width,height,title) {}
            virtual ~TestGLFWImGuiGLADAppInitDelegate()noexcept {}

            virtual void Init() override;
        private:
            void InitGLWindow();
            void InitGLAD();
            void InitImGui();
        };
        class TestGLFWImGuiGLADAppMainDelegate :public TestGLFWAppMainDelegate {
        public:
            TestGLFWImGuiGLADAppMainDelegate(TestLib::TestApplication* parent)noexcept :TestGLFWAppMainDelegate(parent) {
            }
            virtual ~TestGLFWImGuiGLADAppMainDelegate()noexcept {}

            virtual void Main() override;
        private:
            void RenderFrame();
            void RenderImGui();
        };
        class TestGLFWImGuiGLADAppFreeDelegate :public TestGLFWAppFreeDelegate{
        public:
            TestGLFWImGuiGLADAppFreeDelegate(TestLib::TestApplication* parent)noexcept :TestGLFWAppFreeDelegate(parent) {}
            virtual ~TestGLFWImGuiGLADAppFreeDelegate()noexcept {}

            virtual void Free() noexcept override;
        private:
            void FreeImGui()noexcept;
        };
        class TestGLFWImGuiGLADAppExtensionData : public TestGLFWAppExtensionData
        {
        public:
            TestGLFWImGuiGLADAppExtensionData(TestLib::TestApplication* app)noexcept;
            virtual ~TestGLFWImGuiGLADAppExtensionData()noexcept;
            void InitImGui();
            void FreeImGui()noexcept;
            void DrawImGui();
        private:
            friend class TestGLFWImGuiGLADAppInitDelegate;
            friend class TestGLFWImGuiGLADAppFreeDelegate;
            friend class TestGLFWImGuiGLADAppMainDelegate;
            struct Impl;
            std::unique_ptr<Impl> m_Impl;
        };
    }
}
#endif

