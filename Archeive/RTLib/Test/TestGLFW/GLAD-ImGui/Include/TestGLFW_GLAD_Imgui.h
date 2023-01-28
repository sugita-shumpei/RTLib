#ifndef TEST_TEST_GLFW_GLAD_IMGUI_H
#define TEST_TEST_GLFW_GLAD_IMGUI_H
#include <glad/glad.h>
#include <TestGLFW_Common.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <stdexcept>
#include <iostream>
namespace RTLib {
    namespace Test {
        class TestGLFWGLADImGuiAppInitDelegate :public TestGLFWAppInitDelegate {
        public:
            TestGLFWGLADImGuiAppInitDelegate(TestLib::TestApplication* parent, int width, int height, const char* title)noexcept :TestGLFWAppInitDelegate(parent,width,height,title) {}
            virtual ~TestGLFWGLADImGuiAppInitDelegate()noexcept {}

            virtual void Init() override;
        private:
            void InitGLWindow();
            void InitGLAD();
            void InitImGui();
        };
        class TestGLFWGLADImGuiAppMainDelegate :public TestGLFWAppMainDelegate {
        public:
            TestGLFWGLADImGuiAppMainDelegate(TestLib::TestApplication* parent)noexcept :TestGLFWAppMainDelegate(parent) {
            }
            virtual ~TestGLFWGLADImGuiAppMainDelegate()noexcept {}

            virtual void Main() override;
        private:
            void RenderFrame();
            void RenderImGui();
        };
        class TestGLFWGLADImGuiAppFreeDelegate :public TestGLFWAppFreeDelegate{
        public:
            TestGLFWGLADImGuiAppFreeDelegate(TestLib::TestApplication* parent)noexcept :TestGLFWAppFreeDelegate(parent) {}
            virtual ~TestGLFWGLADImGuiAppFreeDelegate()noexcept {}

            virtual void Free() noexcept override;
        private:
            void FreeImGui()noexcept;
        };
        class TestGLFWGLADImGuiAppExtendedData : public TestGLFWAppExtendedData
        {
        public:
            TestGLFWGLADImGuiAppExtendedData(TestLib::TestApplication* app)noexcept;
            virtual ~TestGLFWGLADImGuiAppExtendedData()noexcept;
            void InitImGui();
            void FreeImGui()noexcept;
            void DrawImGui();
        private:
            friend class TestGLFWGLADImGuiAppInitDelegate;
            friend class TestGLFWGLADImGuiAppFreeDelegate;
            friend class TestGLFWGLADImGuiAppMainDelegate;
        private:
            struct Impl;
            std::unique_ptr<Impl> m_Impl;
        };
    }
}
#endif

