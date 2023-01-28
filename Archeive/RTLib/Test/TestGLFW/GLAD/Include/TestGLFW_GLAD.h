#ifndef TEST_TEST_GLFW_GLAD_H
#define TEST_TEST_GLFW_GLAD_H
#include <glad/glad.h>
#include <TestGLFW_Common.h>
#include <stdexcept>
#include <iostream>
namespace RTLib {
	namespace Test {
		class TestGLFWGLADAppInitDelegate :public TestGLFWAppInitDelegate {
		public:
			TestGLFWGLADAppInitDelegate(TestLib::TestApplication* parent, int width, int height, const char* title)noexcept :TestGLFWAppInitDelegate(parent,width,height,title) {}
			virtual ~TestGLFWGLADAppInitDelegate()noexcept {}

			virtual void Init() override;
		private:
			void InitGLWindow();
			void InitGLAD();
		};
		class TestGLFWGLADAppMainDelegate :public TestGLFWAppMainDelegate {
		public:
			TestGLFWGLADAppMainDelegate(TestLib::TestApplication* parent)noexcept :TestGLFWAppMainDelegate(parent) {
			}
			virtual ~TestGLFWGLADAppMainDelegate()noexcept {}

			virtual void Main() override;
		};
	}
}
#endif
