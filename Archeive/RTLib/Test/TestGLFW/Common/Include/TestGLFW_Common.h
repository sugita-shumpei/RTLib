#ifndef TEST_TEST_GLFW_COMMON_H
#define TEST_TEST_GLFW_COMMON_H
#include <TestApplication.h>
#include <GLFW/glfw3.h>
#include <stdexcept>
#include <iostream>
#include <unordered_map>
namespace RTLib {
	namespace Test {
		class TestGLFWAppInitDelegate;
		class TestGLFWAppMainDelegate;
        class TestGLFWAppFreeDelegate;
        class TestGLFWAppExtendedData;
		class TestGLFWApplication : public TestLib::TestApplication {
		public:
			using InitDelegate = TestGLFWAppInitDelegate;
			using MainDelegate = TestGLFWAppMainDelegate;
            using FreeDelegate = TestGLFWAppFreeDelegate;
		public:
			TestGLFWApplication()noexcept;
			virtual ~TestGLFWApplication()noexcept;
		protected:
			virtual void Init()			 override;
			virtual void Main()          override;
			virtual void Free()noexcept  override;
		private:
			friend class TestGLFWAppInitDelegate;
			friend class TestGLFWAppMainDelegate;
            friend class TestGLFWAppFreeDelegate;
            friend class TestGLFWAppExtendedData;
			bool										m_IsGlfwInit;
			GLFWwindow*									m_Window;
		};
		class TestGLFWAppInitDelegate:public TestLib::TestAppInitDelegate {
		public:
			TestGLFWAppInitDelegate(TestLib::TestApplication* parent, int width, int height, const char* title)noexcept:TestAppInitDelegate(parent) {
				m_Width  = width ;
				m_Height = height;
				m_Title  = title ;
			}
			virtual ~TestGLFWAppInitDelegate()noexcept {}
            
            //all resource must be initialized in this function
			virtual void Init() override;

			auto GetWidth ()const noexcept -> int		  { return m_Width;  }
			auto GetHeight()const noexcept -> int         { return m_Height; }
			auto GetTitle ()const noexcept -> const char* { return m_Title;  }

		protected:
			auto GetWindow()const noexcept -> GLFWwindow*;
		protected:
			void InitGLFW();
			void InitWindow( const std::unordered_map<int,int>& windowHints);
			void MakeContext();
			void ShowWindow();
		private:
			int			 m_Width;
			int			 m_Height;
			const char*  m_Title;
		};
		class TestGLFWAppMainDelegate:public TestLib::TestAppMainDelegate {
		public:
			TestGLFWAppMainDelegate(TestLib::TestApplication* parent)noexcept :TestAppMainDelegate(parent) {}

			virtual ~TestGLFWAppMainDelegate()noexcept {}

			virtual void Main() override;
		protected:
			bool ShouldClose();
			void SwapBuffers();
			void PollEvents();
			auto GetWindow() const noexcept -> GLFWwindow*;
		};
        class TestGLFWAppFreeDelegate : public TestLib::TestAppFreeDelegate{
        public:
            TestGLFWAppFreeDelegate(TestLib::TestApplication* parent)noexcept :TestAppFreeDelegate(parent) {}
            virtual ~TestGLFWAppFreeDelegate()noexcept {}
            //all resource must be released in this function
            virtual void Free()noexcept override;
        protected:
            void FreeWindow()noexcept;
            void FreeGLFW()noexcept;
        };
        class TestGLFWAppExtendedData : public TestLib::TestAppExtendedData{
        public:
            TestGLFWAppExtendedData(TestLib::TestApplication* parent)noexcept:TestLib::TestAppExtendedData(parent){}
            virtual ~TestGLFWAppExtendedData()noexcept{}
        protected:
            auto GetWindow()const noexcept -> GLFWwindow*;
        };
	}
}
#endif
