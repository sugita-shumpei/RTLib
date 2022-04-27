#ifndef TEST_TEST_GLFW_H
#define TEST_TEST_GLFW_H
#include <TestApplication.h>
#include <GLFW/glfw3.h>
#include <stdexcept>
#include <iostream>
#include <unordered_map>
namespace RTLib {
	namespace Test {
		class TestGLFWAppInitDelegate;
		class TestGLFWAppMainDelegate;
		class TestGLFWApplicationExtData {
		public:
			virtual ~TestGLFWApplicationExtData()noexcept{
				Free();
			}
			virtual void Init()         = 0;
			virtual void Free()noexcept {}
		};
		class TestGLFWApplication : public TestLib::TestApplication {
		public:
			using ExtData      = TestGLFWApplicationExtData;
			using InitDelegate = TestGLFWAppInitDelegate;
			using MainDelegate = TestGLFWAppMainDelegate;
		public:
			TestGLFWApplication()noexcept;
			virtual ~TestGLFWApplication()noexcept;

			template<typename AppExtData, bool Cond = std::is_base_of_v<ExtData, AppExtData>>
			void AddExtData()noexcept {
				m_ExtData = std::unique_ptr<ExtData>(new AppExtData());
			}

			template<typename AppExtData,typename ...Args, bool Cond = std::is_base_of_v<ExtData, AppExtData>>
			void AddExtData(Args&&... args)noexcept {
				m_ExtData = std::unique_ptr<ExtData>(new AppExtData(args...));
			}
		protected:
			virtual void Init()			 override;
			virtual void Main()          override;
			virtual void Free()noexcept  override;
		private:
			friend class TestGLFWAppInitDelegate;
			friend class TestGLFWAppMainDelegate;
			bool										m_IsGlfwInit;
			GLFWwindow*									m_Window;
			std::unique_ptr<TestGLFWApplicationExtData> m_ExtData;
		};
		class TestGLFWAppInitDelegate:public TestLib::TestAppInitDelegate {
		public:
			TestGLFWAppInitDelegate(TestLib::TestApplication* parent, int width, int height, const char* title)noexcept:TestAppInitDelegate(parent) {
				m_Width  = width ;
				m_Height = height;
				m_Title  = title ;
			}
			virtual ~TestGLFWAppInitDelegate()noexcept {}

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
			void InitExtData();
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
			auto GetExtData()const noexcept -> TestGLFWApplicationExtData*;
		};
	}
}
#endif