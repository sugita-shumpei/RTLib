#ifndef TEST_TEST_APPLICATION_H
#define TEST_TEST_APPLICATION_H
#include <TestAppInitDelegate.h>
#include <TestAppMainDelegate.h>
#include <TestAppFreeDelegate.h>
#include <TestAppExtensionData.h>
#include <memory>
#include <iostream>
#include <stdexcept>
namespace RTLib {
	namespace TestLib {
		class TestApplication {
		public:
			TestApplication()noexcept;
			virtual ~TestApplication()noexcept;
			auto Run(int argc, const char** argv)noexcept -> int;

			template<typename InitDelegate, bool Cond = std::is_base_of_v<TestAppInitDelegate, InitDelegate>>
			void AddInitDelegate()noexcept {
				if (m_InitDelegate) {
					return;
				}
				m_InitDelegate = std::unique_ptr<TestAppInitDelegate>(new InitDelegate(this));
			}
			template<typename InitDelegate, typename... Args, bool Cond = std::is_base_of_v<TestAppInitDelegate, InitDelegate>>
			void AddInitDelegate(Args&& ...args)noexcept {
				if (m_InitDelegate) {
					return;
				}
				m_InitDelegate = std::unique_ptr<TestAppInitDelegate>(new InitDelegate(this, std::forward<Args&&>(args)...));
			}
			template<typename MainDelegate, bool Cond = std::is_base_of_v<TestAppMainDelegate, MainDelegate>>
			void AddMainDelegate()noexcept {
				if (m_MainDelegate) {
					return;
				}
				m_MainDelegate = std::unique_ptr<TestAppMainDelegate>(new MainDelegate(this));
			}
			template<typename MainDelegate, typename... Args, bool Cond = std::is_base_of_v<TestAppMainDelegate, MainDelegate>>
			void AddMainDelegate(Args&& ...args)noexcept {
				if (m_MainDelegate) {
					return;
				}
				m_MainDelegate = std::unique_ptr<TestAppMainDelegate>(new MainDelegate(this, std::forward<Args&&>(args)...));
			}
			template<typename FreeDelegate, bool Cond = std::is_base_of_v<TestAppFreeDelegate, FreeDelegate>>
			void AddFreeDelegate()noexcept {
				if (m_FreeDelegate) {
					return;
				}
				m_FreeDelegate = std::unique_ptr<TestAppFreeDelegate>(new FreeDelegate(this));
			}
			template<typename FreeDelegate, typename... Args, bool Cond = std::is_base_of_v<TestAppFreeDelegate, FreeDelegate>>
			void AddFreeDelegate(Args&& ...args)noexcept {
				if (m_FreeDelegate) {
					return;
				}
				m_FreeDelegate = std::unique_ptr<TestAppFreeDelegate>(new FreeDelegate(this, std::forward<Args&&>(args)...));
			}
            template<typename AppExtData, bool Cond = std::is_base_of_v<TestAppExtensionData, AppExtData>>
            void AddExtensionData()noexcept {
                m_ExtensionData = std::unique_ptr<TestAppExtensionData>(new AppExtData(this));
            }

            template<typename AppExtData,typename ...Args, bool Cond = std::is_base_of_v<TestAppExtensionData, AppExtData>>
            void AddExtensionData(Args&&... args)noexcept {
                m_ExtensionData = std::unique_ptr<TestAppExtensionData>(new AppExtData(this,args...));
            }
            
            auto GetExtensionData()const noexcept -> const TestAppExtensionData*;
            auto GetExtensionData()      noexcept ->       TestAppExtensionData*;

			auto GetArgc()const noexcept -> int;
			auto GetArgv()const noexcept -> const char**;
		protected:
			virtual void Init()         {}
			virtual void Main()         {}
			virtual void Free()noexcept {}
		private:
			void Impl_Init();
			void Impl_Main();
			void Impl_Free()noexcept;
		private:
			std::unique_ptr<TestAppInitDelegate>  m_InitDelegate;
			std::unique_ptr<TestAppMainDelegate>  m_MainDelegate;
			std::unique_ptr<TestAppFreeDelegate>  m_FreeDelegate;
            std::unique_ptr<TestAppExtensionData> m_ExtensionData;
			int									  m_Argc;
			const char**                          m_Argv;
		};
	}
}
#endif
