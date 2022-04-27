#ifndef TEST_TEST_APP_FREE_DELEGATE_H
#define TEST_TEST_APP_FREE_DELEGATE_H
namespace RTLib
{
	namespace TestLib
	{
		class TestApplication;
		class TestAppFreeDelegate
		{
		public:
			TestAppFreeDelegate(TestApplication* app)noexcept;
			virtual ~TestAppFreeDelegate()noexcept;
			virtual void Free()noexcept = 0;

			auto GetParent()const noexcept -> const TestApplication*;
			auto GetParent()      noexcept ->	    TestApplication*;
		private:
			TestApplication* m_Parent;
		};
	}
}
#endif
