#ifndef TEST_TEST_APP_MAIN_DELEGATE_H
#define TEST_TEST_APP_MAIN_DELEGATE_H
namespace RTLib {
	namespace TestLib
	{
		class TestApplication;
		class TestAppMainDelegate
		{
		public:
			TestAppMainDelegate(TestApplication* app)noexcept;
			virtual ~TestAppMainDelegate()noexcept;
			virtual void Main() = 0;

			auto GetParent()const noexcept -> const TestApplication*;
			auto GetParent()      noexcept ->	    TestApplication*;
		private:
			TestApplication* m_Parent;
		};
	}
}
#endif
