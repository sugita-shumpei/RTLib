#ifndef TEST_TEST_APP_INIT_DELEGATE_H
#define TEST_TEST_APP_INIT_DELEGATE_H
namespace RTLib
{
	namespace TestLib
	{
		class TestApplication;
		class TestAppInitDelegate
		{
		public:
			TestAppInitDelegate(TestApplication* app)noexcept;
			virtual ~TestAppInitDelegate()noexcept;
			virtual void Init() = 0;

			auto GetParent()const noexcept -> const TestApplication*;
			auto GetParent()      noexcept ->	    TestApplication*;
		private:
			TestApplication* m_Parent;
		};
	}
}
#endif
