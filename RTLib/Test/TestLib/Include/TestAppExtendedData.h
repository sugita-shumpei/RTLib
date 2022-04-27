#ifndef TEST_TEST_APP_EXTENDED_DATA_H
#define TEST_TEST_APP_EXTENDED_DATA_H
namespace RTLib
{
    namespace TestLib
    {
        class TestApplication;
        class TestAppExtendedData
        {
        public:
            TestAppExtendedData(TestApplication* app)noexcept;
            virtual ~TestAppExtendedData()noexcept;
            
            auto GetParent()const noexcept -> const TestApplication*;
            auto GetParent()      noexcept ->        TestApplication*;
        private:
            TestApplication* m_Parent;
        };
    }
}
#endif

