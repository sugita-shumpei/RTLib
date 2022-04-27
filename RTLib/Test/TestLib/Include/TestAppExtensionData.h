#ifndef TEST_TEST_APP_EXTENSION_DATA_H
#define TEST_TEST_APP_EXTENSION_DATA_H
namespace RTLib
{
    namespace TestLib
    {
        class TestApplication;
        class TestAppExtensionData
        {
        public:
            TestAppExtensionData(TestApplication* app)noexcept;
            virtual ~TestAppExtensionData()noexcept;
            
            auto GetParent()const noexcept -> const TestApplication*;
            auto GetParent()      noexcept ->        TestApplication*;
        private:
            TestApplication* m_Parent;
        };
    }
}
#endif

