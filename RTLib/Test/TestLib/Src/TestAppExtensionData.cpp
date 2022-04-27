#include <TestAppExtensionData.h>
#include <TestApplication.h>
RTLib::TestLib::TestAppExtensionData::TestAppExtensionData(TestApplication* app)noexcept
{
    m_Parent = app;
}
RTLib::TestLib::TestAppExtensionData::~TestAppExtensionData()noexcept{
    m_Parent = nullptr;
}

auto RTLib::TestLib::TestAppExtensionData::GetParent()const noexcept -> const TestApplication* {
    return  m_Parent;
}
auto RTLib::TestLib::TestAppExtensionData::GetParent()      noexcept ->        TestApplication*{
    return  m_Parent;
}
