#include <TestAppExtendedData.h>
#include <TestApplication.h>
RTLib::TestLib::TestAppExtendedData::TestAppExtendedData(TestApplication* app)noexcept
{
    m_Parent = app;
}
RTLib::TestLib::TestAppExtendedData::~TestAppExtendedData()noexcept{
    m_Parent = nullptr;
}

auto RTLib::TestLib::TestAppExtendedData::GetParent()const noexcept -> const TestApplication* {
    return  m_Parent;
}
auto RTLib::TestLib::TestAppExtendedData::GetParent()      noexcept ->        TestApplication*{
    return  m_Parent;
}
