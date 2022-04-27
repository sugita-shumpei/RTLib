#include <TestApplication.h>
#include <TestAppInitDelegate.h>

RTLib::TestLib::TestAppInitDelegate::TestAppInitDelegate(TestApplication* app) noexcept
{
	m_Parent = app;
}

RTLib::TestLib::TestAppInitDelegate::~TestAppInitDelegate() noexcept
{
	m_Parent = nullptr;
}

auto RTLib::TestLib::TestAppInitDelegate::GetParent() const noexcept -> const TestApplication*
{
	return m_Parent;
}

auto RTLib::TestLib::TestAppInitDelegate::GetParent() noexcept -> TestApplication*
{
	return m_Parent;
}
