#include <TestApplication.h>
#include <TestAppFreeDelegate.h>

RTLib::TestLib::TestAppFreeDelegate::TestAppFreeDelegate(TestApplication* app) noexcept
{
	m_Parent = app;
}

RTLib::TestLib::TestAppFreeDelegate::~TestAppFreeDelegate() noexcept
{
	m_Parent = nullptr;
}

auto RTLib::TestLib::TestAppFreeDelegate::GetParent() const noexcept -> const TestApplication*
{
	return m_Parent;
}

auto RTLib::TestLib::TestAppFreeDelegate::GetParent() noexcept -> TestApplication*
{
	return m_Parent;
}
