#include <TestApplication.h>
#include <TestAppMainDelegate.h>

RTLib::TestLib::TestAppMainDelegate::TestAppMainDelegate(TestApplication* app) noexcept
{
	m_Parent = app;
}

RTLib::TestLib::TestAppMainDelegate::~TestAppMainDelegate() noexcept
{
	m_Parent = nullptr;
}

auto RTLib::TestLib::TestAppMainDelegate::GetParent() const noexcept -> const TestApplication*
{
	return m_Parent;
}

auto RTLib::TestLib::TestAppMainDelegate::GetParent() noexcept -> TestApplication*
{
	return m_Parent;
}
