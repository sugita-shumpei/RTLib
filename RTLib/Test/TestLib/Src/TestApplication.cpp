#include <TestApplication.h>

RTLib::TestLib::TestApplication::TestApplication() noexcept
{
	m_Argc = 0;
	m_Argv = nullptr;
	m_InitDelegate = nullptr;
	m_MainDelegate = nullptr;
	m_FreeDelegate = nullptr;
    m_ExtendedData= nullptr;
}

RTLib::TestLib::TestApplication::~TestApplication() noexcept
{
	Impl_Free();
	m_Argc          = 0;
	m_Argv          = nullptr;
	m_InitDelegate  = nullptr;
	m_MainDelegate  = nullptr;
	m_FreeDelegate  = nullptr;
    m_ExtendedData = nullptr;
}

auto RTLib::TestLib::TestApplication::Run(int argc, const char** argv) noexcept -> int
{
	m_Argc  = argc;
	m_Argv  = argv;
	int res = 0;
	try {
		Impl_Init();
		Impl_Main();
	}
	catch (std::exception& err) {
		std::cerr << err.what() << std::endl;
		res = -1;
	}
	Impl_Free();
	return res;
}

auto RTLib::TestLib::TestApplication::GetExtendedData()const noexcept -> const TestAppExtendedData* {
    return m_ExtendedData.get();
}
auto RTLib::TestLib::TestApplication::GetExtendedData()      noexcept ->       TestAppExtendedData* {
    return m_ExtendedData.get();
}
auto RTLib::TestLib::TestApplication::GetArgc() const noexcept -> int
{
	return m_Argc;
}

auto RTLib::TestLib::TestApplication::GetArgv() const noexcept -> const char**
{
	return m_Argv;
}

void RTLib::TestLib::TestApplication::Impl_Init()
{
	if (m_InitDelegate) {
		m_InitDelegate->Init();
	}
	else {
		this->Init();
	}
}

void RTLib::TestLib::TestApplication::Impl_Main()
{
	if (m_MainDelegate) {
		m_MainDelegate->Main();
	}
	else {
		this->Main();
	}
}

void RTLib::TestLib::TestApplication::Impl_Free() noexcept
{
	if (m_FreeDelegate) {
		m_FreeDelegate->Free();
	}
	else {
		this->Free();
	}
}
