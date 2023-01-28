#include <RTLib/Core/BaseObject.h>
#include <sstream>
namespace RTLib
{
	namespace Core {
		struct RTLib::Core::BaseObject::Impl
		{
			Impl()noexcept {
				m_Name = "";
			}
			std::string             m_Name;
		};
		auto TypeId2String(const TypeId& typeId) -> std::string {
			std::stringstream ss;
			for (int i = 0; i < 15; ++i) {
				ss << std::hex << "0x" << (uint32_t)typeId[i] << "-";
			}
			ss << std::hex << "0x" << (uint32_t)typeId[15];
			return ss.str();
		}
	}
}
RTLib::Core::BaseObject::BaseObject() noexcept
{
	m_Impl = std::unique_ptr<Impl>(new Impl());
}

RTLib::Core::BaseObject::~BaseObject() noexcept {
	m_Impl.reset();
}

auto RTLib::Core::BaseObject::GetBaseTypeIdString() const noexcept -> std::string
{
	return TypeId2String(GetBaseTypeId());
}

auto RTLib::Core::BaseObject::GetTypeIdString() const noexcept -> std::string
{
	return TypeId2String(GetTypeId());
}

void RTLib::Core::BaseObject::SetName(const std::string& name) noexcept
{
	m_Impl->m_Name = name;
}

auto RTLib::Core::BaseObject::GetName() const noexcept -> std::string
{
	return m_Impl->m_Name;
}
