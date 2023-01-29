#ifndef RTLIB_CORE_RT_UUID_H
#define RTLIB_CORE_RT_UUID_H
#include <array>
namespace RTLib
{
	namespace Core
	{
		class RTUuid
		{
		public:
			RTUuid() noexcept :m_Value{} {}
			RTUuid(const RTUuid& lhs)noexcept :m_Value{ lhs.m_Value}{}
			RTUuid(     RTUuid&& rhs)noexcept :m_Value{std::move(rhs.m_Value)} {}
			RTUuid& operator=(const RTUuid& lhs)noexcept {
				if (this != &lhs) {
					m_Value = lhs.m_Value;
				}
				return *this;
			}
			RTUuid& operator=(RTUuid&& rhs) noexcept {
				if (this != &rhs) {
					m_Value = std::move(rhs.m_Value);
				}
				return *this;
			}


		private:
			std::array<unsigned char, 16> m_Value;
		};
	}
}
#endif