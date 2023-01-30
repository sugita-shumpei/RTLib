#ifndef RTLIB_CORE_RT_UUID_H
#define RTLIB_CORE_RT_UUID_H
#include <string_view>
#include <array>
namespace RTLib
{
	namespace Core
	{
		class RTUuid
		{
		public:
			constexpr RTUuid() noexcept :m_Value{} {}
			constexpr RTUuid(const std::array<unsigned char, 16>& v) noexcept :m_Value{ v } {}
			constexpr RTUuid(
				std::uint32_t                     data1,
				std::uint16_t                     data2,
				std::uint16_t                     data3,
				const std::array<std::uint8_t, 8>& data4
			)noexcept :m_Value
			{
				/*0x8=0x1000*/
				static_cast<uint8_t>((data1 & static_cast<uint32_t>(0x000000FF)) >> 0),
				static_cast<uint8_t>((data1 & static_cast<uint32_t>(0x0000FF00)) >> 8),
				static_cast<uint8_t>((data1 & static_cast<uint32_t>(0x00FF0000)) >> 16),
				static_cast<uint8_t>((data1 & static_cast<uint32_t>(0xFF000000)) >> 24),
				static_cast<uint8_t>((data2 & static_cast<uint16_t>(0x00FF)) >> 0) ,
				static_cast<uint8_t>((data2 & static_cast<uint16_t>(0xFF00)) >> 8),
				static_cast<uint8_t>((data3 & static_cast<uint16_t>(0x00FF)) >> 0),
				static_cast<uint8_t>((data3 & static_cast<uint16_t>(0xFF00)) >> 8),
				data4[0],data4[1],data4[2],data4[3],
				data4[4],data4[5],data4[6],data4[7]
			} {}

			constexpr RTUuid(const RTUuid& lhs)noexcept :m_Value{ lhs.m_Value } {}
			constexpr RTUuid(RTUuid&& rhs)noexcept :m_Value{ std::move(rhs.m_Value) } {}
			constexpr RTUuid& operator=(const RTUuid& lhs)noexcept {
				if (this != &lhs) {
					m_Value = lhs.m_Value;
				}
				return *this;
			}
			constexpr RTUuid& operator=(RTUuid&& rhs) noexcept {
				if (this != &rhs) {
					m_Value = std::move(rhs.m_Value);
				}
				return *this;
			}

			constexpr auto GetData()const noexcept -> std::array<unsigned char, 16>{
				return m_Value;
			}
			constexpr auto GetData1()const noexcept -> std::uint32_t {
				return
					(static_cast<std::uint32_t>(m_Value[0]) << 0) +
					(static_cast<std::uint32_t>(m_Value[1]) << 8) +
					(static_cast<std::uint32_t>(m_Value[2]) << 16) +
					(static_cast<std::uint32_t>(m_Value[3]) << 24);
			}
			constexpr auto GetData2()const noexcept -> std::uint16_t {
				return
					(static_cast<std::uint16_t>(m_Value[4]) << 0) +
					(static_cast<std::uint16_t>(m_Value[5]) << 8);
			}
			constexpr auto GetData3()const noexcept -> std::uint16_t {
				return
					(static_cast<std::uint16_t>(m_Value[6]) << 0) +
					(static_cast<std::uint16_t>(m_Value[7]) << 8);
			}
			constexpr auto GetData4()const noexcept -> std::array<std::uint8_t, 8>
			{
				return std::array<std::uint8_t, 8>{
					m_Value[8], m_Value[9], m_Value[10], m_Value[11],
						m_Value[12], m_Value[13], m_Value[14], m_Value[15]
				};
			}
			constexpr auto GetHexData1()const noexcept -> std::array<char, 9>
			{
				auto v0 = GetHex(m_Value[0]);
				auto v1 = GetHex(m_Value[1]);
				auto v2 = GetHex(m_Value[2]);
				auto v3 = GetHex(m_Value[3]);
				return std::array<char, 9>{
					v3[1], v3[0], v2[1], v2[0], v1[1], v1[0], v0[1], v0[0], '\0'
				};
			}
			constexpr auto GetHexData2()const noexcept -> std::array<char, 5>
			{
				auto v0 = GetHex(m_Value[4]);
				auto v1 = GetHex(m_Value[5]);
				return std::array<char, 5>{
					v1[1], v1[0], v0[1], v0[0], '\0'
				};
			}
			constexpr auto GetHexData3()const noexcept -> std::array<char, 5>
			{
				auto v0 = GetHex(m_Value[6]);
				auto v1 = GetHex(m_Value[7]);
				return std::array<char, 5>{
					v1[1], v1[0], v0[1], v0[0], '\0'
				};
			}
			constexpr auto GetHexData4()const noexcept -> std::array<char, 24>
			{
				auto v0 = GetHex(m_Value[8]);
				auto v1 = GetHex(m_Value[9]);
				auto v2 = GetHex(m_Value[10]);
				auto v3 = GetHex(m_Value[11]);
				auto v4 = GetHex(m_Value[12]);
				auto v5 = GetHex(m_Value[13]);
				auto v6 = GetHex(m_Value[14]);
				auto v7 = GetHex(m_Value[15]);
				return std::array<char, 24>{
					v0[1], v0[0], '-', v1[1], v1[0], '-', v2[1], v2[0], '-', v3[1], v3[0], '-', v4[1], v4[0], '-', v5[1], v5[0], '-', v6[1], v6[0], '-', v7[1], v7[0], '\0'
				};
			}
			constexpr auto GetHexData()const noexcept -> std::array<char,43>
			{
				auto v1 = GetHexData1();
				auto v2 = GetHexData2();
				auto v3 = GetHexData3();
				auto v4 = GetHexData4();
				return std::array<char, 43>{
					v1[0], v1[1], v1[2], v1[3], v1[4], v1[5], v1[6], v1[7], '-', 
					v2[0], v2[1], v2[2], v2[3], '-',
					v3[0], v3[1], v3[2], v3[3], '-', 
					v4[0], v4[1], v4[2], v4[3], v4[4], v4[5], v4[6], v4[7], v4[8], v4[9], v4[10], v4[11], v4[12], v4[13], v4[14], v4[15], v4[16], v4[17], v4[18], v4[19], v4[20], v4[21], v4[22] ,'\0'
				};
			}
			auto GetString()const noexcept -> std::string {
				return std::string(GetHexData().data());
			}
			constexpr bool operator==(const RTUuid& v) const noexcept {
				if (GetLower() != v.GetLower()) { return false; }
				if (GetUpper() != v.GetUpper()) { return false; }
				return true;
			}
			constexpr bool operator!=(const RTUuid& v) const noexcept {
				return !operator==(v);
			}
		private:
			static constexpr auto GetHex(std::uint8_t v) noexcept -> std::array<char, 2>{
				std::uint8_t v0 = (v & static_cast<std::uint8_t>(0x0F));
				std::uint8_t v1 = (v & static_cast<std::uint8_t>(0xF0))>>4;
				std::array<char, 2> res = {};
				if (v0 <= 9) {
					res[0] = '0' + v0;
				}
				else {
					res[0] = 'a' + (v0 - 10);
				}
				if (v1 <= 9) {
					res[1] = '0' + v1;
				}
				else {
					res[1] = 'a' + (v1 - 10);
				}
				return res;
			}
			constexpr auto GetLower()const noexcept -> uint64_t {
				return (static_cast<std::uint64_t>(m_Value[0]) << 0) +
					   (static_cast<std::uint64_t>(m_Value[1]) << 8) +
					   (static_cast<std::uint64_t>(m_Value[2]) << 16) +
					   (static_cast<std::uint64_t>(m_Value[3]) << 24)+
					   (static_cast<std::uint64_t>(m_Value[4]) << 32) +
					   (static_cast<std::uint64_t>(m_Value[5]) << 40) +
					   (static_cast<std::uint64_t>(m_Value[6]) << 48) +
					   (static_cast<std::uint64_t>(m_Value[7]) << 56);
			}
			constexpr auto GetUpper()const noexcept -> uint64_t {
				return (static_cast<std::uint64_t>(m_Value[8]) << 0) +
					   (static_cast<std::uint64_t>(m_Value[9]) << 8) +
					   (static_cast<std::uint64_t>(m_Value[10]) << 16) +
					   (static_cast<std::uint64_t>(m_Value[11]) << 24)+
					   (static_cast<std::uint64_t>(m_Value[12]) << 32) +
					   (static_cast<std::uint64_t>(m_Value[13]) << 40) +
					   (static_cast<std::uint64_t>(m_Value[14]) << 48) +
					   (static_cast<std::uint64_t>(m_Value[15]) << 56);
			}
		private:
			std::array<unsigned char, 16> m_Value;
		};
	}
}
#endif