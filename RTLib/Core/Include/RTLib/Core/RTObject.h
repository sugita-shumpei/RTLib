#ifndef RTLIB_CORE_RT_OBJECT_H
#define RTLIB_CORE_RT_OBJECT_H
#include <RTLib/Core/RTDataType.h>
#include <RTLib/Core/RTUuid.h>
#include <vector>
namespace RTLib
{
	namespace Core
	{
		class RTObject
		{
		public:
			static inline constexpr auto TypeString = std::string_view("RTLib::Core::RTObject");
			// {DF774468-AAF1-4C2F-A458-36E3ED7113AF}
			static inline constexpr auto TypeID = RTLib::Core::RTUuid(0xdf774468, 0xaaf1, 0x4c2f, { 0xa4, 0x58, 0x36, 0xe3, 0xed, 0x71, 0x13, 0xaf });
			static inline constexpr auto TypeIndices = std::array<RTUuid, 1>{TypeID};

			virtual ~RTObject() noexcept {}

			virtual auto GetTypeID()   const noexcept -> RTUuid     = 0;
			virtual auto GetTypeString() const noexcept -> RTString = 0;
			virtual auto GetTypeIndices()const noexcept -> std::vector<RTUuid> = 0;

			template<typename Derived>
			auto As()const noexcept -> const Derived* {
				const auto typeIndices = GetTypeIndices();
				for (auto& typeID: typeIndices) {
					if (Derived::TypeID == typeID) {
						return static_cast<const Derived*>(this);
					}
				}
				return nullptr;
			}
			template<typename Derived>
			auto As() noexcept -> Derived* {
				const auto typeIndices = GetTypeIndices();
				for (auto& typeID : typeIndices) {
					if (Derived::TypeID == typeID) {
						return static_cast<Derived*>(this);
					}
				}
				return nullptr;
			}
		};
		template<typename RTObjectType>
		struct RTObjectTraits
		{
			static_assert(std::is_base_of_v<RTObject, RTObjectType>,"");
			using base_type = typename RTObjectType::base_type;
			static_assert(std::is_base_of_v<RTObjectType::base_type, RTObjectType>, "");
			static inline constexpr decltype(RTObjectType::TypeID)      TypeID     = RTObjectType::TypeID;
			static inline constexpr decltype(RTObjectType::TypeIndices) TypeIndices= RTObjectType::TypeIndices;
			static inline constexpr decltype(RTObjectType::TypeString)  TypeString = RTLib::Core::RTObject::TypeString;
		};
		template<>
		struct RTObjectTraits<RTLib::Core::RTObject>
		{
			static inline constexpr decltype(RTLib::Core::RTObject::TypeID)      TypeID      = RTLib::Core::RTObject::TypeID;
			static inline constexpr decltype(RTLib::Core::RTObject::TypeIndices) TypeIndices = RTLib::Core::RTObject::TypeIndices;
			static inline constexpr decltype(RTLib::Core::RTObject::TypeString)  TypeString  = RTLib::Core::RTObject::TypeString;
		};

	}
}
#endif