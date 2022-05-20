#ifndef RTLIB_CORE_BASE_OBJECT_H
#define RTLIB_CORE_BASE_OBJECT_H
#include <atomic>
#include <memory>
#include <string>
#include <array>
#define RTLIB_CORE_TYPE_OBJECT_DECLARE_DERIVED_METHOD(OBJECT,BASE_OBJECT, UUID_LIST) \
	static inline constexpr auto ThisTypeId()noexcept -> RTLib::Core::TypeId { return RTLib::Core::TypeId{UUID_LIST};} \
	static inline constexpr auto BaseTypeId()noexcept -> RTLib::Core::TypeId { return BASE_OBJECT::ThisTypeId(); } \
	virtual auto GetBaseTypeId()const noexcept -> RTLib::Core::TypeId override { return BaseTypeId(); } \
	virtual auto GetTypeId()const noexcept -> RTLib::Core::TypeId override { return ThisTypeId(); }

#define RTLIB_CORE_TYPE_OBJECT_DECLARE_BEGIN(OBJECT, BASE_OBJECT, UUID_LIST) \
class OBJECT: public BASE_OBJECT { \
public: \
	RTLIB_CORE_TYPE_OBJECT_DECLARE_DERIVED_METHOD(OBJECT, BASE_OBJECT, UUID_LIST) \
    virtual ~OBJECT()noexcept

#define RTLIB_CORE_TYPE_OBJECT_DECLARE_END() }

namespace RTLib
{
	namespace Core
	{
		using TypeId = std::array<uint8_t, 16>;
		using InstID = uint64_t;

		auto TypeId2String(const TypeId& typeId)->std::string;
		class BaseObject
		{
		public:
			static inline constexpr auto BaseTypeId()noexcept ->TypeId{
				return std::array<uint8_t, 16>{};
			}
			static inline constexpr auto ThisTypeId()noexcept ->TypeId{
				return std::array<uint8_t, 16>{};
			}
			BaseObject()noexcept;
			virtual ~BaseObject()noexcept;

			virtual auto GetBaseTypeId()       const noexcept -> TypeId { return BaseTypeId(); }
				    auto GetBaseTypeIdString() const noexcept -> std::string;

			virtual auto GetTypeId()           const noexcept -> TypeId { return ThisTypeId(); }
					auto GetTypeIdString()     const noexcept -> std::string;

					void SetName(const std::string& name)noexcept;
				    auto GetName()const noexcept -> std::string;
		private:
			struct                Impl;
			std::unique_ptr<Impl> m_Impl;
		};
	}
}
#endif