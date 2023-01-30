#include <RTLib/Core/RTSerialized.h>
#include <RTLib/Core/RTObject.h>
class MyObject : public RTLib::Core::RTObject
{
public:
	using base_type = RTLib::Core::RTObject;
	static inline constexpr std::string_view                   TypeString  = std::string_view("MyObject");
	static inline constexpr RTLib::Core::RTUuid                TypeID      = RTLib::Core::RTUuid(0x4e4797c, 0xa2f8, 0x4070, { 0xb5, 0x40, 0xc1, 0x86, 0x8d, 0xe0, 0xe9, 0x35 });
	static inline constexpr std::array<RTLib::Core::RTUuid, 2> TypeIndices = std::array<RTLib::Core::RTUuid, 2>{RTLib::Core::RTObject::TypeID, TypeID};

	virtual ~MyObject() noexcept {}

	virtual auto GetTypeID()     const noexcept -> RTLib::Core::RTUuid override {
		return MyObject::TypeID;
	}
	virtual auto GetTypeIndices()const noexcept -> std::vector<RTLib::Core::RTUuid> override {
		return std::vector<RTLib::Core::RTUuid>(std::begin(TypeIndices), std::end(TypeIndices));
	}
	virtual auto GetTypeString() const noexcept -> RTLib::Core::RTString override {
		return RTLib::Core::RTString(TypeString.data());
	}

};
int main()
{
	constexpr RTLib::Core::RTUuid v0 = { 0x4e4797c, 0xa2f8, 0x4070, { 0xb5, 0x40, 0xc1, 0x86, 0x8d, 0xe0, 0xe9, 0x35 } };
	constexpr RTLib::Core::RTUuid v1 = { 0xd95f7bbe, 0x44b3, 0x41ca, { 0xb8, 0x4e, 0x7a, 0xbe, 0xcf, 0xfe, 0x5f, 0x9a } };
	constexpr RTLib::Core::RTUuid v2 = { 0xd95f7bbe, 0x44b3, 0x41ca, { 0xb8, 0x4e, 0x7a, 0xbe, 0xcf, 0xfe, 0x5f, 0x9a } };
	static_assert(v0 != v1, ""); 
	static_assert(v1 == v2, "");
	constexpr auto v3 = v0.GetHexData();
	std::cout << v3.data() << std::endl;

	std::cout << RTLib::Core::RTObjectTraits<MyObject>::TypeIndices[0].GetString() << std::endl;
	std::cout << RTLib::Core::RTObjectTraits<MyObject>::TypeIndices[1].GetString() << std::endl;

	auto myObject = MyObject();

	auto ptObject = myObject.As<RTLib::Core::RTObject>();

	std::cout << ptObject->GetTypeID().GetString() << std::endl;
}