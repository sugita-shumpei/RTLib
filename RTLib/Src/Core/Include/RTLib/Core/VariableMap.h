#ifndef RTLIB_CORE_VARIABLE_MAP_H
#define RTLIB_CORE_VARIABLE_MAP_H
#include <string>
#include <array>
#include <unordered_map>
#include <memory>
#define RTLIB_CORE_VARIABLE_MAP_METHOD_DECLARE_SET(Name) void Set##Name(const std::string& keyName, const Internal##Name& value)noexcept { m_##Name##Data[keyName] = value; }
#define RTLIB_CORE_VARIABLE_MAP_METHOD_DECLARE_SET_FLOAT_N_FROM(CNT) template<typename T> void SetFloat##CNT##From(const std::string& keyName, const T& value)noexcept { \
    static_assert(sizeof(T)==sizeof(float)*CNT);\
    Internal##Float##CNT middle; \
    std::memcpy(&middle, &value, sizeof(float)*CNT);\
    return SetFloat##CNT(keyName,middle); \
}
#define RTLIB_CORE_VARIABLE_MAP_METHOD_DECLARE_GET(Name) auto Get##Name(const std::string& keyName)const -> Internal##Name { return m_##Name##Data.at(keyName); }
#define RTLIB_CORE_VARIABLE_MAP_METHOD_DECLARE_GET_FLOAT_N_AS(CNT) template<typename T> auto GetFloat##CNT##As(const std::string& keyName)const noexcept -> T{ \
    static_assert(sizeof(T)==sizeof(float)*CNT);\
    Internal##Float##CNT middle = GetFloat##CNT(keyName); \
    T value {}; \
    std::memcpy(&value, &middle, sizeof(float)*CNT);\
    return value; \
}
#define RTLIB_CORE_VARIABLE_MAP_METHOD_DECLARE_GET_OR(Name) auto Get##Name##Or(const std::string& keyName, Internal##Name defaultValue)const noexcept-> Internal##Name { \
    if (m_##Name##Data.count(keyName)>0){ return m_##Name##Data.at(keyName);} \
    return defaultValue;\
}
#define RTLIB_CORE_VARIABLE_MAP_METHOD_DECLARE_GET_FLOAT_N_AS(CNT) template<typename T> auto GetFloat##CNT##As(const std::string& keyName)const noexcept -> T{ \
    static_assert(sizeof(T)==sizeof(float)*CNT);\
    Internal##Float##CNT middle = GetFloat##CNT(keyName); \
    T value {}; \
    std::memcpy(&value, &middle, sizeof(float)*CNT);\
    return value; \
}
#define RTLIB_CORE_VARIABLE_MAP_METHOD_DECLARE_POP(Name) auto Pop##Name(const std::string& keyName)noexcept -> Internal##Name { \
    auto val = Get##Name(keyName); \
    m_##Name##Data.erase(keyName); \
    return val;\
}
#define RTLIB_CORE_VARIABLE_MAP_METHOD_DECLARE_HAS(Name) bool Has##Name(const std::string& keyName)const noexcept{ return m_##Name##Data.count(keyName) > 0; }

namespace RTLib
{
	namespace Core {
        class  VariableMap
        {
        private:
            using InternalUInt32 = uint32_t;
            using InternalBool = bool;
            using InternalFloat1 = float;
            using InternalFloat2 = std::array<float, 2>;
            using InternalFloat3 = std::array<float, 3>;
            using InternalFloat4 = std::array<float, 4>;
            //For String
            using InternalString = std::string;
        public:

            //Set
            RTLIB_CORE_VARIABLE_MAP_METHOD_DECLARE_SET(UInt32);
            RTLIB_CORE_VARIABLE_MAP_METHOD_DECLARE_SET(Bool);
            RTLIB_CORE_VARIABLE_MAP_METHOD_DECLARE_SET(Float1);
            RTLIB_CORE_VARIABLE_MAP_METHOD_DECLARE_SET(Float2);
            RTLIB_CORE_VARIABLE_MAP_METHOD_DECLARE_SET(Float3);
            RTLIB_CORE_VARIABLE_MAP_METHOD_DECLARE_SET(Float4);
            RTLIB_CORE_VARIABLE_MAP_METHOD_DECLARE_SET(String);
            RTLIB_CORE_VARIABLE_MAP_METHOD_DECLARE_SET_FLOAT_N_FROM(1);
            RTLIB_CORE_VARIABLE_MAP_METHOD_DECLARE_SET_FLOAT_N_FROM(2);
            RTLIB_CORE_VARIABLE_MAP_METHOD_DECLARE_SET_FLOAT_N_FROM(3);
            RTLIB_CORE_VARIABLE_MAP_METHOD_DECLARE_SET_FLOAT_N_FROM(4);
            //Get
            RTLIB_CORE_VARIABLE_MAP_METHOD_DECLARE_GET(UInt32);
            RTLIB_CORE_VARIABLE_MAP_METHOD_DECLARE_GET(Bool);
            RTLIB_CORE_VARIABLE_MAP_METHOD_DECLARE_GET(Float1);
            RTLIB_CORE_VARIABLE_MAP_METHOD_DECLARE_GET(Float2);
            RTLIB_CORE_VARIABLE_MAP_METHOD_DECLARE_GET(Float3);
            RTLIB_CORE_VARIABLE_MAP_METHOD_DECLARE_GET(Float4);
            RTLIB_CORE_VARIABLE_MAP_METHOD_DECLARE_GET(String);
            RTLIB_CORE_VARIABLE_MAP_METHOD_DECLARE_GET_OR(UInt32);
            RTLIB_CORE_VARIABLE_MAP_METHOD_DECLARE_GET_OR(Bool);
            RTLIB_CORE_VARIABLE_MAP_METHOD_DECLARE_GET_OR(Float1);
            RTLIB_CORE_VARIABLE_MAP_METHOD_DECLARE_GET_OR(Float2);
            RTLIB_CORE_VARIABLE_MAP_METHOD_DECLARE_GET_OR(Float3);
            RTLIB_CORE_VARIABLE_MAP_METHOD_DECLARE_GET_OR(Float4);
            RTLIB_CORE_VARIABLE_MAP_METHOD_DECLARE_GET_OR(String);
            RTLIB_CORE_VARIABLE_MAP_METHOD_DECLARE_GET_FLOAT_N_AS(1);
            RTLIB_CORE_VARIABLE_MAP_METHOD_DECLARE_GET_FLOAT_N_AS(2);
            RTLIB_CORE_VARIABLE_MAP_METHOD_DECLARE_GET_FLOAT_N_AS(3);
            RTLIB_CORE_VARIABLE_MAP_METHOD_DECLARE_GET_FLOAT_N_AS(4);
            //Pop
            RTLIB_CORE_VARIABLE_MAP_METHOD_DECLARE_POP(UInt32);
            RTLIB_CORE_VARIABLE_MAP_METHOD_DECLARE_POP(Bool);
            RTLIB_CORE_VARIABLE_MAP_METHOD_DECLARE_POP(Float1);
            RTLIB_CORE_VARIABLE_MAP_METHOD_DECLARE_POP(Float2);
            RTLIB_CORE_VARIABLE_MAP_METHOD_DECLARE_POP(Float3);
            RTLIB_CORE_VARIABLE_MAP_METHOD_DECLARE_POP(Float4);
            RTLIB_CORE_VARIABLE_MAP_METHOD_DECLARE_POP(String);
            //Has
            RTLIB_CORE_VARIABLE_MAP_METHOD_DECLARE_HAS(UInt32);
            RTLIB_CORE_VARIABLE_MAP_METHOD_DECLARE_HAS(Bool);
            RTLIB_CORE_VARIABLE_MAP_METHOD_DECLARE_HAS(Float1);
            RTLIB_CORE_VARIABLE_MAP_METHOD_DECLARE_HAS(Float2);
            RTLIB_CORE_VARIABLE_MAP_METHOD_DECLARE_HAS(Float3);
            RTLIB_CORE_VARIABLE_MAP_METHOD_DECLARE_HAS(Float4);
            RTLIB_CORE_VARIABLE_MAP_METHOD_DECLARE_HAS(String);

            auto EnumerateUInt32Data()const noexcept -> const std::unordered_map<std::string, uint32_t>& { return m_UInt32Data; }
            auto EnumerateBoolData()  const noexcept -> const std::unordered_map<std::string, bool    >& { return m_BoolData; }
            auto EnumerateFloat1Data()const noexcept -> const std::unordered_map<std::string, float               >& { return m_Float1Data; }
            auto EnumerateFloat2Data()const noexcept -> const std::unordered_map<std::string, std::array<float, 2>>& { return m_Float2Data; }
            auto EnumerateFloat3Data()const noexcept -> const std::unordered_map<std::string, std::array<float, 3>>& { return m_Float3Data; }
            auto EnumerateFloat4Data()const noexcept -> const std::unordered_map<std::string, std::array<float, 4>>& { return m_Float4Data; }
            auto EnumerateStringData()const noexcept -> const std::unordered_map<std::string, std::string>& { return m_StringData; }

            bool IsEmpty()const noexcept
            {
                return EnumerateBoolData().empty() &&
                    EnumerateUInt32Data().empty() &&
                    EnumerateFloat1Data().empty() &&
                    EnumerateFloat2Data().empty() &&
                    EnumerateFloat3Data().empty() &&
                    EnumerateFloat4Data().empty() &&
                    EnumerateStringData().empty();
            }
        private:
            std::unordered_map<std::string, uint32_t>              m_UInt32Data;
            std::unordered_map<std::string, bool>                  m_BoolData;
            std::unordered_map<std::string, float>                 m_Float1Data;
            std::unordered_map<std::string, std::array<float, 2>>  m_Float2Data;
            std::unordered_map<std::string, std::array<float, 3>>  m_Float3Data;
            std::unordered_map<std::string, std::array<float, 4>>  m_Float4Data;
            std::unordered_map<std::string, std::string>           m_StringData;
        };
        using  VariableMapList = std::vector<VariableMap>;
        using  VariableMapListPtr = std::shared_ptr<VariableMapList>;

        //Variable
        template<typename JsonType>
        inline void   to_json(JsonType& j, const RTLib::Core::VariableMap& v) {

            for (auto& [key, value] : v.EnumerateBoolData()) {
                j[key] = value;
            }
            for (auto& [key, value] : v.EnumerateFloat1Data()) {
                j[key] = value;
            }
            for (auto& [key, value] : v.EnumerateFloat2Data()) {
                j[key] = value;
            }
            for (auto& [key, value] : v.EnumerateFloat3Data()) {
                j[key] = value;
            }
            for (auto& [key, value] : v.EnumerateFloat4Data()) {
                j[key] = value;
            }
            for (auto& [key, value] : v.EnumerateUInt32Data()) {
                j[key] = value;
            }
            for (auto& [key, value] : v.EnumerateStringData()) {
                j[key] = value;
            }


        }
        template<typename JsonType>
        inline void from_json(const JsonType& j, RTLib::Core::VariableMap& v) {
            for (auto& elem : j.items()) {
                if (elem.value().is_string()) {
                    v.SetString(elem.key(), elem.value().get<std::string>());
                }
                if (elem.value().is_boolean()) {
                    v.SetBool(elem.key(), elem.value().get<bool>());
                }
                if (elem.value().is_number_integer()) {
                    v.SetUInt32(elem.key(), elem.value().get<unsigned int>());
                }
                if (elem.value().is_number_float()) {
                    v.SetFloat1(elem.key(), elem.value().get<float>());
                }
                if (elem.value().is_array()) {
                    auto size = elem.value().size();
                    switch (size) {
                    case 2:
                        v.SetFloat2(elem.key(), elem.value().get < std::array<float, 2>>());
                        break;
                    case 3:
                        v.SetFloat3(elem.key(), elem.value().get < std::array<float, 3>>());
                        break;
                    case 4:
                        v.SetFloat4(elem.key(), elem.value().get < std::array<float, 4>>());
                        break;
                    default:
                        break;
                    }
                }
            }
        }
        //VariableArray
        template<typename JsonType>
        inline void   to_json(JsonType& j, const std::vector<RTLib::Core::VariableMap>& v) {
            for (auto& elem : v) {
                auto elemJson = JsonType(v);
                j.push_back(elemJson);
            }
        }
        template<typename JsonType>
        inline void from_json(const JsonType& j, std::vector<RTLib::Core::VariableMap>& v) {
            v.clear();
            v.reserve(j.size());
            for (auto& elem : j) {
                v.push_back(elem.get<RTLib::Core::VariableMap>());
            }
        }
	}
}
#endif
