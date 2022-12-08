#include <RTLibCoreTestConfig.h>
#include <RTLib/Core/BaseObject.h>
#include <RTLib/Core/BinaryReader.h>
#include <RTLib/Core/Exceptions.h>
#include <array>
#include <iostream>
#include <fstream>
#include <vector>
#include <variant>
#include <unordered_set>
#include <stack>

class  FbxBinary {};
enum  class FbxBinaryAttributeTypeCode :uint8_t {
	eBool = 'C',
	eInt16 = 'Y',
	eInt32 = 'I',
	eInt64 = 'L',
	eFloat32 = 'F',
	eFloat64 = 'D',
	eArrayBool = 'b',
	eArrayInt32 = 'i',
	eArrayInt64 = 'l',
	eArrayFloat32 = 'f',
	eArrayFloat64 = 'd',
	eBin = 'R',
	eStr = 'S',
};
enum class FbxBinaryAttributeArrayEncoding :uint32_t {
	eDefault = 0,
	eZlib    = 1
};
using       FbxBinaryAttributeValueHeader = std::monostate;
struct      FbxBinaryAttributeArrayHeader {
	std::uint32_t                   numElements;
	FbxBinaryAttributeArrayEncoding encoding;
	std::uint32_t                   sizeInBytes;
};
static_assert(sizeof(FbxBinaryAttributeArrayHeader) == 12);
struct      FbxBinaryAttributeSpecialHeader {
	std::uint32_t sizeInBytes;

};
static_assert(sizeof(FbxBinaryAttributeSpecialHeader) == 4);
template<typename T>
struct FbxBinaryAttributeArrayValue {
	std::vector<T>    raw;
	std::vector<char> cmp;
};
struct FbxBinaryAttributeBinaryValue {
	std::vector<char> bin;
};
struct FbxBinaryAttributeStringValue {
	std::string       str;
};
struct FbxBinaryAttribute {
	struct Data {
		FbxBinaryAttributeTypeCode                                     type   = FbxBinaryAttributeTypeCode::eBool;
		std::variant<FbxBinaryAttributeValueHeader, FbxBinaryAttributeArrayHeader, FbxBinaryAttributeSpecialHeader> header = FbxBinaryAttributeValueHeader();
		std::variant<
			char, 
			int16_t, 
			int32_t, 
			int64_t, 
			float, 
			double,
			FbxBinaryAttributeArrayValue<char>,
			FbxBinaryAttributeArrayValue<int32_t>,
			FbxBinaryAttributeArrayValue<int64_t>,
			FbxBinaryAttributeArrayValue<float>,
			FbxBinaryAttributeArrayValue<double>,
			FbxBinaryAttributeBinaryValue,
			FbxBinaryAttributeStringValue
		> value = char('T');
	} data;
	struct Info {
		size_t sizeInBytes  = 0;
		bool   isCompressed = false;
	};
	auto GetBool()const -> char {
		return std::get<char>(data.value);
	}
	auto GetInt16()const -> int16_t {
		return std::get<int16_t>(data.value);
	}
	auto GetInt32()const -> int32_t {
		return std::get<int32_t>(data.value);
	}
	auto GetInt64()const -> int64_t {
		return std::get<int64_t>(data.value);
	}
	auto GetFloat32()const -> float {
		return std::get<float>(data.value);
	}
	auto GetFloat64()const -> double {
		return std::get<double>(data.value);
	}
	auto GetArrayBool()const  -> const FbxBinaryAttributeArrayValue<char>   & { return std::get< FbxBinaryAttributeArrayValue<char>>(data.value); }
	auto GetArrayInt32()const -> const FbxBinaryAttributeArrayValue<int32_t>& { return std::get< FbxBinaryAttributeArrayValue<int32_t>>(data.value); }
	auto GetArrayInt64()const -> const FbxBinaryAttributeArrayValue<int64_t>& { return std::get< FbxBinaryAttributeArrayValue<int64_t>>(data.value); }
	auto GetArrayFloat32()const -> const FbxBinaryAttributeArrayValue<float>& { return std::get< FbxBinaryAttributeArrayValue<float>>(data.value); }
	auto GetArrayFloat64()const -> const FbxBinaryAttributeArrayValue<double>& { return std::get< FbxBinaryAttributeArrayValue<double>>(data.value); }
	auto GetBinary()const -> const FbxBinaryAttributeBinaryValue& { return std::get<FbxBinaryAttributeBinaryValue>(data.value); }
	auto GetString()const -> const FbxBinaryAttributeStringValue& { return std::get<FbxBinaryAttributeStringValue>(data.value); }

	static constexpr bool IsValidType(uint8_t v) noexcept {
		if (v == static_cast<uint8_t>(FbxBinaryAttributeTypeCode::eBool)) { return true; }
		if (v == static_cast<uint8_t>(FbxBinaryAttributeTypeCode::eInt16)) { return true; }
		if (v == static_cast<uint8_t>(FbxBinaryAttributeTypeCode::eInt32)) { return true; }
		if (v == static_cast<uint8_t>(FbxBinaryAttributeTypeCode::eInt64)) { return true; }
		if (v == static_cast<uint8_t>(FbxBinaryAttributeTypeCode::eFloat32)) { return true; }
		if (v == static_cast<uint8_t>(FbxBinaryAttributeTypeCode::eFloat64)) { return true; }
		if (v == static_cast<uint8_t>(FbxBinaryAttributeTypeCode::eArrayBool)) { return true; }
		if (v == static_cast<uint8_t>(FbxBinaryAttributeTypeCode::eArrayInt32)) { return true; }
		if (v == static_cast<uint8_t>(FbxBinaryAttributeTypeCode::eArrayInt64)) { return true; }
		if (v == static_cast<uint8_t>(FbxBinaryAttributeTypeCode::eArrayFloat32)) { return true; }
		if (v == static_cast<uint8_t>(FbxBinaryAttributeTypeCode::eArrayFloat64)) { return true; }
		if (v == static_cast<uint8_t>(FbxBinaryAttributeTypeCode::eBin)) { return true; }
		if (v == static_cast<uint8_t>(FbxBinaryAttributeTypeCode::eStr)) { return true; }
		return false;
	}
	static bool ReadTypeCodeFromData(const char* pSrcData, size_t& pos, FbxBinaryAttributeTypeCode& typeCode, bool isAdvancedAfterRead = true)noexcept
	{
		uint8_t tv = 0;
		std::memcpy((char*) &tv, pSrcData, sizeof(tv));
		if (isAdvancedAfterRead) {
			pos += sizeof(tv);
		}
		if (!IsValidType(tv)) {
			return false;
		}
		typeCode = static_cast<FbxBinaryAttributeTypeCode>(tv);
		return true;
	}
	static bool ReadTypeCodeFromStream(std::istream& stream, FbxBinaryAttributeTypeCode& typeCode, bool isAdvancedAfterRead = true)noexcept
	{
		auto begPos = static_cast<size_t>(stream.tellg());
		uint8_t tv = 0;
		stream.read((char*)&tv, sizeof(tv));
		if (!isAdvancedAfterRead) {
			stream.seekg(begPos, std::ios::beg);
		}
		if (!IsValidType(tv)) {
			return false;
		}
		typeCode = static_cast<FbxBinaryAttributeTypeCode>(tv);
		return true;
	}
	template<typename T>
	static bool ReadValueFromData(const char* pSrcData, size_t& pos, T& singleValue, bool isAdvancedAfterRead = true) noexcept {

	}
	template<typename T>
	static bool ReadValueFromFromStream(std::istream& stream, T& singleValue, bool isAdvancedAfterRead = true) noexcept {

	}
	static bool ReadSpecialHeaderFromData(const char* pSrcData, size_t& pos, FbxBinaryAttributeSpecialHeader& header, bool isAdvancedAfterRead = true) noexcept
	{

	}
	static bool ReadSpecialHeaderFromStream(std::istream& stream, FbxBinaryAttributeSpecialHeader& header, bool isAdvancedAfterRead = true) noexcept
	{

	}
	static bool ReadFromData(const char* pSrcData, size_t& pos, FbxBinaryAttribute& attribute, bool isAdvancedAfterRead = true)noexcept
	{
		auto begPos = pos;
		if (!ReadTypeCodeFromData(pSrcData, pos, attribute.data.type)) {
			if (!isAdvancedAfterRead) {
				pos = begPos;
			}
			return false;
		}
		if (attribute.data.type == FbxBinaryAttributeTypeCode::eBool) {
			attribute.data.header = FbxBinaryAttributeValueHeader();
			char res = 'T';
			bool isLoaded = !ReadValueFromData(pSrcData, pos, res);
			attribute.data.value = res;
			if (!isLoaded) {
				if (!isAdvancedAfterRead) {
					pos = begPos;
				}
			}
			return isLoaded;
		}
		if (attribute.data.type == FbxBinaryAttributeTypeCode::eInt16) {
			attribute.data.header = FbxBinaryAttributeValueHeader();
			int16_t res = 0;
			bool isLoaded = !ReadValueFromData(pSrcData, pos, res);
			attribute.data.value = res;
			if (!isLoaded) {
				if (!isAdvancedAfterRead) {
					pos = begPos;
				}
			}
			return isLoaded;
		}
		if (attribute.data.type == FbxBinaryAttributeTypeCode::eInt32) {
			attribute.data.header = FbxBinaryAttributeValueHeader();
			int32_t res = 0;
			bool isLoaded = !ReadValueFromData(pSrcData, pos, res);
			attribute.data.value = res;
			if (!isLoaded) {
				if (!isAdvancedAfterRead) {
					pos = begPos;
				}
			}
			return isLoaded;
		}
		if (attribute.data.type == FbxBinaryAttributeTypeCode::eInt64) {
			attribute.data.header = FbxBinaryAttributeValueHeader();
			int64_t res = 0;
			bool isLoaded = !ReadValueFromData(pSrcData, pos, res);
			attribute.data.value = res;
			if (!isLoaded) {
				if (!isAdvancedAfterRead) {
					pos = begPos;
				}
			}
			return isLoaded;
		}
		if (attribute.data.type == FbxBinaryAttributeTypeCode::eFloat32) {
			attribute.data.header = FbxBinaryAttributeValueHeader();
			float res = 0;
			bool isLoaded = !ReadValueFromData(pSrcData, pos, res);
			attribute.data.value = res;
			if (!isLoaded) {
				if (!isAdvancedAfterRead) {
					pos = begPos;
				}
			}
			return isLoaded;
		}
		if (attribute.data.type == FbxBinaryAttributeTypeCode::eFloat64) {
			attribute.data.header = FbxBinaryAttributeValueHeader();
			double res = 0;
			bool isLoaded = !ReadValueFromData(pSrcData, pos, res);
			attribute.data.value = res;
			if (!isLoaded) {
				if (!isAdvancedAfterRead) {
					pos = begPos;
				}
			}
			return isLoaded;
		}
		if (attribute.data.type == FbxBinaryAttributeTypeCode::eBin) {

		}
		return true;
	}

};
struct FbxBinaryAttributeArray {
	struct Data {
		std::vector<FbxBinaryAttribute> values;
	} data;
};
struct FbxBinaryNode;
struct FbxBinaryNodeArray {
	struct Data {
		std::vector<FbxBinaryNode> values = {};
		std::array<uint8_t, 25>    endMarker = {};
	} data = {};
	static bool ReadFromData(const char* pSrcData, size_t& pos, FbxBinaryNodeArray& nodeArray, uint32_t fbxVersion, bool isAdvancedAfterRead = true)noexcept;
	static bool ReadFromStream(std::istream& stream, FbxBinaryNodeArray& nodeArray, uint32_t fbxVersion, bool isAdvancedAfterRead = true)noexcept;
};
struct FbxBinaryNodeHeader {
	struct DataV1 {
		std::uint32_t endPos;
		std::uint32_t numAttributes;
		std::uint32_t sumAttributesInBytes;
		std::uint8_t  lenNodeName;
	};
	struct DataV2 {
		std::uint64_t endPos;
		std::uint64_t numAttributes;
		std::uint64_t sumAttributesInBytes;
		std::uint8_t  lenNodeName;
	};
	std::variant<DataV1, DataV2> data = DataV1();
};

struct FbxBinaryNode {
	FbxBinaryNodeHeader        header     = {};
	std::string                name       = "";
	FbxBinaryAttributeArray    attributes = {};
	FbxBinaryNodeArray         nodes      = {};

	static bool ReadHeaderFromData(const char* pSrcData, size_t& pos, FbxBinaryNodeHeader& header, uint32_t fbxVersion, bool isAdvancedAfterRead = true)noexcept
	{
		if ((fbxVersion / 100) >= 75) {
			size_t tmpPos = pos;
			FbxBinaryNodeHeader::DataV2 data = {};
			std::memcpy((char*)&data.endPos              , pSrcData + tmpPos, sizeof(data.endPos)       );
			tmpPos += sizeof(data.endPos);
			std::memcpy((char*)&data.numAttributes       , pSrcData + tmpPos, sizeof(data.numAttributes));
			tmpPos += sizeof(data.numAttributes);
			std::memcpy((char*)&data.sumAttributesInBytes, pSrcData + tmpPos, sizeof(data.endPos)       );
			tmpPos += sizeof(data.sumAttributesInBytes);
			std::memcpy((char*)&data.lenNodeName         , pSrcData + tmpPos, sizeof(data.lenNodeName  ));
			tmpPos += sizeof(data.lenNodeName);
			if (isAdvancedAfterRead) {
				pos = tmpPos;
			}
			header.data = data;
			if (data.endPos      < 27) { return false; }
			if (data.lenNodeName == 0) { return false; }
			return true;
		}
		else {
			size_t tmpPos = pos;
			FbxBinaryNodeHeader::DataV1 data = {};
			std::memcpy((char*)&data.endPos, pSrcData + tmpPos, sizeof(data.endPos));
			tmpPos += sizeof(data.endPos);
			std::memcpy((char*)&data.numAttributes, pSrcData + tmpPos, sizeof(data.numAttributes));
			tmpPos += sizeof(data.numAttributes);
			std::memcpy((char*)&data.sumAttributesInBytes, pSrcData + tmpPos, sizeof(data.endPos));
			tmpPos += sizeof(data.sumAttributesInBytes);
			std::memcpy((char*)&data.lenNodeName, pSrcData + tmpPos, sizeof(data.lenNodeName));
			tmpPos += sizeof(data.lenNodeName);
			if (isAdvancedAfterRead) {
				pos = tmpPos;
			}
			header.data = data;
			if (data.endPos < 27) { return false; }
			if (data.lenNodeName == 0) { return false; }
			return true;
		}
	}
	static bool ReadHeaderFromStream(std::istream& stream, FbxBinaryNodeHeader& header, uint32_t fbxVersion, bool isAdvancedAfterRead = true)noexcept {
		if ((fbxVersion / 100) >= 75) {
			size_t begPos = static_cast<size_t>(stream.tellg());
			FbxBinaryNodeHeader::DataV2 data = {};
			stream.read((char*)&data.endPos              , sizeof(data.endPos)       );
			stream.read((char*)&data.numAttributes       , sizeof(data.numAttributes));
			stream.read((char*)&data.sumAttributesInBytes, sizeof(data.sumAttributesInBytes));;
			stream.read((char*)&data.lenNodeName         , sizeof(data.lenNodeName));
			if (!isAdvancedAfterRead) {
				stream.seekg(begPos, std::ios::beg);
			}
			header.data = data;
			if (data.endPos < 27) { return false; }
			if (data.lenNodeName == 0) { return false; }
			return true;
		}
		else {
			size_t begPos = static_cast<size_t>(stream.tellg());
			FbxBinaryNodeHeader::DataV1 data = {};
			stream.read((char*)&data.endPos, sizeof(data.endPos));
			stream.read((char*)&data.numAttributes, sizeof(data.numAttributes));
			stream.read((char*)&data.sumAttributesInBytes, sizeof(data.sumAttributesInBytes));;
			stream.read((char*)&data.lenNodeName, sizeof(data.lenNodeName));
			if (!isAdvancedAfterRead) {
				stream.seekg(begPos, std::ios::beg);
			}
			header.data = data;
			if (data.endPos < 27) { return false; }
			if (data.lenNodeName == 0) { return false; }
			return true;
		}
	}
	
	static bool ReadStringFromData(const char* pSrcData, size_t& pos, std::string& name, bool isAdvancedAfterRead = true)noexcept
	{
		if (name.empty()) { return false; }
		size_t lenNodeName = name.size();
		std::memcpy(name.data(), pSrcData + pos, lenNodeName);
		name.resize(lenNodeName);
		if (isAdvancedAfterRead)
		{
			pos += lenNodeName;
		}
		return true;
	}
	static bool ReadStringFromStream(std::istream& stream, std::string& name, bool isAdvancedAfterRead = true)noexcept
	{
		if (name.empty()) { return false; }
		size_t lenNodeName = name.size();
		auto begPos = static_cast<size_t>(stream.tellg());
		stream.read((char*)name.data(), lenNodeName);
		name.resize(lenNodeName);
		if (!isAdvancedAfterRead)
		{
			stream.seekg(begPos, std::ios::beg);
		}
		return true;
	}
	
	static bool ReadFromData(const char* pSrcData, size_t& pos, FbxBinaryNode& nodeData, uint32_t fbxVersion, bool isAdvancedAfterRead = true)noexcept {
		auto begPos = pos;
		if (!ReadHeaderFromData(pSrcData, pos, nodeData.header, fbxVersion)) { 
			if (!isAdvancedAfterRead) {
				pos = begPos;
			}
			return false; 
		}

		size_t sumAttributesInBytes = 0;
		size_t endPos = 0;
		if (std::get_if<0>(&nodeData.header.data)) {
			nodeData.name.resize(std::get<0>(nodeData.header.data).lenNodeName);
			sumAttributesInBytes = std::get<0>(nodeData.header.data).sumAttributesInBytes;
			endPos = std::get<0>(nodeData.header.data).endPos;
		}
		else {
			nodeData.name.resize(std::get<1>(nodeData.header.data).lenNodeName);
			sumAttributesInBytes = std::get<1>(nodeData.header.data).sumAttributesInBytes;
			endPos = std::get<0>(nodeData.header.data).endPos;
		}
		if (!ReadStringFromData(pSrcData, pos, nodeData.name)) { 
			if (!isAdvancedAfterRead) {
				pos = begPos;
			}
			return false; 
		}
		pos += sumAttributesInBytes;
		if (pos != endPos) {
			if (!FbxBinaryNodeArray::ReadFromData(pSrcData, pos, nodeData.nodes, fbxVersion)) {
				if (!isAdvancedAfterRead) {
					pos = begPos;
				}
				else {
					pos = endPos;
				}
				return false;
			}
		}
		if (!isAdvancedAfterRead) {
			pos = begPos;
		}
		else {
			pos = endPos;
		}
		return true;
	}
	static bool ReadFromStream(std::istream& stream, FbxBinaryNode& nodeData, uint32_t fbxVersion, bool isAdvancedAfterRead = true)noexcept
	{
		auto begPos = static_cast<size_t>(stream.tellg());
		if (!ReadHeaderFromStream(stream, nodeData.header, fbxVersion)) { 
			if (!isAdvancedAfterRead) {
				stream.seekg(begPos, std::ios::beg);
			}
			return false; 
		}
		size_t sumAttributesInBytes = 0;
		size_t endPos = 0;
		if (std::get_if<0>(&nodeData.header.data)) {
			nodeData.name.resize(std::get<0>(nodeData.header.data).lenNodeName);
			sumAttributesInBytes = std::get<0>(nodeData.header.data).sumAttributesInBytes;
			endPos = std::get<0>(nodeData.header.data).endPos;
		}
		else {
			nodeData.name.resize(std::get<1>(nodeData.header.data).lenNodeName);
			sumAttributesInBytes = std::get<1>(nodeData.header.data).sumAttributesInBytes;
			endPos = std::get<0>(nodeData.header.data).endPos;
		}
		if (!ReadStringFromStream(stream, nodeData.name)) { 
			if (!isAdvancedAfterRead) {
				stream.seekg(begPos, std::ios::beg);
			}
			return false; 
		}
		stream.seekg(static_cast<size_t>(stream.tellg()) + sumAttributesInBytes, std::ios::beg);
		if (static_cast<size_t>(stream.tellg())!=endPos) {
			if (!FbxBinaryNodeArray::ReadFromStream(stream, nodeData.nodes, fbxVersion)) {
				if (!isAdvancedAfterRead) {
					stream.seekg(begPos, std::ios::beg);
				}
				else {
					stream.seekg(endPos, std::ios::beg);
				}
				return false;
			}
		}
		
		if (!isAdvancedAfterRead) {
			stream.seekg(begPos, std::ios::beg);
		}
		else {
			stream.seekg(endPos, std::ios::beg);
		}
		return true;
	}

};

bool FbxBinaryNodeArray::ReadFromData(const char* pSrcData, size_t& pos, FbxBinaryNodeArray& nodeArray, uint32_t fbxVersion, bool isAdvancedAfterRead)noexcept
{
	nodeArray.data.values.clear();
	std::vector<size_t> posOffsets = {};
	size_t begPos = pos;
	size_t tmpPos = begPos;
	bool isEndMarkerValid = false;
	do {
		if ((fbxVersion / 100) >= 75) {
			std::memcpy((char*)&nodeArray.data.endMarker, pSrcData + tmpPos, 25);
			if (nodeArray.data.endMarker == std::array<uint8_t, 25>{}) {
				isEndMarkerValid = true;
				tmpPos += 25;
			}
			else {
				posOffsets.push_back(tmpPos);
				uint64_t endPos = 0;
				std::memcpy((char*)&endPos, pSrcData + tmpPos, sizeof(endPos));
				tmpPos = endPos;
			}
		}
		else {
			std::memcpy((char*)&nodeArray.data.endMarker, pSrcData + tmpPos, 13);
			if (nodeArray.data.endMarker == std::array<uint8_t, 25>{}) {
				isEndMarkerValid = true;
				tmpPos += 13;
			}
			else {
				posOffsets.push_back(tmpPos);
				uint32_t endPos = 0;
				std::memcpy((char*)&endPos, pSrcData + tmpPos, sizeof(endPos));
				tmpPos = endPos;
			}
		}
	} while (!isEndMarkerValid);
	nodeArray.data.values.reserve(posOffsets.size());
	bool isLoadNode = true;
	for (auto& posOffset : posOffsets)
	{
		auto tmpPosOffset = posOffset;
		FbxBinaryNode node;
		if (!FbxBinaryNode::ReadFromData(pSrcData, tmpPosOffset, node, false)) {
			std::cerr << "Failed To Load Binary Node!\n";
			isLoadNode = false;
		}
		else {
			nodeArray.data.values.emplace_back(std::move(node));
		}
	}
	if (!isAdvancedAfterRead) { 
		pos = begPos;
	}
	else {
		pos = tmpPos;
	}
	if (!isLoadNode) { return false; }
	return isEndMarkerValid;
}
bool FbxBinaryNodeArray::ReadFromStream(std::istream& stream, FbxBinaryNodeArray& nodeArray, uint32_t fbxVersion, bool isAdvancedAfterRead)noexcept
{
	nodeArray.data.values.clear();
	std::vector<size_t> posOffsets = {};
	size_t begPos = stream.tellg();
	size_t tmpPos = begPos;
	bool isEndMarkerValid = false;
	do {
		if ((fbxVersion / 100) >= 75) {
			stream.read((char*)&nodeArray.data.endMarker, 25);
			if (nodeArray.data.endMarker == std::array<uint8_t, 25>{}) {
				isEndMarkerValid = true;
				tmpPos += 25;
			}
			else {
				posOffsets.push_back(tmpPos);
				uint64_t endPos = 0;
				stream.seekg(tmpPos, std::ios::beg);
				stream.read((char*)&endPos, sizeof(endPos));
				stream.seekg(endPos, std::ios::beg);
				tmpPos = endPos;
			}
		}
		else {
			stream.read((char*)&nodeArray.data.endMarker, 13);
			if (nodeArray.data.endMarker == std::array<uint8_t, 25>{}) {
				isEndMarkerValid = true;
				tmpPos += 13;
			}
			else {
				posOffsets.push_back(tmpPos);
				uint32_t endPos = 0;
				stream.seekg(tmpPos, std::ios::beg);
				stream.read((char*)&endPos, sizeof(endPos));
				stream.seekg(endPos, std::ios::beg);
				tmpPos = endPos;
			}
		}
	} while (!isEndMarkerValid);;
	nodeArray.data.values.reserve(posOffsets.size());
	bool isLoadNode = true;
	for (auto& posOffset : posOffsets)
	{
		auto tmpPosOffset = posOffset;
		FbxBinaryNode node;
		stream.seekg(tmpPosOffset, std::ios::beg);
		if (!FbxBinaryNode::ReadFromStream(stream, node, false)) {
			std::cerr << "Failed To Load Binary Node!\n";
			isLoadNode = false;
		}
		else {
			nodeArray.data.values.emplace_back(std::move(node));
		}
	}
	if (!isAdvancedAfterRead) {
		stream.seekg(begPos, std::ios::beg);
	}
	else {
		stream.seekg(tmpPos, std::ios::beg);
	}
	if (!isLoadNode) { return false; }
	return isEndMarkerValid;
}

struct FbxBinaryData {
	std::array<uint8_t, 23> magick;
	std::uint32_t           fbxVersion1;
	//Beg+27
	FbxBinaryNodeArray      topLevelNodes;
	std::array<uint8_t, 16> footer1;
	std::vector<char>       padding;
	//Beg-144
	std::array<uint8_t, 4>  footer2;
	std::uint32_t           fbxVersion2;
	std::array<uint8_t,120> footer3;
	std::array<uint8_t,16>  footer4;

	static bool ReadMagickFromData(const char* pSrcData, size_t& pos, std::array<uint8_t, 23>& magick, bool isAdvancedAfterRead = true) noexcept
	{
		std::memcpy((char*)&magick, pSrcData+ pos, sizeof(magick));
		if (magick == kMagick) {
			if (isAdvancedAfterRead) {
				pos += sizeof(magick);
			}
			return true;
		}
		else {
			return false;
		}
	}
	static bool ReadMagickFromStream(std::istream& stream, std::array<uint8_t, 23>& magick, bool isAdvancedAfterRead = true) noexcept
	{
		stream.read((char*)&magick, sizeof(magick));
		if (magick == kMagick) {
			if (!isAdvancedAfterRead) {
				size_t curPos = static_cast<size_t>(stream.tellg());
				stream.seekg(curPos - sizeof(magick), std::ios::beg);
			}
			return true;
		}
		else {
			return false;
		}
	}

	static bool ReadFbxVersionFromData(const char* pSrcData, size_t& pos, uint32_t& fbxVersion, bool isAdvancedAfterRead = true) noexcept
	{
		std::memcpy((char*)&fbxVersion, pSrcData + pos, sizeof(fbxVersion));
		if ((fbxVersion/100==73)||
			(fbxVersion/100==74)||
			(fbxVersion/100==75)) {
			if (isAdvancedAfterRead) {
				pos += sizeof(fbxVersion);
			}
			return true;
		}
		else {
			return false;
		}
	}
	static bool ReadFbxVersionFromStream(std::istream& stream, uint32_t& fbxVersion, bool isAdvancedAfterRead = true) noexcept
	{
		stream.read((char*)&fbxVersion, sizeof(fbxVersion));
		if ((fbxVersion / 100 == 73) ||
			(fbxVersion / 100 == 74) ||
			(fbxVersion / 100 == 75)) {
			if (!isAdvancedAfterRead) {
				size_t curPos = static_cast<size_t>(stream.tellg());
				stream.seekg(curPos - sizeof(fbxVersion), std::ios::beg);
			}
			return true;
		}
		else {
			return false;
		}
	}

	static bool ReadFooter1FromData(const char* pSrcData, size_t& pos, std::array<uint8_t, 16>& footer1, bool isAdvancedAfterRead = true)
	{
		std::memcpy((char*)&footer1, pSrcData + pos, sizeof(footer1));
		if (isAdvancedAfterRead) {
			pos += sizeof(footer1);
		}
		if (((footer1[ 0] & 0xf0) != kFooter1[ 0]) ||
			((footer1[ 1] & 0xf0) != kFooter1[ 1]) ||
			((footer1[ 2] & 0xf0) != kFooter1[ 2]) ||
			((footer1[ 3] & 0xf0) != kFooter1[ 3]) ||
			((footer1[ 4] & 0xf0) != kFooter1[ 4]) ||
			((footer1[ 5] & 0xf0) != kFooter1[ 5]) ||
			((footer1[ 6] & 0xf0) != kFooter1[ 6]) ||
			((footer1[ 7] & 0xf0) != kFooter1[ 7]) ||
			((footer1[ 8] & 0xf0) != kFooter1[ 8]) ||
			((footer1[ 9] & 0xf0) != kFooter1[ 9]) ||
			((footer1[10] & 0xf0) != kFooter1[10]) ||
			((footer1[11] & 0xf0) != kFooter1[11]) ||
			((footer1[12] & 0xf0) != kFooter1[12]) ||
			((footer1[13] & 0xf0) != kFooter1[13]) ||
			((footer1[14] & 0xf0) != kFooter1[14]) ||
			((footer1[15] & 0xf0) != kFooter1[15])) {
			return false;
		}
		else {
			return true;
		}
	}
	static bool ReadFooter1FromStream(std::istream& stream, std::array<uint8_t, 16>& footer1, bool isAdvancedAfterRead = true)
	{
		stream.read((char*)&footer1, sizeof(footer1));
		if (!isAdvancedAfterRead) {
			size_t curPos = static_cast<size_t>(stream.tellg());
			stream.seekg(curPos - sizeof(footer1), std::ios::beg);
		}
		if (((footer1[ 0] & 0xf0) != kFooter1[ 0]) ||
			((footer1[ 1] & 0xf0) != kFooter1[ 1]) ||
			((footer1[ 2] & 0xf0) != kFooter1[ 2]) ||
			((footer1[ 3] & 0xf0) != kFooter1[ 3]) ||
			((footer1[ 4] & 0xf0) != kFooter1[ 4]) ||
			((footer1[ 5] & 0xf0) != kFooter1[ 5]) ||
			((footer1[ 6] & 0xf0) != kFooter1[ 6]) ||
			((footer1[ 7] & 0xf0) != kFooter1[ 7]) ||
			((footer1[ 8] & 0xf0) != kFooter1[ 8]) ||
			((footer1[ 9] & 0xf0) != kFooter1[ 9]) ||
			((footer1[10] & 0xf0) != kFooter1[10]) ||
			((footer1[11] & 0xf0) != kFooter1[11]) ||
			((footer1[12] & 0xf0) != kFooter1[12]) ||
			((footer1[13] & 0xf0) != kFooter1[13]) ||
			((footer1[14] & 0xf0) != kFooter1[14]) ||
			((footer1[15] & 0xf0) != kFooter1[15])) {
			return false;
		}
		else {
			return true;
		}
	}

	static bool ReadFooter2FromData(const char* pSrcData, size_t& pos, std::array<uint8_t, 4>& footer2, bool isAdvancedAfterRead = true)
	{
		std::memcpy((char*)&footer2, pSrcData + pos, sizeof(footer2));
		if (footer2 == kFooter2) {
			if ( isAdvancedAfterRead) {
				pos += sizeof(footer2);
			}
			return true;
		}
		else {
			return false;
		}
	}
	static bool ReadFooter2FromStream(std::istream& stream, std::array<uint8_t, 4>& footer2, bool isAdvancedAfterRead = true)
	{
		stream.read((char*)&footer2, sizeof(footer2));
		if (footer2== kFooter2) {
			if (!isAdvancedAfterRead) {
				size_t curPos = static_cast<size_t>(stream.tellg());
				stream.seekg(curPos - sizeof(footer2), std::ios::beg);
			}
			return true;
		}
		else {
			return false;
		}
	}

	static bool ReadFooter3FromData(const char* pSrcData, size_t& pos, std::array<uint8_t, 120>& footer3, bool isAdvancedAfterRead = true)
	{
		std::memcpy((char*)&footer3, pSrcData + pos, sizeof(footer3));
		if (footer3 == kFooter3) {
			if (isAdvancedAfterRead) {
				pos += sizeof(footer3);
			}
			return true;
		}
		else {
			return false;
		}
	}
	static bool ReadFooter3FromStream(std::istream& stream, std::array<uint8_t, 120>& footer3, bool isAdvancedAfterRead = true)
	{
		stream.read((char*)&footer3, sizeof(footer3));
		if (footer3 == kFooter3) {
			if (!isAdvancedAfterRead) {
				size_t curPos = static_cast<size_t>(stream.tellg());
				stream.seekg(curPos - sizeof(footer3), std::ios::beg);
			}
			return true;
		}
		else {
			return false;
		}
	}

	static bool ReadFooter4FromData(const char* pSrcData, size_t& pos, std::array<uint8_t, 16>& footer4, bool isAdvancedAfterRead = true)
	{
		std::memcpy((char*)&footer4, pSrcData + pos, sizeof(footer4));
		if (footer4 == kFooter4){
			if (isAdvancedAfterRead) {
				pos += sizeof(footer4);
			}
			return true;
		}
		else {
			return false;
		}
	}
	static bool ReadFooter4FromStream(std::istream& stream, std::array<uint8_t, 16>& footer4, bool isAdvancedAfterRead = true)
	{
		stream.read((char*)&footer4, sizeof(footer4));
		if (footer4 == kFooter4){
			if (!isAdvancedAfterRead) {
				size_t curPos = static_cast<size_t>(stream.tellg());
				stream.seekg(curPos - sizeof(footer4), std::ios::beg);
			}
			return true;
		}
		else {
			return false;
		}
	}

	static bool ReadPaddingFromData(const char* pSrcData, size_t& pos, std::vector<char>& padding, bool isAdvancedAfterRead = true)noexcept
	{
		std::memcpy((char*)padding.data(), pSrcData + pos, padding.size());
		bool isZeroPaddings = false;
		for (auto& pad : padding) {
			if (pad != 0) {
				isZeroPaddings = true;
			}
		}
		if (isAdvancedAfterRead) {
			pos += padding.size();
		}
		return !isZeroPaddings;
	}
	static bool ReadPaddingFromStream(std::istream& stream, std::vector<char>& padding, bool isAdvancedAfterRead = true)noexcept
	{
		auto begPos = static_cast<size_t>(stream.tellg());
		stream.read((char*)padding.data(), padding.size());
		bool isZeroPaddings = false;
		for (auto& pad : padding) {
			if (pad != 0) {
				isZeroPaddings = true;
			}
		}
		if (!isAdvancedAfterRead) {
			stream.seekg(begPos, std::ios::beg);
		}
		return !isZeroPaddings;
	}

	static bool ReadFromData(const char* pSrcData, size_t srcDataSizeInBytes, FbxBinaryData& data)noexcept
	{
		size_t pos = 0;
		if (!ReadMagickFromData(pSrcData,pos, data.magick)) { return false; }
		if (!ReadFbxVersionFromData(pSrcData, pos, data.fbxVersion1)) { return false; }
		pos = srcDataSizeInBytes - 144;
		if (!ReadFooter2FromData(pSrcData, pos, data.footer2)) { return false; }
		if (!ReadFbxVersionFromData(pSrcData, pos, data.fbxVersion2)) { return false; }
		if (data.fbxVersion1 != data.fbxVersion2) { return false; }
		if (!ReadFooter3FromData(pSrcData, pos, data.footer3)) { return false; }
		if (!ReadFooter4FromData(pSrcData, pos, data.footer4)) { return false; }
		pos = 27;
		if (!FbxBinaryNodeArray::ReadFromData(pSrcData, pos, data.topLevelNodes, data.fbxVersion1)){return false;}
		if (!ReadFooter1FromData(pSrcData, pos, data.footer1)) { return false; }
		size_t paddingSize = srcDataSizeInBytes - 144- pos;
		data.padding.resize(paddingSize);
		if (!ReadPaddingFromData(pSrcData, pos, data.padding)) { return false; }
		return true;
	}
	static bool ReadFromStream(std::istream& stream, FbxBinaryData& data)noexcept
	{
		if (stream.fail()) { return false; }
		stream.seekg(0L, std::ios::end);
		size_t sizeInBytes = static_cast<size_t>(stream.tellg());
		stream.seekg(0L, std::ios::beg);
		if (!ReadMagickFromStream(stream, data.magick)) { return false; }
		if (!ReadFbxVersionFromStream(stream, data.fbxVersion1)) { return false; }
		stream.seekg(-144L, std::ios::end);
		if (!ReadFooter2FromStream(stream, data.footer2)) { return false; }
		if (!ReadFbxVersionFromStream(stream, data.fbxVersion2)) { return false; }
		if (data.fbxVersion1 != data.fbxVersion2) { return false; }
		if (!ReadFooter3FromStream(stream, data.footer3)) { return false; }
		if (!ReadFooter4FromStream(stream, data.footer4)) { return false; }
		stream.seekg(27, std::ios::beg);
		if (!FbxBinaryNodeArray::ReadFromStream(stream, data.topLevelNodes, data.fbxVersion1)) { return false; }
		if (!ReadFooter1FromStream(stream, data.footer1)) { return false; }
		size_t paddingSize = sizeInBytes - 144 - static_cast<size_t>(stream.tellg());
		data.padding.resize(paddingSize);
		if (!ReadPaddingFromStream(stream, data.padding)) { return false; }
		return true;
	}
private:
	static inline constexpr std::array<uint8_t, 23> kMagick =
	{ 0x4b,0x61,0x79,0x64,0x61,0x72,0x61,0x20,0x46,0x42,0x58,0x20,0x42,0x69,0x6e,0x61,0x72,0x79,0x20,0x20,0x00,0x1a,0x00 };
	static inline constexpr std::array<uint8_t, 16> kFooter1 =
	{ 0xf0,0xb0,0xa0,0x00,0xd0,0xc0,0xd0,0x60,0xb0,0x70,0xf0,0x80,0x10,0xf0,0x20,0x70 };
	static inline constexpr std::array<uint8_t,  4> kFooter2 =
	{ };
	static inline constexpr std::array<uint8_t,120> kFooter3 =
	{ };
	static inline constexpr std::array<uint8_t, 16> kFooter4 = 
	{ 0xf8,0x5a,0x8c,0x6a,0xde,0xf5,0xd9,0x7e,0xec,0xe9,0x0c,0xe3,0x75,0x8f,0x29,0x0b };
};
int main()
{
	{
		FbxBinaryData fbxBinaryData;
		std::ifstream file(RTLIB_CORE_TEST_CONFIG_DATA_PATH"\\Models\\SampleBox\\SampleBox.fbx", std::ios::binary);
		file.seekg(0L, std::ios::end);
		auto fileSize = static_cast<size_t>(file.tellg());
		file.seekg(0L, std::ios::beg);
		std::unique_ptr<char[]> data(new char[fileSize]);
		file.read(data.get(), fileSize);
		if (!FbxBinaryData::ReadFromData(data.get(),fileSize, fbxBinaryData)) {
			
			std::cerr << "Failed To Load Fbx File!\n";
		}
		file.close();
	}
	{
		FbxBinaryData fbxBinaryData;
		std::ifstream file(RTLIB_CORE_TEST_CONFIG_DATA_PATH"\\Models\\SampleBox\\SampleBox.fbx", std::ios::binary);
		if (!FbxBinaryData::ReadFromStream(file,fbxBinaryData)) {
			std::cerr << "Failed To Load Fbx File!\n";
		}
		file.close();
	}
	return 0;
}