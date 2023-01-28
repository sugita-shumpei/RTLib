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
namespace RTLib
{
	namespace Core
	{
		namespace experimental
		{
			struct FbxBinaryAttributeView
			{
				enum class TypeCode : unsigned char
				{
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
					eBinary = 'R',
					eString = 'S',
				};
				enum class TypeArrayEncoding
				{
					eDefault = 0,
					eZlib = 1,
				};

				struct TypeArrayData
				{
					uint32_t count;
					TypeArrayEncoding encoding;
					uint32_t sizeInBytes;
				};
				static_assert(sizeof(TypeArrayData) == 12);
				struct TypeSpecialData
				{
					uint32_t sizeInBytes;
				};
				static_assert(sizeof(TypeSpecialData) == 4);
				static constexpr bool IsValueTypeCode(TypeCode code)
				{
					switch (code)
					{
					case RTLib::Core::experimental::FbxBinaryAttributeView::TypeCode::eBool:
					case RTLib::Core::experimental::FbxBinaryAttributeView::TypeCode::eInt16:
					case RTLib::Core::experimental::FbxBinaryAttributeView::TypeCode::eInt32:
					case RTLib::Core::experimental::FbxBinaryAttributeView::TypeCode::eInt64:
					case RTLib::Core::experimental::FbxBinaryAttributeView::TypeCode::eFloat32:
					case RTLib::Core::experimental::FbxBinaryAttributeView::TypeCode::eFloat64:
						return true;
					default:
						return false;
					}
				}
				static constexpr bool IsSpecialTypeCode(TypeCode code)
				{
					switch (code)
					{
					case RTLib::Core::experimental::FbxBinaryAttributeView::TypeCode::eBinary:
					case RTLib::Core::experimental::FbxBinaryAttributeView::TypeCode::eString:
						return true;
					default:
						return false;
					}
				}
				static constexpr bool IsArrayTypeCode(TypeCode code)
				{
					switch (code)
					{
					case RTLib::Core::experimental::FbxBinaryAttributeView::TypeCode::eArrayBool:
					case RTLib::Core::experimental::FbxBinaryAttributeView::TypeCode::eArrayInt32:
					case RTLib::Core::experimental::FbxBinaryAttributeView::TypeCode::eArrayInt64:
					case RTLib::Core::experimental::FbxBinaryAttributeView::TypeCode::eArrayFloat32:
					case RTLib::Core::experimental::FbxBinaryAttributeView::TypeCode::eArrayFloat64:
						return true;
					default:
						return false;
					}
				}
				static auto ReadFromFile(std::ifstream &fbxFile) noexcept -> FbxBinaryAttributeView
				{
					FbxBinaryAttributeView attrib;
					attrib.begPosInBytes = static_cast<size_t>(fbxFile.tellg());
					fbxFile.read((char *)&attrib.type, sizeof(attrib.type));
					if (IsArrayTypeCode(attrib.type))
					{
						TypeArrayData arrayData;
						fbxFile.read((char *)&arrayData, sizeof(arrayData));
						attrib.sizeInBytes = sizeof(attrib.type) + sizeof(arrayData) + arrayData.sizeInBytes;
						attrib.data = arrayData;
					}
					else if (IsSpecialTypeCode(attrib.type))
					{
						TypeSpecialData specialData;
						fbxFile.read((char *)&specialData, sizeof(specialData));
						attrib.sizeInBytes = sizeof(attrib.type) + sizeof(specialData) + specialData.sizeInBytes;
						attrib.data = specialData;
					}
					else {
						if (attrib.type == RTLib::Core::experimental::FbxBinaryAttributeView::TypeCode::eBool)
						{
							char v;
							fbxFile.read((char*)&v, sizeof(v));
							attrib.data = v;
							attrib.sizeInBytes = sizeof(attrib.type) + sizeof(uint8_t);
						}
						if (attrib.type == RTLib::Core::experimental::FbxBinaryAttributeView::TypeCode::eInt16)
						{
							int16_t v;
							fbxFile.read((char*)&v, sizeof(v));
							attrib.data = v;
							attrib.sizeInBytes = sizeof(attrib.type) + sizeof(int16_t);
						}
						if (attrib.type == RTLib::Core::experimental::FbxBinaryAttributeView::TypeCode::eInt32)
						{
							int32_t v;
							fbxFile.read((char*)&v, sizeof(v));
							attrib.data = v;
							attrib.sizeInBytes = sizeof(attrib.type) + sizeof(int32_t);
						}
						if (attrib.type == RTLib::Core::experimental::FbxBinaryAttributeView::TypeCode::eInt64)
						{
							int64_t v;
							fbxFile.read((char*)&v, sizeof(v));
							attrib.data = v;
							attrib.sizeInBytes = sizeof(attrib.type) + sizeof(int64_t);
						}
						if (attrib.type == RTLib::Core::experimental::FbxBinaryAttributeView::TypeCode::eFloat32)
						{
							float v;
							fbxFile.read((char*)&v, sizeof(v));
							attrib.data = v;
							attrib.sizeInBytes = sizeof(attrib.type) + sizeof(float);
						}
						if (attrib.type == RTLib::Core::experimental::FbxBinaryAttributeView::TypeCode::eFloat64)
						{
							double v;
							fbxFile.read((char*)&v, sizeof(v));
							attrib.data = v;
							attrib.sizeInBytes = sizeof(attrib.type) + sizeof(double);
						}
					}
					fbxFile.seekg(attrib.begPosInBytes + attrib.sizeInBytes, std::ios::beg);
					return attrib;
				}
				TypeCode type = TypeCode::eBool;
				std::variant<std::monostate, char, int16_t, int32_t, int64_t, float, double, TypeArrayData, TypeSpecialData>
					data = std::monostate();
				std::uint32_t begPosInBytes = 0;
				std::uint32_t sizeInBytes = 0;
			};
			struct FbxBinaryNodeView
			{
				enum class HeaderType
				{
					eV1 = 0,
					eV2
				};
				struct HeaderDataV1
				{
					uint32_t endPosInBytes = 0;
					uint32_t numAttributes = 0;
					uint32_t sumAttributesAsBytes = 0;
					uint8_t lenNodeName = 0;
					static auto ReadFromFile(std::ifstream &fbxFile) noexcept -> HeaderDataV1
					{
						HeaderDataV1 data;
						fbxFile.read((char *)&data.endPosInBytes, sizeof(data.endPosInBytes));
						fbxFile.read((char *)&data.numAttributes, sizeof(data.numAttributes));
						fbxFile.read((char *)&data.sumAttributesAsBytes, sizeof(data.sumAttributesAsBytes));
						fbxFile.read((char *)&data.lenNodeName, sizeof(data.lenNodeName));
						return data;
					}
					static inline constexpr auto GetFileSizeInBytes() noexcept -> size_t
					{
						return sizeof(uint32_t) * 3 + 1;
					}
				};
				struct HeaderDataV2
				{
					uint64_t endPosInBytes = 0;
					uint64_t numAttributes = 0;
					uint64_t sumAttributesAsBytes = 0;
					uint8_t lenNodeName = 0;
					static auto ReadFromFile(std::ifstream &fbxFile) noexcept -> HeaderDataV2
					{
						HeaderDataV2 data;
						fbxFile.read((char *)&data.endPosInBytes, sizeof(data.endPosInBytes));
						fbxFile.read((char *)&data.numAttributes, sizeof(data.numAttributes));
						fbxFile.read((char *)&data.sumAttributesAsBytes, sizeof(data.sumAttributesAsBytes));
						fbxFile.read((char *)&data.lenNodeName, sizeof(data.lenNodeName));
						return data;
					}
					static inline constexpr auto GetFileSizeInBytes() noexcept -> size_t
					{
						return sizeof(uint64_t) * 3 + 1;
					}
				};
				struct HeaderData
				{
					HeaderData() noexcept : type{HeaderType::eV1}, data() {}
					~HeaderData() noexcept {}
					HeaderType type;
					union HeaderDataBase
					{
						HeaderDataBase() : v1{} {}
						~HeaderDataBase() {}
						HeaderDataV1 v1;
						HeaderDataV2 v2;
					} data;

					static auto ReadFromFile(std::ifstream &fbxFile, HeaderType type = HeaderType::eV1) noexcept -> HeaderData
					{
						HeaderData header;
						header.type = type;
						if (type == HeaderType::eV1)
						{
							header.data.v1 = HeaderDataV1::ReadFromFile(fbxFile);
						}
						else
						{
							header.data.v2 = HeaderDataV2::ReadFromFile(fbxFile);
						}
						return header;
					}
					auto GetEndPosInBytes() const noexcept -> uint32_t
					{
						if (type == HeaderType::eV1)
						{
							return data.v1.endPosInBytes;
						}
						else
						{
							return data.v2.endPosInBytes;
						}
					}
					auto GetNumAttributes() const noexcept -> uint32_t
					{
						if (type == HeaderType::eV1)
						{
							return data.v1.numAttributes;
						}
						else
						{
							return data.v2.numAttributes;
						}
					}
					auto GetSumAttributesAsBytes() const noexcept -> uint32_t
					{
						if (type == HeaderType::eV1)
						{
							return data.v1.sumAttributesAsBytes;
						}
						else
						{
							return data.v2.sumAttributesAsBytes;
						}
					}
					auto GetLenNodeName() const noexcept -> uint8_t
					{
						if (type == HeaderType::eV1)
						{
							return data.v1.lenNodeName;
						}
						else
						{
							return data.v2.lenNodeName;
						}
					}
					auto GetEndNodeMarkerSize() const noexcept -> uint32_t
					{
						if (type == HeaderType::eV1)
						{
							return 13;
						}
						else
						{
							return 25;
						}
					}
					auto GetFileSizeInBytes() const noexcept -> size_t
					{
						if (type == HeaderType::eV1)
						{
							return HeaderDataV1::GetFileSizeInBytes();
						}
						else
						{
							return HeaderDataV2::GetFileSizeInBytes();
						}
					}
				};

				static void ReadFromFile(std::ifstream &fbxFile, FbxBinaryNodeView &nodeData, HeaderType type = HeaderType::eV1, bool isLoadAttributes = false) noexcept
				{
					auto begPos = static_cast<size_t>(fbxFile.tellg());
					nodeData.begPosInBytes = begPos;
					nodeData.header = HeaderData::ReadFromFile(fbxFile, type);
					nodeData.name.resize(nodeData.header.GetLenNodeName());
					fbxFile.read((char *)nodeData.name.c_str(), nodeData.name.size());
					// std::cout << "node \"" << nodeData.name << "\" Read Begin: " << begPos << "\n";
					auto endPos = nodeData.header.GetEndPosInBytes();
					nodeData.sizeInBytes = endPos - begPos;
					auto endNodeMarkerSize = nodeData.header.GetEndNodeMarkerSize();
					auto begChildNode = begPos + nodeData.header.GetFileSizeInBytes() + nodeData.header.GetLenNodeName() + nodeData.header.GetSumAttributesAsBytes();
					assert(endPos > 0);
					if (isLoadAttributes)
					{
						auto numAttrib = nodeData.header.GetNumAttributes();
						if (numAttrib > 0)
						{
							for (int i = 0; i < numAttrib; ++i)
							{
								nodeData.attributes.emplace_back(FbxBinaryAttributeView::ReadFromFile(fbxFile));
							}
							auto finPos = static_cast<size_t>(fbxFile.tellg());
							assert(finPos == begChildNode);
						}
					}
					fbxFile.seekg(begChildNode, std::ios::beg);
					constexpr std::array<char, 25> compEndMarker = {};
					if (begChildNode == endPos)
					{
						// std::cout << "node \"" << nodeData.name << "\" has No Child Nodes And Has No End Marker\n";
					}
					else if (begChildNode + endNodeMarkerSize == endPos)
					{
						fbxFile.read(nodeData.endMarker.data(), endNodeMarkerSize);
						constexpr std::array<char, 25> compEndMarker = {};
						assert(compEndMarker == nodeData.endMarker);
						// std::cout << "node \"" << nodeData.name << "\" has No Child Nodes And Has End Marker\n";
					}
					else
					{
						size_t finPos = 0;
						do
						{
							// std::cout << "RUN LOOP" << std::endl;
							FbxBinaryNodeView childNode;
							FbxBinaryNodeView::ReadFromFile(fbxFile, childNode, type, isLoadAttributes);
							nodeData.children.emplace_back(std::move(childNode));
							finPos = fbxFile.tellg();
						} while ((finPos < endPos) && (finPos + endNodeMarkerSize != endPos));
						if (finPos + endNodeMarkerSize == endPos)
						{
							fbxFile.read(nodeData.endMarker.data(), endNodeMarkerSize);
							assert(compEndMarker == nodeData.endMarker);
							// std::cout << "node \"" << nodeData.name << "\" has Some Child Nodes And Has End Marker\n";
						}
						else if (finPos == endPos)
						{
							// std::cout << "node \"" << nodeData.name << "\" has Some Child Nodes And Has No End Marker\n";
						}
						else
						{
							assert(false);
						}
					}
				}

				HeaderData header = {};
				std::string name = "";
				std::vector<FbxBinaryAttributeView> attributes = {};
				std::vector<FbxBinaryNodeView> children = {};
				std::array<char, 25> endMarker = {};
				std::uint32_t begPosInBytes = 0;
				std::uint32_t sizeInBytes = 0;
			};
			struct FbxBinary
			{
				static inline constexpr std::array<char, 23> validMagick = {
					"Kaydara FBX Binary\x20\x20\x00\x1a"};
				static inline constexpr std::array<uint8_t, 16> validFooter4 = {
					0xf8, 0x5a, 0x8c, 0x6a, 0xde, 0xf5, 0xd9, 0x7e, 0xec, 0xe9, 0x0c, 0xe3, 0x75, 0x8f, 0x29, 0x0b};
				auto GetNodeHeaderType() const noexcept -> FbxBinaryNodeView::HeaderType
				{
					if (version1 / 100 >= 75)
					{
						return FbxBinaryNodeView::HeaderType::eV2;
					}
					else
					{
						return FbxBinaryNodeView::HeaderType::eV1;
					}
				}
				auto GetEndNodeMarkerSize() const noexcept -> uint32_t
				{
					if (version1 / 100 >= 75)
					{
						return 25;
					}
					else
					{
						return 13;
					}
				}
				static bool Load(const char *filename, FbxBinary &binaryData, bool isLoadAttributes = false) noexcept
				{
					std::ifstream file(filename, std::ios::binary);
					if (file.is_open())
					{
						file.seekg(0, std::ios::end);
						size_t fbxFileSize = static_cast<size_t>(file.tellg());
						file.seekg(0, std::ios::beg);
						file.read((char *)binaryData.magick.data(), sizeof(binaryData.magick));
						assert(validMagick == binaryData.magick);
						file.read((char *)&binaryData.version1, sizeof(binaryData.version1));
						file.seekg(-144, std::ios::end);
						assert(static_cast<size_t>(file.tellg()) % 16 == 0);
						file.read((char *)&binaryData.footer2, sizeof(binaryData.footer2));
						assert(binaryData.footer2 == 0);
						file.read((char *)&binaryData.version2, sizeof(binaryData.version2));
						assert(binaryData.version2 == binaryData.version1);
						auto compFooter3 = std::array<char, 120>();
						file.read((char *)binaryData.footer3.data(), sizeof(binaryData.footer3));
						assert(binaryData.footer3 == compFooter3);
						file.read((char *)binaryData.footer4.data(), sizeof(binaryData.footer4));
						assert(binaryData.footer4 == validFooter4);
						file.seekg(sizeof(binaryData.magick) + sizeof(binaryData.version1), std::ios::beg);
						bool isReachToEnd = false;
						constexpr std::array<char, 25> compEndMarker = {};
						do
						{
							FbxBinaryNodeView nodeData;
							FbxBinaryNodeView::ReadFromFile(file, nodeData, binaryData.GetNodeHeaderType(), isLoadAttributes);
							binaryData.topLevelNodeViews.emplace_back(std::move(nodeData));
							auto curPos = static_cast<size_t>(file.tellg());
							file.read((char *)binaryData.endMarker.data(), binaryData.GetEndNodeMarkerSize());
							if (binaryData.endMarker == compEndMarker)
							{
								isReachToEnd = true;
							}
							else
							{
								file.seekg(curPos, std::ios::beg);
							}

						} while (!isReachToEnd);
						file.read((char *)binaryData.footer1.data(), sizeof(binaryData.footer1));
						for (auto &v : binaryData.footer1)
						{
							std::cout << std::hex << (int)v << " ";
						}
						std::cout << std::endl;
						auto paddingBeg = static_cast<size_t>(file.tellg());
						auto paddingEnd = fbxFileSize - 144;
						std::cout << "padding size: " << std::dec << paddingEnd - paddingBeg << std::endl;
						binaryData.padding.resize(paddingEnd - paddingBeg);
						file.read((char *)binaryData.padding.data(), binaryData.padding.size());
						binaryData.rawData = std::unique_ptr<char[]>(new char[fbxFileSize]);
						file.seekg(0L, std::ios::beg);
						file.read((char *)binaryData.rawData.get(), fbxFileSize);
						binaryData.rawDataSizeInBytes = fbxFileSize;
						file.close();
						return true;
					}
					return false;
				}
				size_t rawDataSizeInBytes = 0;
				std::unique_ptr<char[]> rawData = {};
				std::array<char, 23> magick = {};
				uint32_t version1 = 0;
				std::vector<FbxBinaryNodeView> topLevelNodeViews = {};
				std::array<char, 25> endMarker = {};
				std::array<uint8_t, 16> footer1 = {};
				std::vector<char> padding = {};
				uint32_t footer2 = 0;
				uint32_t version2 = 0;
				std::array<char, 120> footer3 = {};
				std::array<uint8_t, 16> footer4 = {};
			};
			
			struct FbxProperties
			{

			};
			struct FbxGeometry {
				int64_t       id;
				std::string   name;
				std::string   usage;
				FbxProperties properties70;
			};
		}
	}
}
int main()
{
	RTLib::Core::experimental::FbxBinary binary;
	RTLib::Core::experimental::FbxBinary::Load(RTLIB_CORE_TEST_CONFIG_DATA_PATH "/Models/ZeroDay/MEASURE_ONE/MEASURE_ONE.fbx", binary, true);
	std::unordered_map<std::string, const RTLib::Core::experimental::FbxBinaryNodeView *> binaryNodeSet = {};
	std::stack<std::pair<std::string, const RTLib::Core::experimental::FbxBinaryNodeView *>> tmpStack;
	for (auto &v : binary.topLevelNodeViews)
	{
		tmpStack.push({"Root", &v});
	}
	while (!tmpStack.empty())
	{
		auto [name, pView] = tmpStack.top();
		tmpStack.pop();
		binaryNodeSet.insert({name + "/" + pView->name, pView});
		if (!pView->children.empty())
		{
			for (auto &c : pView->children)
			{
				tmpStack.push({name + "/" + pView->name, &c});
			}
		}
	}
	auto ShowNodeAttributes = [](const RTLib::Core::experimental::FbxBinaryNodeView* view, std::string ext = "", const char* pRawData = nullptr)
	{
		size_t i = 0;
		for (auto& attr : view->attributes) {

			if (attr.type == RTLib::Core::experimental::FbxBinaryAttributeView::TypeCode::eBool)
			{
				std::cout << ext<< view->name << ".attributes[" << i << "]: Bool  = " << std::get<char>(attr.data) << " ;\n";
			}
			if (attr.type == RTLib::Core::experimental::FbxBinaryAttributeView::TypeCode::eInt16)
			{
				std::cout << ext<< view->name << ".attributes[" << i << "]: Int16 = " << std::get<int16_t>(attr.data) << " ;\n";
			}
			if (attr.type == RTLib::Core::experimental::FbxBinaryAttributeView::TypeCode::eInt32)
			{
				std::cout << ext<< view->name << ".attributes[" << i << "]: Int32 = " << std::get<int32_t>(attr.data) << " ;\n";
			}
			if (attr.type == RTLib::Core::experimental::FbxBinaryAttributeView::TypeCode::eInt64)
			{
				std::cout << ext<< view->name << ".attributes[" << i << "]: Int64 = " << std::get<int64_t>(attr.data) << " ;\n";
			}
			if (attr.type == RTLib::Core::experimental::FbxBinaryAttributeView::TypeCode::eFloat32)
			{
				std::cout << ext << view->name << ".attributes[" << i << "]: Float32 = " << std::get<float>(attr.data) << " ;\n";
			}
			if (attr.type == RTLib::Core::experimental::FbxBinaryAttributeView::TypeCode::eFloat64)
			{
				std::cout << ext << view->name << ".attributes[" << i << "]: Float64 = " << std::get<double>(attr.data) << " ;\n";
			}
			if (pRawData) {
				if (attr.type == RTLib::Core::experimental::FbxBinaryAttributeView::TypeCode::eString)
				{
					auto specData = std::get<RTLib::Core::experimental::FbxBinaryAttributeView::TypeSpecialData>(attr.data);
					auto v = std::string(specData.sizeInBytes, '\0');
					std::memcpy((void*)v.c_str(), pRawData + attr.begPosInBytes + 1 + sizeof(specData), specData.sizeInBytes);
					std::cout << ext << view->name << ".attributes[" << i << "]: String = \"" << v << "\" ;\n";
				}
			}
			++i;
		}
	};
	{

		auto view = binaryNodeSet.at("Root/Objects"); 
		ShowNodeAttributes(view, "", binary.rawData.get());
		for (const auto& cview : view->children) {
			ShowNodeAttributes(&cview, view->name+".", binary.rawData.get());
			//for (const auto& ccview : cview.children) {
			//	ShowNodeAttributes(&ccview, view->name + "."+ cview.name + ".", binary.rawData.get());
			//	//for (const auto& cccview : ccview.children) {
			//	//	ShowNodeAttributes(&cccview, view->name + "." + cview.name + "." + ccview.name + ".", binary.rawData.get());
			//	//	for (const auto& ccccview : ccview.children) {
			//	//		ShowNodeAttributes(&ccccview, view->name + "." + cview.name + "." + ccview.name + "." + cccview.name + ".", binary.rawData.get());
			//	//	}
			//	//}
			//}
		}
	}

	return 0;
}