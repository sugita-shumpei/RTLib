#ifndef TEST_TEST_VULKAN_H
#define TEST_TEST_VULKAN_H
#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1
#include <vulkan/vulkan.hpp>
#include <algorithm>
#include <vector>
#include <string>
#include <optional>
#include <unordered_map>
#include <tuple>
#include <type_traits>
#include <functional>
#include <any>
namespace RTLib {
	namespace Test {
		namespace Vulkan {
			inline bool findExtName(const std::vector<vk::ExtensionProperties>& extProps, const std::string& extName)noexcept
			{
				return std::find_if(std::begin(extProps), std::end(extProps), [extName](const auto& extProp) {
					return std::string(extProp.extensionName.data()) == extName;
				}) != std::end(extProps);
			}
			inline bool findLyrName(const std::vector<vk::LayerProperties>    & lyrProps, const std::string& lyrName)noexcept
			{
				return std::find_if(std::begin(lyrProps), std::end(lyrProps), [lyrName](const auto& lyrProp) {
					return std::string(lyrProp.layerName.data()) == lyrName;
				}) != std::end(lyrProps);
			}

			class FeaturesChain {
			private:
				class IHolder;
				template<typename VkStructureData, bool isConstPNext = std::is_same_v<vk::StructureType, decltype(VkStructureData::structureType)>>
				class THolder;
			public:
				 FeaturesChain()noexcept {}
				~FeaturesChain()noexcept {}
				 FeaturesChain(const FeaturesChain& featChain)noexcept;
				 FeaturesChain& operator=(const FeaturesChain& featChain)noexcept;

				bool Empty()const noexcept;
				void Clear()noexcept;
				auto Count()const noexcept -> size_t;

				template<typename VkStructureData>
				bool Has()const noexcept;
				template<typename VkStructureData>
				void Add()noexcept;
				template<typename VkStructureData>
				auto Get()const noexcept -> std::optional<VkStructureData>;
				template<typename VkStructureData>
				void Set(const VkStructureData& data)noexcept;

				auto GetPHead()const noexcept -> const void*;
				auto GetPHead()     noexcept ->        void*;
				auto GetPTail()const noexcept -> const void*;
				void SetPTail(void* pTail)noexcept;
			private:
				template<typename VkStructureData>
				auto Impl_Add()noexcept -> THolder<VkStructureData>*;
				std::unordered_map<vk::StructureType, uint32_t> m_Indices = {};
				std::vector<std::unique_ptr<IHolder>>         m_Holders = {};
				void*										  m_PTail   = nullptr;
			};

			class FeaturesChain::IHolder {
			public:
				virtual ~IHolder()noexcept {}
				virtual auto Clone()const noexcept -> IHolder*       = 0;
				virtual void Reset(void* pNext = nullptr)noexcept    = 0;
				virtual auto GetSize() const noexcept -> std::size_t = 0;
				virtual auto GetPData()const noexcept -> const void* = 0;
				virtual auto GetPData()      noexcept ->       void* = 0;
				virtual auto GetSType()const noexcept -> vk::StructureType = 0;
				virtual auto GetPNext()const noexcept -> const void* = 0;
				virtual void SetPNext(void* pNext)noexcept = 0;
			};
			template<typename VkStructureData>
			class FeaturesChain::THolder<VkStructureData,true > : public FeaturesChain::IHolder {
			public:
				THolder(const VkStructureData& data = {})noexcept {
					m_Data = data;
					m_Data.sType = VkStructureData::structureType;
					m_Data.pNext = nullptr;
				}
				virtual ~THolder()noexcept {}
				virtual auto Clone()const noexcept -> IHolder*              override{
					return new THolder(m_Data);
				}
				virtual void Reset(void* pNext = nullptr)noexcept           override { 
					m_Data.sType = VkStructureData::structureType;
					m_Data.pNext = pNext; 
				}
				virtual auto GetSize()const noexcept -> std::size_t			override { return sizeof(m_Data); }
				virtual auto GetPData()const noexcept -> const void*		override { return &m_Data; }
				virtual auto GetPData()      noexcept ->       void*		override { return &m_Data; }
				virtual auto GetSType()const noexcept -> vk::StructureType  override { return VkStructureData::structureType; }
				virtual auto GetPNext()const noexcept -> const void*	    override { return  m_Data.pNext;  }
				virtual void SetPNext(void* pNext)noexcept					override { m_Data.pNext = pNext; }
				auto Read()const noexcept -> VkStructureData {
					VkStructureData data = m_Data;
					data.sType = VkStructureData::structureType;
					data.pNext = nullptr;
					return data;
				}
				void Write(const VkStructureData& data)noexcept {
					auto pNext = m_Data.pNext;
					m_Data = data;
					m_Data.sType = VkStructureData::structureType;
					m_Data.pNext = pNext;
				}
			private:
				VkStructureData m_Data;
			};
			template<typename VkStructureData>
			class FeaturesChain::THolder<VkStructureData,false> : public FeaturesChain::IHolder {
			public:
				THolder(const VkStructureData& data = {})noexcept {
					m_Data = data;
					m_Data.sType = VkStructureData::structureType;
					m_Data.pNext = nullptr;
				}
				virtual ~THolder()noexcept {}
				virtual auto Clone()const noexcept -> IHolder* override {
					return new THolder(m_Data);
				}
				virtual void Reset(void* pNext = nullptr)noexcept           override {
					m_Data.sType = VkStructureData::structureType;
					m_Data.pNext = pNext;
				}
				virtual auto GetSize()const noexcept -> std::size_t        override { return sizeof(m_Data); }
				virtual auto GetPData()const noexcept -> const void*       override { return &m_Data; }
				virtual auto GetPData()      noexcept ->       void*       override { return &m_Data; }
				virtual auto GetSType()const noexcept -> vk::StructureType override { return VkStructureData::structureType; }
				virtual auto GetPNext()const noexcept -> const void*       override { return  m_Data.pNext; }
				virtual void SetPNext(void* pNext)noexcept				   override { m_Data.pNext = pNext; }
				auto Read()const noexcept -> VkStructureData {
					VkStructureData data = m_Data;
					data.sType = VkStructureData::structureType;
					data.pNext = nullptr;
					return data;
				}
				void Write(const VkStructureData& data)noexcept {
					auto pNext   = m_Data.pNext;
					m_Data       = data;
					m_Data.sType = VkStructureData::structureType;
					m_Data.pNext = pNext;
				}
			private:
				VkStructureData m_Data;
			};

			template<typename VkStructureData>
			bool FeaturesChain::Has() const noexcept {
				return m_Indices.count(VkStructureData::structureType) > 0;
			}

			template<typename VkStructureData>
			void FeaturesChain::Add() noexcept {
				(void)Impl_Add();
			}

			template<typename VkStructureData>
			auto FeaturesChain::Get() const noexcept -> std::optional<VkStructureData> {
				if (Has<VkStructureData>()) {
					return static_cast<const THolder<VkStructureData>*>(m_Holders[m_Indices.at(VkStructureData::structureType)].get())->Read();
				}
				else {
					return std::nullopt;
				}
			}

			template<typename VkStructureData>
			void FeaturesChain::Set(const VkStructureData& data) noexcept {
				Impl_Add<VkStructureData>()->Write(data);
			}

			template<typename VkStructureData>
			auto FeaturesChain::Impl_Add() noexcept -> THolder<VkStructureData>* {
				if (Has<VkStructureData>()) {
					return static_cast<THolder<VkStructureData>*>(m_Holders[m_Indices.at(VkStructureData::structureType)].get());
				}
				auto count = m_Holders.size();
				m_Indices[VkStructureData::structureType] = count;
				m_Holders.push_back(std::unique_ptr<IHolder>(new THolder<VkStructureData>()));
				m_Holders.back()->SetPNext(m_PTail);
				if (count > 1) {
					m_Holders[count - 1]->SetPNext(m_Holders[count]->GetPData());
				}
				return static_cast<THolder<VkStructureData>*>(m_Holders[count].get());
			}
}
	}
}
#endif