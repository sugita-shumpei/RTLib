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
#include <variant>
#include <utility>
#include <type_traits>
#include <functional>
#include <tuple>
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
				template<typename T>
				using MapResult = std::conditional_t<std::is_same_v<T, void>, bool, std::optional<T>>;
			public:
				enum class OperationMode {
					eAnd,
					eOr
				};
			public:
				 FeaturesChain()noexcept {}
				~FeaturesChain()noexcept {}
				 FeaturesChain(const FeaturesChain& featChain)noexcept;
				 FeaturesChain& operator=(const FeaturesChain& featChain)noexcept;

				bool Empty()const noexcept;
				void Clear()noexcept;
				auto Count()const noexcept -> size_t;
				void Show ()const noexcept;

				template<typename VkStructureData>
				bool Has()const noexcept;
				template<typename VkStructureData>
				bool Has(size_t idx)const noexcept;
				template<typename VkStructureData>
				void Add()noexcept;
				template<typename VkStructureData>
				auto Get()const noexcept -> std::optional<VkStructureData>;
				template<typename VkStructureData>
				auto Get(size_t idx)const noexcept -> std::optional<VkStructureData>;
				template<typename VkStructureData>
				void Set(const VkStructureData& data)noexcept;
				template<typename VkStructureData>
				bool SetIf(const VkStructureData& data)noexcept;
				template<typename VkStructureData>
				bool SetIf(size_t idx, const VkStructureData& data)noexcept;
				template<typename VkStructureData, typename Func, bool Cond = std::is_same_v<decltype(std::declval<Func>()(std::declval<VkStructureData&>())),void>>
				void Map(Func func)noexcept {
					if (!Has<VkStructureData>()) {
						 Add<VkStructureData>();
					}
					auto idx   = m_Indices.at(VkStructureData::structureType);
					auto sData = Get<VkStructureData>(idx);
					func(*sData);
					(void)Set<VkStructureData>(idx, *sData);
				}
				template<typename VkStructureData, typename Func, typename ...Args, bool Cond = std::is_same_v<decltype(std::declval<Func>()(std::declval<VkStructureData&>(), std::declval<Args>()...)), void >>
				void Map(Func func, Args&&... args)noexcept{
					if (!Has<VkStructureData>()) {
						 Add<VkStructureData>();
					}
					auto idx = m_Indices.at(VkStructureData::structureType);
					auto sData = Get<VkStructureData>(idx);
					func(*sData, std::forward<Args>(args)...);
					(void)SetIf<VkStructureData>(idx, *sData);
				}
				template<typename VkStructureData, typename Func, bool Cond = std::is_same_v<decltype(std::declval<Func>()(std::declval<VkStructureData&>())), void>>
				bool MapIf(Func func)noexcept {
					if (!Has<VkStructureData>()) {
						return false;
					}
					auto idx = m_Indices.at(VkStructureData::structureType);
					auto sData = Get<VkStructureData>(idx);
					func(*sData);
					(void)SetIf<VkStructureData>(idx, *sData);
					return true;
				}
				template<typename VkStructureData, typename Func, typename ...Args, bool Cond = std::is_same_v<decltype(std::declval<Func>()(std::declval<VkStructureData&>(), std::declval<Args>()...)), void >>
				bool MapIf(Func func, Args&&... args)noexcept {
					if (!Has<VkStructureData>()) {
						return false;
					}
					auto idx = m_Indices.at(VkStructureData::structureType);
					auto sData = Get<VkStructureData>(idx);
					func(*sData, std::forward<Args>(args)...);
					(void)SetIf<VkStructureData>(idx, *sData);
					return true;
				}
				template<typename VkStructureData, typename Func, bool Cond = std::is_same_v<decltype(std::declval<Func>()(std::declval<const VkStructureData&>())), bool>>
				bool Filter(Func func)const noexcept {
					if (!Has<VkStructureData>()) {
						return false;
					}
					auto idx   = m_Indices.at(VkStructureData::structureType);
					auto sData = Get<VkStructureData>(idx);
					return func(*sData);
				}
				
				template<typename VkStructureData1, typename VkStructureData2, typename ...VkStructureDatas, typename Func, bool Cond = std::is_same_v<decltype(std::declval<Func>()(std::declval<std::variant<VkStructureData1, VkStructureData2, VkStructureDatas...>&>())), void>>
				void MapForEach(Func func)noexcept {
					constexpr auto count = 2 + sizeof...(VkStructureDatas);
					constexpr auto indices = std::make_index_sequence<count>();
					return Impl_MapForEach<VkStructureData1, VkStructureData2, VkStructureDatas..., Func>(func,indices);
				}
				template<typename VkStructureData1, typename VkStructureData2, typename ...VkStructureDatas, typename Func, bool Cond = std::is_same_v<decltype(std::declval<Func>()(std::declval<      std::variant<VkStructureData1, VkStructureData2, VkStructureDatas...>&>())), void>>
				void MapIfForEach(Func func)noexcept {
					constexpr auto count = 2 + sizeof...(VkStructureDatas);
					constexpr auto indices = std::make_index_sequence<count>();
					return Impl_MapIfForEach<VkStructureData1, VkStructureData2, VkStructureDatas..., Func>(func, indices);
				}

				template<typename VkStructureData1, typename VkStructureData2, typename ...VkStructureDatas, typename Func, bool Cond = std::is_same_v<decltype(std::declval<Func>()(std::declval<const std::variant<VkStructureData1, VkStructureData2, VkStructureDatas...>&>())), bool>>
				bool FilterForEach(OperationMode operation, Func func)const noexcept {
					constexpr auto count   = 2 + sizeof...(VkStructureDatas);
					constexpr auto indices = std::make_index_sequence<count>();
					if (operation == OperationMode::eAnd) {
						return Impl_AndFilterForEach<VkStructureData1, VkStructureData2, VkStructureDatas..., Func>(func, indices);
					}else {
						return Impl_OrFilterForEach<VkStructureData1, VkStructureData2, VkStructureDatas..., Func>(func, indices);
					}
				}

				auto GetPHead()const noexcept -> const void*;
				auto GetPHead()     noexcept ->        void*;
				auto GetPTail()const noexcept -> const void*;
				void SetPTail(void* pTail)noexcept;
			private:
				template<size_t    idx, typename TupleT, typename VariantT, typename Func, bool Cond = std::is_same_v<decltype(std::declval<Func>()(std::declval<VariantT&>())), void>>
				void Impl_MapForElem(Func func) noexcept{
					using cur_type = typename std::tuple_element<idx, TupleT>::type;
					if (!Has<cur_type>()) {
						 Add<cur_type>();
					}
					auto tpIdx = m_Indices.at(cur_type::structureType);
					auto sData = VariantT(*Get<cur_type>(tpIdx));
					func(sData);
					if (std::holds_alternative<cur_type>(sData)) {
						SetIf<cur_type>(tpIdx, std::get<cur_type>(sData));
					}
				}
				template<size_t    idx, typename VkStructureData1, typename VkStructureData2, typename ...VkStructureDatas, typename Func, bool Cond = std::is_same_v<decltype(std::declval<Func>()(std::declval<      std::variant<VkStructureData1, VkStructureData2, VkStructureDatas...>&>())), void>>
				bool Impl_MapIfForElem(Func func)noexcept {
					using tuple_t = std::tuple<VkStructureData1, VkStructureData2, VkStructureDatas...>;
					using cur_type = typename std::tuple_element<idx, tuple_t>::type;
					if (!Has<cur_type>()) {
						return false;
					}
					auto tpIdx = m_Indices.at(cur_type::structureType);
					auto sData = std::variant<VkStructureData1, VkStructureData2, VkStructureDatas...>(*Get<cur_type>(tpIdx));
					func(sData);
					if (std::holds_alternative<cur_type>(sData)) {
						SetIf<cur_type>(tpIdx, std::get<cur_type>(sData));
					}
					return true;
				}

				template<typename VkStructureData1, typename VkStructureData2, typename ...VkStructureDatas, typename Func, size_t... indices, bool Cond = std::is_same_v<decltype(std::declval<Func>()(std::declval<      std::variant<VkStructureData1, VkStructureData2, VkStructureDatas...>&>())), void>>
				void Impl_MapForEach(Func func, std::index_sequence<indices...>)noexcept {
					using tuple_t   = std::tuple  <VkStructureData1, VkStructureData2, VkStructureDatas...>;
					using variant_t = std::variant<VkStructureData1, VkStructureData2, VkStructureDatas...>;
					using swallow = int[];
					(void)swallow {
						(Impl_MapForElem<indices, tuple_t, variant_t,Func>(func), 0)...,
					};
				}
				template<typename VkStructureData1, typename VkStructureData2, typename ...VkStructureDatas, typename Func, size_t... indices, bool Cond = std::is_same_v<decltype(std::declval<Func>()(std::declval<      std::variant<VkStructureData1, VkStructureData2, VkStructureDatas...>&>())), void>>
				void Impl_MapIfForEach(Func func, std::index_sequence<indices...>)noexcept {
					using tuple_t = std::tuple<VkStructureData1, VkStructureData2, VkStructureDatas...>;
					using swallow = int[];
					(void)swallow {
						((void)Impl_MapIfForElem<indices, VkStructureData1, VkStructureData2, VkStructureDatas..., Func>(func), 0)...,
					};
				}
				template<typename VkStructureData1, typename VkStructureData2, typename ...VkStructureDatas, typename Func, size_t... indices, bool Cond = std::is_same_v<decltype(std::declval<Func>()(std::declval<const std::variant<VkStructureData1, VkStructureData2, VkStructureDatas...>&>())), bool>>
				bool Impl_AndFilterForEach(Func func, std::index_sequence<indices...>) const{
					bool res = true;
					{
						using tuple_t = std::tuple<VkStructureData1, VkStructureData2, VkStructureDatas...>;
						using swallow = int[];
						(void)swallow {
							((void)(res = (Has<std::tuple_element<indices, tuple_t>::type>() ? res : false)), 0)...,
						};
						if (!res) { return false; }
					}
					{
						using tuple_t = std::tuple<VkStructureData1, VkStructureData2, VkStructureDatas...>;
						using swallow = int[];
						(void)swallow {
							((void)(res = (func(std::variant<VkStructureData1, VkStructureData2, VkStructureDatas...>(*Get<std::tuple_element<indices, tuple_t>::type>())) ? res : false)), 0)...,
						};
						return res;

					}
					return res;
				}
				template<typename VkStructureData1, typename VkStructureData2, typename ...VkStructureDatas, typename Func, size_t... indices, bool Cond = std::is_same_v<decltype(std::declval<Func>()(std::declval<const std::variant<VkStructureData1, VkStructureData2, VkStructureDatas...>&>())), bool>>
				bool Impl_OrFilterForEach(Func func, std::index_sequence<indices...>) const {
					{
						bool res = false;
						using tuple_t = std::tuple<VkStructureData1, VkStructureData2, VkStructureDatas...>;
						using swallow = int[];
						(void)swallow {
							((void)(res = (res ? true : Has<std::tuple_element<indices, tuple_t>::type>())), 0)...,
						};
						if (!res) { return false; }
					}
					bool res = false;
					{
						using tuple_t = std::tuple<VkStructureData1, VkStructureData2, VkStructureDatas...>;
						using swallow = int[];
						(void)swallow {
							((void)(res = res ? true : (func(std::variant<VkStructureData1, VkStructureData2, VkStructureDatas...>(*Get<std::tuple_element<indices, tuple_t>::type>())))), 0)...,
						};
					}
					return res;
				}

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
			bool FeaturesChain::Has(size_t idx) const noexcept {
				if (m_Holders.size() > idx) {
					return m_Holders[idx]->GetSType() == VkStructureData::structureType;
				}
				return false;
			}

			template<typename VkStructureData>
			void FeaturesChain::Add() noexcept {
				(void)Impl_Add<VkStructureData>();
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
			auto FeaturesChain::Get(size_t idx) const noexcept -> std::optional<VkStructureData> {
				if (Has<VkStructureData>(idx)) {
					return static_cast<const THolder<VkStructureData>*>(m_Holders[idx].get())->Read();
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
			bool FeaturesChain::SetIf(const VkStructureData& data) noexcept {
				if (!Has<VkStructureData>()) {
					return false;
				}
				static_cast<THolder<VkStructureData>*>(m_Holders[m_Indices.at(VkStructureData::structureType)].get())->Write(data);
				return true;
			}

			template<typename VkStructureData>
			bool FeaturesChain::SetIf(size_t idx, const VkStructureData& data) noexcept {
				if (!Has<VkStructureData>(idx)) {
					return false;
				}
				static_cast<THolder<VkStructureData>*>(m_Holders[idx].get())->Write(data);
				return true;
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
				if (count >= 1) {
					m_Holders[count - 1]->SetPNext(m_Holders[count]->GetPData());
				}
				return static_cast<THolder<VkStructureData>*>(m_Holders[count].get());
			}
}
	}
}
#endif