#include <TestVulkan.h>
#include <iostream>
VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE

RTLib::Test::Vulkan::FeaturesChain::FeaturesChain(const FeaturesChain& featChain) noexcept {
	m_Indices = featChain.m_Indices;
	m_Holders.clear();
	m_Holders.reserve(featChain.Count());
	m_PTail = featChain.m_PTail;
	for (auto& holder : featChain.m_Holders) {
		m_Holders.push_back(std::unique_ptr<IHolder>(holder->Clone()));
	}
	if (!m_Holders.empty()) {
		for (size_t i = 0; i < m_Holders.size() - 1; ++i) {
			m_Holders[i]->SetPNext(m_Holders[i + 1]->GetPData());
		}
		m_Holders.back()->SetPNext(m_PTail);
	}
}

RTLib::Test::Vulkan::FeaturesChain& RTLib::Test::Vulkan::FeaturesChain::operator=(const FeaturesChain& featChain)noexcept {
	if (this != &featChain) {
		m_Indices = featChain.m_Indices;
		m_Holders.clear();
		m_Holders.reserve(featChain.Count());
		m_PTail = featChain.m_PTail;
		for (auto& holder : featChain.m_Holders) {
			m_Holders.push_back(std::unique_ptr<IHolder>(holder->Clone()));
		}
		if (!m_Holders.empty()) {
			for (size_t i = 0; i < m_Holders.size() - 1; ++i) {
				m_Holders[i]->SetPNext(m_Holders[i + 1]->GetPData());
			}
			m_Holders.back()->SetPNext(m_PTail);
		}
	}
	return *this;
}

bool RTLib::Test::Vulkan::FeaturesChain::Empty() const noexcept {
	return m_Indices.empty();
}

void RTLib::Test::Vulkan::FeaturesChain::Clear() noexcept {
	m_Indices.clear();
	m_Holders.clear();
	m_PTail = nullptr;
}

auto RTLib::Test::Vulkan::FeaturesChain::Count() const noexcept -> size_t { return m_Indices.size(); }

void RTLib::Test::Vulkan::FeaturesChain::Show() const noexcept
{
	for (size_t i = 0; i < m_Holders.size(); ++i) {
		std::cout << "index " << i << ": " << vk::to_string(m_Holders[i]->GetSType()) << "(" << m_Holders[i]->GetPData() << ")" << " -> " << "(" << m_Holders[i]->GetPNext() << ")" << "\n";
	}
}

auto RTLib::Test::Vulkan::FeaturesChain::GetPHead() const noexcept -> const void* {
	if (m_Holders.empty()) {
		return nullptr;
	}
	else {
		return m_Holders.front()->GetPData();
	}
}

auto RTLib::Test::Vulkan::FeaturesChain::GetPHead() noexcept -> void* {
	if (m_Holders.empty()) {
		return nullptr;
	}
	else {
		return m_Holders.front()->GetPData();
	}
}

auto RTLib::Test::Vulkan::FeaturesChain::GetPTail() const noexcept -> const void* {
	return m_PTail;
}

void RTLib::Test::Vulkan::FeaturesChain::SetPTail(void* pTail) noexcept {
	if (m_Holders.empty()) {
		return;
	}
	m_PTail = pTail;
	m_Holders.back()->SetPNext(pTail);
}
