#include <RTLib/Ext/OPX7/OPX7ShaderTableLayout.h>
#include <iostream>

RTLib::Ext::OPX7::OPX7ShaderTableLayoutGeometryAS::OPX7ShaderTableLayoutGeometryAS(const std::string& name) noexcept :m_Name{ name } {}

RTLib::Ext::OPX7::OPX7ShaderTableLayoutGeometryAS::OPX7ShaderTableLayoutGeometryAS(const OPX7ShaderTableLayoutGeometryAS& gas) noexcept
{
	m_Name = gas.m_Name;
	m_BaseRecordCount = 0;
	m_DwGeometries      = gas.GetDwGeometries();
	for (auto& geometry : m_DwGeometries) {
		geometry.Internal_SetBaseRecordOffset(m_BaseRecordCount);
		geometry.Internal_SetUpGeometryAS(this);
		m_BaseRecordCount += geometry.GetBaseRecordCount();
	}
}

auto RTLib::Ext::OPX7::OPX7ShaderTableLayoutGeometryAS::operator=(const OPX7ShaderTableLayoutGeometryAS& gas) noexcept -> OPX7ShaderTableLayoutGeometryAS&
{
	// TODO: return ステートメントをここに挿入します
	if (this != &gas) {
		m_Name = gas.m_Name;
		m_BaseRecordCount = 0;
		m_DwGeometries = gas.GetDwGeometries();
		for (auto& geometry : m_DwGeometries) {
			geometry.Internal_SetBaseRecordOffset(m_BaseRecordCount);
			geometry.Internal_SetUpGeometryAS(this);
			m_BaseRecordCount += geometry.GetBaseRecordCount();
		}
	}
	return *this;
}

auto RTLib::Ext::OPX7::OPX7ShaderTableLayoutGeometryAS::GetName() const noexcept -> std::string { return m_Name; }

auto RTLib::Ext::OPX7::OPX7ShaderTableLayoutGeometryAS::GetBaseRecordCount() const noexcept -> unsigned int { return m_BaseRecordCount; }

 auto RTLib::Ext::OPX7::OPX7ShaderTableLayoutGeometryAS::GetDwGeometries() const noexcept -> const std::vector<Geometry>& { return m_DwGeometries; }

void RTLib::Ext::OPX7::OPX7ShaderTableLayoutGeometryAS::SetDwGeometry(const OPX7ShaderTableLayoutGeometry& geometry) noexcept
{
	m_DwGeometries.push_back(geometry);
	m_DwGeometries.back().Internal_SetUpGeometryAS(this);
	Update();
}

void RTLib::Ext::OPX7::OPX7ShaderTableLayoutGeometryAS::Update() noexcept
{
	m_BaseRecordCount = 0;
	for (auto& geometry : m_DwGeometries) {
		geometry.Internal_SetBaseRecordOffset(m_BaseRecordCount);
		m_BaseRecordCount += std::max<unsigned int>(geometry.GetBaseRecordCount(),1);
	}
}

RTLib::Ext::OPX7::OPX7ShaderTableLayoutGeometry::OPX7ShaderTableLayoutGeometry(const std::string& name, unsigned int baseRecordcount) noexcept {
	m_Name            = name;
	m_BaseRecordCount = baseRecordcount;
}

RTLib::Ext::OPX7::OPX7ShaderTableLayoutGeometry::OPX7ShaderTableLayoutGeometry(const OPX7ShaderTableLayoutGeometry& geometry) noexcept
	:m_Name{ geometry.m_Name },m_BaseRecordCount(geometry.GetBaseRecordCount()) {}

auto RTLib::Ext::OPX7::OPX7ShaderTableLayoutGeometry::operator=(const OPX7ShaderTableLayoutGeometry& geometry) noexcept -> OPX7ShaderTableLayoutGeometry&
{
	if (this != &geometry) {
		m_UpGeometryAS = nullptr;
		m_Name = geometry.m_Name;
		m_BaseRecordCount = geometry.m_BaseRecordCount;
		m_BaseRecordOffset = 0;
	}
	return *this;
}

auto RTLib::Ext::OPX7::OPX7ShaderTableLayoutGeometry::GetName() const noexcept -> std::string { return m_Name; }

auto RTLib::Ext::OPX7::OPX7ShaderTableLayoutGeometry::GetBaseRecordOffset() const noexcept -> unsigned int { return m_BaseRecordOffset; }

 auto RTLib::Ext::OPX7::OPX7ShaderTableLayoutGeometry::GetBaseRecordCount() const noexcept -> unsigned int { return m_BaseRecordCount; }

void RTLib::Ext::OPX7::OPX7ShaderTableLayoutGeometry::Internal_SetUpGeometryAS(OPX7ShaderTableLayoutGeometryAS* geometryAS) noexcept
{
	m_UpGeometryAS = geometryAS;
}

void RTLib::Ext::OPX7::OPX7ShaderTableLayoutGeometry::Internal_SetBaseRecordOffset(unsigned int offset) noexcept {
	m_BaseRecordOffset = offset;
}

RTLib::Ext::OPX7::OPX7ShaderTableLayoutInstance::OPX7ShaderTableLayoutInstance(const std::string& name,GeometryAS* geometryAS) noexcept {
	m_Name = name;
	m_DwGeometryAS = geometryAS;
}

RTLib::Ext::OPX7::OPX7ShaderTableLayoutInstance::OPX7ShaderTableLayoutInstance(const std::string& name, InstanceAS* instanceAS) noexcept {
	m_Name = name;
	m_DwInstanceAS = instanceAS;
}

RTLib::Ext::OPX7::OPX7ShaderTableLayoutInstance::OPX7ShaderTableLayoutInstance(const OPX7ShaderTableLayoutInstance& instance) noexcept
{
	m_Name = instance.m_Name;
	m_UpInstanceAS = nullptr;
	m_DwGeometryAS = instance.m_DwGeometryAS;
	m_DwInstanceAS = instance.m_DwInstanceAS;
	m_RecordCount = 0;
	m_RecordStride = 0;
	m_RecordOffset = 0;
}

auto RTLib::Ext::OPX7::OPX7ShaderTableLayoutInstance::operator=(const OPX7ShaderTableLayoutInstance& instance) noexcept -> OPX7ShaderTableLayoutInstance&
{
	// TODO: return ステートメントをここに挿入します
	if (this != &instance) {
		m_Name = instance.m_Name;
		m_UpInstanceAS = nullptr;
		m_DwGeometryAS = instance.m_DwGeometryAS;
		m_DwInstanceAS = instance.m_DwInstanceAS;
		m_RecordCount  = 0;
		m_RecordStride = 0;
		m_RecordOffset = 0;
	}
	return *this;
}



auto RTLib::Ext::OPX7::OPX7ShaderTableLayoutInstance::GetName() const noexcept -> std::string { return m_Name; }

auto RTLib::Ext::OPX7::OPX7ShaderTableLayoutInstance::GetRecordStride() const noexcept -> unsigned int
{
	return m_RecordStride;
}

void RTLib::Ext::OPX7::OPX7ShaderTableLayoutInstance::SetRecordStride(unsigned int recordStride) noexcept {
	auto rootInstanceAS = RootInstanceAS();
	if (!rootInstanceAS) {
		return;
	}
	rootInstanceAS->Internal_SetRecordStride(recordStride);
}

auto RTLib::Ext::OPX7::OPX7ShaderTableLayoutInstance::GetRecordCount() const noexcept -> unsigned int
{
	return m_RecordCount;
}

auto RTLib::Ext::OPX7::OPX7ShaderTableLayoutInstance::GetRecordOffset() const noexcept -> unsigned int
{
	return m_RecordOffset;
}

auto RTLib::Ext::OPX7::OPX7ShaderTableLayoutInstance::GetDwGeometryAS() const noexcept -> const GeometryAS*
{
	return m_DwGeometryAS;
}

auto RTLib::Ext::OPX7::OPX7ShaderTableLayoutInstance::GetDwInstanceAS() const noexcept -> const InstanceAS*
{
	return m_DwInstanceAS;
}

auto RTLib::Ext::OPX7::OPX7ShaderTableLayoutInstance::RootInstanceAS() noexcept -> InstanceAS*
{
	if (m_UpInstanceAS) {
		return m_UpInstanceAS->RootInstanceAS();
	}
	else {
		return nullptr;
	}
}

auto RTLib::Ext::OPX7::OPX7ShaderTableLayoutInstance::RootInstanceAS() const noexcept -> const InstanceAS*
{
	if (m_UpInstanceAS) {
		return m_UpInstanceAS->RootInstanceAS();
	}
	else {
		return nullptr;
	}
}

void RTLib::Ext::OPX7::OPX7ShaderTableLayoutInstance::Show() noexcept
{
	std::cout << "Instance";
	if (m_DwInstanceAS) {
		std::cout << "(InstanceAS)\n";
	}
	else {
		std::cout << "(GeometryAS)\n";
	}
	std::cout << "Offset: " << m_RecordOffset << " Count: " << m_RecordCount << std::endl;
	if (m_DwInstanceAS) {
		m_DwInstanceAS->Show();
	}
}

void RTLib::Ext::OPX7::OPX7ShaderTableLayoutInstance::Internal_SetRecordStride(unsigned int recordStride) noexcept {
	m_RecordStride = recordStride;
	if (!m_DwInstanceAS) {
		return;
	}
	m_DwInstanceAS->Internal_SetRecordStride(recordStride);
}

void RTLib::Ext::OPX7::OPX7ShaderTableLayoutInstance::Internal_SetRecordOffset(unsigned int recordOffset) noexcept
{
	if (m_DwInstanceAS) {
		m_DwInstanceAS->Internal_SetRecordOffset(recordOffset);
		m_RecordCount  = m_DwInstanceAS->GetRecordCount();
	}
	if (m_DwGeometryAS) {
		m_RecordCount = m_DwGeometryAS->GetBaseRecordCount() * m_RecordStride;
	}
	m_RecordOffset = recordOffset;
}

void RTLib::Ext::OPX7::OPX7ShaderTableLayoutInstance::Internal_SetUpInstanceAS(InstanceAS* instanceAS) noexcept
{
	m_UpInstanceAS = instanceAS;
}

RTLib::Ext::OPX7::OPX7ShaderTableLayoutInstanceAS::OPX7ShaderTableLayoutInstanceAS(const std::string& name) noexcept :m_Name{ name } {}

RTLib::Ext::OPX7::OPX7ShaderTableLayoutInstanceAS::OPX7ShaderTableLayoutInstanceAS(const OPX7ShaderTableLayoutInstanceAS& instance) noexcept
{
	m_Name = instance.m_Name;
	m_UpInstance   = nullptr;
	m_DwInstances  = instance.GetInstances();
	m_RecordStride = 0;
	SetRecordStride(instance.GetRecordStride());
}

auto RTLib::Ext::OPX7::OPX7ShaderTableLayoutInstanceAS::operator=(const OPX7ShaderTableLayoutInstanceAS& instanceAS) noexcept -> OPX7ShaderTableLayoutInstanceAS&
{
	// TODO: return ステートメントをここに挿入します
	if (this != &instanceAS) {
		m_Name = instanceAS.m_Name;
		m_UpInstance = nullptr;
		m_DwInstances = instanceAS.GetInstances();
		m_RecordStride = 0;
		SetRecordStride(instanceAS.GetRecordStride());
	}
	return *this;
}

auto RTLib::Ext::OPX7::OPX7ShaderTableLayoutInstanceAS::GetName() const noexcept -> std::string
{
	return m_Name;
}

void RTLib::Ext::OPX7::OPX7ShaderTableLayoutInstanceAS::SetName(const std::string& name) noexcept
{
	m_Name = name;
}

auto RTLib::Ext::OPX7::OPX7ShaderTableLayoutInstanceAS::GetInstanceCount() const noexcept -> unsigned int
{
	return m_DwInstances.size();
}

auto RTLib::Ext::OPX7::OPX7ShaderTableLayoutInstanceAS::GetInstances() const noexcept -> const Instances&
{
	// TODO: return ステートメントをここに挿入します
	return m_DwInstances;
}

void RTLib::Ext::OPX7::OPX7ShaderTableLayoutInstanceAS::SetInstance(const Instance& instance) noexcept
{
	m_DwInstances.push_back(instance);
	m_DwInstances.back().Internal_SetUpInstanceAS(this);
	if (m_RecordStride != 0) {
		auto rootInstanceAS = RootInstanceAS();
		if (rootInstanceAS) {
			rootInstanceAS->Internal_SetRecordOffset(0);
		}
	}
}

auto RTLib::Ext::OPX7::OPX7ShaderTableLayoutInstanceAS::GetRecordStride() const noexcept -> unsigned int
{
	return m_RecordStride;
}

void RTLib::Ext::OPX7::OPX7ShaderTableLayoutInstanceAS::SetRecordStride(unsigned int recordStride) noexcept
{
	auto rootInstanceAS = RootInstanceAS();
	if (rootInstanceAS) {
		rootInstanceAS->Internal_SetRecordStride(recordStride);
		rootInstanceAS->Internal_SetRecordOffset(0);
	}
}

auto RTLib::Ext::OPX7::OPX7ShaderTableLayoutInstanceAS::GetRecordCount() const noexcept -> unsigned int
{
	return m_RecordCount;
}

auto RTLib::Ext::OPX7::OPX7ShaderTableLayoutInstanceAS::RootInstanceAS() noexcept -> InstanceAS*
{
	if (m_UpInstance) {
		return m_UpInstance->RootInstanceAS();
	}
	else {
		return this;
	}
}

auto RTLib::Ext::OPX7::OPX7ShaderTableLayoutInstanceAS::RootInstanceAS() const noexcept -> const InstanceAS*
{
	if (m_UpInstance) {
		return m_UpInstance->RootInstanceAS();
	}
	else {
		return this;
	}
}

void RTLib::Ext::OPX7::OPX7ShaderTableLayoutInstanceAS::Show() noexcept
{
	std::cout << "InstanceAS\n";
	std::cout << " Count: " << m_RecordCount << std::endl;
	for (auto & instance : m_DwInstances) {
		instance.Show();
	}
}

void RTLib::Ext::OPX7::OPX7ShaderTableLayoutInstanceAS::Internal_SetRecordStride(unsigned int recordStride) noexcept {
	m_RecordStride = recordStride;
	for (auto& instance : m_DwInstances) {
		instance.Internal_SetRecordStride(recordStride);
	}
}

void RTLib::Ext::OPX7::OPX7ShaderTableLayoutInstanceAS::Internal_SetRecordOffset(unsigned int recordOffset) noexcept
{
	m_RecordCount = 0;
	for (auto& instance : m_DwInstances) {
		instance.Internal_SetRecordOffset(recordOffset);
		m_RecordCount += instance.GetRecordCount();
		recordOffset  += instance.GetRecordCount();
	}
}

RTLib::Ext::OPX7::OPX7ShaderTableLayout::OPX7ShaderTableLayout(const OPX7ShaderTableLayoutInstanceAS& tlas) noexcept {
	auto gasSet = std::unordered_set<const OPX7ShaderTableLayoutGeometryAS*>();
	auto iasSet = std::unordered_set<const OPX7ShaderTableLayoutInstanceAS*>();
	auto gasMap = std::unordered_map<const OPX7ShaderTableLayoutGeometryAS*, uint32_t>();
	auto iasMap = std::unordered_map<const OPX7ShaderTableLayoutInstanceAS*, uint32_t>();
	EnumerateImpl(&tlas, gasSet, iasSet);
	m_GeometryASLayouts.reserve(std::size(gasSet));
	for (auto& pGAS : gasSet) {
		gasMap[pGAS] = m_GeometryASLayouts.size();
		m_GeometryASLayouts.push_back(std::make_unique<OPX7ShaderTableLayoutGeometryAS>(*pGAS));
	}
	m_InstanceASLayouts.reserve(std::size(iasSet) + 1);
	iasMap[&tlas] = 0;
	m_InstanceASLayouts.push_back(std::make_unique<OPX7ShaderTableLayoutInstanceAS>(tlas));
	m_InstanceASLayouts.front()->SetName("Root");

	for (auto& pIAS : iasSet) {
		iasMap[pIAS] = m_InstanceASLayouts.size();
		m_InstanceASLayouts.push_back(std::make_unique<OPX7ShaderTableLayoutInstanceAS>(*pIAS));
	}


	for (auto& pIAS : iasSet) {
		for (auto& instance : m_InstanceASLayouts[iasMap[pIAS]]->m_DwInstances)
		{
			if (instance.m_UpInstanceAS) {
				instance.m_UpInstanceAS = m_InstanceASLayouts[iasMap[instance.m_UpInstanceAS]].get();
			}
			if (instance.m_DwGeometryAS) {
				instance.m_DwGeometryAS = m_GeometryASLayouts[gasMap[instance.m_DwGeometryAS]].get();
			}
			if (instance.m_DwInstanceAS) {
				instance.m_DwInstanceAS = m_InstanceASLayouts[iasMap[instance.m_DwInstanceAS]].get();
				instance.m_DwInstanceAS->m_UpInstance = &instance;
			}
		}
	}
	m_InstanceASLayouts[0]->SetRecordStride(tlas.GetRecordStride());

}

auto RTLib::Ext::OPX7::OPX7ShaderTableLayout::GetRecordCount() const noexcept -> unsigned int
{
	return m_InstanceASLayouts[0]->GetRecordCount();
}

auto RTLib::Ext::OPX7::OPX7ShaderTableLayout::GetRecordStride() const noexcept -> unsigned int
{
	return m_InstanceASLayouts[0]->GetRecordStride();
}

auto RTLib::Ext::OPX7::OPX7ShaderTableLayout::RootInstanceAS() noexcept -> OPX7ShaderTableLayoutInstanceAS*
{
	return m_InstanceASLayouts[0].get();
}

auto RTLib::Ext::OPX7::OPX7ShaderTableLayout::RootInstanceAS() const noexcept -> const OPX7ShaderTableLayoutInstanceAS*
{
	return m_InstanceASLayouts[0].get();
}

auto RTLib::Ext::OPX7::OPX7ShaderTableLayout::FindInstance(const std::string& name) const -> const OPX7ShaderTableLayoutInstance*
{
	return FindInstance(RootInstanceAS(),name);
}

auto RTLib::Ext::OPX7::OPX7ShaderTableLayout::FindInstanceAS(const std::string& name) const -> const OPX7ShaderTableLayoutInstanceAS*
{
	return FindInstanceAS(RootInstanceAS(), name);
}

auto RTLib::Ext::OPX7::OPX7ShaderTableLayout::FindGeometryAS(const std::string& name) const -> const OPX7ShaderTableLayoutGeometryAS*
{
	return FindGeometryAS(RootInstanceAS(), name);
}

auto RTLib::Ext::OPX7::OPX7ShaderTableLayout::FindGeometry(const std::string& name) const -> const OPX7ShaderTableLayoutGeometry*
{
	return FindGeometry(RootInstanceAS(), name);
}

auto RTLib::Ext::OPX7::OPX7ShaderTableLayout::FindInstance(const OPX7ShaderTableLayoutInstanceAS* instanceAS, const std::string& name) const -> const OPX7ShaderTableLayoutInstance*
{
	auto[ instanceName,second ] = SplitFirstOf(name);
	auto instance = FindChildInstance(instanceAS, instanceName);
	if (second.empty()) {
		return instance;
	}
	auto [instanceASName, second2] = SplitFirstOf(second);
	auto instanceAS2 = FindChildInstanceAS(instance, instanceASName);
	if (second2.empty()) {
		return nullptr;
	}
	return FindInstance(instanceAS2,second2);
}

auto RTLib::Ext::OPX7::OPX7ShaderTableLayout::FindInstanceAS(const OPX7ShaderTableLayoutInstanceAS* instanceAS, const std::string& name) const -> const OPX7ShaderTableLayoutInstanceAS*
{
	auto [instanceName, second] = SplitFirstOf(name);
	auto instance = FindChildInstance(instanceAS, instanceName);
	if (!instance) { return nullptr; }
	if (second.empty()) { return nullptr; }
	return FindInstanceAS(instance,second);
}

auto RTLib::Ext::OPX7::OPX7ShaderTableLayout::FindInstanceAS(const OPX7ShaderTableLayoutInstance* instance, const std::string& name) const -> const OPX7ShaderTableLayoutInstanceAS*
{
	auto [instanceASName, second] = SplitFirstOf(name);
	auto instanceAS = FindChildInstanceAS(instance, instanceASName);
	if (second.empty()) {
		return instanceAS;
	}
	auto [instanceName, second2] = SplitFirstOf(second);
	auto instance2 = FindChildInstance(instanceAS, instanceName);
	if (second2.empty()) {
		return nullptr;
	}
	return FindInstanceAS(instance2,second2);
}

auto RTLib::Ext::OPX7::OPX7ShaderTableLayout::FindGeometryAS(const OPX7ShaderTableLayoutInstance* instance, const std::string& name) const -> const OPX7ShaderTableLayoutGeometryAS*
{
	auto [asName, second] = SplitFirstOf(name);
	auto geometryAS = instance->GetDwGeometryAS();
	if (geometryAS) {
		if (!second.empty()) {
			return nullptr;
		}
		if (geometryAS->GetName() == asName) { 
			return geometryAS; 
		}
		else {
			return nullptr;
		}
	}
	auto instanceAS = instance->GetDwInstanceAS();
	if (instanceAS->GetName() != asName) {
		return nullptr;
	}
	if (second.empty()) {
		return nullptr;
	}
	return FindGeometryAS(instanceAS, second);
}

auto RTLib::Ext::OPX7::OPX7ShaderTableLayout::FindGeometry(const OPX7ShaderTableLayoutInstanceAS* instanceAS, const std::string& name) const -> const OPX7ShaderTableLayoutGeometry*
{
	auto [geometryASName, geometryName] = SplitLastOf(name);
	if (geometryName.empty()) {
		return nullptr;
	}
	auto geometryAS = FindGeometryAS(instanceAS, geometryASName);
	if (!geometryAS) { return nullptr; }
	return FindGeometry(geometryAS, geometryName);
}

auto RTLib::Ext::OPX7::OPX7ShaderTableLayout::FindGeometry(const OPX7ShaderTableLayoutInstance* instance, const std::string& name) const -> const OPX7ShaderTableLayoutGeometry*
{
	auto [geometryASName, geometryName] = SplitLastOf(name);
	if (geometryName.empty()) {
		return nullptr;
	}
	auto geometryAS = FindGeometryAS(instance, geometryASName);
	if (!geometryAS) { return nullptr; }
	return FindGeometry(geometryAS,geometryName);
}

auto RTLib::Ext::OPX7::OPX7ShaderTableLayout::FindGeometry(const OPX7ShaderTableLayoutGeometryAS* gas, const std::string& name) const -> const OPX7ShaderTableLayoutGeometry*
{
	return FindChildGeometry(gas,name);
}

auto RTLib::Ext::OPX7::OPX7ShaderTableLayout::FindGeometryAS(const OPX7ShaderTableLayoutInstanceAS* instanceAS, const std::string& name) const -> const OPX7ShaderTableLayoutGeometryAS*
{
	auto [instanceName, geometryASName] = SplitLastOf(name);

	if (geometryASName.empty()) { return nullptr; }
	auto instance = FindInstance(instanceAS, instanceName);

	if (!instance) { return nullptr; }
	auto geometryAS = FindChildGeometryAS(instance, geometryASName);

	return geometryAS;
}

auto RTLib::Ext::OPX7::OPX7ShaderTableLayout::FindChildInstance(const OPX7ShaderTableLayoutInstanceAS* instanceAS, const std::string& name) const -> const OPX7ShaderTableLayoutInstance*
{
	if (!instanceAS) { return nullptr; }
	for (auto& instance : instanceAS->GetInstances()) {
		if (instance.GetName() == name) { return &instance; }
	}
	return nullptr;
}

auto RTLib::Ext::OPX7::OPX7ShaderTableLayout::FindChildInstanceAS(const OPX7ShaderTableLayoutInstance* instance, const std::string& name) const -> const OPX7ShaderTableLayoutInstanceAS*
{
	if (!instance) { return nullptr; }
	if (instance->GetDwGeometryAS()) {
		return nullptr;
	}
	auto instanceAS = instance->GetDwInstanceAS();
	if (instanceAS->GetName() == name) {
		return instanceAS;
	}
	else {
		return nullptr;
	}
}

auto RTLib::Ext::OPX7::OPX7ShaderTableLayout::FindChildGeometryAS(const OPX7ShaderTableLayoutInstance* instance, const std::string& name) const -> const OPX7ShaderTableLayoutGeometryAS*
{
	if (!instance) { return nullptr; }

	if (instance->GetDwInstanceAS()) {
		return nullptr;
	}

	auto geometryAS = instance->GetDwGeometryAS();
	if (geometryAS->GetName() == name) {
		return geometryAS;
	}
	else {
		return nullptr;
	}
}

auto RTLib::Ext::OPX7::OPX7ShaderTableLayout::FindChildGeometry(const OPX7ShaderTableLayoutGeometryAS* gas, const std::string& name) const -> const OPX7ShaderTableLayoutGeometry*
{
	if (!gas) { return nullptr; }
	for (auto& geometry : gas->GetDwGeometries()) {
		if (geometry.GetName() == name) {
			return &geometry;
		}
	}
	return nullptr;
}

