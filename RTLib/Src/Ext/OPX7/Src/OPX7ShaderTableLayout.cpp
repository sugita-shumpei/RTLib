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
	{
		auto gasSet = std::unordered_set<const OPX7ShaderTableLayoutGeometryAS*>();
		auto iasSet = std::unordered_set<const OPX7ShaderTableLayoutInstanceAS*>();
		auto gasMap = std::unordered_map<const OPX7ShaderTableLayoutGeometryAS*, uint32_t>();
		auto iasMap = std::unordered_map<const OPX7ShaderTableLayoutInstanceAS*, uint32_t>();
		EnumerateImpl(&tlas, gasSet, iasSet);
		m_GeometryASLayouts.reserve(std::size(gasSet));
		for (auto& pGAS : gasSet) {
			gasMap[pGAS] = m_GeometryASLayouts.size();
			m_GeometryASLayouts.push_back(std::make_unique<OPX7ShaderTableLayoutGeometryAS>(*pGAS));
			m_GeometryASLayouts.back()->m_Name = pGAS->m_Name;
		}
		m_InstanceASLayouts.reserve(std::size(iasSet) + 1);
		iasMap[&tlas] = 0;
		m_InstanceASLayouts.push_back(std::make_unique<OPX7ShaderTableLayoutInstanceAS>(tlas));
		m_InstanceASLayouts.front()->SetName("Root");

		for (auto& pIAS : iasSet) {
			iasMap[pIAS] = m_InstanceASLayouts.size();
			m_InstanceASLayouts.push_back(std::make_unique<OPX7ShaderTableLayoutInstanceAS>(*pIAS));
		}
		iasSet.insert(&tlas);
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
	{
		for (auto& geometryAS : m_GeometryASLayouts) {
			m_BaseGeometryASNames.push_back(geometryAS->GetName());
			m_BaseDescs[geometryAS->GetName()].pData = geometryAS.get();
			m_BaseDescs[geometryAS->GetName()].baseRecordOffset = 0;
			m_BaseDescs[geometryAS->GetName()].baseRecordCount  = geometryAS->GetBaseRecordCount();
			for (auto& geometry : geometryAS->GetDwGeometries()) {
				m_BaseGeometryNames.push_back(geometryAS->GetName() + "/" + geometry.GetName());
				m_BaseDescs[geometryAS->GetName() + "/" + geometry.GetName()].pData = &geometry;
				m_BaseDescs[geometryAS->GetName() + "/" + geometry.GetName()].baseRecordOffset = geometry.GetBaseRecordOffset();
				m_BaseDescs[geometryAS->GetName() + "/" + geometry.GetName()].baseRecordCount  = geometry.GetBaseRecordCount();
			}
		}

	}
	{
		m_Descs["Root"].pData = m_InstanceASLayouts[0].get();
		m_Descs["Root"].recordOffset = 0;
		m_Descs["Root"].recordCount  = m_InstanceASLayouts[0]->GetRecordCount();
		m_Descs["Root"].recordStride = m_InstanceASLayouts[0]->GetRecordStride();
		auto instanceASMap = std::unordered_map<std::string, const OPX7ShaderTableLayoutInstanceAS*>();
		for (auto& instance : m_InstanceASLayouts[0]->GetInstances()) {
			m_InstanceNames.push_back("Root/" + instance.GetName());
			m_Descs["Root/" + instance.GetName()].pData = &instance;
			m_Descs["Root/" + instance.GetName()].recordCount  = instance.GetRecordCount ();
			m_Descs["Root/" + instance.GetName()].recordOffset = instance.GetRecordOffset();
			m_Descs["Root/" + instance.GetName()].recordStride = instance.GetRecordStride();
			if (instance.GetDwGeometryAS()) {
				for (auto& geometry : instance.GetDwGeometryAS()->GetDwGeometries()) {
					m_GeometryNames.push_back("Root/" + instance.GetName() + "/" + geometry.GetName());
					m_Descs["Root/" + instance.GetName()+"/"+geometry.GetName()].pData        = &geometry;
					m_Descs["Root/" + instance.GetName()+"/"+geometry.GetName()].recordCount  = geometry.GetBaseRecordCount() * instance.GetRecordStride();
					m_Descs["Root/" + instance.GetName()+"/"+geometry.GetName()].recordOffset = instance.GetRecordOffset()    + geometry.GetBaseRecordOffset() * instance.GetRecordStride();
					m_Descs["Root/" + instance.GetName()+"/"+geometry.GetName()].recordStride = instance.GetRecordStride() ;
				}
			}
			else {
				instanceASMap["Root/" + instance.GetName()] = instance.GetDwInstanceAS();
			}
		}
		while (!instanceASMap.empty()) {
			auto newInstanceASMap = std::unordered_map<std::string, const OPX7ShaderTableLayoutInstanceAS*>();
			for (auto& [name, instanceAS] : instanceASMap) {
				for (auto& instance : instanceAS->GetInstances()) {
					m_InstanceNames.push_back(name + "/" + instance.GetName());
					auto& desc = m_Descs[name + "/" + instance.GetName()];
					desc.pData        = &instance;
					desc.recordCount  = instance.GetRecordCount();
					desc.recordOffset = instance.GetRecordOffset();
					desc.recordStride = instance.GetRecordStride();
					if (instance.GetDwGeometryAS())
					{
						auto baseGASName = instance.GetDwGeometryAS()->GetName();
						for (auto& geometry : instance.GetDwGeometryAS()->GetDwGeometries()) {
							m_GeometryNames.push_back(baseGASName + "/" + geometry.GetName());
							auto& baseDesc = m_BaseDescs[baseGASName + "/" + geometry.GetName()];
							m_Descs[name + "/" + instance.GetName() + "/" + geometry.GetName()].pData = &geometry;
							m_Descs[name + "/" + instance.GetName() + "/" + geometry.GetName()].recordCount  = baseDesc.baseRecordCount  * desc.recordStride;
							m_Descs[name + "/" + instance.GetName() + "/" + geometry.GetName()].recordOffset = baseDesc.baseRecordOffset * desc.recordStride + desc.recordOffset;
							m_Descs[name + "/" + instance.GetName() + "/" + geometry.GetName()].recordStride = desc.recordStride;
						}
					}
					else {
						newInstanceASMap[name + "/" + instance.GetName()] = instance.GetDwInstanceAS();
					}
				}
			}
			instanceASMap = newInstanceASMap;
		}

		m_MaxTraversalDepth = 1;
		for (auto& name : m_InstanceNames) {
			size_t traversableDepth = std::count(std::begin(name), std::end(name), '/');
			m_MaxTraversalDepth = std::max<size_t>(m_MaxTraversalDepth, traversableDepth + 1);
		}
	}
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

