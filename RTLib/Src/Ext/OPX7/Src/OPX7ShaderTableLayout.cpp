#include <RTLib/Ext/OPX7/OPX7ShaderTableLayout.h>
#include <iostream>

RTLib::Ext::OPX7::OPX7ShaderTableLayoutGeometryAS::OPX7ShaderTableLayoutGeometryAS() noexcept {}

RTLib::Ext::OPX7::OPX7ShaderTableLayoutGeometryAS::OPX7ShaderTableLayoutGeometryAS(const OPX7ShaderTableLayoutGeometryAS& gas) noexcept
{
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

RTLib::Ext::OPX7::OPX7ShaderTableLayoutGeometry::OPX7ShaderTableLayoutGeometry(unsigned int baseRecordcount) noexcept {
	m_BaseRecordCount = baseRecordcount;
}

RTLib::Ext::OPX7::OPX7ShaderTableLayoutGeometry::OPX7ShaderTableLayoutGeometry(const OPX7ShaderTableLayoutGeometry& geometry) noexcept
	:m_BaseRecordCount(geometry.GetBaseRecordCount()) {}

auto RTLib::Ext::OPX7::OPX7ShaderTableLayoutGeometry::operator=(const OPX7ShaderTableLayoutGeometry& geometry) noexcept -> OPX7ShaderTableLayoutGeometry&
{
	if (this != &geometry) {
		m_UpGeometryAS = nullptr;
		m_BaseRecordCount = geometry.m_BaseRecordCount;
		m_BaseRecordOffset = 0;
	}
	return *this;
}

 auto RTLib::Ext::OPX7::OPX7ShaderTableLayoutGeometry::GetBaseRecordOffset() const noexcept -> unsigned int { return m_BaseRecordOffset; }

 auto RTLib::Ext::OPX7::OPX7ShaderTableLayoutGeometry::GetBaseRecordCount() const noexcept -> unsigned int { return m_BaseRecordCount; }

void RTLib::Ext::OPX7::OPX7ShaderTableLayoutGeometry::Internal_SetUpGeometryAS(OPX7ShaderTableLayoutGeometryAS* geometryAS) noexcept
{
	m_UpGeometryAS = geometryAS;
}

void RTLib::Ext::OPX7::OPX7ShaderTableLayoutGeometry::Internal_SetBaseRecordOffset(unsigned int offset) noexcept {
	m_BaseRecordOffset = offset;
}

RTLib::Ext::OPX7::OPX7ShaderTableLayoutInstance::OPX7ShaderTableLayoutInstance(GeometryAS* geometryAS) noexcept {
	m_DwGeometryAS = geometryAS;
}

RTLib::Ext::OPX7::OPX7ShaderTableLayoutInstance::OPX7ShaderTableLayoutInstance(InstanceAS* instanceAS) noexcept {
	m_DwInstanceAS = instanceAS;
}

RTLib::Ext::OPX7::OPX7ShaderTableLayoutInstance::OPX7ShaderTableLayoutInstance(const OPX7ShaderTableLayoutInstance& instance) noexcept
{
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
		m_UpInstanceAS = nullptr;
		m_DwGeometryAS = instance.m_DwGeometryAS;
		m_DwInstanceAS = instance.m_DwInstanceAS;
		m_RecordCount  = 0;
		m_RecordStride = 0;
		m_RecordOffset = 0;
	}
	return *this;
}

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

RTLib::Ext::OPX7::OPX7ShaderTableLayoutInstanceAS::OPX7ShaderTableLayoutInstanceAS() noexcept {}

RTLib::Ext::OPX7::OPX7ShaderTableLayoutInstanceAS::OPX7ShaderTableLayoutInstanceAS(const OPX7ShaderTableLayoutInstanceAS& instance) noexcept
{
	m_UpInstance   = nullptr;
	m_DwInstances  = instance.GetInstances();
	m_RecordStride = 0;
	SetRecordStride(instance.GetRecordStride());
}

auto RTLib::Ext::OPX7::OPX7ShaderTableLayoutInstanceAS::operator=(const OPX7ShaderTableLayoutInstanceAS& instanceAS) noexcept -> OPX7ShaderTableLayoutInstanceAS&
{
	// TODO: return ステートメントをここに挿入します
	if (this != &instanceAS) {
		m_UpInstance = nullptr;
		m_DwInstances = instanceAS.GetInstances();
		m_RecordStride = 0;
		SetRecordStride(instanceAS.GetRecordStride());
	}
	return *this;
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
