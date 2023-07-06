#include <RTLib/Core/Mesh.h>

RTLib::Core::Mesh::Mesh(String name,UInt32 numVertices, UInt32 numIndices, UInt32 flags)
    :
    Core::Object(),
    m_Name{name},
    m_NumVertices{numVertices},
    m_NumIndices{numIndices},
    m_Flags{flags},
    m_Positions{ numVertices },
    m_Indices{ numIndices }
{
    m_Positions.resize(numVertices);
    m_Indices.resize(numIndices);
    if (flags & MeshFlagsNormal)
    {
        m_Normals.resize(numVertices);
    }
    if (flags & MeshFlagsBinormal)
    {
        m_Binormals.resize(numVertices);
    }
    if (flags & MeshFlagsTangent)
    {
        m_Tangents.resize(numVertices);
    }
    for (UInt32 i = 0; i < 8; ++i)
    {
        if (flags & (MeshFlagsTexCoord<<i))
        {
            m_TexCoords[i].resize(numVertices);
        }
    }
    for (UInt32 i = 0; i < 8; ++i)
    {
        if (flags & (MeshFlagsColor << i))
        {
            m_Colors[i].resize(numVertices);
        }
    }
}
auto RTLib::Core::Mesh::New(
    String name = "",
    UInt32 numVertices = 0,
    UInt32 numIndices = 0,
    UInt32 flags = 0
) -> std::shared_ptr<Mesh> {
    return std::shared_ptr<Mesh>(new Mesh(name, numVertices, numIndices, flags));
}
RTLib::Core::Mesh::~Mesh(){
    for (auto& [tag, resource] : m_Resources) {
        resource->free();
    }
    
}
auto RTLib::Core::Mesh::query_object(const TypeID& typeID) -> std::shared_ptr<Core::Object> {
    if (typeID == ObjectTypeID_Unknown || typeID == ObjectTypeID_Mesh) {
        return shared_from_this();
    }
    else {
        return nullptr;
    }
}

auto RTLib::Core::Mesh::get_type_id() const noexcept -> TypeID
{
    return ObjectTypeID_Mesh;
}

auto RTLib::Core::Mesh::get_name() const noexcept -> String
{
    return m_Name;
}

void RTLib::Core::Mesh::set_name(String name) noexcept
{
    m_Name = name;
}

auto RTLib::Core::Mesh::get_num_vertices() const noexcept -> UInt32
{
    return m_NumVertices;
}

auto RTLib::Core::Mesh::get_num_indices() const noexcept -> UInt32
{
    return m_NumIndices;
}

auto RTLib::Core::Mesh::get_vertices() const noexcept -> const Vector3*
{
    return m_Positions.data();
}

auto RTLib::Core::Mesh::get_vertices() noexcept -> Vector3*
{
    return m_Positions.data();
}

auto RTLib::Core::Mesh::get_normals() const noexcept -> const Vector3*
{
    if (!has_normal()) { return nullptr; }
    return m_Normals.data();
}

auto RTLib::Core::Mesh::get_normals() noexcept -> Vector3*
{
    if (!has_normal()) { return nullptr; }
    return m_Normals.data();
}

auto RTLib::Core::Mesh::get_binormals() const noexcept -> const Vector3*
{
    if (!has_binormal()) { return nullptr; }
    return m_Binormals.data();
}

auto RTLib::Core::Mesh::get_binormals() noexcept -> Vector3*
{
    if (!has_binormal()) { return nullptr; }
    return m_Binormals.data();
}

auto RTLib::Core::Mesh::get_tangents() const noexcept -> const Vector3*
{
    if (!has_tangent()) { return nullptr; }
    return m_Tangents.data();
}

auto RTLib::Core::Mesh::get_tangents() noexcept -> Vector3*
{
    if (!has_tangent()) { return nullptr; }
    return m_Tangents.data();
}

auto RTLib::Core::Mesh::get_texcoords(UInt32 idx) const noexcept -> const Vector3*
{
    if (!has_texcoord(idx)) { return nullptr; }
    return m_TexCoords[idx].data();
}

auto RTLib::Core::Mesh::get_texcoords(UInt32 idx) noexcept -> Vector3*
{
    if (!has_texcoord(idx)) { return nullptr; }
    return m_TexCoords[idx].data();
}

auto RTLib::Core::Mesh::get_colors(UInt32 idx) const noexcept -> const Vector3*
{
    if (!has_color(idx)) { return nullptr; }
    return m_Colors[idx].data();
}

auto RTLib::Core::Mesh::get_colors(UInt32 idx) noexcept -> Vector3*
{
    if (!has_color(idx)) { return nullptr; }
    return m_Colors[idx].data();
}

auto RTLib::Core::Mesh::get_flags() const noexcept
{
    return m_Flags;
}

RTLib::Core::Bool RTLib::Core::Mesh::has_normal() const noexcept
{
    return (m_Flags&MeshFlagsNormal);
}

RTLib::Core::Bool RTLib::Core::Mesh::has_binormal() const noexcept
{
    return (m_Flags & MeshFlagsBinormal);
}

RTLib::Core::Bool RTLib::Core::Mesh::has_tangent() const noexcept
{
    return (m_Flags & MeshFlagsTangent);
}

RTLib::Core::Bool RTLib::Core::Mesh::has_texcoord(UInt32 idx) const noexcept
{
    return (m_Flags & (MeshFlagsTexCoord<<idx));
}

RTLib::Core::Bool RTLib::Core::Mesh::has_color(UInt32 idx) const noexcept
{
    return (m_Flags & (MeshFlagsColor << idx));
}

auto RTLib::Core::Mesh::get_resources(String resourceTag) -> std::shared_ptr<MeshResource>
{
    if (m_Resources.count(resourceTag) > 0) {
        return m_Resources.at(resourceTag);
    }
    else { return nullptr; }
}
void RTLib::Core::Mesh::upload()
{
    for (auto& [tag, resource] : m_Resources)
    {
        resource->upload();
    }
}

void RTLib::Core::Mesh::alloc()
{
    for (auto& [tag, resource] : m_Resources)
    {
        resource->alloc();
    }
}
void RTLib::Core::Mesh::free()
{
    for (auto& [tag, resource] : m_Resources)
    {
        resource->free();
    }
}

RTLib::Core::VertexAttributeDescriptor::VertexAttributeDescriptor(VertexAttribute attribute_, VertexAttributeFormat format_, UInt32 dimmension_, UInt32 offset_, UInt32 binding_) noexcept
    :attribute{attribute_ },format{format_ },dimension{dimmension_ },offset{offset_},binding{binding_}
{
}
