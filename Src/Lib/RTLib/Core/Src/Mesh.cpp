#include <RTLib/Core/Mesh.h>

RTLib::Core::Bool RTLib::Core::Mesh::has_normal() const noexcept
{
    return m_Flags & MeshFlagsNormal;
}

RTLib::Core::Bool RTLib::Core::Mesh::has_tangent() const noexcept
{
    return m_Flags & MeshFlagsTangent;
}

RTLib::Core::Bool RTLib::Core::Mesh::has_binormal() const noexcept
{
    return m_Flags & MeshFlagsBinormal;
}

RTLib::Core::Bool RTLib::Core::Mesh::has_texcrd(UInt32 index) const noexcept
{
    if (index >= 8) { return false; }
    return m_Flags & (MeshFlagsTexCoord0 << index);
}

RTLib::Core::Bool RTLib::Core::Mesh::has_color(UInt32 index) const noexcept
{
    if (index >= 8) { return false; }
    return m_Flags & (MeshFlagsColor0 << index);
}

auto RTLib::Core::Mesh::get_vertices() const noexcept -> const Vector3*
{
    return m_Vertices.get();
}

auto RTLib::Core::Mesh::get_vertices() noexcept -> Vector3*
{
    return m_Vertices.get();
}

auto RTLib::Core::Mesh::get_normals() const noexcept -> const Vector3*
{
    return m_Normals.get();
}

auto RTLib::Core::Mesh::get_normals() noexcept -> Vector3*
{
    return m_Normals.get();
}

auto RTLib::Core::Mesh::get_binormals() const noexcept -> const Vector3*
{
    return m_Binormals.get();
}

auto RTLib::Core::Mesh::get_binormals() noexcept -> Vector3*
{
    return m_Binormals.get();
}

auto RTLib::Core::Mesh::get_tangents() const noexcept -> const Vector3*
{
    return m_Tangents.get();
}

auto RTLib::Core::Mesh::get_tangents() noexcept -> Vector3*
{
    return m_Tangents.get();
}

auto RTLib::Core::Mesh::get_texcoords(UInt32 idx) const noexcept -> const Vector3*
{
    return m_TexCoords[idx].get();
}

auto RTLib::Core::Mesh::get_texcoords(UInt32 idx) noexcept -> Vector3*
{
    return m_TexCoords[idx].get();
}

auto RTLib::Core::Mesh::get_colors(UInt32 idx) const noexcept -> const Vector4*
{
    return m_Colors[idx].get();
}

auto RTLib::Core::Mesh::get_colors(UInt32 idx) noexcept -> Vector4*
{
    return m_Colors[idx].get();
}

auto RTLib::Core::Mesh::get_indices() const noexcept -> const UInt32*
{
    return m_Indices.get();
}

auto RTLib::Core::Mesh::get_indices() noexcept -> UInt32*
{
    return m_Indices.get();
}

RTLib::Core::Bool RTLib::Core::Mesh::has_native_handle(String name) const noexcept
{
    return m_NativeHandles.count(name) > 0;
}

void RTLib::Core::Mesh::upload_native_handle(String nativeName)
{
    if (m_NativeHandles.count(nativeName) == 0)
    {
        return;
    }
    auto nativeHandle = m_NativeHandles.at(nativeName);
    nativeHandle->upload();
}

void RTLib::Core::Mesh::download_native_handle(String nativeName)
{
    if (m_NativeHandles.count(nativeName) == 0) { return; }
    auto nativeHandle = m_NativeHandles.at(nativeName);
    nativeHandle->download();
}

auto RTLib::Core::Mesh::get_native_vertex_buffer_handle(String nativeName, UInt32 idx) const noexcept -> UInt64
{
    if (!has_native_handle(nativeName)) { return 0; }
    return m_NativeHandles.at(nativeName)->get_native_vertex_buffer_handle(idx);
}

auto RTLib::Core::Mesh::get_native_index_buffer_handle(String nativeName) const noexcept -> UInt64
{
    if (!has_native_handle(nativeName)) { return 0; }
    return m_NativeHandles.at(nativeName)->get_native_index_buffer_handle();
}

auto RTLib::Core::Mesh::get_native_num_vertex_attributes(String nativeName) const noexcept -> UInt32
{
    if (!has_native_handle(nativeName)) { return 0; }
    return m_NativeHandles.at(nativeName)->get_num_vertex_attribues();
}

auto RTLib::Core::Mesh::get_native_num_vertex_bindings(String nativeName) const noexcept -> UInt32
{
    if (!has_native_handle(nativeName)) { return 0; }
    return m_NativeHandles.at(nativeName)->get_num_vertex_bindings();
}

RTLib::Core::Bool RTLib::Core::Mesh::get_native_vertex_attribues(String nativeName, std::vector<VertexAttributeDescriptor>& attributes) const noexcept
{

    if (!has_native_handle(nativeName)) { return false; }
    auto nativeHandle = m_NativeHandles.at(nativeName);
    attributes = nativeHandle->get_vertex_attribues();
    return true;
}

RTLib::Core::Bool RTLib::Core::Mesh::get_native_vertex_bindings(String nativeName, std::vector<VertexInputBindingDescriptor>& bindings) const noexcept
{
    if (!has_native_handle(nativeName)) { return false; }
    auto nativeHandle = m_NativeHandles.at(nativeName);
    bindings = nativeHandle->get_vertex_bindings();
    return true;
}

RTLib::Core::Mesh::Mesh(UInt32 numVertices, UInt32 numIndices, UInt32 flags) noexcept
    :m_NumVertices{numVertices},m_NumIndices{numIndices},m_Flags{ flags }
{
    m_Vertices = std::unique_ptr<Vector3[]>(new Vector3[numVertices]);
    m_Indices  = std::unique_ptr<UInt32[]>(new UInt32[numIndices]);
    if (flags & MeshFlagsNormal)
    {
        m_Normals   = std::unique_ptr<Vector3[]>(new Vector3[numVertices]);
    }
    if (flags & MeshFlagsBinormal)
    {
        m_Binormals = std::unique_ptr<Vector3[]>(new Vector3[numVertices]);
    }
    if (flags & MeshFlagsTangent)
    {
        m_Tangents  = std::unique_ptr<Vector3[]>(new Vector3[numVertices]);
    }
    for (UInt32 i = 0; i < 8; ++i)
    {
        if (flags & (MeshFlagsTexCoord0<<i))
        {
            m_TexCoords[i] = std::unique_ptr<Vector3[]>(new Vector3[numVertices]);
        }
    }
    for (UInt32 i = 0; i < 8; ++i)
    {
        if (flags & (MeshFlagsColor0 << i))
        {
            m_Colors[i] = std::unique_ptr<Vector4[]>(new Vector4[numVertices]);
        }
    }
}

RTLib::Core::Mesh::Mesh(const Mesh& mesh)
    :m_NumVertices{ mesh.m_NumVertices},
     m_NumIndices{ mesh.m_NumIndices},
     m_Flags{mesh.m_Flags}
{
    m_Vertices = std::unique_ptr<Vector3[]>(new Vector3[m_NumVertices]);
    std::memcpy(m_Vertices.get(), mesh.m_Vertices.get(), sizeof(Vector3) * m_NumVertices);

    m_Indices = std::unique_ptr<UInt32[]>(new UInt32[m_NumIndices]);
    std::memcpy(m_Indices.get(), mesh.m_Indices.get(), sizeof(UInt32) * m_NumIndices);

    if (m_Flags & MeshFlagsNormal)
    {
        m_Normals = std::unique_ptr<Vector3[]>(new Vector3[m_NumVertices]);
        std::memcpy(m_Normals.get(), mesh.m_Normals.get(), sizeof(Vector3) * m_NumVertices);
    }
    if (m_Flags & MeshFlagsBinormal)
    {
        m_Binormals = std::unique_ptr<Vector3[]>(new Vector3[m_NumVertices]);
        std::memcpy(m_Binormals.get(), mesh.m_Binormals.get(), sizeof(Vector3) * m_NumVertices);
    }
    if (m_Flags & MeshFlagsTangent)
    {
        m_Tangents = std::unique_ptr<Vector3[]>(new Vector3[m_NumVertices]);
        std::memcpy(m_Tangents.get(), mesh.m_Tangents.get(), sizeof(Vector3) * m_NumVertices);
    }
    for (UInt32 i = 0; i < 8; ++i)
    {
        if (m_Flags & (MeshFlagsTexCoord0 << i))
        {
            m_TexCoords[i] = std::unique_ptr<Vector3[]>(new Vector3[m_NumVertices]);
            std::memcpy(m_TexCoords[i].get(), mesh.m_TexCoords[i].get(), sizeof(Vector3) * m_NumVertices);
        }
    }
    for (UInt32 i = 0; i < 8; ++i)
    {
        if (m_Flags & (MeshFlagsColor0 << i))
        {
            m_Colors[i] = std::unique_ptr<Vector4[]>(new Vector4[m_NumVertices]);
            std::memcpy(m_Colors[i].get(), mesh.m_Colors[i].get(), sizeof(Vector4) * m_NumVertices);
        }
    }

    for (auto& [name, nativeHandle] : mesh.m_NativeHandles)
    {
        m_NativeHandles[name] = std::shared_ptr<MeshNativeHandle>(nativeHandle->clone(this));
    }
}

RTLib::Core::Mesh& RTLib::Core::Mesh::operator=(const Mesh& mesh)
{
    // TODO: return ステートメントをここに挿入します
    if (this != &mesh)
    {
        m_NumVertices = mesh.m_NumVertices; 
        m_NumIndices  = mesh.m_NumIndices;
        m_Flags = mesh.m_Flags;
        m_NativeHandles = {};
        m_Vertices = std::unique_ptr<Vector3[]>(new Vector3[m_NumVertices]);
        std::memcpy(m_Vertices.get(), mesh.m_Vertices.get(), sizeof(Vector3) * m_NumVertices);

        m_Indices = std::unique_ptr<UInt32[]>(new UInt32[m_NumIndices]);
        std::memcpy(m_Indices.get(), mesh.m_Indices.get(), sizeof(UInt32) * m_NumIndices);

        if (m_Flags & MeshFlagsNormal)
        {
            m_Normals = std::unique_ptr<Vector3[]>(new Vector3[m_NumVertices]);
            std::memcpy(m_Normals.get(), mesh.m_Normals.get(), sizeof(Vector3) * m_NumVertices);
        }
        if (m_Flags & MeshFlagsBinormal)
        {
            m_Binormals = std::unique_ptr<Vector3[]>(new Vector3[m_NumVertices]);
            std::memcpy(m_Binormals.get(), mesh.m_Binormals.get(), sizeof(Vector3) * m_NumVertices);
        }
        if (m_Flags & MeshFlagsTangent)
        {
            m_Tangents = std::unique_ptr<Vector3[]>(new Vector3[m_NumVertices]);
            std::memcpy(m_Tangents.get(), mesh.m_Tangents.get(), sizeof(Vector3) * m_NumVertices);
        }
        for (UInt32 i = 0; i < 8; ++i)
        {
            if (m_Flags & (MeshFlagsTexCoord0 << i))
            {
                m_TexCoords[i] = std::unique_ptr<Vector3[]>(new Vector3[m_NumVertices]);
                std::memcpy(m_TexCoords[i].get(), mesh.m_TexCoords[i].get(), sizeof(Vector3) * m_NumVertices);
            }
        }
        for (UInt32 i = 0; i < 8; ++i)
        {
            if (m_Flags & (MeshFlagsColor0 << i))
            {
                m_Colors[i] = std::unique_ptr<Vector4[]>(new Vector4[m_NumVertices]);
                std::memcpy(m_Colors[i].get(), mesh.m_Colors[i].get(), sizeof(Vector4) * m_NumVertices);
            }
        }

        for (auto& [name, nativeHandle] : mesh.m_NativeHandles)
        {
            m_NativeHandles[name] = std::shared_ptr<MeshNativeHandle>(nativeHandle->clone(this));
        }
    }
    return *this;
}

RTLib::Core::Mesh::Mesh(Mesh&& mesh)
    :m_NumVertices{ mesh.m_NumVertices },
    m_NumIndices{ mesh.m_NumIndices },
    m_Flags{ mesh.m_Flags }
{
    for (auto& [name, nativeHandle] : mesh.m_NativeHandles)
    {
        m_NativeHandles[name] = std::shared_ptr<MeshNativeHandle>(nativeHandle->move(this));
    }

    m_Vertices  = std::move(mesh.m_Vertices);
    m_Indices   = std::move(mesh.m_Indices);
    m_Normals   = std::move(mesh.m_Normals);
    m_Binormals = std::move(mesh.m_Binormals);
    m_Tangents  = std::move(mesh.m_Tangents);

    for (UInt32 i = 0; i < 8; ++i) {
        m_TexCoords[i] = std::move(mesh.m_TexCoords[i]);
    }
    for (UInt32 i = 0; i < 8; ++i) {
        m_Colors[i] = std::move(mesh.m_Colors[i]);
    }

    mesh.m_NativeHandles.clear();
}

RTLib::Core::Mesh& RTLib::Core::Mesh::operator=(Mesh&& mesh)
{
    // TODO: return ステートメントをここに挿入します
    if (this != &mesh)
    {
        m_NumVertices = mesh.m_NumVertices;
        m_NumIndices  = mesh.m_NumIndices;
        m_Flags       = mesh.m_Flags;

        for (auto& [name, nativeHandle] : mesh.m_NativeHandles)
        {
            m_NativeHandles[name] = std::shared_ptr<MeshNativeHandle>(nativeHandle->move(this));
        }

        m_Vertices = std::move(mesh.m_Vertices);
        m_Indices = std::move(mesh.m_Indices);
        m_Normals = std::move(mesh.m_Normals);
        m_Binormals = std::move(mesh.m_Binormals);
        m_Tangents = std::move(mesh.m_Tangents);

        for (UInt32 i = 0; i < 8; ++i) {
            m_TexCoords[i] = std::move(mesh.m_TexCoords[i]);
        }
        for (UInt32 i = 0; i < 8; ++i) {
            m_Colors[i] = std::move(mesh.m_Colors[i]);
        }

        mesh.m_NativeHandles.clear();
    }
    return *this;
}

auto RTLib::Core::Mesh::get_num_vertices() const noexcept -> UInt32 { return m_NumVertices; }

auto RTLib::Core::Mesh::get_num_indices() const noexcept -> UInt32 { return m_NumIndices; }

auto RTLib::Core::MeshNativeHandle::get_num_vertex_attribues() const noexcept -> UInt32
{
    return m_VertexAttributes.size();
}

auto RTLib::Core::MeshNativeHandle::get_num_vertex_bindings() const noexcept -> UInt32
{
    return m_VertexBindings.size();
}

auto RTLib::Core::MeshNativeHandle::get_vertex_attribues() const noexcept -> const std::vector<VertexAttributeDescriptor>&
{
    // TODO: return ステートメントをここに挿入します
    return m_VertexAttributes;
}

auto RTLib::Core::MeshNativeHandle::get_vertex_bindings() const noexcept -> const std::vector<VertexInputBindingDescriptor>&
{
    // TODO: return ステートメントをここに挿入します
    return m_VertexBindings;
}

auto RTLib::Core::MeshNativeHandle::get_name() const noexcept -> String
{
    return m_Name;
}

