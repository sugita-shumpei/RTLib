#ifndef RTLIB_CORE_MESH__H
#define RTLIB_CORE_MESH__H

#include <RTLib/Core/Vector.h>

#ifndef __CUDACC__
#include <unordered_set>
#include <memory>
#include <array>

namespace RTLib
{
	inline namespace Core
	{
		enum class VertexAttribute
		{
			ePosition,
			eNormal,
			eTangent,
			eColor,
			eTexCoord0,
			eTexCoord1,
			eTexCoord2,
			eTexCoord3,
			eTexCoord4,
			eTexCoord5,
			eTexCoord6,
			eTexCoord7,
			eBlendWeight,
			eBlendIndices
		};
		enum class VertexAttributeFormat
		{
			eFloat32,
			eFloat16,
			eUNorm8,
			eSNorm8,
			eUNorm16,
			eSNorm16,
			eUInt8,
			eSInt8,
			eUInt16,
			eSInt16,
			eUInt32,
			eSInt32
		};
		struct     VertexAttributeDescriptor
		{
			VertexAttribute    attribute = VertexAttribute::ePosition;
			UInt32             dimension = 0;
			VertexAttributeFormat format = VertexAttributeFormat::eFloat32;
			UInt32               binding = 0;
			UInt32                offset = 0;
		};
		struct     VertexInputBindingDescriptor
		{
			UInt32               binding = 0;
			UInt32                stride = 0;
		};
		enum       MeshFlags: UInt32
		{
			MeshFlagsNone       = 0,
			MeshFlagsNormal     = (1<<0),
			MeshFlagsBinormal   = (1<<1),
			MeshFlagsTangent    = (1<<2),
			MeshFlagsTexCoord   = (1<<3),
			MeshFlagsTexCoord0  = MeshFlagsTexCoord,
			MeshFlagsTexCoord1  = (1<<4),
			MeshFlagsTexCoord2  = (1<<5),
			MeshFlagsTexCoord3  = (1<<6),
			MeshFlagsTexCoord4  = (1<<7),
			MeshFlagsTexCoord5  = (1<<8),
			MeshFlagsTexCoord6  = (1<<9),
			MeshFlagsTexCoord7  = (1<<10),
			MeshFlagsColor      = (1<<11),
			MeshFlagsColor0     = MeshFlagsColor,
			MeshFlagsColor1     = (1 << 12),
			MeshFlagsColor2     = (1 << 13),
			MeshFlagsColor3     = (1 << 14),
			MeshFlagsColor4     = (1 << 15),
			MeshFlagsColor5     = (1 << 16),
			MeshFlagsColor6     = (1 << 17),
			MeshFlagsColor7     = (1 << 18)
		};
		struct     MeshNativeHandle;
		struct     Mesh
		{
			using MeshNativeHandleMap = std::unordered_map<String, std::shared_ptr<MeshNativeHandle>>;

		     Mesh(UInt32 numVertices, UInt32 numIndices, UInt32 flags = 0) noexcept;
			~Mesh() noexcept{}

			Mesh(const Mesh& mesh);
			Mesh& operator=(const Mesh& mesh);

			Mesh(Mesh&& mesh);
			Mesh& operator=(Mesh&& mesh);

			template<typename MeshNativeHandleT>
			auto add_native_handle(
				const std::vector<VertexAttributeDescriptor>&    attributes,
				const std::vector<VertexInputBindingDescriptor>& bindings,
				String                                           name) -> std::shared_ptr<MeshNativeHandleT>
			{
				if (m_NativeHandles.count(name) > 0) { return nullptr; }
				auto handle = std::shared_ptr<MeshNativeHandleT>(MeshNativeHandleT(attributes, bindings, name));
				m_NativeHandles[name] = handle;
				m_NativeHandles[name]->set_mesh(this);
				return handle;
			}

			template<typename MeshNativeHandleT>
			auto get_native_handle(String                        name) -> std::shared_ptr<MeshNativeHandleT>
			{
				if (m_NativeHandles.count(name) > 0) { return nullptr; }
				return std::static_pointer_cast<MeshNativeHandleT, MeshNativeHandle>(m_NativeHandles.at(name));
			}

			auto get_num_vertices() const noexcept -> UInt32;
			auto get_num_indices()  const noexcept -> UInt32;

			auto get_flags() const noexcept -> UInt32 { return m_Flags; }
		
			Bool has_normal()   const noexcept;
			Bool has_tangent()  const noexcept;
			Bool has_binormal() const noexcept;
			Bool has_texcrd(UInt32 index) const noexcept;
			Bool has_color(UInt32 index) const noexcept;

			auto get_vertices() const noexcept -> const Vector3*;
			auto get_vertices() noexcept -> Vector3*;

			auto get_normals() const noexcept  -> const Vector3*;
			auto get_normals() noexcept  -> Vector3*;

			auto get_binormals() const noexcept  -> const Vector3*;
			auto get_binormals() noexcept  -> Vector3*;

			auto get_tangents() const noexcept  -> const Vector3*;
			auto get_tangents() noexcept  -> Vector3*;

			auto get_texcoords(UInt32 idx) const noexcept  -> const Vector3*;
			auto get_texcoords(UInt32 idx) noexcept  -> Vector3*;

			auto get_colors(UInt32 idx) const noexcept  -> const Vector4*;
			auto get_colors(UInt32 idx) noexcept  -> Vector4*;

			auto get_indices() const noexcept  -> const UInt32*;
			auto get_indices() noexcept  -> UInt32*;

			Bool has_native_handle(String name) const noexcept;

			void upload_native_handle(String nativeName);
			void download_native_handle(String nativeName);

			auto get_native_vertex_buffer_handle(String nativeName, UInt32 idx) const noexcept -> UInt64;
			auto get_native_index_buffer_handle(String nativeName) const noexcept -> UInt64;

			auto get_native_num_vertex_attributes(String nativeName) const noexcept -> UInt32;
			auto get_native_num_vertex_bindings(String nativeName) const noexcept -> UInt32;

			Bool get_native_vertex_attribues(String nativeName, std::vector<VertexAttributeDescriptor>& attributes) const noexcept;
			Bool get_native_vertex_bindings(String nativeName, std::vector<VertexInputBindingDescriptor>& bindings) const noexcept;
		private:
			std::unique_ptr<Vector3[]> m_Vertices;
			std::unique_ptr<Vector3[]> m_Normals;
			std::unique_ptr<Vector3[]> m_Binormals;
			std::unique_ptr<Vector3[]> m_Tangents;
			std::unique_ptr<Vector3[]> m_TexCoords[8];
			std::unique_ptr<Vector4[]> m_Colors[8];
			std::unique_ptr<UInt32[]>  m_Indices;
			UInt32                     m_NumVertices;
			UInt32                     m_NumIndices;
			UInt32                     m_Flags;
			MeshNativeHandleMap        m_NativeHandles;
		};
		struct MeshNativeHandle
		{
			friend class Mesh;

			MeshNativeHandle(
				const std::vector<VertexAttributeDescriptor>&    attributes,
				const std::vector<VertexInputBindingDescriptor>& bindings,
				String                                           name)
				:
				m_VertexAttributes{ attributes },
				m_VertexBindings{ bindings },
				m_Name{name}
			{}

			virtual ~MeshNativeHandle() 
			{}

			auto get_num_vertex_attribues() const noexcept -> UInt32;
			auto get_num_vertex_bindings() const noexcept -> UInt32;

			auto get_vertex_attribues() const noexcept -> const std::vector<VertexAttributeDescriptor>&;
			auto get_vertex_bindings() const noexcept -> const std::vector<VertexInputBindingDescriptor>&;

			auto get_name() const noexcept -> String;

			virtual auto clone(Mesh* newMesh = nullptr) -> MeshNativeHandle* = 0;
			virtual auto move (Mesh* newMesh = nullptr) -> MeshNativeHandle* = 0;

			virtual void upload() = 0;
			virtual void download() = 0;

			virtual auto get_mesh() const noexcept -> const Mesh* = 0;
			virtual auto get_mesh() noexcept -> Mesh* = 0;

			virtual auto get_native_vertex_buffer_handle(UInt32 idx) const noexcept -> UInt64 = 0;
			virtual auto get_native_index_buffer_handle() const noexcept -> UInt64 = 0;

		protected:
			virtual void set_mesh(Mesh* mesh) = 0;
		private:
			std::vector<VertexAttributeDescriptor   > m_VertexAttributes;
			std::vector<VertexInputBindingDescriptor> m_VertexBindings;
			String                                    m_Name;

		};
	}
}

#endif
#endif
