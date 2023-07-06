#ifndef RTLIB_CORE_MESH__H
#define RTLIB_CORE_MESH__H
#include <RTLib/Core/Vector.h>
#ifndef __CUDACC__
#include <RTLib/Core/Object.h>
#include <memory>
#include <unordered_map>
namespace RTLib
{
	namespace Core
	{
		enum       MeshFlags
		{
			MeshFlagsUnknown  = 0,
			MeshFlagsNormal   = (1 << 0),
			MeshFlagsBinormal = (1 << 1),
			MeshFlagsTangent  = (1 << 2),
			MeshFlagsTexCoord = (1 << 3),
			MeshFlagsTexCoord0= MeshFlagsTexCoord,
			MeshFlagsTexCoord1= (1 << 4),
			MeshFlagsTexCoord2= (1 << 5),
			MeshFlagsTexCoord3= (1 << 6),
			MeshFlagsTexCoord4= (1 << 7),
			MeshFlagsTexCoord5= (1 << 8),
			MeshFlagsTexCoord6= (1 << 9),
			MeshFlagsTexCoord7= (1 <<10),
			MeshFlagsColor  = (1 << 11),
			MeshFlagsColor0 = MeshFlagsColor,
			MeshFlagsColor1 = (1 << 12),
			MeshFlagsColor2 = (1 << 13),
			MeshFlagsColor3 = (1 << 14),
			MeshFlagsColor4 = (1 << 15),
			MeshFlagsColor5 = (1 << 16),
			MeshFlagsColor6 = (1 << 17),
			MeshFlagsColor7 = (1 << 18),
			MeshFlagsBlendWeight = (1<<19),
			MeshFlagsBlendIndices= (1<<20),

		};
		enum class VertexAttributeFormat
		{
			eUnknown,
			eFloat32,
			eFloat16,
			eUNorm8,
			eSNorm8,
			eUNorm16,
			eSNorm16,
			eUInt8,
			eSInt8,
			eUInt16,
			eSInt16
		};
		enum class VertexAttribute
		{
			ePosition ,
			eNormal   ,
			eBinormal ,
			eTangent  ,
			eTexCoord0,
			eTexCoord1,
			eTexCoord2,
			eTexCoord3,
			eTexCoord4,
			eTexCoord5,
			eTexCoord6,
			eTexCoord7,
			eColor0,
			eColor1,
			eColor2,
			eColor3,
			eColor4,
			eColor5,
			eColor6,
			eColor7,
			eBlendWeight,
			eBlendIndices 
		};
		struct     VertexAttributeDescriptor
		{
			VertexAttributeDescriptor(
				VertexAttribute    attribute = VertexAttribute::ePosition,
				VertexAttributeFormat format = VertexAttributeFormat::eUnknown,
				UInt32            dimension = 0,
				UInt32                offset = 0,
				UInt32               binding = 0
			) noexcept;

			VertexAttribute       attribute;
			VertexAttributeFormat format;
			UInt32                dimension;
			UInt32                offset;
			UInt32                binding;
		};
		struct     VertexBindingDescriptor
		{
			UInt32               binding;
			UInt32               stride;
		};
		struct     MeshResourceDescriptor
		{
			std::vector< VertexAttributeDescriptor> attributes = {};
			std::vector< VertexBindingDescriptor>   bindings   = {};
		};

		class      MeshResource;

		RTLIB_CORE_DEFINE_OBJECT_TYPE_ID(Mesh, "E4D1F213-5868-47C1-874D-028BCBC1F7AA");
		class Mesh : public Core::Object
		{
			using MeshResourceMap = std::unordered_map<String, std::shared_ptr<MeshResource>>;
		public:
			static auto New(
				String name,
				UInt32 numVertices,
				UInt32 numIndices,
				UInt32 flags
			) -> std::shared_ptr<Mesh>;
			virtual ~Mesh() noexcept;

			virtual auto query_object(const TypeID& typeID) -> std::shared_ptr<Object> override;
			virtual auto get_type_id() const noexcept -> TypeID override;
			virtual auto get_name() const noexcept -> String override;

			void set_name(String name) noexcept;

			auto get_num_vertices() const noexcept -> UInt32;
			auto get_num_indices()  const noexcept -> UInt32;
			auto get_flags() const noexcept;

			auto get_vertices() const noexcept -> const Vector3*;
			auto get_vertices() noexcept -> Vector3*;

			auto get_normals() const noexcept -> const Vector3*;
			auto get_normals() noexcept -> Vector3*;

			auto get_binormals() const noexcept -> const Vector3*;
			auto get_binormals() noexcept -> Vector3*;

			auto get_tangents() const noexcept -> const Vector3*;
			auto get_tangents() noexcept -> Vector3*;

			auto get_texcoords(UInt32 idx) const noexcept -> const Vector3*;
			auto get_texcoords(UInt32 idx) noexcept -> Vector3*;

			auto get_colors(UInt32 idx) const noexcept -> const Vector3*;
			auto get_colors(UInt32 idx) noexcept -> Vector3*;

			auto get_indices() const noexcept -> const UInt32* { return m_Indices.data(); }
			auto get_indices() noexcept -> UInt32* { return m_Indices.data(); }

			Bool has_normal()   const noexcept;
			Bool has_binormal() const noexcept;
			Bool has_tangent()  const noexcept;
			Bool has_texcoord(UInt32 idx)  const noexcept;
			Bool has_color(UInt32 idx)  const noexcept;

			template<typename MeshResourceType>
			auto add_resources(String resourceTag, const MeshResourceDescriptor& desc) -> std::shared_ptr<MeshResourceType>
			{
				auto meshResource = MeshResourceType::New(std::dynamic_pointer_cast<RTLib::Core::Mesh>(shared_from_this()), resourceTag, desc);
				m_Resources[resourceTag] = std::static_pointer_cast<MeshResource>(meshResource);
				return meshResource;
			}
			auto get_resources(String resourceTag) -> std::shared_ptr<MeshResource>;

			void upload();
			void alloc();
			void free();
		private:
			Mesh(
				String name = "",
				UInt32 numVertices = 0,
				UInt32 numIndices = 0,
				UInt32 flags = 0
			);
		private:
			UInt32               m_NumVertices;
			UInt32               m_NumIndices;
			std::vector<Vector3> m_Positions;
			std::vector<Vector3> m_Normals;
			std::vector<Vector3> m_Binormals;
			std::vector<Vector3> m_Tangents;
			std::vector<Vector3> m_TexCoords[8];
			std::vector<Vector3> m_Colors[8];
			std::vector<UInt32>  m_Indices;
			MeshResourceMap      m_Resources = {};
			UInt32               m_Flags;
			String               m_Name;
		};

		RTLIB_CORE_DEFINE_OBJECT_TYPE_ID(MeshResource, "C16D35F4-81D9-4BA0-8C62-F133C49F20A5");
		class MeshResource: public Core::Object
		{
		public:
			using VertexAttributeDescs = std::vector<VertexAttributeDescriptor>;
			using VertexBindingDescs   = std::vector<VertexBindingDescriptor  >;

			virtual ~MeshResource() {}

			virtual void upload() = 0;
			virtual void alloc()  = 0;
			virtual void free()   = 0;

			virtual auto get_mesh() const noexcept ->  std::shared_ptr<Mesh> = 0;

			virtual auto get_attributes() const noexcept -> const VertexAttributeDescs& = 0;
			virtual auto get_bindings() const noexcept   -> const VertexBindingDescs&   = 0;
			virtual auto get_tag() const noexcept        -> String                      = 0;
		};
	}
}
#endif
#endif
