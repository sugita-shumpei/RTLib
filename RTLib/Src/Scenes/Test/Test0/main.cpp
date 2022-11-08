#include <iostream>
#include <array>
#include <vector>
namespace RTLib {
	struct Color {
		
	};
	using  String = std::string;
	using  Float  = float;
	using  Float2 = std::array<float, 2>;
	using  Float3 = std::array<float, 4>;
	using  Float4 = std::array<float, 4>;
	using  UInt3  = std::array<uint32_t, 3>;
	using  Transform = std::array<float, 12>;
	template<typename T>
	using  Array  = std::vector<T>;
	struct Texture {

	};
	struct ColorTexture {
		Color          m_color;
	};
	struct ImageTexture {
		String         m_file_name;
	};
	struct CheckTexture {
		ColorOrTexture m_base_color;
		ColorOrTexture m_comp_color;
		Float          m_frequency;
	};
	struct FloatOrTexture {
		enum Type {
			FloatOrTextureTypeFloat   = 0,
			FloatOrTextureTypeTexture = 1,
		} m_type;
		union {
			Float   m_float;
			Texture m_texture;
		};
	};
	struct ColorOrTexture {
		enum Type {
			ColorOrTextureTypeColor   = 0,
			ColorOrTextureTypeTexture = 1,
		} m_type;
		union {
			Color   m_color;
			Texture m_texture;
		};
	};
	struct DiffuseMaterial {
		ColorOrTexture m_diffuse;
	};
	struct PhongMaterial {
		ColorOrTexture m_diffuse;
		ColorOrTexture m_specular;
		FloatOrTexture m_shinness;
	};
	struct DeltaMetalMaterial {
		ColorOrTexture m_specular;
	};
	struct DeltaGlassMaterial {
		ColorOrTexture m_specular;
		ColorOrTexture m_transmit;
		FloatOrTexture m_ior;
	};
	struct Geometry {

	};
	struct TriangleGeometry {
		Float3 m_vertex0;
		Float3 m_vertex1;
		Float3 m_vertex2;
	};
	struct TriangleArrayGeometry {
		Array<Float3>  m_array_vertices  ;
		Array<Float3>  m_array_normals   ;
		Array<Float3>  m_array_binormals ;
		Array<Float3>  m_array_tangents  ;
		Array<Float2>  m_array_texcoords ;
		Array< UInt3>  m_array_indices   ;
		Array<Texture> m_array_alpha_maps;
	};
	struct SphereGeometry {
		Float3 m_center;
		Float  m_radius;
	};
	struct SphereArrayGeometry {
		Array<Float3> m_array_center;
		Array<Float > m_array_radius;
	};
	struct TransformGeometry {
		Transform  m_transform;
		Geometry   m_base;
	};
	struct Model {};
	struct ObjModel {

	};
	struct FbxModel {

	};
	struct GltfModel {

	};
}

int main() {

	return 0;
}