#ifndef TEST_TESTLIB_CORNEL_BOX__H
#define TEST_TESTLIB_CORNEL_BOX__H
#include <OptiXToolkit/ShaderUtil/vec_math.h>
#include <unordered_map>
#include <array>

#define TESTLIB_CORNEL_BOX_ADD(NAME) \
	void add_##NAME() { \
		if (verticesMap.count(#NAME)>0){ return;} \
		size_t curGroupCount = groupNames.size(); \
		size_t curVertexCount = std::size(vertices); \
		size_t curIndexCount  = std::size(indices); \
		size_t NAME##VertexCount = std::size(vertices##NAME); \
		size_t NAME##IndexCount = std::size(indices##NAME); \
		groupNames.push_back(#NAME); \
		verticesMap[#NAME] = {(unsigned int)curVertexCount,(unsigned int)NAME##VertexCount}; \
		indicesMap[#NAME] = {(unsigned int)curIndexCount,(unsigned int)NAME##IndexCount}; \
		vertices.reserve(curVertexCount+NAME##VertexCount); \
		std::copy(std::begin(vertices##NAME),std::end(vertices##NAME),std::back_inserter(vertices)); \
		indices.reserve(curIndexCount+NAME##IndexCount); \
		std::copy(std::begin(indices##NAME),std::end(indices##NAME),std::back_inserter(indices)); \
		diffuses.push_back(defaultDiffuse##NAME);\
		emissions.push_back(defaultEmission##NAME);\
	}

namespace TestLib
{
// # The original Cornell Box in OBJ format.
// # Note that the real box is not a perfect cube, so
// # the faces are imperfect in this data set.
// #
// # Created by Guedis Cardenas and Morgan McGuire at Williams College, 2011
// # Released into the Public Domain.
// #
// # http://graphics.cs.williams.edu/data
// # http ://www.graphics.cornell.edu/online/box/data.html
// #
	//vertexBuffer->at(0) = make_float3(+0.5f, -0.5f, -1.0f);
	//vertexBuffer->at(1) = make_float3(-0.5f, -0.5f, -1.0f);
	//vertexBuffer->at(2) = make_float3(0.0f, +0.5f, -1.0f);
	struct CornelBox {
		// Vertices
		static inline constexpr float3 verticesFloor[]     = {
			//## Object floor
			{-1.01,0.00, 0.99 },
			{ 1.00,0.00, 0.99 },
			{ 1.00,0.00,-1.04 },
			{-0.99,0.00,-1.04 },
			//g floor
			//usemtl floor
			//f - 4 - 3 - 2 - 1
		};
		static inline constexpr float3 verticesCeiling[]   = {
			//## Object ceiling
			{-1.02,1.99, 0.99 },
			{-1.02,1.99,-1.04 },
			{ 1.00,1.99,-1.04 },
			{ 1.00,1.99, 0.99 },
			//g ceiling
			//usemtl ceiling
			//f - 4 - 3 - 2 - 1
		};
		static inline constexpr float3 verticesBackwall[]  = {
			//## Object backwall
			{-0.99,0.00,-1.04},
			{ 1.00,0.00,-1.04},
			{ 1.00,1.99,-1.04},
			{-1.02,1.99,-1.04},			//g backWall
			//{-0.5f, -0.5f,-1.0f},
			//{+0.5f, -0.5f,-1.0f},
			//{+0.5f, +0.5f,-1.0f},
			//{-0.5f, +0.5f,-1.0f},

			//usemtl backWall
			//f - 4 - 3 - 2 - 1
		};
		static inline constexpr float3 verticesRightwall[] = {
			//## Object rightwall
			{1.00,0.00,-1.04},
			{1.00,0.00, 0.99},
			{1.00,1.99, 0.99},
			{1.00,1.99,-1.04},
			//g rightWall
			//usemtl rightWall
			//f - 4 - 3 - 2 - 1
		};
		static inline constexpr float3 verticesLeftwall[]  = {
			//## Object leftWall
			{-1.01,0.00, 0.99},
			{-0.99,0.00,-1.04},
			{-1.02,1.99,-1.04},
			{-1.02,1.99, 0.99},
			//g leftWall
			//usemtl leftWall
			//f - 4 - 3 - 2 - 1
		};
		static inline constexpr float3 verticesShortbox[]  = {
			//## Object shortBox
			//usemtl shortBox

			//# Top Face
			{ 0.53, 0.60,0.75 },
			{ 0.70, 0.60,0.17 },
			{ 0.13, 0.60,0.00 },
			{-0.05, 0.60,0.57 },
			//f - 4 - 3 - 2 - 1

			//# Left Face
			{-0.05,0.00,0.57 },
			{-0.05,0.60,0.57 },
			{ 0.13,0.60,0.00 },
			{ 0.13,0.00,0.00 },
			//f - 4 - 3 - 2 - 1

			//# Front Face
			{ 0.53,0.00,0.75 },
			{ 0.53,0.60,0.75 },
			{-0.05,0.60,0.57 },
			{-0.05,0.00,0.57 },
			//f - 4 - 3 - 2 - 1

			//# Right Face
			{0.70,0.00,0.17},
			{0.70,0.60,0.17},
			{0.53,0.60,0.75},
			{0.53,0.00,0.75},
			//f - 4 - 3 - 2 - 1

			//# Back Face
			{ 0.13,0.00,0.00},
			{ 0.13,0.60,0.00},
			{ 0.70,0.60,0.17},
			{ 0.70,0.00,0.17},
			//f - 4 - 3 - 2 - 1
			//# Bottom Face
			{ 0.53,0.00,0.75},
			{ 0.70,0.00,0.17},
			{ 0.13,0.00,0.00},
			{-0.05,0.00,0.57},
			//f - 12 - 11 - 10 - 9
			//g shortBox
			//usemtl shortBox
		};
		static inline constexpr float3 verticesTallbox[]   = {
			//## Object tallBox
			//usemtl tallBox
			//# Top Face
			{ -0.53,1.20, 0.09 },
			{  0.04,1.20,-0.09 },
			{ -0.14,1.20,-0.67 },
			{ -0.71,1.20,-0.49 },
			//f - 4 - 3 - 2 - 1

			//# Left Face
			{-0.53,  0.00, 0.09},
			{-0.53,  1.20, 0.09},
			{-0.71,  1.20,-0.49},
			{-0.71,  0.00,-0.49},
			//f - 4 - 3 - 2 - 1

			//# Back Face
			{-0.71,0.00,-0.49},
			{-0.71,1.20,-0.49},
			{-0.14,1.20,-0.67},
			{-0.14,0.00,-0.67},
			//f - 4 - 3 - 2 - 1

			//# Right Face
			{-0.14,0.00,-0.67},
			{-0.14,1.20,-0.67},
			{ 0.04,1.20,-0.09},
			{ 0.04,0.00,-0.09},
			//f - 4 - 3 - 2 - 1

			//# Front Face
			{ 0.04,0.00,-0.09},
			{ 0.04,1.20,-0.09},
			{-0.53,1.20, 0.09},
			{-0.53,0.00, 0.09},
			//f - 4 - 3 - 2 - 1

			//# Bottom Face
			{-0.53,0.00, 0.09},
			{ 0.04,0.00,-0.09},
			{-0.14,0.00,-0.67},
			{-0.71,0.00,-0.49},
			//f - 8 - 7 - 6 - 5},
			//g tallBox
			//usemtl tallBox
		};
		static inline constexpr float3 verticesLight[]     = {
			//## Object light
			{-0.24,1.98, 0.16},
			{-0.24,1.98,-0.22},
			{ 0.23,1.98,-0.22},
			{ 0.23,1.98, 0.16},
			//g light
			//usemtl light
			//f - 4 - 3 - 2 - 1
		};
		// Indices
		static inline constexpr uint3 indicesFloor[]       = {
			{0,1,2},{2,3,0}
		};
		static inline constexpr uint3 indicesCeiling[]     = {
			{0,1,2},{2,3,0}
		};
		static inline constexpr uint3 indicesBackwall[]    = {
			{0,1,2},{2,3,0}
		};
		static inline constexpr uint3 indicesRightwall[]   = {
			{0,1,2},{2,3,0}
		};
		static inline constexpr uint3 indicesLeftwall[]    = {
			{0,1,2},{2,3,0}
		};
		static inline constexpr uint3 indicesShortbox[]    = {
			{0,1,2},{2,3,0},
			{4,5,6},{6,7,4},
			{8,9,10},{10,11,8},
			{12,13,14},{14,15,12},
			{16,17,18},{18,19,16},
			{20,21,22},{22,23,20},
		};
		static inline constexpr uint3 indicesTallbox[]     = {
			{0,1,2},{2,3,0},
			{4,5,6},{6,7,4},
			{8,9,10},{10,11,8},
			{12,13,14},{14,15,12},
			{16,17,18},{18,19,16},
			{20,21,22},{22,23,20},
		};
		static inline constexpr uint3 indicesLight[]       = {
			{0,1,2},{2,3,0}
		};
		// Diffuse
		static inline constexpr float3 defaultDiffuseFloor     = { 0.725, 0.71, 0.68  };
		static inline constexpr float3 defaultDiffuseCeiling   = { 0.725, 0.71, 0.68  };
		static inline constexpr float3 defaultDiffuseBackwall  = { 0.725, 0.71, 0.68  };
		static inline constexpr float3 defaultDiffuseRightwall = {  0.14, 0.45, 0.091 };
		static inline constexpr float3 defaultDiffuseLeftwall  = {  0.63,0.065, 0.05  };
		static inline constexpr float3 defaultDiffuseShortbox  = { 0.725, 0.71, 0.68  };
		static inline constexpr float3 defaultDiffuseTallbox   = { 0.725, 0.71, 0.68  };
		static inline constexpr float3 defaultDiffuseLight     = { 0.78 , 0.78, 0.78  };
		// Emission
		static inline constexpr float3 defaultEmissionFloor     = { 0, 0,0 };
		static inline constexpr float3 defaultEmissionCeiling   = { 0, 0,0 };
		static inline constexpr float3 defaultEmissionBackwall  = { 0, 0,0 };
		static inline constexpr float3 defaultEmissionRightwall = { 0, 0,0 };
		static inline constexpr float3 defaultEmissionLeftwall  = { 0, 0,0 };
		static inline constexpr float3 defaultEmissionShortbox  = { 0, 0,0 };
		static inline constexpr float3 defaultEmissionTallbox   = { 0, 0,0 };
		static inline constexpr float3 defaultEmissionLight     = {17,12,4 };

		TESTLIB_CORNEL_BOX_ADD(Floor);
		TESTLIB_CORNEL_BOX_ADD(Ceiling);
		TESTLIB_CORNEL_BOX_ADD(Backwall);
		TESTLIB_CORNEL_BOX_ADD(Rightwall);
		TESTLIB_CORNEL_BOX_ADD(Leftwall);
		TESTLIB_CORNEL_BOX_ADD(Shortbox);
		TESTLIB_CORNEL_BOX_ADD(Tallbox);
		TESTLIB_CORNEL_BOX_ADD(Light);

		std::vector<std::string> groupNames = {};

		std::unordered_map<std::string, uint2> verticesMap = {};
		std::unordered_map<std::string, uint2> indicesMap = {};

		std::vector<float3>   vertices   = {};
		std::vector<uint3>    indices    = {};

		std::vector<float3>   diffuses   = {};
		std::vector<float3>   emissions  = {};
	};
}
#endif
