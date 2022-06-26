#ifndef RTLIB_EXT_OPX7_OPX7_ACCELERATION_STRUCTURE_H
#define RTLIB_EXT_OPX7_OPX7_ACCELERATION_STRUCTURE_H
#include <RTLib/Core/BaseObject.h>
#include <RTLib/Ext/OPX7/OPX7Common.h>
#include <RTLib/Ext/OPX7/OPX7ShaderTableLayout.h>
#include <RTLib/Ext/OPX7/UuidDefinitions.h>
#include <RTLib/Ext/CUDA/CUDABuffer.h>
#include <array>
#include <memory>
#include <tuple>
namespace RTLib {
	namespace Ext
	{
		namespace OPX7
		{
			/*LAYOUT<->BUILD_INPUT*/
			class OPX7Context;
			class OPX7Geometry;
			class OPX7GeometryTriangle;
			class OPX7GeometrySphere  ;
			class OPX7GeometryCustom  ;
			class OPX7BuildInputInstance;
			class OPX7AccelerationStructure;
			class OPX7GeometryAccelerationStructure;
			class OPX7InstanceAccelerationStructure;
			class OPX7Instance;
			class OPX7GeometryAccelerationStructureInstance;
			class OPX7InstanceAccelerationStructureInstance;

			class OPX7Geometry
			{
			public:
				struct Holder
				{
					virtual ~Holder()noexcept {}
				};
			public:
				virtual ~OPX7Geometry()noexcept {}
				virtual auto GetOptixBuildInputWithHolder()const noexcept -> std::tuple<OptixBuildInput, std::unique_ptr<Holder>> = 0;
			};
			class OPX7GeometryTriangle:public OPX7Geometry
			{
			public:
				struct    VertexView
				{
					CUDA::CUDABufferView   view   = {};
					unsigned int           stride = 0;
					OPX7::OPX7VertexFormat format = OPX7::OPX7VertexFormat::eNone;
					auto GetNumVertices()const noexcept -> unsigned int
					{
						if ((view.GetSizeInBytes() == 0)||(stride==0)) {
							return 0;
						}
						else {
							return view.GetSizeInBytes() / stride;
						}
					}
				};
				struct    TriIdxView
				{
					CUDA::CUDABufferView   view   = {};
					unsigned int           stride = 0;
					OPX7::OPX7TriIdxFormat format = OPX7::OPX7TriIdxFormat::eNone;
					auto GetNumIndices()const noexcept -> unsigned int
					{
						if ((view.GetSizeInBytes() == 0) || (stride == 0)) {
							return 0;
						}
						else {
							return view.GetSizeInBytes() / stride;
						}
					}
				};
				struct SbtOffsetView {
					CUDA::CUDABufferView      view       = {};
					unsigned int              stride     = 0;
					OPX7::OPX7SbtOffsetFormat format     = OPX7::OPX7SbtOffsetFormat::eNone;
					unsigned int              numRecords = 0;
					auto GetNumIndices()const noexcept -> unsigned int
					{
						if ((view.GetSizeInBytes() == 0) || (stride == 0)) {
							return 0;
						}
						else {
							return view.GetSizeInBytes() / stride;
						}
					}
				};
				struct TransformView {
					CUDA::CUDABufferView      view   = {};
					OPX7::OPX7TransformFormat format = OPX7::OPX7TransformFormat::eNone;
				};
			public:
				virtual ~OPX7GeometryTriangle()noexcept {}
				virtual auto GetOptixBuildInputWithHolder()const noexcept -> std::tuple<OptixBuildInput, std::unique_ptr<Holder>> override;
				//Setter
				void    SetVertexView(const    VertexView&       vertexView)noexcept
				{
					m_VbView = vertexView;
				}
				void    SetTriIdxView(const    TriIdxView&       triIdxView)noexcept
				{
					m_IbView = triIdxView;
				}
				void    SetSbtOffsetView(const SbtOffsetView&    sbtOffsetView)noexcept {
					m_SbView = sbtOffsetView;
					m_GeometryFlags.resize(m_SbView.numRecords, OPX7::OPX7GeometryFlagsNone);
				}
				void    SetTransformView(const TransformView&    transformView)noexcept {
					m_TbView = transformView;
				}
				void    SetPrimIdxOffset(unsigned int            primIdxOffset)noexcept {
					m_PrimIdxOffset = primIdxOffset;
				}
				void    SetNumSbtRecords(unsigned int            numSbtRecords)noexcept
				{
					m_SbView.numRecords = numSbtRecords;
					m_GeometryFlags.resize(numSbtRecords, OPX7::OPX7GeometryFlagsNone);
				}
				void    SetGeometryFlags(OPX7::OPX7GeometryFlags geometryFlags)noexcept {
					for (auto& flags : m_GeometryFlags) {
						geometryFlags = flags;
					}
				}
				void    SetGeometryFlags(size_t idx, OPX7::OPX7GeometryFlags geometryFlags)noexcept {
					m_GeometryFlags[idx] = geometryFlags;
				}
				//Getter
				auto    GetVertexView()const noexcept -> const VertexView& {
					return m_VbView;
				}
				auto    GetTriIdxView()const noexcept -> const TriIdxView& {
					return m_IbView;
				}
				auto    GetSbtOffsetView()const noexcept -> const SbtOffsetView& {
					return m_SbView;
				}
				auto    GetTransformView()const noexcept -> const TransformView& {
					return m_TbView;
				}
				auto    GetPrimIdxOffset()const noexcept -> unsigned int
				{
					return m_PrimIdxOffset;
				}
				auto    GetNumSbtRecords()const noexcept -> unsigned int { return m_GeometryFlags.size(); }
				auto    GetGeometryFlags()const noexcept -> const std::vector<OPX7::OPX7GeometryFlags>&
				{
					return m_GeometryFlags;
				}
			private:
				VertexView                m_VbView        = {};
				TriIdxView                m_IbView        = {};
				SbtOffsetView             m_SbView        = {};
				TransformView             m_TbView        = {};
				unsigned int              m_PrimIdxOffset = 0;
				std::vector<OPX7::OPX7GeometryFlags>  m_GeometryFlags = {};
			};
			class OPX7GeometrySphere :public OPX7Geometry {
			public:
				struct    VertexView
				{
					CUDA::CUDABufferView   view = {};
					unsigned int           stride = 0;
					OPX7::OPX7VertexFormat format = OPX7::OPX7VertexFormat::eNone;
					auto GetNumVertices()const noexcept -> unsigned int
					{
						if ((view.GetSizeInBytes() == 0) || (stride == 0)) {
							return 0;
						}
						else {
							return view.GetSizeInBytes() / stride;
						}
					}
				};
				struct SbtOffsetView {
					CUDA::CUDABufferView      view = {};
					unsigned int              stride = 0;
					OPX7::OPX7SbtOffsetFormat format = OPX7::OPX7SbtOffsetFormat::eNone;
					unsigned int              numRecords = 0;
					auto GetNumIndices()const noexcept -> unsigned int
					{
						if ((view.GetSizeInBytes() == 0) || (stride == 0)) {
							return 0;
						}
						else {
							return view.GetSizeInBytes() / stride;
						}
					}
				};
				struct    RadiusView 
				{
					CUDA::CUDABufferView   view         = {};
					unsigned int           stride       = 0;
					bool                   singleRadius = false;
					auto GetNumRadius()const noexcept -> unsigned int
					{
						if ((view.GetSizeInBytes() == 0) || (stride == 0)) {
							return 0;
						}
						else {
							return view.GetSizeInBytes() / stride;
						}
					}
				};
			public:
				virtual ~OPX7GeometrySphere()noexcept {}
				virtual auto GetOptixBuildInputWithHolder()const noexcept -> std::tuple<OptixBuildInput, std::unique_ptr<Holder>> override;
				//Setter
				void    SetVertexView(const    VertexView& vertexView)noexcept
				{
					m_VbView = vertexView;
				}
				void    SetRadiusView(const    RadiusView& radiusView)noexcept
				{
					m_RbView = radiusView;
				}
				void    SetSbtOffsetView(const SbtOffsetView& sbtOffsetView)noexcept {
					m_SbView = sbtOffsetView;
					m_GeometryFlags.resize(m_SbView.numRecords, OPX7::OPX7GeometryFlagsNone);
				}
				void    SetPrimIdxOffset(unsigned int            primIdxOffset)noexcept {
					m_PrimIdxOffset = primIdxOffset;
				}
				void    SetNumSbtRecords(unsigned int            numSbtRecords)noexcept
				{
					m_SbView.numRecords = numSbtRecords;
					m_GeometryFlags.resize(numSbtRecords, OPX7::OPX7GeometryFlagsNone);
				}
				void    SetGeometryFlags(OPX7::OPX7GeometryFlags geometryFlags)noexcept {
					for (auto& flags : m_GeometryFlags) {
						geometryFlags = flags;
					}
				}
				void    SetGeometryFlags(size_t idx, OPX7::OPX7GeometryFlags geometryFlags)noexcept {
					m_GeometryFlags[idx] = geometryFlags;
				}
				//Getter
				auto    GetVertexView()const noexcept -> const VertexView& {
					return m_VbView;
				}
				auto    GetRadiusView()const noexcept -> const RadiusView& {
					return m_RbView;
				}
				auto    GetSbtOffsetView()const noexcept -> const SbtOffsetView& {
					return m_SbView;
				}
				auto    GetPrimIdxOffset()const noexcept -> unsigned int
				{
					return m_PrimIdxOffset;
				}
				auto    GetNumSbtRecords()const noexcept -> unsigned int { return m_GeometryFlags.size(); }
				auto    GetGeometryFlags()const noexcept -> const std::vector<OPX7::OPX7GeometryFlags>&
				{
					return m_GeometryFlags;
				}
			private:
				VertexView                m_VbView = {};
				RadiusView                m_RbView = {};
				SbtOffsetView             m_SbView = {};
				unsigned int              m_PrimIdxOffset = 0;
				std::vector<OPX7::OPX7GeometryFlags>  m_GeometryFlags = {};
			};
			class OPX7GeometryCustom : public OPX7Geometry
			{
			public:
				struct      AabbView
				{
					CUDA::CUDABufferView   view   = {};
					unsigned int           stride = 0;
					auto GetNumPrimitives()const noexcept -> unsigned int
					{
						if ((view.GetSizeInBytes() == 0) || (stride == 0)) {
							return 0;
						}
						else {
							return view.GetSizeInBytes() / stride;
						}
					}
				};
				struct SbtOffsetView {
					CUDA::CUDABufferView      view = {};
					unsigned int              stride = 0;
					OPX7::OPX7SbtOffsetFormat format = OPX7::OPX7SbtOffsetFormat::eNone;
					unsigned int              numRecords = 0;
					auto GetNumIndices()const noexcept -> unsigned int
					{
						if ((view.GetSizeInBytes() == 0) || (stride == 0)) {
							return 0;
						}
						else {
							return view.GetSizeInBytes() / stride;
						}
					}
				};
				struct TransformView {
					CUDA::CUDABufferView      view = {};
					OPX7::OPX7TransformFormat format = OPX7::OPX7TransformFormat::eNone;
				};
			public:
				virtual ~OPX7GeometryCustom()noexcept {}
				virtual auto GetOptixBuildInputWithHolder()const noexcept -> std::tuple<OptixBuildInput, std::unique_ptr<Holder>> override;
				//Setter
				void    SetAabbView(const AabbView& aabbView)noexcept {
					m_AbView = aabbView;
				}
				void    SetSbtOffsetView(const SbtOffsetView& sbtOffsetView)noexcept {
					m_SbView = sbtOffsetView;
					m_GeometryFlags.resize(m_SbView.numRecords, OPX7::OPX7GeometryFlagsNone);
				}
				void    SetPrimIdxOffset(unsigned int            primIdxOffset)noexcept {
					m_PrimIdxOffset = primIdxOffset;
				}
				void    SetNumSbtRecords(unsigned int            numSbtRecords)noexcept
				{
					m_SbView.numRecords = numSbtRecords;
					m_GeometryFlags.resize(numSbtRecords, OPX7::OPX7GeometryFlagsNone);
				}
				void    SetGeometryFlags(OPX7::OPX7GeometryFlags geometryFlags)noexcept {
					for (auto& flags : m_GeometryFlags) {
						geometryFlags = flags;
					}
				}
				void    SetGeometryFlags(size_t idx, OPX7::OPX7GeometryFlags geometryFlags)noexcept {
					m_GeometryFlags[idx] = geometryFlags;
				}
				//Getter
				auto    GetAabbView()const noexcept -> AabbView {
					return m_AbView;
				}
				auto    GetSbtOffsetView()const noexcept -> const SbtOffsetView& {
					return m_SbView;
				}
				auto    GetPrimIdxOffset()const noexcept -> unsigned int
				{
					return m_PrimIdxOffset;
				}
				auto    GetNumSbtRecords()const noexcept -> unsigned int { return m_GeometryFlags.size(); }
				auto    GetGeometryFlags()const noexcept -> const std::vector<OPX7::OPX7GeometryFlags>&
				{
					return m_GeometryFlags;
				}
			private:
				AabbView                  m_AbView = {};
				SbtOffsetView             m_SbView = {};
				unsigned int              m_PrimIdxOffset = 0;
				std::vector<OPX7::OPX7GeometryFlags>  m_GeometryFlags = {};

			};
			
			class OPX7AccelerationStructure
			{
			public:
				virtual ~OPX7AccelerationStructure()noexcept{}
				
				virtual bool IsBuilt()const noexcept = 0;
				virtual void Build()  = 0;
				virtual void Update() = 0;

				virtual auto GetOptixTraversableHandle()const noexcept -> OptixTraversableHandle = 0;
				virtual auto GetCUDABuffer()const noexcept -> const CUDA::CUDABuffer* = 0;
				virtual auto GetCUDABuffer()      noexcept ->       CUDA::CUDABuffer* = 0;
			};
			class OPX7GeometryAccelerationStructure: public OPX7AccelerationStructure
			{
			public:
				virtual ~OPX7GeometryAccelerationStructure()noexcept {}

				virtual bool IsBuilt()const noexcept { return m_Buffer.get(); }
				virtual void Build()  override;
				virtual void Update() override;

				virtual auto GetOptixTraversableHandle()const noexcept -> OptixTraversableHandle override
				{
					return m_Handle;
				}
				virtual auto GetCUDABuffer()const noexcept -> const CUDA::CUDABuffer* {
					return m_Buffer.get();
				}
				virtual auto GetCUDABuffer()      noexcept ->       CUDA::CUDABuffer* {
					return m_Buffer.get();
				}

			private:
				OptixTraversableHandle                     m_Handle     = 0;
				std::unique_ptr<CUDA::CUDABuffer>          m_Buffer     = nullptr;
			};
			class OPX7InstanceAccelerationStructure: public OPX7AccelerationStructure
			{
			public:
				virtual ~OPX7InstanceAccelerationStructure()noexcept {}

				virtual auto GetOptixTraversableHandle()const noexcept -> OptixTraversableHandle override
				{
					return m_Handle;
				}
				virtual auto GetCUDABuffer()const noexcept -> const CUDA::CUDABuffer* {
					return m_Buffer.get();
				}
				virtual auto GetCUDABuffer()      noexcept ->       CUDA::CUDABuffer* {
					return m_Buffer.get();
				}
			private:
				OptixTraversableHandle                     m_Handle = 0;
				std::unique_ptr<CUDA::CUDABuffer>          m_Buffer = nullptr;
			};


		}
	}
}
#endif