#ifndef RTLIB_EXT_OPX7_OPX7_PIPELINE_H
#define RTLIB_EXT_OPX7_OPX7_PIPELINE_H
#include <RTLib/Core/BaseObject.h>
#include <RTLib/Ext/OPX7/OPX7Common.h>
#include <RTLib/Ext/OPX7/UuidDefinitions.h>
#include <memory>
namespace RTLib {
	namespace Ext
	{
		namespace OPX7
		{
			class OPX7Context;
			class OPX7ProgramGroup;
			class OPX7Pipeline : public RTLib::Core::BaseObject
			{
			private:
				friend class OPX7ProgramGroup;
				RTLIB_CORE_TYPE_OBJECT_DECLARE_DERIVED_METHOD(OPX7Pipeline, Core::BaseObject, RTLIB_TYPE_UUID_RTLIB_EXT_OPX7_OPX7_PIPELINE);
			public:
				static auto New(OPX7Context* context, const OPX7PipelineCreateDesc& desc)->OPX7Pipeline*;
				virtual ~OPX7Pipeline()noexcept;
				virtual void Destroy();

				auto GetCompileOptions()const noexcept -> const OPX7PipelineCompileOptions&;
				auto GetLinkOptions()   const noexcept -> const OPX7PipelineLinkOptions&;
			private:
				OPX7Pipeline(const OPX7PipelineCreateDesc& desc)noexcept;
				auto GetOptixPipeline()const noexcept ->OptixPipeline;
			private:
				struct Impl;
				std::unique_ptr<Impl> m_Impl;
			};
		}
	}
}
#endif
