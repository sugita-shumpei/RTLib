#ifndef RTLIB_EXT_OPX7_OPX7_MODULE_H
#define RTLIB_EXT_OPX7_OPX7_MODULE_H
#include <RTLib/Core/BaseObject.h>
#include <RTLib/Ext/OPX7/OPX7Common.h>
#include <RTLib/Ext/OPX7/UuidDefinitions.h>
#include <memory>
namespace RTLib {
	namespace Ext
	{
		namespace OPX7
		{
			class OPX7Natives;
			class OPX7Context;
			class OPX7ProgramGroup;
			class OPX7Module: public RTLib::Core::BaseObject
			{
			private:
				friend class OPX7ProgramGroup;
				friend class OPX7Natives;
				RTLIB_CORE_TYPE_OBJECT_DECLARE_DERIVED_METHOD(OPX7Module, Core::BaseObject, RTLIB_TYPE_UUID_RTLIB_EXT_OPX7_OPX7_MODULE);
			public:
				static auto BuiltInTriangleIS(OPX7Context* context, const OPX7ModuleCreateDesc& desc, bool useMotionBlur = false)->OPX7Module*;
				static auto BuiltInSphereIS(  OPX7Context* context, const OPX7ModuleCreateDesc& desc, bool useMotionBlur = false)->OPX7Module*;
				static auto New(OPX7Context* context, const OPX7ModuleCreateDesc& desc)->OPX7Module*;
				virtual ~OPX7Module()noexcept;
				virtual void Destroy() ;

				auto GetModuleCompileOptions()const noexcept -> const OPX7ModuleCompileOptions&;
				auto GetPipelineCompileOptions()const noexcept -> const OPX7PipelineCompileOptions&;
			private:
				OPX7Module(const OPX7ModuleCreateDesc& desc)noexcept;
				auto GetOptixModule()const noexcept ->OptixModule;
			private:
				struct Impl;
				std::unique_ptr<Impl> m_Impl;
			};
		}
	}
}
#endif
