#ifndef RTLIB_EXT_OPX7_OPX7_PROGRAM_GROUP_H
#define RTLIB_EXT_OPX7_OPX7_PROGRAM_GROUP_H
#include <RTLib/Core/BaseObject.h>
#include <RTLib/Ext/OPX7/OPX7Common.h>
#include <RTLib/Ext/OPX7/UuidDefinitions.h>
#include <vector>
#include <memory>
namespace RTLib {
	namespace Ext
	{
		namespace OPX7
		{
			class OPX7Context;
			class OPX7Pipeline;
			class OPX7ProgramGroup : public RTLib::Core::BaseObject
			{
				friend class OPX7Pipeline;
				RTLIB_CORE_TYPE_OBJECT_DECLARE_DERIVED_METHOD(OPX7ProgramGroup, Core::BaseObject, RTLIB_TYPE_UUID_RTLIB_EXT_OPX7_OPX7_PROGRAM_GROUP);
			public:
				static auto Enumerate(OPX7Context* context, const std::vector<OPX7ProgramGroupCreateDesc>& descs, const OPX7ProgramGroupOptions& options)->std::vector<OPX7ProgramGroup*>;
				virtual ~OPX7ProgramGroup()noexcept;
				virtual void Destroy();

				auto GetKind()const noexcept -> OPX7ProgramGroupKind;
			private:
				OPX7ProgramGroup()noexcept;
				auto GetOptixProgramGroup()const noexcept -> OptixProgramGroup;
			private:
				struct Impl;
				std::unique_ptr<Impl> m_Impl;
			};
		}
	}
}
#endif
