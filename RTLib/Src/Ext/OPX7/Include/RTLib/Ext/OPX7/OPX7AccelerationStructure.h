#ifndef RTLIB_EXT_OPX7_OPX7_ACCELERATION_STRUCTURE_H
#define RTLIB_EXT_OPX7_OPX7_ACCELERATION_STRUCTURE_H
#include <RTLib/Core/BaseObject.h>
#include <RTLib/Ext/OPX7/OPX7Common.h>
#include <RTLib/Ext/OPX7/OPX7AccelerationStructureBuildInput.h>
#include <RTLib/Ext/OPX7/UuidDefinitions.h>
#include <memory>
namespace RTLib {
	namespace Ext
	{
		namespace OPX7
		{
			class OPX7Context;
			class OPX7AccelerationStructure : public RTLib::Core::BaseObject
			{
			private:
				RTLIB_CORE_TYPE_OBJECT_DECLARE_DERIVED_METHOD(OPX7AccelerationStructure, Core::BaseObject, RTLIB_TYPE_UUID_RTLIB_EXT_OPX7_OPX7_ACCELERATION_STRUCTURE);
			public:
				static auto New(OPX7Context* context)->OPX7AccelerationStructure*;
				virtual ~OPX7AccelerationStructure()noexcept;
				virtual void Destroy();
			private:
				OPX7AccelerationStructure()noexcept;
				auto GetOptixTraversableHandle()const noexcept ->OptixTraversableHandle;
			private:
				struct Impl;
				std::unique_ptr<Impl> m_Impl;
			};
		}
	}
}
#endif