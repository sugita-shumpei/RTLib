#ifndef RTLIB_EXT_OPX7_OPX7_ACCELERATION_STRUCTURE_INSTANCE_H
#define RTLIB_EXT_OPX7_OPX7_ACCELERATION_STRUCTURE_INSTANCE_H
#include <RTLib/Core/BaseObject.h>
#include <RTLib/Ext/CUDA/CUDABuffer.h>
#include <RTLib/Ext/OPX7/OPX7Common.h>
#include <RTLib/Ext/OPX7/OPX7AccelerationStructure.h>
#include <RTLib/Ext/OPX7/UuidDefinitions.h>
#include <memory>
namespace RTLib {
	namespace Ext
	{
		namespace OPX7
		{
			class OPX7Instance
			{
				using AccelerationStructure = OPX7AccelerationStructure;
			private:
				AccelerationStructure* m_BaseAS;
				float                  m_Transforms[12];
				unsigned int           m_SbtOffset;
				unsigned int           m_VisibilityMask;
				unsigned int           m_Flags;
			};
		}
	}
}
#endif
