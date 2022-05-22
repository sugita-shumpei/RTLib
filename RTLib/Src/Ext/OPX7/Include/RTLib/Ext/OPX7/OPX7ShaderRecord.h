#ifndef RTLIB_EXT_OPX7_OPX7_SHADER_RECORD_H
#define RTLIB_EXT_OPX7_OPX7_SHADER_RECORD_H
#include <optix.h>
namespace RTLib {
	namespace Ext
	{
		namespace OPX7
		{
            template<typename T>
			struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) OPX7ShaderRecord
            {
				char header[OPTIX_SBT_RECORD_HEADER_SIZE];
				T    data;
            };
		}
	}
}
#endif
