#ifndef RTLIB_CORE_COMMON_H
#define RTLIB_CORE_COMMON_H
#include <typeinfo>
#include <cstdint>
#include <typeinfo>
#include <cassert>
#define RTLIB_DEBUG_ASSERT_IF_FAILED(EXEC) \
do { \
    auto rtlib_debug_check_tmp_result = EXEC; \
    assert(rtlib_debug_check_tmp_result); \
} while(0)
namespace RTLib {
    namespace Core {

    }
}
#endif