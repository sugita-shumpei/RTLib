#ifndef RTLIB_SCENE_OBJECT_TYPE_ID__H
#define RTLIB_SCENE_OBJECT_TYPE_ID__H

#ifndef __CUDACC__
#include <RTLib/Core/ObjectTypeID.h>

#define RTLIB_SCENE_DEFINE_OBJECT_TYPE_ID(TYPE, TYPE_ID_STR) \
namespace Scene { struct TYPE; } \
inline namespace Core { RTLIB_CORE_DEFINE_OBJECT_TYPE_ID_2(Scene::TYPE,Scene##TYPE,TYPE_ID_STR); }

#endif
#endif
