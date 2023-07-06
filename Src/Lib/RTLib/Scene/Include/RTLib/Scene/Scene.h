#ifndef RTLIB_SCENE_SCENE__H
#define RTLIB_SCENE_SCENE__H
#include <RTLib/Scene/TransformGraph.h>
namespace RTLib {
	namespace Scene
	{
		struct Scene
		{

		private:
			RTLib::Scene::TransformGraph m_Graph;
		};
	}
}
#endif
