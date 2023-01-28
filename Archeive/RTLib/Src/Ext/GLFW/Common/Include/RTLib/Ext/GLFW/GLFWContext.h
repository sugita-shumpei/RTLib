#ifndef RTLIB_EXT_GLFW_GLFW_CONTEXT_H
#define RTLIB_EXT_GLFW_GLFW_CONTEXT_H
#include <RTLib/Core/Context.h>
#include <RTLib/Core/BaseObject.h>
#include <RTLib/Ext/GLFW/UuidDefinitions.h>
namespace RTLib
{
	namespace Ext
	{
		namespace GLFW
		{
			class GLFWContext : public Core::Context
			{
			public:
				RTLIB_CORE_TYPE_OBJECT_DECLARE_DERIVED_METHOD(Ext::GLFW::GLFWContext, Core::Context, RTLIB_TYPE_UUID_RTLIB_EXT_GLFW_GLFW_CONTEXT);

				static auto New()->GLFWContext*;
				virtual ~GLFWContext()noexcept;

				virtual bool Initialize()override;
				virtual void Terminate() override;

				void Update();
			private:
				GLFWContext()noexcept;
			private:
				struct Impl;
				std::unique_ptr<Impl> m_Impl;
			};
		}
	}
}
#endif
