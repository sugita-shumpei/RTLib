#ifndef RTLIB_EXT_GL_INTERNAL_IMPL_GL_RESOURCE_H
#define RTLIB_EXT_GL_INTERNAL_IMPL_GL_RESOURCE_H
#include <glad/glad.h>
#include <vector>
#include <cstdint>
#include <memory>
#include <unordered_set>
#include <iostream>
#include <stdexcept>
namespace RTLib {
	namespace Ext {
		namespace GL {
			namespace Internal {
				class ImplGLResource;
				class ImplGLResourceTable {
				public:
					friend class ImplGLResource;
				public:
					~ImplGLResourceTable()noexcept;
				private:
					bool   Register(ImplGLResource* resource);
					bool Unregister(ImplGLResource* resource);
				private:
					std::unordered_set<ImplGLResource*> m_Resources = {};
				};
				class ImplGLResourceBase {
				public:
					friend class ImplGLResource;
				public:
					virtual ~ImplGLResourceBase()noexcept {}
				protected:
					virtual bool  Create()noexcept = 0;
					virtual void Destroy()noexcept = 0;
				protected:
					auto GetResId()const noexcept -> GLuint { return m_ResId; }
					void SetResId(GLuint resId)noexcept { m_ResId = resId; }
				private:
					GLuint m_ResId = 0;
				};
				class ImplGLResource {
				public:
					friend class ImplGLResourceTable;
					friend class ImplGLResource;
				public:
					virtual ~ImplGLResource()noexcept;
				protected:
					bool Create();
					void Destroy()noexcept;
				protected:
					ImplGLResource(ImplGLResourceTable* table)noexcept : m_Table{ table }, m_Base{nullptr}{}
					template<typename ResourceBase, typename ...Args, bool Cond = std::is_base_of_v<ImplGLResourceBase,ResourceBase>>
					void  InitBase(Args&&... args) {
						m_Base = std::unique_ptr<ImplGLResourceBase>(new ResourceBase(std::forward<Args>(args)...));
					}
					void ResetBase()noexcept{ m_Base.reset(); }
					auto   GetResId()const noexcept -> GLuint { return m_Base ? m_Base->GetResId(): 0; }
					auto   GetBase()const noexcept -> const ImplGLResourceBase* { return m_Base.get(); }
					auto   GetBase()      noexcept ->       ImplGLResourceBase* { return m_Base.get(); }
				private:
					ImplGLResourceTable*                m_Table = nullptr;
					std::unique_ptr<ImplGLResourceBase> m_Base  = nullptr;
				};

			}
		}
	}
}
#endif