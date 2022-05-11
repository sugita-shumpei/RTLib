#ifndef RTLIB_EXT_GL_INTERNAL_IMPL_GL_BINDABLE_H
#define RTLIB_EXT_GL_INTERNAL_IMPL_GL_BINDABLE_H
#include "ImplGLResource.h"
#include <optional>
#include <stdexcept>
#include <unordered_map>
#include <iostream>
namespace RTLib {
	namespace Ext {
		namespace GL {
			namespace Internal {
				class ImplGLBindable;
				class ImplGLBindingPoint {
				public:
					virtual ~ImplGLBindingPoint()noexcept {}
					void   AddTarget(GLenum target) noexcept {
						if (!HasTarget(target)) {
							m_Bindables[target] = nullptr;
						}
					}
					bool   HasTarget(GLenum target)const noexcept {
						return m_Bindables.count(target) > 0;
					}
					bool  IsBindable(GLenum target)const noexcept {
						if (!HasTarget(target)) {
							return false;
						}
						return m_Bindables.at(target) == nullptr;
					}
					auto GetBindable(GLenum target)->ImplGLBindable*;
					bool    Register(GLenum target, ImplGLBindable* bindable);
					bool  Unregister(GLenum target);
				private:
					std::unordered_map<GLenum, ImplGLBindable*> m_Bindables = {};
				};
				class ImplGLBindableBase: public ImplGLResourceBase {
				public:
					friend class ImplGLBindable;
				public:
					virtual ~ImplGLBindableBase()noexcept {}
				protected:
					virtual bool      Create()noexcept = 0;
					virtual void     Destroy()noexcept = 0;
					virtual void   Bind(GLenum target) = 0;
					virtual void Unbind(GLenum target) = 0;
				};
				class ImplGLBindable : public ImplGLResource {
				public:
					virtual ~ImplGLBindable()noexcept {
						Unbind();
						m_BPoint = nullptr;
					}
					void   Unbind()noexcept {
						auto base = GetBase();
						if (!IsBinded() || !base) {
							return;
						}
						bool res = m_BPoint->Unregister(*m_Target);
						if (res) {
							static_cast<ImplGLBindableBase*>(base)->Unbind(*m_Target);
							m_Target = std::nullopt;
						}
					}
					bool IsBinded()const noexcept { return m_Target != std::nullopt; }
				protected:
					bool Bind(GLenum target) {
						auto base = GetBase();
						if (!m_BPoint || !base) {
							return false;
						}
						if (IsBinded()) {
							return m_Target == target;
						}
						bool res = m_BPoint->Register(target, this);
						if ( res) {
							static_cast<ImplGLBindableBase*>(base)->Bind(target);
							m_Target = target;
						}
						return res;
					}
					bool IsBindable(GLenum target)const noexcept {
						if (!m_BPoint || !GetBase()) {
							return false;
						}
						return m_BPoint->IsBindable(target);
					}
				protected:
					ImplGLBindable(ImplGLResourceTable* table, ImplGLBindingPoint* bPoint)noexcept :ImplGLResource(table),m_BPoint{bPoint}{}
					template<typename BindableBase, typename ...Args, bool Cond = std::is_base_of_v<ImplGLBindable, BindableBase>>
					void InitBase(Args&&... args) {
						ImplGLResource::InitBase<BindableBase>(std::forward<Args>(args)...);
					}
					auto GetBindedTarget()const noexcept -> std::optional<GLenum> {
						return m_Target;
					}
					auto GetBindingPoint()const noexcept -> const ImplGLBindingPoint* {
						return m_BPoint;
					}					
					auto GetBindingPoint()      noexcept ->       ImplGLBindingPoint* {
						return m_BPoint;
					}
				private:
					ImplGLBindingPoint*       m_BPoint = nullptr;
					std::optional<GLenum>     m_Target = std::nullopt;
				};
			}
		}
	}
}
#endif
