#include <RTLib/Ext/GL/GLContext.h>
#include <glad/glad.h>
#include <unordered_map>
#include <unordered_set>
#include <optional>
namespace RTLib {
	namespace Ext {
		namespace GL {
			namespace Internal {
				class GLResource {
				public:
					auto GetResID()const noexcept -> GLuint { return m_ResID; }
				private:
					GLuint m_ResID;
				};
				class GLBindable : public GLResource {
				public:

				private:
					std::optional<GLenum> m_Target;
				};
			}
		}
	}
}