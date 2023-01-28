#ifndef RTLIB_EXT_GL_GL_UNIQUE_IDENTIFIER_H
#define RTLIB_EXT_GL_GL_UNIQUE_IDENTIFIER_H
#include <atomic>
#include <cstdint>
namespace RTLib
{
    namespace Ext
    {
        namespace GL
        {
            using GLUniqueId = int32_t;
            class GLUniqueIdHolder {
            public:
                GLUniqueIdHolder() noexcept : m_UniqueIdentifier {0}{}
                ~GLUniqueIdHolder() noexcept{}
                
                GLUniqueIdHolder(const GLUniqueIdHolder &id) = delete;
                auto operator=(const GLUniqueIdHolder &id) -> GLUniqueIdHolder & = delete;

                GLUniqueIdHolder( GLUniqueIdHolder &&id)noexcept {
                    m_UniqueIdentifier = id.m_UniqueIdentifier;
                    id.m_UniqueIdentifier = 0;
                }
                auto operator=( GLUniqueIdHolder &&id) noexcept-> GLUniqueIdHolder &
                {
                    if (this!=&id)
                    {
                        m_UniqueIdentifier = id.m_UniqueIdentifier;
                        id.m_UniqueIdentifier = 0;
                    }
                    return *this;
                }

                explicit operator GLUniqueId() const noexcept { return GetID(); }
                auto GetID()const noexcept -> GLUniqueId
                {
                    if (m_UniqueIdentifier == 0){
                        static std::atomic<int32_t> counter{1};
                        m_UniqueIdentifier = counter.fetch_add(1);
                    }
                    return m_UniqueIdentifier;
                }
            private:
                mutable GLUniqueId m_UniqueIdentifier;
            };
        }
    }
}
#endif

