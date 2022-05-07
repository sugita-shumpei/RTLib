#ifndef RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_H
#define RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_H
#include <glad/glad.h>
#include <half.h>
#include <numeric>
#define RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_GL_FORMAT_INFO_DEF_RED(FORMAT, BASE_TYPE, NUM_BASES) \
    template <>                                                                                      \
    struct GLFormatInfo<GL_##FORMAT>                                                                   \
    {                                                                                                \
        static inline constexpr GLFormatTypeInfo formats[] = {\
            { GL_##FORMAT,GL_RED, GL_##BASE_TYPE, NUM_BASES}\
        }; \
        static inline constexpr GLenum base_format = formats[0].base_format; \
        static inline constexpr GLsizei channels = GLBaseFormatInfo<base_format>::channels; \
        static inline constexpr GLsizei size = sizeof(GLTypeInfo<formats[0].base_type>::type) * formats[0].num_bases; \
    }

#define RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_GL_FORMAT_INFO_DEF_INT_RED(FORMAT, BASE_TYPE, NUM_BASES) \
    template <>                                                                                      \
    struct GLFormatInfo<GL_##FORMAT>                                                                   \
    {                                                                                                \
        static inline constexpr GLFormatTypeInfo formats[] = {\
            { GL_##FORMAT,GL_RED_INTEGER, GL_##BASE_TYPE, NUM_BASES}\
        }; \
        static inline constexpr GLenum base_format = formats[0].base_format; \
        static inline constexpr GLsizei channels = GLBaseFormatInfo<base_format>::channels; \
        static inline constexpr GLsizei size = sizeof(GLTypeInfo<formats[0].base_type>::type) * formats[0].num_bases; \
    }

#define RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_GL_FORMAT_INFO_DEF_RG(FORMAT, BASE_TYPE, NUM_BASES) \
    template <>                                                                                      \
    struct GLFormatInfo<GL_##FORMAT>                                                                   \
    {                                                                                                \
        static inline constexpr GLFormatTypeInfo formats[] = {\
            {GL_##FORMAT, GL_RG, GL_##BASE_TYPE, NUM_BASES}\
        }; \
        static inline constexpr GLenum base_format = formats[0].base_format; \
        static inline constexpr GLsizei channels = GLBaseFormatInfo<base_format>::channels; \
        static inline constexpr GLsizei size = sizeof(GLTypeInfo<formats[0].base_type>::type) * formats[0].num_bases; \
    }

#define RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_GL_FORMAT_INFO_DEF_INT_RG(FORMAT, BASE_TYPE, NUM_BASES) \
    template <>                                                                                      \
    struct GLFormatInfo<GL_##FORMAT>                                                                   \
    {                                                                                                \
        static inline constexpr GLFormatTypeInfo formats[] = {\
            {GL_##FORMAT, GL_RG_INTEGER, GL_##BASE_TYPE, NUM_BASES}\
        }; \
        static inline constexpr GLenum base_format = formats[0].base_format; \
        static inline constexpr GLsizei channels = GLBaseFormatInfo<base_format>::channels; \
        static inline constexpr GLsizei size = sizeof(GLTypeInfo<formats[0].base_type>::type) * formats[0].num_bases; \
    }

#define RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_GL_FORMAT_INFO_DEF_RGB(FORMAT, BASE_TYPE, NUM_BASES) \
    template <>                                                                                      \
    struct GLFormatInfo<GL_##FORMAT>                                                                   \
    {                                                                                                \
        static inline constexpr GLFormatTypeInfo formats[] = {\
            {GL_##FORMAT, GL_RGB, GL_##BASE_TYPE, NUM_BASES}, \
            {GL_##FORMAT, GL_BGR, GL_##BASE_TYPE, NUM_BASES}  \
        }; \
        static inline constexpr GLenum base_format = formats[0].base_format; \
        static inline constexpr GLsizei channels = GLBaseFormatInfo<base_format>::channels; \
        static inline constexpr GLsizei size = sizeof(GLTypeInfo<formats[0].base_type>::type) * formats[0].num_bases; \
    }

#define RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_GL_FORMAT_INFO_DEF_INT_RGB(FORMAT, BASE_TYPE, NUM_BASES) \
    template <>                                                                                      \
    struct GLFormatInfo<GL_##FORMAT>                                                                   \
    {                                                                                                \
        static inline constexpr GLFormatTypeInfo formats[] = {\
            {GL_##FORMAT, GL_RGB_INTEGER, GL_##BASE_TYPE, NUM_BASES}, \
            {GL_##FORMAT, GL_BGR_INTEGER, GL_##BASE_TYPE, NUM_BASES}  \
        }; \
        static inline constexpr GLenum base_format = formats[0].base_format; \
        static inline constexpr GLsizei channels = GLBaseFormatInfo<base_format>::channels; \
        static inline constexpr GLsizei size = sizeof(GLTypeInfo<formats[0].base_type>::type) * formats[0].num_bases; \
    }

#define RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_GL_FORMAT_INFO_DEF_RGBA(FORMAT, BASE_TYPE, NUM_BASES) \
    template <>                                                                                      \
    struct GLFormatInfo<GL_##FORMAT>                                                                   \
    {                                                                                                \
        static inline constexpr GLFormatTypeInfo formats[] = {\
            {GL_##FORMAT,GL_RGBA, GL_##BASE_TYPE, NUM_BASES}, \
            {GL_##FORMAT,GL_BGRA, GL_##BASE_TYPE, NUM_BASES}  \
        }; \
        static inline constexpr GLenum base_format = formats[0].base_format; \
        static inline constexpr GLsizei channels = GLBaseFormatInfo<base_format>::channels; \
        static inline constexpr GLsizei size = sizeof(GLTypeInfo<formats[0].base_type>::type) * formats[0].num_bases; \
    }

#define RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_GL_FORMAT_INFO_DEF_INT_RGBA(FORMAT, BASE_TYPE, NUM_BASES) \
    template <>                                                                                      \
    struct GLFormatInfo<GL_##FORMAT>                                                                   \
    {                                                                                                \
        static inline constexpr GLFormatTypeInfo formats[] = {\
            {GL_##FORMAT,GL_RGBA_INTEGER, GL_##BASE_TYPE, NUM_BASES}, \
            {GL_##FORMAT,GL_BGRA_INTEGER, GL_##BASE_TYPE, NUM_BASES}  \
        }; \
        static inline constexpr GLenum base_format = formats[0].base_format; \
        static inline constexpr GLsizei channels = GLBaseFormatInfo<base_format>::channels; \
        static inline constexpr GLsizei size = sizeof(GLTypeInfo<formats[0].base_type>::type) * formats[0].num_bases; \
    }

#define RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_GL_FORMAT_INFO_DEF_DEPTH(FORMAT, BASE_TYPE, NUM_BASES) \
    template <>                                                                                      \
    struct GLFormatInfo<GL_##FORMAT>                                                                   \
    {                                                                                                \
        static inline constexpr GLFormatTypeInfo formats[] = {\
            {GL_##FORMAT,GL_DEPTH_COMPONENT, GL_##BASE_TYPE, NUM_BASES}\
        }; \
        static inline constexpr GLenum base_format = formats[0].base_format; \
        static inline constexpr GLsizei channels = GLBaseFormatInfo<base_format>::channels; \
        static inline constexpr GLsizei size = sizeof(GLTypeInfo<formats[0].base_type>::type) * formats[0].num_bases; \
    }

#define RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_GL_FORMAT_INFO_DEF_DEPTH_STENCIL(FORMAT, BASE_TYPE, NUM_BASES) \
    template <>                                                                                      \
    struct GLFormatInfo<GL_##FORMAT>                                                                   \
    {                                                                                                \
        static inline constexpr GLFormatTypeInfo formats[] = {\
            {GL_##FORMAT,GL_DEPTH_STENCIL, GL_##BASE_TYPE, NUM_BASES}\
        }; \
        static inline constexpr GLenum base_format = formats[0].base_format; \
        static inline constexpr GLsizei channels = GLBaseFormatInfo<base_format>::channels; \
        static inline constexpr GLsizei size = sizeof(GLTypeInfo<formats[0].base_type>::type) * formats[0].num_bases; \
    }

#define RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_SIZE(FORMAT) \
    case GL_##FORMAT:                                                         \
        return GLFormatInfo<GL_##FORMAT>::size

#define RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_TYPE_INFO(FORMAT, BASE_TYPE, CNT) \
    case GL_##FORMAT: \
        for (CNT = 0; CNT<sizeof(GLFormatInfo<GL_##FORMAT>::formats)/sizeof(GLFormatInfo<GL_##FORMAT>::formats[0]); ++CNT) { \
            if (GLFormatInfo<GL_##FORMAT>::formats[CNT].base_type == BASE_TYPE){ \
                return GLFormatInfo<GL_##FORMAT>::formats[CNT]; \
            } \
        } \
        return {}

#define RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_FORMAT(FORMAT) \
    case GL_##FORMAT:                                                         \
        return GLFormatInfo<GL_##FORMAT>::formats[0].base_format

#define RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_TYPE(FORMAT)   \
    case GL_##FORMAT:                                                         \
        return GLFormatInfo<GL_##FORMAT>::formats[0].base_type

#define RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_TYPE_SIZE(TYPE) \
    case GL_##TYPE:                                                  \
        return sizeof(GLTypeInfo<GL_##TYPE>::type)

namespace RTLib
{
    namespace Ext
    {
        namespace GL
        {

            namespace Internal
            {
                template <typename UnsignedType,
                          size_t d_bit_count, size_t s_bit_count,
                          size_t d_bit_offset, size_t s_bit_offset>
                struct DepthStencilType
                {
                public:
                    constexpr DepthStencilType() noexcept : m_val{0} {}
                    constexpr DepthStencilType(UnsignedType d, UnsignedType s) noexcept : m_val{pack(d, s)} {}

                    constexpr DepthStencilType(const DepthStencilType &val) noexcept : m_val(val.m_val) {}
                    constexpr DepthStencilType &operator=(const DepthStencilType &val) noexcept
                    {
                        m_val = val.m_val;
                        return *this;
                    }

                    explicit constexpr operator UnsignedType() const noexcept
                    {
                        return m_val;
                    }
                    // R
                    constexpr auto GetDepth() const noexcept -> UnsignedType
                    {
                        UnsignedType d = static_cast<UnsignedType>(m_val & static_cast<UnsignedType>(mask_d)) >> offset_d;
                        return (d <= static_cast<UnsignedType>(max_d)) ? d : static_cast<UnsignedType>(max_d);
                    }
                    constexpr void SetDepth(UnsignedType depth) noexcept
                    {
                        UnsignedType td = (depth <= static_cast<UnsignedType>(max_d)) ? depth : static_cast<UnsignedType>(max_d);
                        m_val &= static_cast<UnsignedType>(mask_d);
                        m_val |= static_cast<UnsignedType>(td << offset_d);
                    }
                    // G
                    constexpr auto GetStencil() const noexcept -> UnsignedType
                    {
                        UnsignedType s = static_cast<UnsignedType>(m_val & static_cast<UnsignedType>(mask_s)) >> offset_s;
                        return (s <= static_cast<UnsignedType>(max_s)) ? s : static_cast<UnsignedType>(max_s);
                    }
                    constexpr void SetStencil(UnsignedType stencil) noexcept
                    {
                        UnsignedType ts = (stencil <= static_cast<UnsignedType>(max_s)) ? stencil : static_cast<UnsignedType>(max_s);
                        m_val &= static_cast<UnsignedType>(mask_s);
                        m_val |= static_cast<UnsignedType>(ts << offset_s);
                    }

                private:
                    static constexpr auto pack(UnsignedType d, UnsignedType s) noexcept -> UnsignedType
                    {
                        // 3
                        UnsignedType td = (d <= static_cast<UnsignedType>(max_d)) ? d : static_cast<UnsignedType>(max_d);
                        UnsignedType ts = (s <= static_cast<UnsignedType>(max_s)) ? s : static_cast<UnsignedType>(max_s);
                        return static_cast<UnsignedType>(
                            static_cast<UnsignedType>(static_cast<UnsignedType>(td) << offset_d) +
                            static_cast<UnsignedType>(static_cast<UnsignedType>(ts) << offset_s));
                    }

                private:
                    static constexpr inline size_t offset_d = d_bit_offset;
                    static constexpr inline size_t offset_s = s_bit_offset;
                    static constexpr inline UnsignedType max_d = static_cast<UnsignedType>(static_cast<UnsignedType>(1) << d_bit_count) - static_cast<UnsignedType>(1);
                    static constexpr inline UnsignedType max_s = static_cast<UnsignedType>(static_cast<UnsignedType>(1) << s_bit_count) - static_cast<UnsignedType>(1);
                    static constexpr inline UnsignedType mask_d = static_cast<UnsignedType>(static_cast<UnsignedType>(max_d) << offset_d);
                    static constexpr inline UnsignedType mask_s = static_cast<UnsignedType>(static_cast<UnsignedType>(max_s) << offset_s);

                private:
                    UnsignedType m_val;
                };

                template <typename UnsignedType,
                          size_t r_bit_count, size_t g_bit_count, size_t b_bit_count,
                          size_t r_bit_offset, size_t g_bit_offset, size_t b_bit_offset>
                struct RGBType
                {
                public:
                    constexpr RGBType() noexcept : m_val{0} {}
                    constexpr RGBType(UnsignedType r, UnsignedType g, UnsignedType b) noexcept : m_val{pack(r, g, b)} {}

                    constexpr RGBType(const RGBType &val) noexcept : m_val(val.m_val) {}
                    constexpr RGBType &operator=(const RGBType &val) noexcept
                    {
                        m_val = val.m_val;
                        return *this;
                    }

                    explicit constexpr operator UnsignedType() const noexcept
                    {
                        return m_val;
                    }
                    // R
                    constexpr auto GetR() const noexcept -> UnsignedType
                    {
                        UnsignedType r = static_cast<UnsignedType>(static_cast<UnsignedType>(m_val & static_cast<UnsignedType>(mask_r)) >> offset_r);
                        return (r <= static_cast<UnsignedType>(max_r)) ? r : static_cast<UnsignedType>(max_r);
                    }
                    constexpr void SetR(UnsignedType r) noexcept
                    {
                        UnsignedType tr = (r <= static_cast<UnsignedType>(max_r)) ? r : static_cast<UnsignedType>(max_r);
                        m_val &= static_cast<UnsignedType>(mask_r);
                        m_val |= static_cast<UnsignedType>(tr << offset_r);
                    }
                    // G
                    constexpr auto GetG() const noexcept -> UnsignedType
                    {
                        UnsignedType g = static_cast<UnsignedType>(m_val & static_cast<UnsignedType>(mask_g)) >> offset_g;
                        return (g <= static_cast<UnsignedType>(max_g)) ? g : static_cast<UnsignedType>(max_g);
                    }
                    constexpr void SetG(UnsignedType g) noexcept
                    {
                        UnsignedType tg = (g <= static_cast<UnsignedType>(max_g)) ? g : static_cast<UnsignedType>(max_g);
                        m_val &= static_cast<UnsignedType>(mask_g);
                        m_val |= static_cast<UnsignedType>(tg << offset_g);
                    }
                    // B
                    constexpr auto GetB() const noexcept -> UnsignedType
                    {
                        UnsignedType b = static_cast<UnsignedType>(m_val & static_cast<UnsignedType>(mask_b)) >> offset_b;
                        return (b <= static_cast<UnsignedType>(max_b)) ? b : static_cast<UnsignedType>(max_b);
                    }
                    constexpr void SetB(UnsignedType b) noexcept
                    {
                        UnsignedType tb = (b <= static_cast<UnsignedType>(max_b)) ? b : static_cast<UnsignedType>(max_b);
                        m_val &= static_cast<UnsignedType>(mask_b);
                        m_val |= static_cast<UnsignedType>(tb << offset_b);
                    }

                public:
                    static constexpr auto pack(UnsignedType r, UnsignedType g, UnsignedType b) noexcept -> UnsignedType
                    {
                        // 3
                        UnsignedType tr = (r <= static_cast<UnsignedType>(max_r)) ? r : static_cast<UnsignedType>(max_r);
                        UnsignedType tg = (g <= static_cast<UnsignedType>(max_g)) ? g : static_cast<UnsignedType>(max_g);
                        UnsignedType tb = (b <= static_cast<UnsignedType>(max_b)) ? b : static_cast<UnsignedType>(max_b);
                        return static_cast<UnsignedType>(
                            static_cast<UnsignedType>(static_cast<UnsignedType>(tr) << offset_r) +
                            static_cast<UnsignedType>(static_cast<UnsignedType>(tg) << offset_g) +
                            static_cast<UnsignedType>(static_cast<UnsignedType>(tb) << offset_b));
                    }

                public:
                    static constexpr inline size_t offset_r = r_bit_offset;
                    static constexpr inline size_t offset_g = g_bit_offset;
                    static constexpr inline size_t offset_b = b_bit_offset;
                    static constexpr inline UnsignedType max_r = static_cast<UnsignedType>(static_cast<UnsignedType>(1) << r_bit_count) - static_cast<UnsignedType>(1);
                    static constexpr inline UnsignedType max_g = static_cast<UnsignedType>(static_cast<UnsignedType>(1) << g_bit_count) - static_cast<UnsignedType>(1);
                    static constexpr inline UnsignedType max_b = static_cast<UnsignedType>(static_cast<UnsignedType>(1) << b_bit_count) - static_cast<UnsignedType>(1);
                    static constexpr inline UnsignedType mask_r = static_cast<UnsignedType>(static_cast<UnsignedType>(max_r) << offset_r);
                    static constexpr inline UnsignedType mask_g = static_cast<UnsignedType>(static_cast<UnsignedType>(max_g) << offset_g);
                    static constexpr inline UnsignedType mask_b = static_cast<UnsignedType>(static_cast<UnsignedType>(max_b) << offset_b);

                private:
                    UnsignedType m_val;
                };

                template <typename UnsignedType,
                          size_t r_bit_count, size_t g_bit_count, size_t b_bit_count, size_t a_bit_count,
                          size_t r_bit_offset, size_t g_bit_offset, size_t b_bit_offset, size_t a_bit_offset>
                struct RGBAType
                {
                public:
                    constexpr RGBAType() noexcept : m_val{0} {}
                    constexpr RGBAType(UnsignedType r, UnsignedType g, UnsignedType b, UnsignedType a) noexcept : m_val{pack(r, g, b, a)} {}

                    constexpr RGBAType(const RGBAType &val) noexcept : m_val(val.m_val) {}
                    constexpr RGBAType &operator=(const RGBAType &val) noexcept
                    {
                        m_val = val.m_val;
                        return *this;
                    }

                    explicit constexpr operator UnsignedType() const noexcept
                    {
                        return m_val;
                    }
                    // R
                    constexpr auto GetR() const noexcept -> UnsignedType
                    {
                        UnsignedType r = static_cast<UnsignedType>(m_val & static_cast<UnsignedType>(mask_r)) >> offset_r;
                        return (r <= static_cast<UnsignedType>(max_r)) ? r : static_cast<UnsignedType>(max_r);
                    }
                    constexpr void SetR(UnsignedType r) noexcept
                    {
                        UnsignedType tr = (r <= static_cast<UnsignedType>(max_r)) ? r : static_cast<UnsignedType>(max_r);
                        m_val &= static_cast<UnsignedType>(mask_r);
                        m_val |= static_cast<UnsignedType>(tr << offset_r);
                    }
                    // G
                    constexpr auto GetG() const noexcept -> UnsignedType
                    {
                        UnsignedType g = static_cast<UnsignedType>(m_val & static_cast<UnsignedType>(mask_g)) >> offset_g;
                        return (g <= static_cast<UnsignedType>(max_g)) ? g : static_cast<UnsignedType>(max_g);
                    }
                    constexpr void SetG(UnsignedType g) noexcept
                    {
                        UnsignedType tg = (g <= static_cast<UnsignedType>(max_g)) ? g : static_cast<UnsignedType>(max_g);
                        m_val &= static_cast<UnsignedType>(mask_g);
                        m_val |= static_cast<UnsignedType>(tg << offset_g);
                    }
                    // B
                    constexpr auto GetB() const noexcept -> UnsignedType
                    {
                        UnsignedType b = static_cast<UnsignedType>(m_val & static_cast<UnsignedType>(mask_b)) >> offset_b;
                        return (b <= static_cast<UnsignedType>(max_b)) ? b : static_cast<UnsignedType>(max_b);
                    }
                    constexpr void SetB(UnsignedType b) noexcept
                    {
                        UnsignedType tb = (b <= static_cast<UnsignedType>(max_b)) ? b : static_cast<UnsignedType>(max_b);
                        m_val &= static_cast<UnsignedType>(mask_b);
                        m_val |= static_cast<UnsignedType>(tb << offset_b);
                    }
                    // A
                    constexpr auto GetA() const noexcept -> UnsignedType
                    {
                        UnsignedType a = static_cast<UnsignedType>(m_val & static_cast<UnsignedType>(mask_a)) >> offset_a;
                        return (a <= static_cast<UnsignedType>(max_a)) ? a : static_cast<UnsignedType>(max_a);
                    }
                    constexpr void SetA(UnsignedType a) noexcept
                    {
                        UnsignedType tb = (b <= static_cast<UnsignedType>(max_a)) ? a : static_cast<UnsignedType>(max_a);
                        m_val &= static_cast<UnsignedType>(mask_a);
                        m_val |= static_cast<UnsignedType>(ta << offset_a);
                    }

                private:
                    static constexpr auto pack(UnsignedType r, UnsignedType g, UnsignedType b, UnsignedType a) noexcept -> UnsignedType
                    {
                        // 3
                        UnsignedType tr = (r <= static_cast<UnsignedType>(max_r)) ? r : static_cast<UnsignedType>(max_r);
                        UnsignedType tg = (g <= static_cast<UnsignedType>(max_g)) ? g : static_cast<UnsignedType>(max_g);
                        UnsignedType tb = (b <= static_cast<UnsignedType>(max_b)) ? b : static_cast<UnsignedType>(max_b);
                        UnsignedType ta = (a <= static_cast<UnsignedType>(max_a)) ? a : static_cast<UnsignedType>(max_a);
                        return static_cast<UnsignedType>(
                            static_cast<UnsignedType>(static_cast<UnsignedType>(tr) << offset_r) +
                            static_cast<UnsignedType>(static_cast<UnsignedType>(tg) << offset_g) +
                            static_cast<UnsignedType>(static_cast<UnsignedType>(tb) << offset_b) +
                            static_cast<UnsignedType>(static_cast<UnsignedType>(ta) << offset_a));
                    }

                public:
                    static constexpr inline size_t offset_r = r_bit_offset;
                    static constexpr inline size_t offset_g = g_bit_offset;
                    static constexpr inline size_t offset_b = b_bit_offset;
                    static constexpr inline size_t offset_a = a_bit_offset;
                    static constexpr inline UnsignedType max_r = static_cast<UnsignedType>(static_cast<UnsignedType>(1) << r_bit_count) - static_cast<UnsignedType>(1);
                    static constexpr inline UnsignedType max_g = static_cast<UnsignedType>(static_cast<UnsignedType>(1) << g_bit_count) - static_cast<UnsignedType>(1);
                    static constexpr inline UnsignedType max_b = static_cast<UnsignedType>(static_cast<UnsignedType>(1) << b_bit_count) - static_cast<UnsignedType>(1);
                    static constexpr inline UnsignedType max_a = static_cast<UnsignedType>(static_cast<UnsignedType>(1) << a_bit_count) - static_cast<UnsignedType>(1);
                    static constexpr inline UnsignedType mask_r = static_cast<UnsignedType>(static_cast<UnsignedType>(max_r) << offset_r);
                    static constexpr inline UnsignedType mask_g = static_cast<UnsignedType>(static_cast<UnsignedType>(max_g) << offset_g);
                    static constexpr inline UnsignedType mask_b = static_cast<UnsignedType>(static_cast<UnsignedType>(max_b) << offset_b);
                    static constexpr inline UnsignedType mask_a = static_cast<UnsignedType>(static_cast<UnsignedType>(max_a) << offset_a);

                private:
                    UnsignedType m_val;
                };

                template <GLenum typeEnum>
                struct GLTypeInfo;
                template <>
                struct GLTypeInfo<GL_UNSIGNED_BYTE>
                {
                    using type = GLubyte;
                };
                template <>
                struct GLTypeInfo<GL_BYTE>
                {
                    using type = GLbyte;
                };
                template <>
                struct GLTypeInfo<GL_UNSIGNED_SHORT>
                {
                    using type = GLushort;
                };
                template <>
                struct GLTypeInfo<GL_SHORT>
                {
                    using type = GLshort;
                };
                template <>
                struct GLTypeInfo<GL_UNSIGNED_INT>
                {
                    using type = GLuint;
                };
                template <>
                struct GLTypeInfo<GL_INT>
                {
                    using type = GLint;
                };
                template <>
                struct GLTypeInfo<GL_FLOAT>
                {
                    using type = GLfloat;
                };
                template <>
                struct GLTypeInfo<GL_HALF_FLOAT>
                {
                    using type = Imath::half;
                };
                template <>
                struct GLTypeInfo<GL_UNSIGNED_BYTE_3_3_2>
                {
                    using type = RGBType<GLubyte, 3, 3, 2, 5, 2, 0>;
                    static_assert(sizeof(type) == sizeof(GLubyte));
                };
                template <>
                struct GLTypeInfo<GL_UNSIGNED_BYTE_2_3_3_REV>
                {
                    using type = RGBType<GLubyte, 2, 3, 3, 0, 3, 6>;
                    static_assert(sizeof(type) == sizeof(GLubyte));
                };
                template <>
                struct GLTypeInfo<GL_UNSIGNED_SHORT_5_6_5>
                {
                    using type = RGBType<GLushort, 5, 6, 5, 11, 5, 0>;
                    static_assert(sizeof(type) == sizeof(GLushort));
                };
                template <>
                struct GLTypeInfo<GL_UNSIGNED_SHORT_5_6_5_REV>
                {
                    using type = RGBType<GLushort, 5, 6, 5, 0, 5, 11>;
                    static_assert(sizeof(type) == sizeof(GLushort));
                };
                template <>
                struct GLTypeInfo<GL_UNSIGNED_SHORT_4_4_4_4>
                {
                    using type = RGBAType<GLushort, 4, 4, 4, 4, 12, 8, 4, 0>;
                    static_assert(sizeof(type) == sizeof(GLushort));
                };
                template <>
                struct GLTypeInfo<GL_UNSIGNED_SHORT_4_4_4_4_REV>
                {
                    using type = RGBAType<GLushort, 4, 4, 4, 4, 0, 4, 8, 12>;
                    static_assert(sizeof(type) == sizeof(GLushort));
                };
                template <>
                struct GLTypeInfo<GL_UNSIGNED_SHORT_5_5_5_1>
                {
                    using type = RGBAType<GLushort, 5, 5, 5, 1, 11, 6, 1, 0>;
                    static_assert(sizeof(type) == sizeof(GLushort));
                };
                template <>
                struct GLTypeInfo<GL_UNSIGNED_SHORT_1_5_5_5_REV>
                {
                    using type = RGBAType<GLushort, 1, 5, 5, 5, 0, 5, 10, 15>;
                    static_assert(sizeof(type) == sizeof(GLushort));
                };
                template <>
                struct GLTypeInfo<GL_UNSIGNED_INT_8_8_8_8>
                {
                    using type = RGBAType<GLuint, 8, 8, 8, 8, 24, 16, 8, 0>;
                    static_assert(sizeof(type) == sizeof(GLuint));
                };
                template <>
                struct GLTypeInfo<GL_UNSIGNED_INT_8_8_8_8_REV>
                {
                    using type = RGBAType<GLuint, 8, 8, 8, 8, 0, 8, 16, 24>;
                    static_assert(sizeof(type) == sizeof(GLuint));
                };
                template <>
                struct GLTypeInfo<GL_UNSIGNED_INT_10_10_10_2>
                {
                    using type = RGBAType<GLuint, 10, 10, 10, 2, 22, 12, 2, 0>;
                    static_assert(sizeof(type) == sizeof(GLuint));
                };
                template <>
                struct GLTypeInfo<GL_UNSIGNED_INT_2_10_10_10_REV>
                {
                    using type = RGBAType<GLuint, 2, 10, 10, 10, 0, 10, 20, 30>;
                    static_assert(sizeof(type) == sizeof(GLuint));
                };
                template <>
                struct GLTypeInfo<GL_UNSIGNED_INT_5_9_9_9_REV>
                {
                    using type = RGBAType<GLuint, 9, 9, 9, 5, 0, 9, 18, 27>;
                    static_assert(sizeof(type) == sizeof(GLuint));
                };
                template <>
                struct GLTypeInfo<GL_UNSIGNED_INT_10F_11F_11F_REV>
                {
                    // R11:G11:B10
                    // B 0:G10:R21
                    using type = RGBType<GLuint, 11, 11, 10, 21, 10, 0>;
                    static_assert(sizeof(type) == sizeof(GLuint));
                };
                template <>
                struct GLTypeInfo<GL_FLOAT_32_UNSIGNED_INT_24_8_REV>
                {
                    struct type
                    {
                    public:
                        constexpr type() noexcept : m_Depth{0.0f}, m_Stencil{0} {}
                        constexpr type(float d, unsigned char s) noexcept : m_Depth{d}, m_Stencil{static_cast<unsigned int>(s) << 24} {}
                        constexpr type(const type &type) noexcept : m_Depth{type.m_Depth}, m_Stencil{type.m_Stencil} {}
                        constexpr type &operator=(const type &type) noexcept
                        {
                            m_Depth = type.m_Depth;
                            m_Stencil = type.m_Stencil;
                            return *this;
                        }
                        constexpr auto GetDepth() const noexcept -> float { return m_Depth; }
                        constexpr auto GetStencil() const noexcept -> unsigned char { return static_cast<unsigned char>(m_Stencil >> 24); }
                        constexpr void SetDepth(float d) noexcept { m_Depth = d; }
                        constexpr void SetStencil(unsigned char s) noexcept { m_Stencil = static_cast<unsigned int>(s) << 24; }

                    private:
                        float m_Depth;
                        unsigned int m_Stencil;
                    };
                    static_assert(sizeof(type) == sizeof(float) * 2);
                };
                template <>
                struct GLTypeInfo<GL_UNSIGNED_INT_24_8>
                {
                    // D24S8
                    // D 8S0
                    using type = DepthStencilType<GLuint, 24, 8, 8, 0>;
                };

                struct GLFormatTypeInfo
                {
                    GLenum format;
                    GLenum base_format;
                    GLenum base_type;
                    GLsizei num_bases;
                };

                template <GLenum formatEnum>
                struct GLBaseFormatInfo;
                template <>
                struct GLBaseFormatInfo<GL_RED>
                {
                    static inline constexpr GLsizei channels = 1;
                };
                template <>
                struct GLBaseFormatInfo<GL_RG>
                {
                    static inline constexpr GLsizei channels = 2;
                };
                template <>
                struct GLBaseFormatInfo<GL_RGB>
                {
                    static inline constexpr GLsizei channels = 3;
                };
                template <>
                struct GLBaseFormatInfo<GL_BGR>
                {
                    static inline constexpr GLsizei channels = 3;
                };
                template <>
                struct GLBaseFormatInfo<GL_RGBA>
                {
                    static inline constexpr GLsizei channels = 4;
                };
                template <>
                struct GLBaseFormatInfo<GL_BGRA>
                {
                    static inline constexpr GLsizei channels = 4;
                };
                template <>
                struct GLBaseFormatInfo<GL_RED_INTEGER>
                {
                    static inline constexpr GLsizei channels = 1;
                };
                template <>
                struct GLBaseFormatInfo<GL_RG_INTEGER>
                {
                    static inline constexpr GLsizei channels = 2;
                };
                template <>
                struct GLBaseFormatInfo<GL_RGB_INTEGER>
                {
                    static inline constexpr GLsizei channels = 3;
                };
                template <>
                struct GLBaseFormatInfo<GL_BGR_INTEGER>
                {
                    static inline constexpr GLsizei channels = 3;
                };
                template <>
                struct GLBaseFormatInfo<GL_RGBA_INTEGER>
                {
                    static inline constexpr GLsizei channels = 4;
                };
                template <>
                struct GLBaseFormatInfo<GL_BGRA_INTEGER>
                {
                    static inline constexpr GLsizei channels = 4;
                };
                template <>
                struct GLBaseFormatInfo<GL_DEPTH_COMPONENT>
                {
                    static inline constexpr GLsizei channels = 1;
                };
                template <>
                struct GLBaseFormatInfo<GL_DEPTH_STENCIL>
                {
                    static inline constexpr GLsizei channels = 2;
                };

                template <GLenum formatEnum>
                struct GLFormatInfo;
                // R
                RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_GL_FORMAT_INFO_DEF_RED(R8, UNSIGNED_BYTE, 1);
                RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_GL_FORMAT_INFO_DEF_RED(R8_SNORM, BYTE, 1);
                RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_GL_FORMAT_INFO_DEF_RED(R16, UNSIGNED_SHORT, 1);
                RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_GL_FORMAT_INFO_DEF_RED(R16_SNORM, SHORT, 1);
                //RG
                RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_GL_FORMAT_INFO_DEF_RG(RG8, UNSIGNED_BYTE, 2);
                RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_GL_FORMAT_INFO_DEF_RG(RG8_SNORM, BYTE, 2);
                RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_GL_FORMAT_INFO_DEF_RG(RG16, UNSIGNED_SHORT, 2);
                RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_GL_FORMAT_INFO_DEF_RG(RG16_SNORM, SHORT, 2);
                //RGB
                RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_GL_FORMAT_INFO_DEF_RGB(RGB8, UNSIGNED_BYTE, 3);
                RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_GL_FORMAT_INFO_DEF_RGB(RGB8_SNORM, BYTE, 3);
                RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_GL_FORMAT_INFO_DEF_RGB(RGB16, UNSIGNED_SHORT, 3);
                RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_GL_FORMAT_INFO_DEF_RGB(RGB16_SNORM, SHORT, 3);
                //RGBA
                template <>
                struct GLFormatInfo<GL_RGBA8>
                {
                    static inline constexpr GLFormatTypeInfo formats[] = {
                        {GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE, 4},
                        {GL_RGBA8, GL_RGBA, GL_UNSIGNED_INT_8_8_8_8, 1},
                        {GL_RGBA8, GL_UNSIGNED_INT_8_8_8_8_REV, 1},
                        {GL_RGBA8, GL_BGRA, GL_UNSIGNED_BYTE, 4},
                        {GL_RGBA8, GL_BGRA, GL_UNSIGNED_INT_8_8_8_8, 1},
                        {GL_RGBA8, GL_BGRA, GL_UNSIGNED_INT_8_8_8_8_REV, 1},
                    };
                    static inline constexpr GLenum base_format = formats[0].base_format;
                    static inline constexpr GLsizei channels = GLBaseFormatInfo<base_format>::channels;
                    static inline constexpr GLsizei size = sizeof(GLTypeInfo<formats[0].base_type>::type) * formats[0].num_bases;
                };
                RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_GL_FORMAT_INFO_DEF_RGBA(RGBA8_SNORM, BYTE, 4);
                RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_GL_FORMAT_INFO_DEF_RGBA(RGBA16, UNSIGNED_SHORT, 4);
                RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_GL_FORMAT_INFO_DEF_RGBA(RGBA16_SNORM, SHORT, 4);

                template <>
                struct GLFormatInfo<GL_R3_G3_B2>
                {
                    static inline constexpr GLFormatTypeInfo formats[] = {
                        {GL_R3_G3_B2, GL_RGB, GL_UNSIGNED_BYTE_3_3_2, 1},
                        {GL_R3_G3_B2, GL_RGB, GL_UNSIGNED_BYTE_2_3_3_REV, 1},
                        {GL_R3_G3_B2, GL_RGB, GL_UNSIGNED_BYTE, 3},
                        {GL_R3_G3_B2, GL_BGR, GL_UNSIGNED_BYTE_3_3_2, 1},
                        {GL_R3_G3_B2, GL_BGR, GL_UNSIGNED_BYTE_2_3_3_REV, 1},
                        {GL_R3_G3_B2, GL_BGR, GL_UNSIGNED_BYTE, 3}};
                    static inline constexpr GLenum base_format = formats[0].base_format;
                    static inline constexpr GLsizei channels = GLBaseFormatInfo<base_format>::channels;
                    static inline constexpr GLsizei size = sizeof(GLTypeInfo<formats[0].base_type>::type) * formats[0].num_bases;
                };
                RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_GL_FORMAT_INFO_DEF_RGB(RGB4, UNSIGNED_BYTE, 3);
                RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_GL_FORMAT_INFO_DEF_RGB(RGB5, UNSIGNED_BYTE, 3);
                template <>
                struct GLFormatInfo<GL_RGB10>
                {
                    static inline constexpr GLFormatTypeInfo formats[] = {
                        {GL_RGB10, GL_RGB, GL_UNSIGNED_SHORT, 3},
                        {GL_RGB10, GL_BGR, GL_UNSIGNED_SHORT, 3}};
                    static inline constexpr GLenum base_format = formats[0].base_format;
                    static inline constexpr GLsizei channels = GLBaseFormatInfo<base_format>::channels;
                    static inline constexpr GLsizei size = sizeof(GLTypeInfo<formats[0].base_type>::type) * formats[0].num_bases;
                };
                template <>
                struct GLFormatInfo<GL_RGB12>
                {
                    static inline constexpr GLFormatTypeInfo formats[] = {
                        {GL_RGB12, GL_RGB, GL_UNSIGNED_SHORT, 3},
                        {GL_RGB12, GL_BGR, GL_UNSIGNED_SHORT, 3}};
                    static inline constexpr GLenum base_format = formats[0].base_format;
                    static inline constexpr GLsizei channels = GLBaseFormatInfo<base_format>::channels;
                    static inline constexpr GLsizei size = sizeof(GLTypeInfo<formats[0].base_type>::type) * formats[0].num_bases;
                };
                RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_GL_FORMAT_INFO_DEF_RGB(R11F_G11F_B10F, UNSIGNED_INT_10F_11F_11F_REV, 1);
                RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_GL_FORMAT_INFO_DEF_RGB(RGB9_E5, UNSIGNED_INT_5_9_9_9_REV, 1);
                RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_GL_FORMAT_INFO_DEF_RGBA(RGBA2, UNSIGNED_BYTE, 4);
                template <> // OK
                struct GLFormatInfo<GL_RGBA4>
                {
                    static inline constexpr GLFormatTypeInfo formats[] = {
                        {GL_RGBA4, GL_RGBA, GL_UNSIGNED_SHORT_4_4_4_4, 1},
                        {GL_RGBA4, GL_RGBA, GL_UNSIGNED_SHORT_4_4_4_4_REV, 1},
                        {GL_RGBA4, GL_RGBA, GL_UNSIGNED_BYTE, 4},
                        {GL_RGBA4, GL_BGRA, GL_UNSIGNED_SHORT_4_4_4_4, 1},
                        {GL_RGBA4, GL_BGRA, GL_UNSIGNED_SHORT_4_4_4_4_REV, 1},
                        {GL_RGBA4, GL_BGRA, GL_UNSIGNED_BYTE, 4},
                    };
                    static inline constexpr GLenum base_format = formats[0].base_format;
                    static inline constexpr GLsizei channels = GLBaseFormatInfo<base_format>::channels;
                    static inline constexpr GLsizei size = sizeof(GLTypeInfo<formats[0].base_type>::type) * formats[0].num_bases;
                };
                template <>
                struct GLFormatInfo<GL_SRGB8>
                {
                    static inline constexpr GLFormatTypeInfo formats[] = {
                        {GL_SRGB8, GL_RGB, GL_UNSIGNED_BYTE, 3},
                        {GL_SRGB8, GL_BGR, GL_UNSIGNED_BYTE, 3},
                    };
                    static inline constexpr GLenum base_format = formats[0].base_format;
                    static inline constexpr GLsizei channels = GLBaseFormatInfo<base_format>::channels;
                    static inline constexpr GLsizei size = sizeof(GLTypeInfo<formats[0].base_type>::type) * formats[0].num_bases;
                };
                template <>
                struct GLFormatInfo<GL_SRGB8_ALPHA8>
                {
                    static inline constexpr GLFormatTypeInfo formats[] = {
                        {GL_SRGB8_ALPHA8, GL_RGBA, GL_UNSIGNED_BYTE, 4},
                        {GL_SRGB8_ALPHA8, GL_BGRA, GL_UNSIGNED_BYTE, 4},
                    };
                    static inline constexpr GLenum base_format = formats[0].base_format;
                    static inline constexpr GLsizei channels = GLBaseFormatInfo<base_format>::channels;
                    static inline constexpr GLsizei size = sizeof(GLTypeInfo<formats[0].base_type>::type) * formats[0].num_bases;
                };
                template <>
                struct GLFormatInfo<GL_RGB5_A1>
                {
                    static inline constexpr GLFormatTypeInfo formats[] = {
                        {GL_RGB5_A1, GL_RGBA, GL_UNSIGNED_SHORT_5_5_5_1, 1},
                        {GL_RGB5_A1, GL_RGBA, GL_UNSIGNED_SHORT_1_5_5_5_REV, 1},
                        {GL_RGB5_A1, GL_RGBA, GL_UNSIGNED_INT_2_10_10_10_REV, 1},
                        {GL_RGB5_A1, GL_BGRA, GL_UNSIGNED_SHORT_5_5_5_1, 1},
                        {GL_RGB5_A1, GL_RGBA, GL_UNSIGNED_SHORT_1_5_5_5_REV, 1},
                        {GL_RGB5_A1, GL_BGRA, GL_UNSIGNED_INT_2_10_10_10_REV, 1},
                    };
                    static inline constexpr GLenum base_format = formats[0].base_format;
                    static inline constexpr GLsizei channels = GLBaseFormatInfo<base_format>::channels;
                    static inline constexpr GLsizei size = sizeof(GLTypeInfo<formats[0].base_type>::type) * formats[0].num_bases;
                };
                template <> // OK
                struct GLFormatInfo<GL_RGB10_A2>
                {
                    static inline constexpr GLFormatTypeInfo formats[] = {
                        {GL_RGB10_A2, GL_RGBA, GL_UNSIGNED_INT_10_10_10_2, 1},
                        {GL_RGB10_A2, GL_RGBA, GL_UNSIGNED_INT_2_10_10_10_REV, 1},
                        {GL_RGB10_A2, GL_RGBA, GL_UNSIGNED_SHORT, 4},
                        {GL_RGB10_A2, GL_BGRA, GL_UNSIGNED_INT_10_10_10_2, 1},
                        {GL_RGB10_A2, GL_BGRA, GL_UNSIGNED_INT_2_10_10_10_REV, 1},
                        {GL_RGB10_A2, GL_BGRA, GL_UNSIGNED_SHORT, 4},
                    };
                    static inline constexpr GLenum base_format = formats[0].base_format;
                    static inline constexpr GLsizei channels = GLBaseFormatInfo<base_format>::channels;
                    static inline constexpr GLsizei size = sizeof(GLTypeInfo<formats[0].base_type>::type) * formats[0].num_bases;
                };
                template <> // OK
                struct GLFormatInfo<GL_RGB10_A2UI>
                {
                    static inline constexpr GLFormatTypeInfo formats[] = {
                        {GL_RGB10_A2UI, GL_RGBA_INTEGER, GL_UNSIGNED_INT_10_10_10_2, 1},
                        {GL_RGB10_A2UI, GL_RGBA_INTEGER, GL_UNSIGNED_INT_2_10_10_10_REV, 1},
                        {GL_RGB10_A2UI, GL_RGBA_INTEGER, GL_UNSIGNED_SHORT, 4},
                        {GL_RGB10_A2UI, GL_BGRA_INTEGER, GL_UNSIGNED_INT_10_10_10_2, 1},
                        {GL_RGB10_A2UI, GL_BGRA_INTEGER, GL_UNSIGNED_INT_2_10_10_10_REV, 1},
                        {GL_RGB10_A2UI, GL_BGRA_INTEGER, GL_UNSIGNED_SHORT, 4},
                    };
                    static inline constexpr GLenum base_format = formats[0].base_format;
                    static inline constexpr GLsizei channels = GLBaseFormatInfo<base_format>::channels;
                    static inline constexpr GLsizei size = sizeof(GLTypeInfo<formats[0].base_type>::type) * formats[0].num_bases;
                };
                RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_GL_FORMAT_INFO_DEF_RGBA(RGBA12, UNSIGNED_SHORT, 4);
                //FLOAT
                RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_GL_FORMAT_INFO_DEF_RED(R16F, HALF_FLOAT, 1);
                RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_GL_FORMAT_INFO_DEF_RED(R32F, FLOAT, 1);
                RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_GL_FORMAT_INFO_DEF_RG(RG16F, HALF_FLOAT, 2);
                RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_GL_FORMAT_INFO_DEF_RG(RG32F, FLOAT, 2);
                RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_GL_FORMAT_INFO_DEF_RGB(RGB16F, HALF_FLOAT, 3);
                RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_GL_FORMAT_INFO_DEF_RGB(RGB32F, FLOAT, 3);
                RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_GL_FORMAT_INFO_DEF_RGBA(RGBA16F, HALF_FLOAT, 4);
                RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_GL_FORMAT_INFO_DEF_RGBA(RGBA32F, FLOAT, 4);
                // RED
                RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_GL_FORMAT_INFO_DEF_INT_RG(R8I, BYTE, 1);
                RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_GL_FORMAT_INFO_DEF_INT_RG(R8UI, UNSIGNED_BYTE, 1);
                RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_GL_FORMAT_INFO_DEF_INT_RG(R16I, SHORT, 1);
                RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_GL_FORMAT_INFO_DEF_INT_RG(R16UI, UNSIGNED_SHORT, 1);
                RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_GL_FORMAT_INFO_DEF_INT_RG(R32I, INT, 1);
                RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_GL_FORMAT_INFO_DEF_INT_RG(R32UI, UNSIGNED_INT, 1);
                // RG
                RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_GL_FORMAT_INFO_DEF_INT_RG(RG8I, BYTE, 2);
                RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_GL_FORMAT_INFO_DEF_INT_RG(RG8UI, UNSIGNED_BYTE, 2);
                RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_GL_FORMAT_INFO_DEF_INT_RG(RG16I, SHORT, 2);
                RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_GL_FORMAT_INFO_DEF_INT_RG(RG16UI, UNSIGNED_SHORT, 2);
                RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_GL_FORMAT_INFO_DEF_INT_RG(RG32I, INT, 2);
                RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_GL_FORMAT_INFO_DEF_INT_RG(RG32UI, UNSIGNED_INT, 2);
                // RGB
                RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_GL_FORMAT_INFO_DEF_INT_RGB(RGB8I, BYTE, 3);
                RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_GL_FORMAT_INFO_DEF_INT_RGB(RGB8UI, UNSIGNED_BYTE, 3);
                RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_GL_FORMAT_INFO_DEF_INT_RGB(RGB16I, SHORT, 3);
                RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_GL_FORMAT_INFO_DEF_INT_RGB(RGB16UI, UNSIGNED_SHORT, 3);
                RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_GL_FORMAT_INFO_DEF_INT_RGB(RGB32I, INT, 3);
                RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_GL_FORMAT_INFO_DEF_INT_RGB(RGB32UI, UNSIGNED_INT, 3);
                // RGBA
                RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_GL_FORMAT_INFO_DEF_INT_RGBA(RGBA8I, BYTE, 4);
                RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_GL_FORMAT_INFO_DEF_INT_RGBA(RGBA8UI, UNSIGNED_BYTE, 3);
                RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_GL_FORMAT_INFO_DEF_INT_RGBA(RGBA16I, SHORT, 4);
                RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_GL_FORMAT_INFO_DEF_INT_RGBA(RGBA16UI, UNSIGNED_SHORT, 4);
                RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_GL_FORMAT_INFO_DEF_INT_RGBA(RGBA32I, INT, 4);
                RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_GL_FORMAT_INFO_DEF_INT_RGBA(RGBA32UI, UNSIGNED_INT, 4);
                // Depth
                template <>
                struct GLFormatInfo<GL_DEPTH_COMPONENT16>
                {
                    static inline constexpr GLFormatTypeInfo formats[] = {
                        {GL_DEPTH_COMPONENT16, GL_DEPTH_COMPONENT, GL_UNSIGNED_SHORT, 1},
                        {GL_DEPTH_COMPONENT16, GL_DEPTH_COMPONENT, GL_UNSIGNED_INT, 1},
                    };
                    static inline constexpr GLenum base_format = formats[0].base_format;
                    static inline constexpr GLsizei channels = GLBaseFormatInfo<base_format>::channels;
                    static inline constexpr GLsizei size = sizeof(GLTypeInfo<formats[0].base_type>::type) * formats[0].num_bases;
                };
                RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_GL_FORMAT_INFO_DEF_DEPTH(DEPTH_COMPONENT24, UNSIGNED_INT, 1);
                RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_GL_FORMAT_INFO_DEF_DEPTH(DEPTH_COMPONENT32F, FLOAT, 1);
                RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_GL_FORMAT_INFO_DEF_DEPTH_STENCIL(DEPTH24_STENCIL8, UNSIGNED_INT_24_8, 1);
                RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_GL_FORMAT_INFO_DEF_DEPTH_STENCIL(DEPTH32F_STENCIL8, FLOAT_32_UNSIGNED_INT_24_8_REV, 1);

                inline constexpr auto GetGLFormatSize(GLenum format) -> size_t
                {
                    switch (format)
                    {
                        /*R*/
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_SIZE(R8);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_SIZE(R8_SNORM);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_SIZE(R16);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_SIZE(R16_SNORM);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_SIZE(R8UI);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_SIZE(R16UI);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_SIZE(R32UI);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_SIZE(R8I);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_SIZE(R16I);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_SIZE(R32I);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_SIZE(R16F);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_SIZE(R32F);
                        /*RG*/
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_SIZE(RG8);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_SIZE(RG8_SNORM);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_SIZE(RG16);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_SIZE(RG16_SNORM);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_SIZE(RG8UI);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_SIZE(RG16UI);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_SIZE(RG32UI);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_SIZE(RG8I);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_SIZE(RG16I);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_SIZE(RG32I);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_SIZE(RG16F);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_SIZE(RG32F);
                        /*RGB*/
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_SIZE(RGB8);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_SIZE(RGB8_SNORM);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_SIZE(RGB16);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_SIZE(RGB16_SNORM);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_SIZE(RGB8UI);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_SIZE(RGB16UI);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_SIZE(RGB32UI);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_SIZE(RGB8I);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_SIZE(RGB16I);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_SIZE(RGB32I);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_SIZE(RGB16F);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_SIZE(RGB32F);
                        /*RGBA*/
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_SIZE(RGBA8);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_SIZE(RGBA8_SNORM);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_SIZE(RGBA16);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_SIZE(RGBA16_SNORM);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_SIZE(RGBA8UI);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_SIZE(RGBA16UI);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_SIZE(RGBA32UI);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_SIZE(RGBA8I);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_SIZE(RGBA16I);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_SIZE(RGBA32I);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_SIZE(RGBA16F);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_SIZE(RGBA32F);
                        /*CUSTOM*/
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_SIZE(R3_G3_B2);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_SIZE(RGB4);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_SIZE(RGB5);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_SIZE(SRGB8);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_SIZE(SRGB8_ALPHA8);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_SIZE(RGB10);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_SIZE(RGB12);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_SIZE(R11F_G11F_B10F);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_SIZE(RGB9_E5);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_SIZE(RGBA2);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_SIZE(RGBA4);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_SIZE(RGB5_A1);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_SIZE(RGB10_A2);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_SIZE(RGB10_A2UI);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_SIZE(RGBA12);
                        /*DEPTH*/
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_SIZE(DEPTH_COMPONENT16);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_SIZE(DEPTH_COMPONENT24);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_SIZE(DEPTH_COMPONENT32F);
                        /*DEPTH_STENCIL*/
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_SIZE(DEPTH24_STENCIL8);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_SIZE(DEPTH32F_STENCIL8);
                    default:
                        return 0;
                    }
                }
                inline constexpr auto GetGLBaseFormat(GLenum format) -> GLenum
                {
                    switch (format)
                    {
                        /*R*/
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_FORMAT(R8);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_FORMAT(R8_SNORM);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_FORMAT(R16);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_FORMAT(R16_SNORM);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_FORMAT(R8UI);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_FORMAT(R16UI);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_FORMAT(R32UI);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_FORMAT(R8I);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_FORMAT(R16I);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_FORMAT(R32I);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_FORMAT(R16F);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_FORMAT(R32F);
                        /*RG*/
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_FORMAT(RG8);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_FORMAT(RG8_SNORM);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_FORMAT(RG16);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_FORMAT(RG16_SNORM);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_FORMAT(RG8UI);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_FORMAT(RG16UI);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_FORMAT(RG32UI);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_FORMAT(RG8I);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_FORMAT(RG16I);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_FORMAT(RG32I);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_FORMAT(RG16F);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_FORMAT(RG32F);
                        /*RGB*/
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_FORMAT(RGB8);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_FORMAT(RGB8_SNORM);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_FORMAT(RGB16);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_FORMAT(RGB16_SNORM);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_FORMAT(RGB8UI);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_FORMAT(RGB16UI);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_FORMAT(RGB32UI);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_FORMAT(RGB8I);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_FORMAT(RGB16I);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_FORMAT(RGB32I);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_FORMAT(RGB16F);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_FORMAT(RGB32F);
                        /*RGBA*/
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_FORMAT(RGBA8);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_FORMAT(RGBA8_SNORM);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_FORMAT(RGBA16);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_FORMAT(RGBA16_SNORM);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_FORMAT(RGBA8UI);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_FORMAT(RGBA16UI);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_FORMAT(RGBA32UI);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_FORMAT(RGBA8I);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_FORMAT(RGBA16I);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_FORMAT(RGBA32I);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_FORMAT(RGBA16F);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_FORMAT(RGBA32F);
                        /*CUSTOM*/
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_FORMAT(R3_G3_B2);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_FORMAT(RGB4);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_FORMAT(RGB5);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_FORMAT(SRGB8);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_FORMAT(SRGB8_ALPHA8);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_FORMAT(RGB10);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_FORMAT(RGB12);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_FORMAT(R11F_G11F_B10F);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_FORMAT(RGB9_E5);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_FORMAT(RGBA2);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_FORMAT(RGBA4);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_FORMAT(RGB5_A1);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_FORMAT(RGB10_A2);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_FORMAT(RGB10_A2UI);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_FORMAT(RGBA12);
                        /*DEPTH*/
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_FORMAT(DEPTH_COMPONENT16);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_FORMAT(DEPTH_COMPONENT24);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_FORMAT(DEPTH_COMPONENT32F);
                        /*DEPTH_STENCIL*/
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_FORMAT(DEPTH24_STENCIL8);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_FORMAT(DEPTH32F_STENCIL8);
                    default:
                        return GL_RGBA;
                    }
                }
                inline constexpr auto GetGLBaseType(GLenum format) -> GLenum
                {
                    switch (format)
                    {
                        /*R*/
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_TYPE(R8);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_TYPE(R8_SNORM);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_TYPE(R16);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_TYPE(R16_SNORM);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_TYPE(R8UI);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_TYPE(R16UI);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_TYPE(R32UI);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_TYPE(R8I);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_TYPE(R16I);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_TYPE(R32I);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_TYPE(R16F);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_TYPE(R32F);
                        /*RG*/
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_TYPE(RG8);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_TYPE(RG8_SNORM);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_TYPE(RG16);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_TYPE(RG16_SNORM);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_TYPE(RG8UI);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_TYPE(RG16UI);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_TYPE(RG32UI);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_TYPE(RG8I);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_TYPE(RG16I);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_TYPE(RG32I);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_TYPE(RG16F);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_TYPE(RG32F);
                        /*RGB*/
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_TYPE(RGB8);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_TYPE(RGB8_SNORM);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_TYPE(RGB16);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_TYPE(RGB16_SNORM);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_TYPE(RGB8UI);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_TYPE(RGB16UI);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_TYPE(RGB32UI);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_TYPE(RGB8I);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_TYPE(RGB16I);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_TYPE(RGB32I);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_TYPE(RGB16F);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_TYPE(RGB32F);
                        /*RGBA*/
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_TYPE(RGBA8);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_TYPE(RGBA8_SNORM);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_TYPE(RGBA16);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_TYPE(RGBA16_SNORM);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_TYPE(RGBA8UI);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_TYPE(RGBA16UI);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_TYPE(RGBA32UI);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_TYPE(RGBA8I);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_TYPE(RGBA16I);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_TYPE(RGBA32I);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_TYPE(RGBA16F);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_TYPE(RGBA32F);
                        /*CUSTOM*/
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_TYPE(R3_G3_B2);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_TYPE(RGB4);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_TYPE(RGB5);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_TYPE(SRGB8);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_TYPE(SRGB8_ALPHA8);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_TYPE(RGB10);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_TYPE(RGB12);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_TYPE(R11F_G11F_B10F);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_TYPE(RGB9_E5);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_TYPE(RGBA2);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_TYPE(RGBA4);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_TYPE(RGB5_A1);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_TYPE(RGB10_A2);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_TYPE(RGB10_A2UI);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_TYPE(RGBA12);
                        /*DEPTH*/
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_TYPE(DEPTH_COMPONENT16);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_TYPE(DEPTH_COMPONENT24);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_TYPE(DEPTH_COMPONENT32F);
                        /*DEPTH_STENCIL*/
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_TYPE(DEPTH24_STENCIL8);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_TYPE(DEPTH32F_STENCIL8);
                    default:
                        return GL_RGBA;
                    }
                }
                inline constexpr auto GetGLTypeSize(GLenum type)->size_t {
                    switch (type)
                    {
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_TYPE_SIZE(UNSIGNED_BYTE);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_TYPE_SIZE(BYTE);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_TYPE_SIZE(UNSIGNED_SHORT);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_TYPE_SIZE(SHORT);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_TYPE_SIZE(UNSIGNED_INT);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_TYPE_SIZE(INT);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_TYPE_SIZE(HALF_FLOAT);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_TYPE_SIZE(FLOAT);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_TYPE_SIZE(UNSIGNED_BYTE_3_3_2);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_TYPE_SIZE(UNSIGNED_BYTE_2_3_3_REV);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_TYPE_SIZE(UNSIGNED_SHORT_5_6_5);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_TYPE_SIZE(UNSIGNED_SHORT_5_6_5_REV);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_TYPE_SIZE(UNSIGNED_SHORT_4_4_4_4);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_TYPE_SIZE(UNSIGNED_SHORT_4_4_4_4_REV);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_TYPE_SIZE(UNSIGNED_SHORT_5_5_5_1);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_TYPE_SIZE(UNSIGNED_SHORT_1_5_5_5_REV);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_TYPE_SIZE(UNSIGNED_INT_8_8_8_8);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_TYPE_SIZE(UNSIGNED_INT_8_8_8_8_REV);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_TYPE_SIZE(UNSIGNED_INT_10_10_10_2);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_TYPE_SIZE(UNSIGNED_INT_2_10_10_10_REV);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_TYPE_SIZE(UNSIGNED_INT_5_9_9_9_REV);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_TYPE_SIZE(UNSIGNED_INT_10F_11F_11F_REV);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_TYPE_SIZE(FLOAT_32_UNSIGNED_INT_24_8_REV);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_TYPE_SIZE(UNSIGNED_INT_24_8);
                    default: return 0;
                    }
                }
                inline constexpr auto GetGLFormatTypeInfo(GLenum format, GLenum baseType) -> GLFormatTypeInfo {
                    size_t cnt = 0;
                    switch (format) {
                        /*R*/
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_TYPE_INFO(R8, baseType, cnt);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_TYPE_INFO(R8_SNORM, baseType, cnt);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_TYPE_INFO(R16, baseType, cnt);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_TYPE_INFO(R16_SNORM, baseType, cnt);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_TYPE_INFO(R8UI, baseType, cnt);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_TYPE_INFO(R16UI, baseType, cnt);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_TYPE_INFO(R32UI, baseType, cnt);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_TYPE_INFO(R8I, baseType, cnt);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_TYPE_INFO(R16I, baseType, cnt);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_TYPE_INFO(R32I, baseType, cnt);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_TYPE_INFO(R16F, baseType, cnt);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_TYPE_INFO(R32F, baseType, cnt);
                        /*RG*/
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_TYPE_INFO(RG8, baseType, cnt);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_TYPE_INFO(RG8_SNORM, baseType, cnt);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_TYPE_INFO(RG16, baseType, cnt);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_TYPE_INFO(RG16_SNORM, baseType, cnt);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_TYPE_INFO(RG8UI, baseType, cnt);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_TYPE_INFO(RG16UI, baseType, cnt);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_TYPE_INFO(RG32UI, baseType, cnt);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_TYPE_INFO(RG8I, baseType, cnt);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_TYPE_INFO(RG16I, baseType, cnt);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_TYPE_INFO(RG32I, baseType, cnt);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_TYPE_INFO(RG16F, baseType, cnt);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_TYPE_INFO(RG32F, baseType, cnt);
                        /*RGB*/
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_TYPE_INFO(RGB8, baseType, cnt);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_TYPE_INFO(RGB8_SNORM, baseType, cnt);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_TYPE_INFO(RGB16, baseType, cnt);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_TYPE_INFO(RGB16_SNORM, baseType, cnt);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_TYPE_INFO(RGB8UI, baseType, cnt);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_TYPE_INFO(RGB16UI, baseType, cnt);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_TYPE_INFO(RGB32UI, baseType, cnt);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_TYPE_INFO(RGB8I, baseType, cnt);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_TYPE_INFO(RGB16I, baseType, cnt);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_TYPE_INFO(RGB32I, baseType, cnt);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_TYPE_INFO(RGB16F, baseType, cnt);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_TYPE_INFO(RGB32F, baseType, cnt);
                        /*RGBA*/
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_TYPE_INFO(RGBA8, baseType, cnt);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_TYPE_INFO(RGBA8_SNORM, baseType, cnt);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_TYPE_INFO(RGBA16, baseType, cnt);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_TYPE_INFO(RGBA16_SNORM, baseType, cnt);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_TYPE_INFO(RGBA8UI, baseType, cnt);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_TYPE_INFO(RGBA16UI, baseType, cnt);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_TYPE_INFO(RGBA32UI, baseType, cnt);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_TYPE_INFO(RGBA8I, baseType, cnt);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_TYPE_INFO(RGBA16I, baseType, cnt);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_TYPE_INFO(RGBA32I, baseType, cnt);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_TYPE_INFO(RGBA16F, baseType, cnt);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_TYPE_INFO(RGBA32F, baseType, cnt);
                        /*CUSTOM*/
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_TYPE_INFO(R3_G3_B2, baseType, cnt);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_TYPE_INFO(RGB4, baseType, cnt);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_TYPE_INFO(RGB5, baseType, cnt);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_TYPE_INFO(SRGB8, baseType, cnt);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_TYPE_INFO(SRGB8_ALPHA8, baseType, cnt);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_TYPE_INFO(RGB10, baseType, cnt);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_TYPE_INFO(RGB12, baseType, cnt);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_TYPE_INFO(R11F_G11F_B10F, baseType, cnt);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_TYPE_INFO(RGB9_E5, baseType, cnt);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_TYPE_INFO(RGBA2, baseType, cnt);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_TYPE_INFO(RGBA4, baseType, cnt);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_TYPE_INFO(RGB5_A1, baseType, cnt);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_TYPE_INFO(RGB10_A2, baseType, cnt);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_TYPE_INFO(RGB10_A2UI, baseType, cnt);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_TYPE_INFO(RGBA12, baseType, cnt);
                        /*DEPTH*/
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_TYPE_INFO(DEPTH_COMPONENT16, baseType, cnt);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_TYPE_INFO(DEPTH_COMPONENT24, baseType, cnt);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_TYPE_INFO(DEPTH_COMPONENT32F, baseType, cnt);
                        /*DEPTH_STENCIL*/
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_TYPE_INFO(DEPTH24_STENCIL8, baseType, cnt);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_TYPE_INFO(DEPTH32F_STENCIL8, baseType, cnt);

                    default: return {};
                    }
                }
                inline constexpr auto GetGLFormatTypeSize(GLenum format, GLenum baseType) -> size_t {
                    return GetGLFormatTypeInfo(format, baseType).num_bases * GetGLTypeSize(baseType);
                }
            }
        }
    }
}
#undef RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_GL_FORMAT_INFO_DEF_RED
#undef RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_GL_FORMAT_INFO_DEF_RG
#undef RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_GL_FORMAT_INFO_DEF_RGB
#undef RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_GL_FORMAT_INFO_DEF_RGBA
#undef RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_GL_FORMAT_INFO_DEF_INT_RED
#undef RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_GL_FORMAT_INFO_DEF_INT_RG
#undef RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_GL_FORMAT_INFO_DEF_INT_RGB
#undef RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_GL_FORMAT_INFO_DEF_INT_RGBA
#undef RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_GL_FORMAT_INFO_DEF_DEPTH
#undef RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_GL_FORMAT_INFO_DEF_DEPTH_STENCIL
#undef RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_SIZE
#undef RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_FORMAT
#undef RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_TYPE
#undef RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_TYPE_SIZE
#endif