#ifndef RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_H
#define RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_H
#include <glad/glad.h>
#include <half.h>
#include <numeric>
#include <string>
#define RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_GL_FORMAT_INFO_DEF_RED(FORMAT, BASE_TYPE, NUM_BASES) \
    template <>                                                                                      \
    struct GLFormatInfo<FORMAT>                                                                   \
    {                                                                                                \
        static inline constexpr GLFormatTypeInfo formats[] = {\
            { FORMAT,GL_RED, BASE_TYPE, NUM_BASES}\
        }; \
        static inline constexpr GLenum base_format = formats[0].base_format; \
        static inline constexpr GLsizei channels = GLBaseFormatInfo<base_format>::channels; \
        static inline constexpr GLsizei size = sizeof(GLTypeInfo<formats[0].base_type>::type) * formats[0].num_bases; \
    }

#define RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_GL_FORMAT_INFO_DEF_INT_RED(FORMAT, BASE_TYPE, NUM_BASES) \
    template <>                                                                                      \
    struct GLFormatInfo<FORMAT>                                                                   \
    {                                                                                                \
        static inline constexpr GLFormatTypeInfo formats[] = {\
            { FORMAT,GL_RED_INTEGER, BASE_TYPE, NUM_BASES}\
        }; \
        static inline constexpr GLenum base_format = formats[0].base_format; \
        static inline constexpr GLsizei channels = GLBaseFormatInfo<base_format>::channels; \
        static inline constexpr GLsizei size = sizeof(GLTypeInfo<formats[0].base_type>::type) * formats[0].num_bases; \
    }

#define RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_GL_FORMAT_INFO_DEF_RG(FORMAT, BASE_TYPE, NUM_BASES) \
    template <>                                                                                      \
    struct GLFormatInfo<FORMAT>                                                                   \
    {                                                                                                \
        static inline constexpr GLFormatTypeInfo formats[] = {\
            {FORMAT, GL_RG, BASE_TYPE, NUM_BASES}\
        }; \
        static inline constexpr GLenum base_format = formats[0].base_format; \
        static inline constexpr GLsizei channels = GLBaseFormatInfo<base_format>::channels; \
        static inline constexpr GLsizei size = sizeof(GLTypeInfo<formats[0].base_type>::type) * formats[0].num_bases; \
    }

#define RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_GL_FORMAT_INFO_DEF_INT_RG(FORMAT, BASE_TYPE, NUM_BASES) \
    template <>                                                                                      \
    struct GLFormatInfo<FORMAT>                                                                   \
    {                                                                                                \
        static inline constexpr GLFormatTypeInfo formats[] = {\
            {FORMAT, GL_RG_INTEGER, BASE_TYPE, NUM_BASES}\
        }; \
        static inline constexpr GLenum base_format = formats[0].base_format; \
        static inline constexpr GLsizei channels = GLBaseFormatInfo<base_format>::channels; \
        static inline constexpr GLsizei size = sizeof(GLTypeInfo<formats[0].base_type>::type) * formats[0].num_bases; \
    }

#define RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_GL_FORMAT_INFO_DEF_RGB(FORMAT, BASE_TYPE, NUM_BASES) \
    template <>                                                                                      \
    struct GLFormatInfo<FORMAT>                                                                   \
    {                                                                                                \
        static inline constexpr GLFormatTypeInfo formats[] = {\
            {FORMAT, GL_RGB, BASE_TYPE, NUM_BASES}, \
            {FORMAT, GL_BGR, BASE_TYPE, NUM_BASES}  \
        }; \
        static inline constexpr GLenum base_format = formats[0].base_format; \
        static inline constexpr GLsizei channels = GLBaseFormatInfo<base_format>::channels; \
        static inline constexpr GLsizei size = sizeof(GLTypeInfo<formats[0].base_type>::type) * formats[0].num_bases; \
    }

#define RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_GL_FORMAT_INFO_DEF_INT_RGB(FORMAT, BASE_TYPE, NUM_BASES) \
    template <>                                                                                      \
    struct GLFormatInfo<FORMAT>                                                                   \
    {                                                                                                \
        static inline constexpr GLFormatTypeInfo formats[] = {\
            {FORMAT, GL_RGB_INTEGER, BASE_TYPE, NUM_BASES}, \
            {FORMAT, GL_BGR_INTEGER, BASE_TYPE, NUM_BASES}  \
        }; \
        static inline constexpr GLenum base_format = formats[0].base_format; \
        static inline constexpr GLsizei channels = GLBaseFormatInfo<base_format>::channels; \
        static inline constexpr GLsizei size = sizeof(GLTypeInfo<formats[0].base_type>::type) * formats[0].num_bases; \
    }

#define RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_GL_FORMAT_INFO_DEF_RGBA(FORMAT, BASE_TYPE, NUM_BASES) \
    template <>                                                                                      \
    struct GLFormatInfo<FORMAT>                                                                   \
    {                                                                                                \
        static inline constexpr GLFormatTypeInfo formats[] = {\
            {FORMAT,GL_RGBA, BASE_TYPE, NUM_BASES}, \
            {FORMAT,GL_BGRA, BASE_TYPE, NUM_BASES}  \
        }; \
        static inline constexpr GLenum base_format = formats[0].base_format; \
        static inline constexpr GLsizei channels = GLBaseFormatInfo<base_format>::channels; \
        static inline constexpr GLsizei size = sizeof(GLTypeInfo<formats[0].base_type>::type) * formats[0].num_bases; \
    }

#define RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_GL_FORMAT_INFO_DEF_INT_RGBA(FORMAT, BASE_TYPE, NUM_BASES) \
    template <>                                                                                      \
    struct GLFormatInfo<FORMAT>                                                                   \
    {                                                                                                \
        static inline constexpr GLFormatTypeInfo formats[] = {\
            {FORMAT,GL_RGBA_INTEGER, BASE_TYPE, NUM_BASES}, \
            {FORMAT,GL_BGRA_INTEGER, BASE_TYPE, NUM_BASES}  \
        }; \
        static inline constexpr GLenum base_format = formats[0].base_format; \
        static inline constexpr GLsizei channels = GLBaseFormatInfo<base_format>::channels; \
        static inline constexpr GLsizei size = sizeof(GLTypeInfo<formats[0].base_type>::type) * formats[0].num_bases; \
    }

#define RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_GL_FORMAT_INFO_DEF_DEPTH(FORMAT, BASE_TYPE, NUM_BASES) \
    template <>                                                                                      \
    struct GLFormatInfo<FORMAT>                                                                   \
    {                                                                                                \
        static inline constexpr GLFormatTypeInfo formats[] = {\
            {FORMAT,GL_DEPTH_COMPONENT, BASE_TYPE, NUM_BASES}\
        }; \
        static inline constexpr GLenum base_format = formats[0].base_format; \
        static inline constexpr GLsizei channels = GLBaseFormatInfo<base_format>::channels; \
        static inline constexpr GLsizei size = sizeof(GLTypeInfo<formats[0].base_type>::type) * formats[0].num_bases; \
    }

#define RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_GL_FORMAT_INFO_DEF_DEPTH_STENCIL(FORMAT, BASE_TYPE, NUM_BASES) \
    template <>                                                                                      \
    struct GLFormatInfo<FORMAT>                                                                   \
    {                                                                                                \
        static inline constexpr GLFormatTypeInfo formats[] = {\
            {FORMAT,GL_DEPTH_STENCIL, BASE_TYPE, NUM_BASES}\
        }; \
        static inline constexpr GLenum base_format = formats[0].base_format; \
        static inline constexpr GLsizei channels = GLBaseFormatInfo<base_format>::channels; \
        static inline constexpr GLsizei size = sizeof(GLTypeInfo<formats[0].base_type>::type) * formats[0].num_bases; \
    }

#define RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_SIZE(FORMAT) \
    case FORMAT:                                                         \
        return GLFormatInfo<FORMAT>::size

#define RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_TYPE_INFO(FORMAT, BASE_TYPE, CNT) \
    case FORMAT: \
        for (CNT = 0; CNT<sizeof(GLFormatInfo<FORMAT>::formats)/sizeof(GLFormatInfo<FORMAT>::formats[0]); ++CNT) { \
            if (GLFormatInfo<FORMAT>::formats[CNT].base_type == BASE_TYPE){ \
                return GLFormatInfo<FORMAT>::formats[CNT]; \
            } \
        } \
        return {}

#define RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_NUM_BASES(FORMAT, BASE_FORMAT, BASE_TYPE, CNT) \
    case FORMAT: \
        for (CNT = 0; CNT<sizeof(GLFormatInfo<FORMAT>::formats)/sizeof(GLFormatInfo<FORMAT>::formats[0]); ++CNT) { \
            if (GLFormatInfo<FORMAT>::formats[CNT].base_type == BASE_TYPE && GLFormatInfo<FORMAT>::formats[CNT].base_format == BASE_FORMAT){ \
                return GLFormatInfo<FORMAT>::formats[CNT].num_bases; \
            } \
        } \
        return 0

#define RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_FORMAT(FORMAT) \
    case FORMAT:                                                         \
        return GLFormatInfo<FORMAT>::formats[0].base_format

#define RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_TYPE(FORMAT)   \
    case FORMAT:                                                         \
        return GLFormatInfo<FORMAT>::formats[0].base_type

#define RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_TYPE_SIZE(TYPE) \
    case TYPE:                                                  \
        return sizeof(GLTypeInfo<TYPE>::type)

#define RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_TO_STRING(ENUM, STRING)  \
    case ENUM:                                                  \
        return STRING
#define RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_CONVERT_SHADER_TYPE_TO_SHADER_STAGE(SHADER_TYPE) \
    case SHADER_TYPE: \
        return SHADER_TYPE##_BIT

#define RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_CONVERT_SHADER_STAGE_TO_SHADER_TYPE(SHADER_TYPE) \
    case SHADER_TYPE##_BIT: \
        return SHADER_TYPE

namespace RTLib
{
    namespace Ext
    {
        namespace GL
        {

            namespace Internal
            {
                inline auto ToString(GLenum glEnum) -> std::string {
                    switch (glEnum) {
                        /*GLShader*/
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_TO_STRING(GL_COMPUTE_SHADER, "ComputeShader");
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_TO_STRING(GL_VERTEX_SHADER  , "VertexShader");
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_TO_STRING(GL_GEOMETRY_SHADER, "GeometryShader");
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_TO_STRING(GL_TESS_CONTROL_SHADER, "TessControlShader");
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_TO_STRING(GL_TESS_EVALUATION_SHADER, "TessEvaluationShader");
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_TO_STRING(GL_FRAGMENT_SHADER, "FragmentShader");
                        /*GLUsage*/
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_TO_STRING(GL_STATIC_DRAW, "StaticDraw");
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_TO_STRING(GL_STATIC_READ, "StaticRead");
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_TO_STRING(GL_STATIC_COPY, "StaticCopy");
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_TO_STRING(GL_DYNAMIC_DRAW,"DynamicDraw");
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_TO_STRING(GL_DYNAMIC_READ,"DynamicRead");
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_TO_STRING(GL_DYNAMIC_COPY,"DynamicCopy");
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_TO_STRING(GL_STREAM_DRAW, "StreamDraw");
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_TO_STRING(GL_STREAM_READ, "StreamRead");
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_TO_STRING(GL_STREAM_COPY, "StreamCopy");
                        /*GL_BUFFER*/
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_TO_STRING(GL_ARRAY_BUFFER, "ArrayBuffer");
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_TO_STRING(GL_ELEMENT_ARRAY_BUFFER, "ElementArrayBuffer");
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_TO_STRING(GL_ATOMIC_COUNTER_BUFFER, "AtomicCounterBuffer");
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_TO_STRING(GL_COPY_READ_BUFFER, "CopyReadBuffer");
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_TO_STRING(GL_COPY_WRITE_BUFFER, "CopyWriteBuffer");
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_TO_STRING(GL_DISPATCH_INDIRECT_BUFFER, "DispatchIndirectBuffer");
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_TO_STRING(GL_DRAW_INDIRECT_BUFFER, "DrawIndirectBuffer");
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_TO_STRING(GL_PIXEL_UNPACK_BUFFER, "PixelUnpackBuffer");
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_TO_STRING(GL_PIXEL_PACK_BUFFER, "PixelPackBuffer");
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_TO_STRING(GL_QUERY_BUFFER, "QueryBuffer");
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_TO_STRING(GL_TEXTURE_BUFFER, "TextureBuffer");
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_TO_STRING(GL_TRANSFORM_FEEDBACK_BUFFER, "TransformFeedbackBuffer");
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_TO_STRING(GL_SHADER_STORAGE_BUFFER, "ShaderStorageBuffer");
                        /*GL_TEXTURE*/
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_TO_STRING(GL_TEXTURE_1D, "Texture1D");
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_TO_STRING(GL_TEXTURE_2D, "Texture2D");
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_TO_STRING(GL_TEXTURE_3D, "Texture3D");
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_TO_STRING(GL_TEXTURE_1D_ARRAY, "Texture1DArray");
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_TO_STRING(GL_TEXTURE_2D_ARRAY, "Texture2DArray");
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_TO_STRING(GL_TEXTURE_RECTANGLE,"TextureRectangle");
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_TO_STRING(GL_TEXTURE_2D_MULTISAMPLE, "Texture2DMultisample");
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_TO_STRING(GL_TEXTURE_2D_MULTISAMPLE_ARRAY, "Texture2DMultisampleArray");
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_TO_STRING(GL_TEXTURE_CUBE_MAP, "TextureCubemap");
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_TO_STRING(GL_TEXTURE_CUBE_MAP_ARRAY, "TextureCubemapArray");
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_TO_STRING(GL_TEXTURE_CUBE_MAP_POSITIVE_X, "TextureCubemapPositiveX");
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_TO_STRING(GL_TEXTURE_CUBE_MAP_NEGATIVE_X, "TextureCubemapNegativeX");
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_TO_STRING(GL_TEXTURE_CUBE_MAP_POSITIVE_Y, "TextureCubemapPositiveY");
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_TO_STRING(GL_TEXTURE_CUBE_MAP_NEGATIVE_Y, "TextureCubemapNegativeY");
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_TO_STRING(GL_TEXTURE_CUBE_MAP_POSITIVE_Z, "TextureCubemapPositiveZ");
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_TO_STRING(GL_TEXTURE_CUBE_MAP_NEGATIVE_Z, "TextureCubemapNegativeZ");
                        /*GL_FRAMEBUFFER*/
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_TO_STRING(GL_FRAMEBUFFER, "Framebuffer");
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_TO_STRING(GL_DRAW_FRAMEBUFFER, "DrawFramebuffer");
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_TO_STRING(GL_READ_FRAMEBUFFER, "ReadFramebuffer");
                        /*GL_PROGRAM_PIPELINE*/
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_TO_STRING(GL_PROGRAM_PIPELINE, "ProgramPipeline");
                    default:
                        return "Undefined";
                    }
                }

                inline constexpr auto ConvertShaderType2ShaderStage(GLenum shaderType)->GLbitfield {
                    switch (shaderType) {
                    RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_CONVERT_SHADER_TYPE_TO_SHADER_STAGE(GL_VERTEX_SHADER);
                    RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_CONVERT_SHADER_TYPE_TO_SHADER_STAGE(GL_FRAGMENT_SHADER);
                    RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_CONVERT_SHADER_TYPE_TO_SHADER_STAGE(GL_GEOMETRY_SHADER);
                    RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_CONVERT_SHADER_TYPE_TO_SHADER_STAGE(GL_TESS_EVALUATION_SHADER);
                    RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_CONVERT_SHADER_TYPE_TO_SHADER_STAGE(GL_TESS_CONTROL_SHADER);
                    RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_CONVERT_SHADER_TYPE_TO_SHADER_STAGE(GL_COMPUTE_SHADER);
                    default:
                        return 0;
                    }
                }

                inline constexpr auto ConvertShaderStage2ShaderType(GLbitfield shaderStage)->GLenum {
                    switch (shaderStage) {
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_CONVERT_SHADER_STAGE_TO_SHADER_TYPE(GL_VERTEX_SHADER);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_CONVERT_SHADER_STAGE_TO_SHADER_TYPE(GL_FRAGMENT_SHADER);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_CONVERT_SHADER_STAGE_TO_SHADER_TYPE(GL_GEOMETRY_SHADER);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_CONVERT_SHADER_STAGE_TO_SHADER_TYPE(GL_TESS_EVALUATION_SHADER);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_CONVERT_SHADER_STAGE_TO_SHADER_TYPE(GL_TESS_CONTROL_SHADER);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_CONVERT_SHADER_STAGE_TO_SHADER_TYPE(GL_COMPUTE_SHADER);
                    default:
                        return 0;
                    }
                }

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
                RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_GL_FORMAT_INFO_DEF_RED(GL_R8, GL_UNSIGNED_BYTE, 1);
                RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_GL_FORMAT_INFO_DEF_RED(GL_R8_SNORM, GL_BYTE, 1);
                RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_GL_FORMAT_INFO_DEF_RED(GL_R16, GL_UNSIGNED_SHORT, 1);
                RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_GL_FORMAT_INFO_DEF_RED(GL_R16_SNORM, GL_SHORT, 1);
                //RG
                RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_GL_FORMAT_INFO_DEF_RG(GL_RG8, GL_UNSIGNED_BYTE, 2);
                RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_GL_FORMAT_INFO_DEF_RG(GL_RG8_SNORM, GL_BYTE, 2);
                RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_GL_FORMAT_INFO_DEF_RG(GL_RG16, GL_UNSIGNED_SHORT, 2);
                RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_GL_FORMAT_INFO_DEF_RG(GL_RG16_SNORM, GL_SHORT, 2);
                //RGB
                RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_GL_FORMAT_INFO_DEF_RGB(GL_RGB8, GL_UNSIGNED_BYTE, 3);
                RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_GL_FORMAT_INFO_DEF_RGB(GL_RGB8_SNORM, GL_BYTE, 3);
                RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_GL_FORMAT_INFO_DEF_RGB(GL_RGB16, GL_UNSIGNED_SHORT, 3);
                RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_GL_FORMAT_INFO_DEF_RGB(GL_RGB16_SNORM, GL_SHORT, 3);
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
                RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_GL_FORMAT_INFO_DEF_RGBA(GL_RGBA8_SNORM, GL_BYTE, 4);
                RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_GL_FORMAT_INFO_DEF_RGBA(GL_RGBA16, GL_UNSIGNED_SHORT, 4);
                RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_GL_FORMAT_INFO_DEF_RGBA(GL_RGBA16_SNORM, GL_SHORT, 4);

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
                RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_GL_FORMAT_INFO_DEF_RGB(GL_RGB4, GL_UNSIGNED_BYTE, 3);
                RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_GL_FORMAT_INFO_DEF_RGB(GL_RGB5, GL_UNSIGNED_BYTE, 3);
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
                RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_GL_FORMAT_INFO_DEF_RGB(GL_R11F_G11F_B10F, GL_UNSIGNED_INT_10F_11F_11F_REV, 1);
                RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_GL_FORMAT_INFO_DEF_RGB(GL_RGB9_E5, GL_UNSIGNED_INT_5_9_9_9_REV, 1);
                RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_GL_FORMAT_INFO_DEF_RGBA(GL_RGBA2, GL_UNSIGNED_BYTE, 4);
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
                RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_GL_FORMAT_INFO_DEF_RGBA(GL_RGBA12, GL_UNSIGNED_SHORT, 4);
                //FLOAT
                RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_GL_FORMAT_INFO_DEF_RED(GL_R16F, GL_HALF_FLOAT, 1);
                RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_GL_FORMAT_INFO_DEF_RED(GL_R32F, GL_FLOAT, 1);
                RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_GL_FORMAT_INFO_DEF_RG(GL_RG16F, GL_HALF_FLOAT, 2);
                RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_GL_FORMAT_INFO_DEF_RG(GL_RG32F, GL_FLOAT, 2);
                RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_GL_FORMAT_INFO_DEF_RGB(GL_RGB16F, GL_HALF_FLOAT, 3);
                RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_GL_FORMAT_INFO_DEF_RGB(GL_RGB32F, GL_FLOAT, 3);
                RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_GL_FORMAT_INFO_DEF_RGBA(GL_RGBA16F, GL_HALF_FLOAT, 4);
                RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_GL_FORMAT_INFO_DEF_RGBA(GL_RGBA32F, GL_FLOAT, 4);
                // RED
                RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_GL_FORMAT_INFO_DEF_INT_RG(GL_R8I, GL_BYTE, 1);
                RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_GL_FORMAT_INFO_DEF_INT_RG(GL_R8UI, GL_UNSIGNED_BYTE, 1);
                RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_GL_FORMAT_INFO_DEF_INT_RG(GL_R16I, GL_SHORT, 1);
                RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_GL_FORMAT_INFO_DEF_INT_RG(GL_R16UI, GL_UNSIGNED_SHORT, 1);
                RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_GL_FORMAT_INFO_DEF_INT_RG(GL_R32I, GL_INT, 1);
                RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_GL_FORMAT_INFO_DEF_INT_RG(GL_R32UI, GL_UNSIGNED_INT, 1);
                // RG
                RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_GL_FORMAT_INFO_DEF_INT_RG(GL_RG8I, GL_BYTE, 2);
                RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_GL_FORMAT_INFO_DEF_INT_RG(GL_RG8UI, GL_UNSIGNED_BYTE, 2);
                RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_GL_FORMAT_INFO_DEF_INT_RG(GL_RG16I, GL_SHORT, 2);
                RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_GL_FORMAT_INFO_DEF_INT_RG(GL_RG16UI, GL_UNSIGNED_SHORT, 2);
                RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_GL_FORMAT_INFO_DEF_INT_RG(GL_RG32I, GL_INT, 2);
                RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_GL_FORMAT_INFO_DEF_INT_RG(GL_RG32UI, GL_UNSIGNED_INT, 2);
                // RGB
                RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_GL_FORMAT_INFO_DEF_INT_RGB(GL_RGB8I, GL_BYTE, 3);
                RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_GL_FORMAT_INFO_DEF_INT_RGB(GL_RGB8UI, GL_UNSIGNED_BYTE, 3);
                RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_GL_FORMAT_INFO_DEF_INT_RGB(GL_RGB16I, GL_SHORT, 3);
                RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_GL_FORMAT_INFO_DEF_INT_RGB(GL_RGB16UI, GL_UNSIGNED_SHORT, 3);
                RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_GL_FORMAT_INFO_DEF_INT_RGB(GL_RGB32I, GL_INT, 3);
                RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_GL_FORMAT_INFO_DEF_INT_RGB(GL_RGB32UI, GL_UNSIGNED_INT, 3);
                // RGBA
                RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_GL_FORMAT_INFO_DEF_INT_RGBA(GL_RGBA8I, GL_BYTE, 4);
                RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_GL_FORMAT_INFO_DEF_INT_RGBA(GL_RGBA8UI, GL_UNSIGNED_BYTE, 3);
                RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_GL_FORMAT_INFO_DEF_INT_RGBA(GL_RGBA16I, GL_SHORT, 4);
                RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_GL_FORMAT_INFO_DEF_INT_RGBA(GL_RGBA16UI, GL_UNSIGNED_SHORT, 4);
                RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_GL_FORMAT_INFO_DEF_INT_RGBA(GL_RGBA32I, GL_INT, 4);
                RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_GL_FORMAT_INFO_DEF_INT_RGBA(GL_RGBA32UI, GL_UNSIGNED_INT, 4);
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
                RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_GL_FORMAT_INFO_DEF_DEPTH(GL_DEPTH_COMPONENT24, GL_UNSIGNED_INT, 1);
                RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_GL_FORMAT_INFO_DEF_DEPTH(GL_DEPTH_COMPONENT32F, GL_FLOAT, 1);
                RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_GL_FORMAT_INFO_DEF_DEPTH_STENCIL(GL_DEPTH24_STENCIL8, GL_UNSIGNED_INT_24_8, 1);
                RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_GL_FORMAT_INFO_DEF_DEPTH_STENCIL(GL_DEPTH32F_STENCIL8, GL_FLOAT_32_UNSIGNED_INT_24_8_REV, 1);

                inline constexpr auto GetGLFormatSize(GLenum format) -> size_t
                {
                    switch (format)
                    {
                        /*R*/
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_SIZE(GL_R8);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_SIZE(GL_R8_SNORM);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_SIZE(GL_R16);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_SIZE(GL_R16_SNORM);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_SIZE(GL_R8UI);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_SIZE(GL_R16UI);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_SIZE(GL_R32UI);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_SIZE(GL_R8I);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_SIZE(GL_R16I);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_SIZE(GL_R32I);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_SIZE(GL_R16F);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_SIZE(GL_R32F);
                        /*RG*/
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_SIZE(GL_RG8);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_SIZE(GL_RG8_SNORM);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_SIZE(GL_RG16);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_SIZE(GL_RG16_SNORM);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_SIZE(GL_RG8UI);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_SIZE(GL_RG16UI);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_SIZE(GL_RG32UI);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_SIZE(GL_RG8I);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_SIZE(GL_RG16I);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_SIZE(GL_RG32I);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_SIZE(GL_RG16F);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_SIZE(GL_RG32F);
                        /*RGB*/
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_SIZE(GL_RGB8);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_SIZE(GL_RGB8_SNORM);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_SIZE(GL_RGB16);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_SIZE(GL_RGB16_SNORM);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_SIZE(GL_RGB8UI);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_SIZE(GL_RGB16UI);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_SIZE(GL_RGB32UI);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_SIZE(GL_RGB8I);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_SIZE(GL_RGB16I);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_SIZE(GL_RGB32I);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_SIZE(GL_RGB16F);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_SIZE(GL_RGB32F);
                        /*RGBA*/
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_SIZE(GL_RGBA8);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_SIZE(GL_RGBA8_SNORM);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_SIZE(GL_RGBA16);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_SIZE(GL_RGBA16_SNORM);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_SIZE(GL_RGBA8UI);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_SIZE(GL_RGBA16UI);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_SIZE(GL_RGBA32UI);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_SIZE(GL_RGBA8I);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_SIZE(GL_RGBA16I);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_SIZE(GL_RGBA32I);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_SIZE(GL_RGBA16F);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_SIZE(GL_RGBA32F);
                        /*CUSTOM*/
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_SIZE(GL_R3_G3_B2);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_SIZE(GL_RGB4);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_SIZE(GL_RGB5);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_SIZE(GL_SRGB8);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_SIZE(GL_SRGB8_ALPHA8);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_SIZE(GL_RGB10);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_SIZE(GL_RGB12);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_SIZE(GL_R11F_G11F_B10F);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_SIZE(GL_RGB9_E5);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_SIZE(GL_RGBA2);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_SIZE(GL_RGBA4);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_SIZE(GL_RGB5_A1);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_SIZE(GL_RGB10_A2);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_SIZE(GL_RGB10_A2UI);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_SIZE(GL_RGBA12);
                        /*DEPTH*/
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_SIZE(GL_DEPTH_COMPONENT16);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_SIZE(GL_DEPTH_COMPONENT24);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_SIZE(GL_DEPTH_COMPONENT32F);
                        /*DEPTH_STENCIL*/
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_SIZE(GL_DEPTH24_STENCIL8);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_SIZE(GL_DEPTH32F_STENCIL8);
                    default:
                        return 0;
                    }
                }
                inline constexpr auto GetGLBaseFormat(GLenum format) -> GLenum
                {
                    switch (format)
                    {
                        /*R*/
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_FORMAT(GL_R8);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_FORMAT(GL_R8_SNORM);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_FORMAT(GL_R16);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_FORMAT(GL_R16_SNORM);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_FORMAT(GL_R8UI);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_FORMAT(GL_R16UI);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_FORMAT(GL_R32UI);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_FORMAT(GL_R8I);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_FORMAT(GL_R16I);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_FORMAT(GL_R32I);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_FORMAT(GL_R16F);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_FORMAT(GL_R32F);
                        /*RG*/
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_FORMAT(GL_RG8);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_FORMAT(GL_RG8_SNORM);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_FORMAT(GL_RG16);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_FORMAT(GL_RG16_SNORM);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_FORMAT(GL_RG8UI);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_FORMAT(GL_RG16UI);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_FORMAT(GL_RG32UI);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_FORMAT(GL_RG8I);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_FORMAT(GL_RG16I);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_FORMAT(GL_RG32I);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_FORMAT(GL_RG16F);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_FORMAT(GL_RG32F);
                        /*RGB*/
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_FORMAT(GL_RGB8);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_FORMAT(GL_RGB8_SNORM);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_FORMAT(GL_RGB16);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_FORMAT(GL_RGB16_SNORM);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_FORMAT(GL_RGB8UI);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_FORMAT(GL_RGB16UI);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_FORMAT(GL_RGB32UI);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_FORMAT(GL_RGB8I);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_FORMAT(GL_RGB16I);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_FORMAT(GL_RGB32I);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_FORMAT(GL_RGB16F);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_FORMAT(GL_RGB32F);
                        /*RGBA*/
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_FORMAT(GL_RGBA8);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_FORMAT(GL_RGBA8_SNORM);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_FORMAT(GL_RGBA16);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_FORMAT(GL_RGBA16_SNORM);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_FORMAT(GL_RGBA8UI);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_FORMAT(GL_RGBA16UI);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_FORMAT(GL_RGBA32UI);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_FORMAT(GL_RGBA8I);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_FORMAT(GL_RGBA16I);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_FORMAT(GL_RGBA32I);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_FORMAT(GL_RGBA16F);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_FORMAT(GL_RGBA32F);
                        /*CUSTOM*/
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_FORMAT(GL_R3_G3_B2);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_FORMAT(GL_RGB4);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_FORMAT(GL_RGB5);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_FORMAT(GL_SRGB8);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_FORMAT(GL_SRGB8_ALPHA8);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_FORMAT(GL_RGB10);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_FORMAT(GL_RGB12);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_FORMAT(GL_R11F_G11F_B10F);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_FORMAT(GL_RGB9_E5);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_FORMAT(GL_RGBA2);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_FORMAT(GL_RGBA4);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_FORMAT(GL_RGB5_A1);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_FORMAT(GL_RGB10_A2);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_FORMAT(GL_RGB10_A2UI);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_FORMAT(GL_RGBA12);
                        /*DEPTH*/
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_FORMAT(GL_DEPTH_COMPONENT16);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_FORMAT(GL_DEPTH_COMPONENT24);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_FORMAT(GL_DEPTH_COMPONENT32F);
                        /*DEPTH_STENCIL*/
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_FORMAT(GL_DEPTH24_STENCIL8);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_FORMAT(GL_DEPTH32F_STENCIL8);
                    default:
                        return GL_RGBA;
                    }
                }
                inline constexpr auto GetGLBaseType(GLenum format) -> GLenum
                {
                    switch (format)
                    {
                        /*R*/
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_TYPE(GL_R8);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_TYPE(GL_R8_SNORM);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_TYPE(GL_R16);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_TYPE(GL_R16_SNORM);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_TYPE(GL_R8UI);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_TYPE(GL_R16UI);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_TYPE(GL_R32UI);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_TYPE(GL_R8I);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_TYPE(GL_R16I);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_TYPE(GL_R32I);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_TYPE(GL_R16F);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_TYPE(GL_R32F);
                        /*RG*/
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_TYPE(GL_RG8);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_TYPE(GL_RG8_SNORM);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_TYPE(GL_RG16);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_TYPE(GL_RG16_SNORM);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_TYPE(GL_RG8UI);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_TYPE(GL_RG16UI);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_TYPE(GL_RG32UI);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_TYPE(GL_RG8I);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_TYPE(GL_RG16I);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_TYPE(GL_RG32I);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_TYPE(GL_RG16F);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_TYPE(GL_RG32F);
                        /*RGB*/
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_TYPE(GL_RGB8);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_TYPE(GL_RGB8_SNORM);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_TYPE(GL_RGB16);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_TYPE(GL_RGB16_SNORM);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_TYPE(GL_RGB8UI);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_TYPE(GL_RGB16UI);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_TYPE(GL_RGB32UI);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_TYPE(GL_RGB8I);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_TYPE(GL_RGB16I);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_TYPE(GL_RGB32I);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_TYPE(GL_RGB16F);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_TYPE(GL_RGB32F);
                        /*RGBA*/
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_TYPE(GL_RGBA8);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_TYPE(GL_RGBA8_SNORM);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_TYPE(GL_RGBA16);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_TYPE(GL_RGBA16_SNORM);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_TYPE(GL_RGBA8UI);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_TYPE(GL_RGBA16UI);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_TYPE(GL_RGBA32UI);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_TYPE(GL_RGBA8I);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_TYPE(GL_RGBA16I);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_TYPE(GL_RGBA32I);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_TYPE(GL_RGBA16F);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_TYPE(GL_RGBA32F);
                        /*CUSTOM*/
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_TYPE(GL_R3_G3_B2);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_TYPE(GL_RGB4);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_TYPE(GL_RGB5);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_TYPE(GL_SRGB8);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_TYPE(GL_SRGB8_ALPHA8);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_TYPE(GL_RGB10);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_TYPE(GL_RGB12);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_TYPE(GL_R11F_G11F_B10F);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_TYPE(GL_RGB9_E5);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_TYPE(GL_RGBA2);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_TYPE(GL_RGBA4);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_TYPE(GL_RGB5_A1);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_TYPE(GL_RGB10_A2);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_TYPE(GL_RGB10_A2UI);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_TYPE(GL_RGBA12);
                        /*DEPTH*/
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_TYPE(GL_DEPTH_COMPONENT16);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_TYPE(GL_DEPTH_COMPONENT24);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_TYPE(GL_DEPTH_COMPONENT32F);
                        /*DEPTH_STENCIL*/
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_TYPE(GL_DEPTH24_STENCIL8);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_TYPE(GL_DEPTH32F_STENCIL8);
                    default:
                        return GL_RGBA;
                    }
                }
                inline constexpr auto GetGLTypeSize(GLenum type)->size_t {
                    switch (type)
                    {
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_TYPE_SIZE(GL_UNSIGNED_BYTE);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_TYPE_SIZE(GL_BYTE);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_TYPE_SIZE(GL_UNSIGNED_SHORT);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_TYPE_SIZE(GL_SHORT);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_TYPE_SIZE(GL_UNSIGNED_INT);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_TYPE_SIZE(GL_INT);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_TYPE_SIZE(GL_HALF_FLOAT);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_TYPE_SIZE(GL_FLOAT);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_TYPE_SIZE(GL_UNSIGNED_BYTE_3_3_2);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_TYPE_SIZE(GL_UNSIGNED_BYTE_2_3_3_REV);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_TYPE_SIZE(GL_UNSIGNED_SHORT_5_6_5);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_TYPE_SIZE(GL_UNSIGNED_SHORT_5_6_5_REV);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_TYPE_SIZE(GL_UNSIGNED_SHORT_4_4_4_4);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_TYPE_SIZE(GL_UNSIGNED_SHORT_4_4_4_4_REV);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_TYPE_SIZE(GL_UNSIGNED_SHORT_5_5_5_1);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_TYPE_SIZE(GL_UNSIGNED_SHORT_1_5_5_5_REV);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_TYPE_SIZE(GL_UNSIGNED_INT_8_8_8_8);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_TYPE_SIZE(GL_UNSIGNED_INT_8_8_8_8_REV);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_TYPE_SIZE(GL_UNSIGNED_INT_10_10_10_2);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_TYPE_SIZE(GL_UNSIGNED_INT_2_10_10_10_REV);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_TYPE_SIZE(GL_UNSIGNED_INT_5_9_9_9_REV);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_TYPE_SIZE(GL_UNSIGNED_INT_10F_11F_11F_REV);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_TYPE_SIZE(GL_FLOAT_32_UNSIGNED_INT_24_8_REV);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_TYPE_SIZE(GL_UNSIGNED_INT_24_8);
                    default: return 0;
                    }
                }
                inline constexpr auto GetGLFormatTypeInfo(GLenum format, GLenum baseType) -> GLFormatTypeInfo {
                    size_t cnt = 0;
                    switch (format) {
                        /*R*/
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_TYPE_INFO(GL_R8, baseType, cnt);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_TYPE_INFO(GL_R8_SNORM, baseType, cnt);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_TYPE_INFO(GL_R16, baseType, cnt);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_TYPE_INFO(GL_R16_SNORM, baseType, cnt);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_TYPE_INFO(GL_R8UI, baseType, cnt);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_TYPE_INFO(GL_R16UI, baseType, cnt);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_TYPE_INFO(GL_R32UI, baseType, cnt);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_TYPE_INFO(GL_R8I, baseType, cnt);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_TYPE_INFO(GL_R16I, baseType, cnt);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_TYPE_INFO(GL_R32I, baseType, cnt);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_TYPE_INFO(GL_R16F, baseType, cnt);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_TYPE_INFO(GL_R32F, baseType, cnt);
                        /*RG*/
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_TYPE_INFO(GL_RG8, baseType, cnt);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_TYPE_INFO(GL_RG8_SNORM, baseType, cnt);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_TYPE_INFO(GL_RG16, baseType, cnt);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_TYPE_INFO(GL_RG16_SNORM, baseType, cnt);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_TYPE_INFO(GL_RG8UI, baseType, cnt);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_TYPE_INFO(GL_RG16UI, baseType, cnt);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_TYPE_INFO(GL_RG32UI, baseType, cnt);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_TYPE_INFO(GL_RG8I, baseType, cnt);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_TYPE_INFO(GL_RG16I, baseType, cnt);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_TYPE_INFO(GL_RG32I, baseType, cnt);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_TYPE_INFO(GL_RG16F, baseType, cnt);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_TYPE_INFO(GL_RG32F, baseType, cnt);
                        /*RGB*/
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_TYPE_INFO(GL_RGB8, baseType, cnt);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_TYPE_INFO(GL_RGB8_SNORM, baseType, cnt);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_TYPE_INFO(GL_RGB16, baseType, cnt);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_TYPE_INFO(GL_RGB16_SNORM, baseType, cnt);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_TYPE_INFO(GL_RGB8UI, baseType, cnt);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_TYPE_INFO(GL_RGB16UI, baseType, cnt);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_TYPE_INFO(GL_RGB32UI, baseType, cnt);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_TYPE_INFO(GL_RGB8I, baseType, cnt);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_TYPE_INFO(GL_RGB16I, baseType, cnt);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_TYPE_INFO(GL_RGB32I, baseType, cnt);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_TYPE_INFO(GL_RGB16F, baseType, cnt);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_TYPE_INFO(GL_RGB32F, baseType, cnt);
                        /*RGBA*/
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_TYPE_INFO(GL_RGBA8, baseType, cnt);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_TYPE_INFO(GL_RGBA8_SNORM, baseType, cnt);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_TYPE_INFO(GL_RGBA16, baseType, cnt);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_TYPE_INFO(GL_RGBA16_SNORM, baseType, cnt);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_TYPE_INFO(GL_RGBA8UI, baseType, cnt);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_TYPE_INFO(GL_RGBA16UI, baseType, cnt);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_TYPE_INFO(GL_RGBA32UI, baseType, cnt);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_TYPE_INFO(GL_RGBA8I, baseType, cnt);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_TYPE_INFO(GL_RGBA16I, baseType, cnt);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_TYPE_INFO(GL_RGBA32I, baseType, cnt);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_TYPE_INFO(GL_RGBA16F, baseType, cnt);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_TYPE_INFO(GL_RGBA32F, baseType, cnt);
                        /*CUSTOM*/
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_TYPE_INFO(GL_R3_G3_B2, baseType, cnt);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_TYPE_INFO(GL_RGB4, baseType, cnt);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_TYPE_INFO(GL_RGB5, baseType, cnt);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_TYPE_INFO(GL_SRGB8, baseType, cnt);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_TYPE_INFO(GL_SRGB8_ALPHA8, baseType, cnt);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_TYPE_INFO(GL_RGB10, baseType, cnt);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_TYPE_INFO(GL_RGB12, baseType, cnt);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_TYPE_INFO(GL_R11F_G11F_B10F, baseType, cnt);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_TYPE_INFO(GL_RGB9_E5, baseType, cnt);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_TYPE_INFO(GL_RGBA2, baseType, cnt);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_TYPE_INFO(GL_RGBA4, baseType, cnt);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_TYPE_INFO(GL_RGB5_A1, baseType, cnt);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_TYPE_INFO(GL_RGB10_A2, baseType, cnt);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_TYPE_INFO(GL_RGB10_A2UI, baseType, cnt);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_TYPE_INFO(GL_RGBA12, baseType, cnt);
                        /*DEPTH*/
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_TYPE_INFO(GL_DEPTH_COMPONENT16, baseType, cnt);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_TYPE_INFO(GL_DEPTH_COMPONENT24, baseType, cnt);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_TYPE_INFO(GL_DEPTH_COMPONENT32F, baseType, cnt);
                        /*DEPTH_STENCIL*/
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_TYPE_INFO(GL_DEPTH24_STENCIL8, baseType, cnt);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_TYPE_INFO(GL_DEPTH32F_STENCIL8, baseType, cnt);

                    default: return {};
                    }
                }
                inline constexpr auto GetGLFormatTypeSize(GLenum format, GLenum baseType) -> size_t {
                    return GetGLFormatTypeInfo(format, baseType).num_bases * GetGLTypeSize(baseType);
                }
                inline constexpr auto GetGLNumBases(GLenum format, GLenum baseFormat, GLenum baseType) -> GLsizei {
                    size_t cnt = 0;
                    switch (format) {
                        /*R*/
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_NUM_BASES(GL_R8,baseFormat, baseType, cnt);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_NUM_BASES(GL_R8_SNORM,baseFormat, baseType, cnt);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_NUM_BASES(GL_R16,baseFormat, baseType, cnt);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_NUM_BASES(GL_R16_SNORM,baseFormat, baseType, cnt);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_NUM_BASES(GL_R8UI,baseFormat, baseType, cnt);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_NUM_BASES(GL_R16UI,baseFormat, baseType, cnt);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_NUM_BASES(GL_R32UI,baseFormat, baseType, cnt);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_NUM_BASES(GL_R8I,baseFormat, baseType, cnt);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_NUM_BASES(GL_R16I,baseFormat, baseType, cnt);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_NUM_BASES(GL_R32I,baseFormat, baseType, cnt);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_NUM_BASES(GL_R16F,baseFormat, baseType, cnt);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_NUM_BASES(GL_R32F,baseFormat, baseType, cnt);
                        /*RG*/
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_NUM_BASES(GL_RG8,baseFormat, baseType, cnt);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_NUM_BASES(GL_RG8_SNORM,baseFormat, baseType, cnt);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_NUM_BASES(GL_RG16,baseFormat, baseType, cnt);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_NUM_BASES(GL_RG16_SNORM,baseFormat, baseType, cnt);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_NUM_BASES(GL_RG8UI,baseFormat, baseType, cnt);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_NUM_BASES(GL_RG16UI,baseFormat, baseType, cnt);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_NUM_BASES(GL_RG32UI,baseFormat, baseType, cnt);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_NUM_BASES(GL_RG8I,baseFormat, baseType, cnt);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_NUM_BASES(GL_RG16I,baseFormat, baseType, cnt);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_NUM_BASES(GL_RG32I,baseFormat, baseType, cnt);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_NUM_BASES(GL_RG16F,baseFormat, baseType, cnt);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_NUM_BASES(GL_RG32F,baseFormat, baseType, cnt);
                        /*RGB*/
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_NUM_BASES(GL_RGB8,baseFormat, baseType, cnt);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_NUM_BASES(GL_RGB8_SNORM,baseFormat, baseType, cnt);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_NUM_BASES(GL_RGB16,baseFormat, baseType, cnt);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_NUM_BASES(GL_RGB16_SNORM,baseFormat, baseType, cnt);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_NUM_BASES(GL_RGB8UI,baseFormat, baseType, cnt);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_NUM_BASES(GL_RGB16UI,baseFormat, baseType, cnt);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_NUM_BASES(GL_RGB32UI,baseFormat, baseType, cnt);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_NUM_BASES(GL_RGB8I,baseFormat, baseType, cnt);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_NUM_BASES(GL_RGB16I,baseFormat, baseType, cnt);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_NUM_BASES(GL_RGB32I,baseFormat, baseType, cnt);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_NUM_BASES(GL_RGB16F,baseFormat, baseType, cnt);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_NUM_BASES(GL_RGB32F,baseFormat, baseType, cnt);
                        /*RGBA*/
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_NUM_BASES(GL_RGBA8,baseFormat, baseType, cnt);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_NUM_BASES(GL_RGBA8_SNORM,baseFormat, baseType, cnt);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_NUM_BASES(GL_RGBA16,baseFormat, baseType, cnt);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_NUM_BASES(GL_RGBA16_SNORM,baseFormat, baseType, cnt);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_NUM_BASES(GL_RGBA8UI,baseFormat, baseType, cnt);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_NUM_BASES(GL_RGBA16UI,baseFormat, baseType, cnt);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_NUM_BASES(GL_RGBA32UI,baseFormat, baseType, cnt);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_NUM_BASES(GL_RGBA8I,baseFormat, baseType, cnt);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_NUM_BASES(GL_RGBA16I,baseFormat, baseType, cnt);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_NUM_BASES(GL_RGBA32I,baseFormat, baseType, cnt);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_NUM_BASES(GL_RGBA16F,baseFormat, baseType, cnt);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_NUM_BASES(GL_RGBA32F,baseFormat, baseType, cnt);
                        /*CUSTOM*/
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_NUM_BASES(GL_R3_G3_B2,baseFormat, baseType, cnt);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_NUM_BASES(GL_RGB4,baseFormat, baseType, cnt);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_NUM_BASES(GL_RGB5,baseFormat, baseType, cnt);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_NUM_BASES(GL_SRGB8,baseFormat, baseType, cnt);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_NUM_BASES(GL_SRGB8_ALPHA8,baseFormat, baseType, cnt);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_NUM_BASES(GL_RGB10,baseFormat, baseType, cnt);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_NUM_BASES(GL_RGB12,baseFormat, baseType, cnt);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_NUM_BASES(GL_R11F_G11F_B10F,baseFormat, baseType, cnt);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_NUM_BASES(GL_RGB9_E5,baseFormat, baseType, cnt);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_NUM_BASES(GL_RGBA2,baseFormat, baseType, cnt);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_NUM_BASES(GL_RGBA4,baseFormat, baseType, cnt);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_NUM_BASES(GL_RGB5_A1,baseFormat, baseType, cnt);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_NUM_BASES(GL_RGB10_A2,baseFormat, baseType, cnt);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_NUM_BASES(GL_RGB10_A2UI,baseFormat, baseType, cnt);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_NUM_BASES(GL_RGBA12,baseFormat, baseType, cnt);
                        /*DEPTH*/
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_NUM_BASES(GL_DEPTH_COMPONENT16,baseFormat, baseType, cnt);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_NUM_BASES(GL_DEPTH_COMPONENT24,baseFormat, baseType, cnt);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_NUM_BASES(GL_DEPTH_COMPONENT32F,baseFormat, baseType, cnt);
                        /*DEPTH_STENCIL*/
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_NUM_BASES(GL_DEPTH24_STENCIL8,baseFormat, baseType, cnt);
                        RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_NUM_BASES(GL_DEPTH32F_STENCIL8,baseFormat, baseType, cnt);

                    default: return 0;
                    }
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
#undef RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_CONVERT_SHADER_STAGE_TO_SHADER_TYPE
#undef RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_CONVERT_SHADER_TYPE_TO_SHADER_STAGE
#undef RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_TO_STRING
#undef RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_TYPE_INFO
#undef RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_NUM_BASES
#undef RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_FORMAT_SIZE
#undef RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_FORMAT
#undef RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_BASE_TYPE
#undef RTLIB_EXT_GL_INTERNAL_IMPL_GL_UTILS_MACRO_CASE_GL_TYPE_SIZE
#endif