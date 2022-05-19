#version 460 core
#extension GL_ARB_separate_shader_objects : enable
layout(location = 0) in  vec3 vertexOutColor;
layout(location = 0) out vec4 fragColor;
layout(set = 0, binding = 0) uniform UBO
{
    vec4 color;
} ubo;
void main(){
    fragColor = vec4(vertexOutColor,1.0f);
}
