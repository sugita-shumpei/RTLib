#version 420 core
#extension GL_ARB_separate_shader_objects : enable
layout(location = 0) in  vec3 vertexOutColor;
layout(location = 0) out vec4 fragColor;
layout(binding = 0,std140) uniform UBO
{
    vec4 color;
} ubo;
void main(){
    fragColor = ubo.color;
}
