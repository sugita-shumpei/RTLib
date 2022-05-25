#version 330 core
#extension GL_ARB_separate_shader_objects : enable
layout(location = 0) in  vec3 vertexOutColor;
layout(location = 0) out vec4 fragColor;
layout(std140) uniform UBO
{
    vec4 color;
} ubo;
void main(){
    //fragColor = ubo.color;
    fragColor = vec4(vertexOutColor,1.0f);
}
