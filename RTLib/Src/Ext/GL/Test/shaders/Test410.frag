#version 410 core
#extension GL_ARB_separate_shader_objects : enable
layout(location = 0) in  vec3 vertexOutTexCoord;
layout(location = 1) in  vec3 vertexOutColor;
layout(location = 0) out vec4 fragColor;
layout(std140) uniform UBO
{
    vec4 color;
} ubo;
uniform sampler2D smp;
void main(){
   fragColor = texture(smp,vertexOutTexCoord.xy);
    //fragColor = vec4(vertexOutTexCoord.xy,1.0f- dot(vertexOutTexCoord.xy,vec2(1.0f))/2.0f,1.0f);
}
