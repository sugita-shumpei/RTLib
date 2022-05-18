#version 420 core
#extension GL_ARB_separate_shader_objects : enable
layout(location = 0) in vec3  vertexInPosition;
layout(location = 1) in vec3  vertexInColor;
layout(location = 0) out vec3 vertexOutColor;
out gl_PerVertex {
    vec4 gl_Position;
};
void main(){
    gl_Position = vec4(vertexInPosition, 1.0);
    vertexOutColor = vertexInColor;
}
