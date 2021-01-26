#version 450

layout(location=0) in vec3 inPos;
layout(location=1) in vec3 inColor;

layout(location=0)out vec3 fragColor;

//std140
layout(binding=0) uniform UBO_MVP{
    vec2 foo;
    mat4 M;
    mat4 V;
    mat4 P;
}uboMVP;

void main(){
    gl_Position=uboMVP.P*uboMVP.V*uboMVP.M*vec4(inPos,1);
    fragColor=inColor;
}
