#version 450

layout(location=0) in vec3 inPos;
layout(location=1) in vec3 inColor;
layout(location=2) in vec2 inTexCoord;
layout(location=3) in mat4 inInstanceTranslation;

layout(location=0)out vec3 fragColor;
layout(location=1)out vec2 fragTexCoord;

//std140
layout(binding=0) uniform UBO_MVP{
    mat4 M;
    mat4 V;
    mat4 P;
}uboMVP;

void main(){
    gl_Position=uboMVP.P*uboMVP.V*uboMVP.M*inInstanceTranslation*vec4(inPos,1);
    fragColor=inColor;
    fragTexCoord=inTexCoord;
}
