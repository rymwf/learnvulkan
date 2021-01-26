#version 450

layout(location=0)in vec3 fragColor;
layout(location=1)in vec2 fragTexCoord;

layout(location=0) out vec4 outColor;

layout(binding=1)uniform sampler2D testTex;

void main(){
//    outColor=vec4(fragColor,1);
outColor=texture(testTex,fragTexCoord);
//outColor=textureLod(testTex,fragTexCoord,5);
//outColor=vec4(fragColor,1)*texture(testTex,fragTexCoord);
}