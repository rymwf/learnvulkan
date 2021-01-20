#version 450

//layout(location=0) in vec3 pos;
//layout(location=1) in vec3 normal;
//struct VS_OUT{
//    vec3 normal;
//};
//
//layout(location=0)out  VS_OUT vs_out;

vec2 temppos[3]={{0,0},{1,0},{0,1}};


void main(){
//    vs_out.normal=normal;
//    gl_Position=vec4(pos,1);
    gl_Position=vec4(temppos[gl_VertexIndex],0,1);    //gl_VertexID in opengl
}
