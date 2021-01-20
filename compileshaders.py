import os

PATH_SHADER_VULKAN = "./shaders"
PATH_SHADER_COMPILE_DST = "./build/bin"

def compileShaders():
    if not os.path.exists(PATH_SHADER_COMPILE_DST):
        os.mkdir(PATH_SHADER_COMPILE_DST)
    shaders = os.listdir(PATH_SHADER_VULKAN)
    for f in shaders:
        os.system('cmd /c "glslc ' + PATH_SHADER_VULKAN+"/" +
                  f+" -o "+PATH_SHADER_COMPILE_DST+"/"+f+".spv" + '"')

if __name__ == "__main__":
    compileShaders()
