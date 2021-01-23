import os
import sys
import getopt

PATH_SHADER_VULKAN = "./shaders"
PATH_SHADER_COMPILE_DST = "./build/bin"

def compileShader(filename):
    os.system('cmd /c "glslc ' + filename + " -o " +
              PATH_SHADER_COMPILE_DST+"/"+os.path.basename(filename)+".spv" + '"')

def processArg(argv):
    if not os.path.exists(PATH_SHADER_COMPILE_DST):
        os.mkdir(PATH_SHADER_COMPILE_DST)
    if len(argv) == 0:
        shaders = os.listdir(PATH_SHADER_VULKAN)
        for f in shaders:
            compileShader(PATH_SHADER_VULKAN+"/" + f)
    else:
        for f in argv:
            compileShader(f)

if __name__ == "__main__":
    processArg(sys.argv[1:])
