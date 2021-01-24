cmake_minimum_required(VERSION 3.16)

project(learnvulkan)

message("-- CMAKE_SYSTEM_INFO_FILE:     " ${CMAKE_SYSTEM_INFO_FILE})
message("-- CMAKE_SYSTEM_NAME:          " ${CMAKE_SYSTEM_NAME})
message("-- CMAKE_SYSTEM_PROCESSOR:     " ${CMAKE_SYSTEM_PROCESSOR})
message("-- CMAKE_SYSTEM:               " ${CMAKE_SYSTEM})
message("-- CMAKE_GENERATOR:            " ${CMAKE_GENERATOR})
message("-- CMAKE_GENERATOR_INSTANCE:   " ${CMAKE_GENERATOR_INSTANCE})
message("-- CMAKE_GENERATOR_PLATFORM:   " ${CMAKE_GENERATOR_PLATFORM})
message("-- CMAKE_GENERATOR_TOOLSET:    " ${CMAKE_GENERATOR_TOOLSET})
message("-- CMAKE_VS_PLATFORM_NAME:     " ${CMAKE_VS_PLATFORM_NAME})
message("-- CMAKE_VS_PLATFORM_TOOLSET:  " ${CMAKE_VS_PLATFORM_TOOLSET})

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED on)
set(CMAKE_CXX_EXTENSIONS off)

if(MSVC) 
add_compile_options(/W4 /WX /wd4201 /wd4996 /wd4100 /wd4189 /wd4819 /wd4324)

if(${CMAKE_VS_PLATFORM_NAME} STREQUAL "x64")
    set(VULKAN_LIB_DIR $ENV{VULKAN_SDK}/lib)
    set(MYLIB_DIR ${CMAKE_SOURCE_DIR}/lib)
else()
    set(VULKAN_LIB_DIR  $ENV{VULKAN_SDK}/lib32)
    set(MYLIB32_DIR ${CMAKE_SOURCE_DIR}/lib32)
endif()
endif()

set(MYOUTPUT_DIR ${CMAKE_BINARY_DIR}/bin/)
set(CMAKE_DEBUG_POSTFIX d)

foreach(VAR ${CMAKE_CONFIGURATION_TYPES})
string(TOUPPER ${VAR} VAR2)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${VAR2} ${MYOUTPUT_DIR})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_${VAR2} ${MYOUTPUT_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_${VAR2} ${MYOUTPUT_DIR})
endforeach()

configure_file(config.h.in config.h)

#==========example template=======
function(newexample examplename)
message(${examplename})
add_executable(${examplename} ${examplename}.cpp common.cpp)

target_include_directories(${examplename} PRIVATE 
${CMAKE_SOURCE_DIR}
${CMAKE_BINARY_DIR}
${CMAKE_SOURCE_DIR}/3rdparty/glm
${CMAKE_SOURCE_DIR}/3rdparty/stb
$ENV{VULKAN_SDK}/include
)

if(WIN32)
target_link_libraries(${examplename} PRIVATE
${MYLIB_DIR}/glfw3.lib
${VULKAN_LIB_DIR}/vulkan-1.lib
)
endif()

endfunction()
newexample(01-triangle)
newexample(02-triangle)
newexample(03-image)
newexample(04-depth)