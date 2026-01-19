# Generated Cmake Pico project file

cmake_minimum_required(VERSION 3.13)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Initialise pico_sdk from installed location
# (note this can come from environment, CMake cache etc)

# == DO NOT EDIT THE FOLLOWING LINES for the Raspberry Pi Pico VS Code Extension to work ==
if(WIN32)
    set(USERHOME $ENV{USERPROFILE})
else()
    set(USERHOME $ENV{HOME})
endif()
set(sdkVersion 2.2.0)
set(toolchainVersion 14_2_Rel1)
set(picotoolVersion 2.2.0-a4)
set(picoVscode ${USERHOME}/.pico-sdk/cmake/pico-vscode.cmake)
if (EXISTS ${picoVscode})
    include(${picoVscode})
endif()
# ====================================================================================
set(PICO_BOARD pico2 CACHE STRING "Board type")

# Pull in Raspberry Pi Pico SDK (must be before project)
include(pico_sdk_import.cmake)

project(test_ia_pico C CXX ASM)

# Initialise the Raspberry Pi Pico SDK
pico_sdk_init()

add_definitions(-DARM_MATH_CM33)
add_definitions(-D__ARM_FEATURE_DSP=1)
add_definitions(-DTF_LITE_STATIC_MEMORY)
add_definitions(-DCMSIS_NN)
add_definitions(-DPROJECT_GENERATION) # Debo definir mi Timer

# Add executable. Default name is the project name, version 0.1

file(GLOB_RECURSE IA_SOURCES 
    "tflite_arm_m33_lib/*.cpp"
    "tflite_arm_m33_lib/*.cc"
    "tflite_arm_m33_lib/*.c"
)

add_library(ia_embed STATIC ${IA_SOURCES})

target_include_directories(ia_embed PUBLIC 
    tflite_arm_m33_lib                      
    tflite_arm_m33_lib/third_party/flatbuffers/include
    tflite_arm_m33_lib/third_party/gemmlowp
    tflite_arm_m33_lib/third_party/ruy
    tflite_arm_m33_lib/third_party/cmsis
    tflite_arm_m33_lib/third_party/cmsis_nn/Include
    tflite_arm_m33_lib/third_party/kissfft
)

target_link_libraries(ia_embed PUBLIC 
    pico_stdlib
)

add_executable(test_ia_pico test_ia_pico.cpp model/logic_or_test.cc)

target_include_directories(test_ia_pico PRIVATE
    model
)

target_compile_options(ia_embed PRIVATE 
    -fno-threadsafe-statics
)

pico_set_program_name(test_ia_pico "test_ia_pico")
pico_set_program_version(test_ia_pico "0.1")

# Modify the below lines to enable/disable output over UART/USB
pico_enable_stdio_uart(test_ia_pico 0)
pico_enable_stdio_usb(test_ia_pico 1)

# Add the standard library to the build
target_link_libraries(test_ia_pico
        pico_stdlib
        ia_embed
        )

# Add the standard include files to the build
target_include_directories(test_ia_pico PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}
)

pico_add_extra_outputs(test_ia_pico)

