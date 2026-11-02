cmake_minimum_required(VERSION 3.22.1)

project("vision_jni" VERSION 1.0.0 LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED true)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED true)

if(DEFINED ANDROID_ABI)
    if(ANDROID_ABI STREQUAL "arm64-v8a")
        set(GGML_SYSTEM_ARCH "ARM")
        set(GGML_OPENMP OFF)
    endif()
endif()

set(LLAMA_BUILD_MTMD ON CACHE BOOL "Build mtmd standalone" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "Do not build all tools" FORCE)
set(LLAMA_BUILD_COMMON ON CACHE BOOL "Build common" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "Do not build tests" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "Do not build examples" FORCE)
set(GGML_LLAMAFILE OFF CACHE BOOL "Disable llamafile" FORCE)

set(LLAMA_SRC ${CMAKE_CURRENT_LIST_DIR}/../../../../../llama.cpp)
add_subdirectory(${LLAMA_SRC} build-llama)

add_library(${CMAKE_PROJECT_NAME} SHARED vision_jni.cpp)

target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE
        ${LLAMA_SRC}
        ${LLAMA_SRC}/common
        ${LLAMA_SRC}/include
        ${LLAMA_SRC}/ggml/include
        ${LLAMA_SRC}/ggml/src
        ${LLAMA_SRC}/tools/mtmd)

target_link_libraries(${CMAKE_PROJECT_NAME}
        llama
        llama-common
        ggml
        mtmd
        log
        android
        jnigraphics)
