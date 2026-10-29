cmake_minimum_required(VERSION 3.18)
project(OIDNExternalWebGPU LANGUAGES C CXX)

set(LT_ROOT "" CACHE PATH "Light Tracer checkout containing libs/dawn")
if(NOT EXISTS "${LT_ROOT}/libs/dawn/include/webgpu/webgpu.h")
  message(FATAL_ERROR "Set LT_ROOT to the Light Tracer checkout")
endif()

add_library(lt_webgpu SHARED IMPORTED GLOBAL)
set_target_properties(lt_webgpu PROPERTIES
  INTERFACE_INCLUDE_DIRECTORIES "${LT_ROOT}/libs/dawn/include"
  INTERFACE_COMPILE_DEFINITIONS OIDN_WEBGPU_DAWN_LEGACY_NATIVE)
if(WIN32)
  set_target_properties(lt_webgpu PROPERTIES
    IMPORTED_LOCATION "${LT_ROOT}/libs/dawn/lib/Release/webgpu_dawn.dll"
    IMPORTED_IMPLIB "${LT_ROOT}/libs/dawn/lib/Release/webgpu_dawn.lib")
else()
  set_target_properties(lt_webgpu PROPERTIES
    IMPORTED_LOCATION "${LT_ROOT}/libs/dawn/lib/Release/libwebgpu_dawn${CMAKE_SHARED_LIBRARY_SUFFIX}")
endif()

set(OIDN_STATIC_LIB ON CACHE BOOL "" FORCE)
set(OIDN_DEVICE_WEBGPU ON CACHE BOOL "" FORCE)
foreach(option CPU SYCL CUDA HIP METAL)
  set(OIDN_DEVICE_${option} OFF CACHE BOOL "" FORCE)
endforeach()
set(OIDN_FILTER_RT OFF CACHE BOOL "" FORCE)
set(OIDN_FILTER_RTLIGHTMAP OFF CACHE BOOL "" FORCE)
set(OIDN_ENABLE_RTLIGHTMAP OFF CACHE BOOL "" FORCE)
set(OIDN_APPS OFF CACHE BOOL "" FORCE)
set(OIDN_LIBRARY_VERSIONED OFF CACHE BOOL "" FORCE)
set(OIDN_WEBGPU_PROVIDER EXTERNAL CACHE STRING "" FORCE)
set(OIDN_WEBGPU_PROVIDER_TARGET lt_webgpu CACHE STRING "" FORCE)
set(OIDN_WEBGPU_STANDALONE OFF CACHE BOOL "" FORCE)
set(OIDN_WEBGPU_TESTS OFF CACHE BOOL "" FORCE)
add_subdirectory(../.. oidn)

add_executable(oidn_external_test external_test.cpp)
target_link_libraries(oidn_external_test PRIVATE OIDN::WebGPU_RT lt_webgpu)
if(WIN32)
  add_custom_command(TARGET oidn_external_test POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:lt_webgpu> $<TARGET_FILE_DIR:oidn_external_test>)
endif()
enable_testing()
add_test(NAME WebGPU.External COMMAND oidn_external_test)
