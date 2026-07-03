## Copyright 2026 Intel Corporation
## SPDX-License-Identifier: Apache-2.0

# Emscripten web build: compiles the minimal C shim and links it against the
# (static) OpenImageDenoise library, producing oidn_web.js + oidn_web.wasm.

if(NOT EMSCRIPTEN)
  message(FATAL_ERROR "The web/ target requires an Emscripten toolchain")
endif()

add_executable(oidn_web oidn_web.cpp)

target_link_libraries(oidn_web PRIVATE OpenImageDenoise wgpu_native)
# The shim publishes the device/output buffer JS objects to the page (for
# canvas display), which requires the WebGPU device internals
target_include_directories(oidn_web PRIVATE ${PROJECT_SOURCE_DIR} ${PROJECT_BINARY_DIR})

# Exported C functions callable from JS (leading underscore is the wasm symbol)
set(OIDN_WEB_EXPORTS
  "_oidnWebInit,_oidnWebSetWeights,_oidnWebDenoise,_oidnWebShutdown,_malloc,_free")

target_link_options(oidn_web PRIVATE
  # WebGPU from the browser
  "--use-port=emdawnwebgpu"
  # Match the -fexceptions the library is compiled with
  "-fexceptions"
  # Let the synchronous C API block on async WebGPU (adapter/device request,
  # buffer maps) by unwinding to the browser event loop and resuming
  "-sASYNCIFY"
  "-sASYNCIFY_STACK_SIZE=131072"
  # OIDN's filter execution has deep call chains; the 64 KB default is too small
  "-sSTACK_SIZE=1048576"
  # The denoiser allocates sizeable GPU staging/host buffers on demand
  "-sALLOW_MEMORY_GROWTH=1"
  "-sINITIAL_MEMORY=134217728"
  # In browsers with resizable-ArrayBuffer support, growable wasm memory makes
  # TextDecoder.decode() throw on heap views ("value must not be resizable",
  # hit in emdawnwebgpu's WGSL string handling); use classic memory growth
  "-sGROWABLE_ARRAYBUFFERS=0"
  # A loadable ES-ish module with a named factory
  "-sMODULARIZE=1"
  "-sEXPORT_NAME=createOIDN"
  "-sEXPORTED_FUNCTIONS=${OIDN_WEB_EXPORTS}"
  "-sEXPORTED_RUNTIME_METHODS=ccall,cwrap,HEAPF32,HEAPU8,ENV"
  "-sENVIRONMENT=web"
  # Fail loudly instead of silently corrupting on OOB during bring-up
  "-sASSERTIONS=1"
)

set_target_properties(oidn_web PROPERTIES
  OUTPUT_NAME "oidn_web"
  SUFFIX ".js"
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")

# Copy the static test page next to the build output for convenience
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/index.html
               ${CMAKE_CURRENT_BINARY_DIR}/index.html COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/denoise.js
               ${CMAKE_CURRENT_BINARY_DIR}/denoise.js COPYONLY)

# The wasm module ships without embedded weights; the page fetches these .tza
# files at runtime and passes them to oidnWebSetWeights
set(OIDN_WEB_WEIGHTS
  rt_ldr.tza
  rt_hdr.tza
  rt_ldr_alb_nrm.tza
  rt_hdr_alb_nrm.tza
)
foreach(_tza ${OIDN_WEB_WEIGHTS})
  configure_file(${PROJECT_SOURCE_DIR}/weights/${_tza}
                 ${CMAKE_CURRENT_BINARY_DIR}/weights/${_tza} COPYONLY)
endforeach()
