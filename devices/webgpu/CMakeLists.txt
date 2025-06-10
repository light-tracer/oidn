set(_srcs
  webgpu_device.cpp
  webgpu_buffer.cpp
  webgpu_engine.cpp
  webgpu_conv.cpp
  webgpu_pool.cpp
  webgpu_upsample.cpp
  webgpu_module.cpp
  webgpu_helper.cpp)
set(OIDN_WEBGPU_SOURCES ${_srcs} PARENT_SCOPE)
list(TRANSFORM _srcs PREPEND ${CMAKE_CURRENT_SOURCE_DIR}/)
set(OIDN_WEBGPU_SOURCE_FILES ${_srcs} PARENT_SCOPE)

# Sources are added to OpenImageDenoise target from top-level CMake
