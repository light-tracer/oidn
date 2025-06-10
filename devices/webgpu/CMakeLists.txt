set(OIDN_WEBGPU_SOURCES
  webgpu_device.cpp
  webgpu_buffer.cpp
  webgpu_heap.cpp
  webgpu_arena.cpp
  webgpu_engine.cpp
  webgpu_conv.cpp
  webgpu_pool.cpp
  webgpu_upsample.cpp
  webgpu_input_process.cpp
  webgpu_output_process.cpp
  webgpu_image_copy.cpp
  webgpu_autoexposure.cpp
  webgpu_module.cpp
  webgpu_helper.cpp)

add_library(OpenImageDenoise_device_webgpu ${OIDN_LIB_TYPE} ${OIDN_WEBGPU_SOURCES} ${OIDN_RESOURCE_FILE})

set_target_properties(OpenImageDenoise_device_webgpu PROPERTIES
  OUTPUT_NAME ${OIDN_LIBRARY_NAME}_device_webgpu
  CXX_STANDARD 17)
if(OIDN_LIBRARY_VERSIONED)
  set_target_properties(OpenImageDenoise_device_webgpu PROPERTIES VERSION ${PROJECT_VERSION})
endif()

target_include_directories(OpenImageDenoise_device_webgpu PRIVATE ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(OpenImageDenoise_device_webgpu PRIVATE OpenImageDenoise_core wgpu_native)

if(OIDN_STATIC_LIB)
  oidn_install_static_module(OpenImageDenoise_device_webgpu)
  target_link_libraries(OpenImageDenoise PRIVATE OpenImageDenoise_device_webgpu)
else()
  oidn_install_module(OpenImageDenoise_device_webgpu)
endif()
