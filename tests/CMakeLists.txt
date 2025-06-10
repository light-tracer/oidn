find_package(GTest REQUIRED)
add_executable(webgpu_conv_test test_webgpu_conv.cpp
  ${PROJECT_SOURCE_DIR}/devices/webgpu/webgpu_helper.cpp
  ${PROJECT_SOURCE_DIR}/devices/webgpu/webgpu_engine.cpp
  ${PROJECT_SOURCE_DIR}/devices/webgpu/webgpu_conv.cpp
  ${PROJECT_SOURCE_DIR}/devices/webgpu/webgpu_pool.cpp
  ${PROJECT_SOURCE_DIR}/devices/webgpu/webgpu_upsample.cpp
  ${PROJECT_SOURCE_DIR}/devices/webgpu/webgpu_input_process.cpp
  ${PROJECT_SOURCE_DIR}/devices/webgpu/webgpu_output_process.cpp
  ${PROJECT_SOURCE_DIR}/devices/webgpu/webgpu_image_copy.cpp
  ${PROJECT_SOURCE_DIR}/devices/webgpu/webgpu_autoexposure.cpp
  ${PROJECT_SOURCE_DIR}/devices/webgpu/webgpu_buffer.cpp
  ${PROJECT_SOURCE_DIR}/devices/webgpu/webgpu_device.cpp)
target_include_directories(webgpu_conv_test PRIVATE ${PROJECT_SOURCE_DIR}/devices/webgpu)
target_include_directories(webgpu_conv_test PRIVATE ${PROJECT_SOURCE_DIR}/devices/cpu)
target_include_directories(webgpu_conv_test PRIVATE ${PROJECT_SOURCE_DIR})
target_link_libraries(webgpu_conv_test PRIVATE OpenImageDenoise OpenImageDenoise_core OpenImageDenoise_device_cpu GTest::GTest GTest::Main)
add_test(NAME WebGPU.Conv2d COMMAND webgpu_conv_test)

add_executable(webgpu_upsample_test test_webgpu_upsample.cpp
  ${PROJECT_SOURCE_DIR}/devices/webgpu/webgpu_helper.cpp
  ${PROJECT_SOURCE_DIR}/devices/webgpu/webgpu_engine.cpp
  ${PROJECT_SOURCE_DIR}/devices/webgpu/webgpu_conv.cpp
  ${PROJECT_SOURCE_DIR}/devices/webgpu/webgpu_pool.cpp
  ${PROJECT_SOURCE_DIR}/devices/webgpu/webgpu_upsample.cpp
  ${PROJECT_SOURCE_DIR}/devices/webgpu/webgpu_input_process.cpp
  ${PROJECT_SOURCE_DIR}/devices/webgpu/webgpu_output_process.cpp
  ${PROJECT_SOURCE_DIR}/devices/webgpu/webgpu_image_copy.cpp
  ${PROJECT_SOURCE_DIR}/devices/webgpu/webgpu_autoexposure.cpp
  ${PROJECT_SOURCE_DIR}/devices/webgpu/webgpu_buffer.cpp
  ${PROJECT_SOURCE_DIR}/devices/webgpu/webgpu_device.cpp)
target_include_directories(webgpu_upsample_test PRIVATE ${PROJECT_SOURCE_DIR}/devices/webgpu)
target_include_directories(webgpu_upsample_test PRIVATE ${PROJECT_SOURCE_DIR}/devices/cpu)
target_include_directories(webgpu_upsample_test PRIVATE ${PROJECT_SOURCE_DIR})
target_link_libraries(webgpu_upsample_test PRIVATE OpenImageDenoise OpenImageDenoise_core OpenImageDenoise_device_cpu GTest::GTest GTest::Main)
add_test(NAME WebGPU.Upsample COMMAND webgpu_upsample_test)

add_executable(webgpu_pool_test test_webgpu_pool.cpp
  ${PROJECT_SOURCE_DIR}/devices/webgpu/webgpu_helper.cpp
  ${PROJECT_SOURCE_DIR}/devices/webgpu/webgpu_engine.cpp
  ${PROJECT_SOURCE_DIR}/devices/webgpu/webgpu_conv.cpp
  ${PROJECT_SOURCE_DIR}/devices/webgpu/webgpu_pool.cpp
  ${PROJECT_SOURCE_DIR}/devices/webgpu/webgpu_upsample.cpp
  ${PROJECT_SOURCE_DIR}/devices/webgpu/webgpu_input_process.cpp
  ${PROJECT_SOURCE_DIR}/devices/webgpu/webgpu_output_process.cpp
  ${PROJECT_SOURCE_DIR}/devices/webgpu/webgpu_image_copy.cpp
  ${PROJECT_SOURCE_DIR}/devices/webgpu/webgpu_autoexposure.cpp
  ${PROJECT_SOURCE_DIR}/devices/webgpu/webgpu_buffer.cpp
  ${PROJECT_SOURCE_DIR}/devices/webgpu/webgpu_device.cpp)
target_include_directories(webgpu_pool_test PRIVATE ${PROJECT_SOURCE_DIR}/devices/webgpu)
target_include_directories(webgpu_pool_test PRIVATE ${PROJECT_SOURCE_DIR}/devices/cpu)
target_include_directories(webgpu_pool_test PRIVATE ${PROJECT_SOURCE_DIR})
target_link_libraries(webgpu_pool_test PRIVATE OpenImageDenoise OpenImageDenoise_core OpenImageDenoise_device_cpu GTest::GTest GTest::Main)
add_test(NAME WebGPU.Pool COMMAND webgpu_pool_test)

add_executable(webgpu_input_process_test test_webgpu_input_process.cpp
  ${PROJECT_SOURCE_DIR}/devices/webgpu/webgpu_helper.cpp
  ${PROJECT_SOURCE_DIR}/devices/webgpu/webgpu_engine.cpp
  ${PROJECT_SOURCE_DIR}/devices/webgpu/webgpu_conv.cpp
  ${PROJECT_SOURCE_DIR}/devices/webgpu/webgpu_pool.cpp
  ${PROJECT_SOURCE_DIR}/devices/webgpu/webgpu_upsample.cpp
  ${PROJECT_SOURCE_DIR}/devices/webgpu/webgpu_input_process.cpp
  ${PROJECT_SOURCE_DIR}/devices/webgpu/webgpu_output_process.cpp
  ${PROJECT_SOURCE_DIR}/devices/webgpu/webgpu_image_copy.cpp
  ${PROJECT_SOURCE_DIR}/devices/webgpu/webgpu_autoexposure.cpp
  ${PROJECT_SOURCE_DIR}/devices/webgpu/webgpu_buffer.cpp
  ${PROJECT_SOURCE_DIR}/devices/webgpu/webgpu_device.cpp)
target_include_directories(webgpu_input_process_test PRIVATE ${PROJECT_SOURCE_DIR}/devices/webgpu)
target_include_directories(webgpu_input_process_test PRIVATE ${PROJECT_SOURCE_DIR}/devices/cpu)
target_include_directories(webgpu_input_process_test PRIVATE ${PROJECT_SOURCE_DIR})
target_link_libraries(webgpu_input_process_test PRIVATE OpenImageDenoise OpenImageDenoise_core OpenImageDenoise_device_cpu GTest::GTest GTest::Main)
add_test(NAME WebGPU.InputProcess COMMAND webgpu_input_process_test)

add_executable(webgpu_output_process_test test_webgpu_output_process.cpp
  ${PROJECT_SOURCE_DIR}/devices/webgpu/webgpu_helper.cpp
  ${PROJECT_SOURCE_DIR}/devices/webgpu/webgpu_engine.cpp
  ${PROJECT_SOURCE_DIR}/devices/webgpu/webgpu_conv.cpp
  ${PROJECT_SOURCE_DIR}/devices/webgpu/webgpu_pool.cpp
  ${PROJECT_SOURCE_DIR}/devices/webgpu/webgpu_upsample.cpp
  ${PROJECT_SOURCE_DIR}/devices/webgpu/webgpu_input_process.cpp
  ${PROJECT_SOURCE_DIR}/devices/webgpu/webgpu_output_process.cpp
  ${PROJECT_SOURCE_DIR}/devices/webgpu/webgpu_image_copy.cpp
  ${PROJECT_SOURCE_DIR}/devices/webgpu/webgpu_autoexposure.cpp
  ${PROJECT_SOURCE_DIR}/devices/webgpu/webgpu_buffer.cpp
  ${PROJECT_SOURCE_DIR}/devices/webgpu/webgpu_device.cpp)
target_include_directories(webgpu_output_process_test PRIVATE ${PROJECT_SOURCE_DIR}/devices/webgpu)
target_include_directories(webgpu_output_process_test PRIVATE ${PROJECT_SOURCE_DIR}/devices/cpu)
target_include_directories(webgpu_output_process_test PRIVATE ${PROJECT_SOURCE_DIR})
target_link_libraries(webgpu_output_process_test PRIVATE OpenImageDenoise OpenImageDenoise_core OpenImageDenoise_device_cpu GTest::GTest GTest::Main)
add_test(NAME WebGPU.OutputProcess COMMAND webgpu_output_process_test)

add_executable(webgpu_image_copy_test test_webgpu_image_copy.cpp
  ${PROJECT_SOURCE_DIR}/devices/webgpu/webgpu_helper.cpp
  ${PROJECT_SOURCE_DIR}/devices/webgpu/webgpu_engine.cpp
  ${PROJECT_SOURCE_DIR}/devices/webgpu/webgpu_conv.cpp
  ${PROJECT_SOURCE_DIR}/devices/webgpu/webgpu_pool.cpp
  ${PROJECT_SOURCE_DIR}/devices/webgpu/webgpu_upsample.cpp
  ${PROJECT_SOURCE_DIR}/devices/webgpu/webgpu_input_process.cpp
  ${PROJECT_SOURCE_DIR}/devices/webgpu/webgpu_output_process.cpp
  ${PROJECT_SOURCE_DIR}/devices/webgpu/webgpu_image_copy.cpp
  ${PROJECT_SOURCE_DIR}/devices/webgpu/webgpu_autoexposure.cpp
  ${PROJECT_SOURCE_DIR}/devices/webgpu/webgpu_buffer.cpp
  ${PROJECT_SOURCE_DIR}/devices/webgpu/webgpu_device.cpp)
target_include_directories(webgpu_image_copy_test PRIVATE ${PROJECT_SOURCE_DIR}/devices/webgpu)
target_include_directories(webgpu_image_copy_test PRIVATE ${PROJECT_SOURCE_DIR}/devices/cpu)
target_include_directories(webgpu_image_copy_test PRIVATE ${PROJECT_SOURCE_DIR})
target_link_libraries(webgpu_image_copy_test PRIVATE OpenImageDenoise OpenImageDenoise_core OpenImageDenoise_device_cpu GTest::GTest GTest::Main)
add_test(NAME WebGPU.ImageCopy COMMAND webgpu_image_copy_test)

add_executable(webgpu_autoexposure_test test_webgpu_autoexposure.cpp
  ${PROJECT_SOURCE_DIR}/devices/webgpu/webgpu_helper.cpp
  ${PROJECT_SOURCE_DIR}/devices/webgpu/webgpu_engine.cpp
  ${PROJECT_SOURCE_DIR}/devices/webgpu/webgpu_conv.cpp
  ${PROJECT_SOURCE_DIR}/devices/webgpu/webgpu_pool.cpp
  ${PROJECT_SOURCE_DIR}/devices/webgpu/webgpu_upsample.cpp
  ${PROJECT_SOURCE_DIR}/devices/webgpu/webgpu_input_process.cpp
  ${PROJECT_SOURCE_DIR}/devices/webgpu/webgpu_output_process.cpp
  ${PROJECT_SOURCE_DIR}/devices/webgpu/webgpu_image_copy.cpp
  ${PROJECT_SOURCE_DIR}/devices/webgpu/webgpu_autoexposure.cpp
  ${PROJECT_SOURCE_DIR}/devices/webgpu/webgpu_buffer.cpp
  ${PROJECT_SOURCE_DIR}/devices/webgpu/webgpu_device.cpp)
target_include_directories(webgpu_autoexposure_test PRIVATE ${PROJECT_SOURCE_DIR}/devices/webgpu)
target_include_directories(webgpu_autoexposure_test PRIVATE ${PROJECT_SOURCE_DIR}/devices/cpu)
target_include_directories(webgpu_autoexposure_test PRIVATE ${PROJECT_SOURCE_DIR})
target_link_libraries(webgpu_autoexposure_test PRIVATE OpenImageDenoise OpenImageDenoise_core OpenImageDenoise_device_cpu GTest::GTest GTest::Main)
add_test(NAME WebGPU.Autoexposure COMMAND webgpu_autoexposure_test)
