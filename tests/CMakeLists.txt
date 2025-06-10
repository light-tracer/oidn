find_package(GTest REQUIRED)
add_executable(webgpu_conv_test test_webgpu_conv.cpp
  ${PROJECT_SOURCE_DIR}/devices/webgpu/webgpu_helper.cpp
  ${PROJECT_SOURCE_DIR}/devices/webgpu/webgpu_engine.cpp
  ${PROJECT_SOURCE_DIR}/devices/webgpu/webgpu_device.cpp)
target_include_directories(webgpu_conv_test PRIVATE ${PROJECT_SOURCE_DIR}/devices/webgpu)
target_include_directories(webgpu_conv_test PRIVATE ${PROJECT_SOURCE_DIR})
target_link_libraries(webgpu_conv_test PRIVATE OpenImageDenoise OpenImageDenoise_core GTest::GTest GTest::Main)
add_test(NAME WebGPU.Conv2d COMMAND webgpu_conv_test)
