find_package(GTest REQUIRED)
add_executable(webgpu_conv_test test_webgpu_conv.cpp)
target_link_libraries(webgpu_conv_test PRIVATE OpenImageDenoise GTest::GTest GTest::Main)
add_test(NAME WebGPU.Conv2d COMMAND webgpu_conv_test)
