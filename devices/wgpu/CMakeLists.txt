cmake_minimum_required(VERSION 3.15)
project(wgpu_device C)

add_executable(wgpuIdentity wgpuIdentity.c)

target_include_directories(wgpuIdentity PRIVATE ${OIDN_WGPU_DIR}/include)

target_link_libraries(wgpuIdentity PRIVATE ${OIDN_WGPU_DIR}/lib/libwgpu_native.a dl pthread m)

install(TARGETS wgpuIdentity DESTINATION bin)
