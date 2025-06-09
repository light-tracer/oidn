cmake_minimum_required(VERSION 3.15)
project(wgpu_device C)

add_executable(wgpuIdentity wgpuIdentity.c)
add_executable(wgpuConv2d wgpuConv2d.c)

foreach(tgt IN ITEMS wgpuIdentity wgpuConv2d)
  target_include_directories(${tgt} PRIVATE ${OIDN_WGPU_DIR}/include)
  target_link_libraries(${tgt} PRIVATE ${OIDN_WGPU_DIR}/lib/libwgpu_native.a dl pthread m)
endforeach()

install(TARGETS wgpuIdentity wgpuConv2d DESTINATION bin)
