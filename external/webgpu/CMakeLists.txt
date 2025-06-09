cmake_minimum_required(VERSION 3.15)
project(webgpu_external)

include(oidn_wgpu)
oidn_download_wgpu(${CMAKE_CURRENT_BINARY_DIR})

add_library(wgpu_native STATIC IMPORTED)
set_target_properties(wgpu_native PROPERTIES
  IMPORTED_LOCATION_RELEASE ${OIDN_WGPU_DIR}/lib/libwgpu_native.a
  INTERFACE_INCLUDE_DIRECTORIES ${OIDN_WGPU_DIR}/include
)
