# ──────────────────────────────────────────────────────────────────────────────
# 1.  CMake ≥3.18 is required because the helper uses file(ARCHIVE_EXTRACT …)
# ──────────────────────────────────────────────────────────────────────────────
cmake_minimum_required(VERSION 3.18)
project(webgpu_external LANGUAGES C CXX)

# ──────────────────────────────────────────────────────────────────────────────
# 2.  Bring in the helper that knows how to download the right archive
# ──────────────────────────────────────────────────────────────────────────────
include(oidn_wgpu)          # <-- file that defines oidn_download_wgpu()

set(WGPU_ROOT "${CMAKE_CURRENT_BINARY_DIR}")
oidn_download_wgpu("${WGPU_ROOT}")   # <-- downloads + unpacks into WGPU_ROOT

# ──────────────────────────────────────────────────────────────────────────────
# 3.  Import the pre-built binary shipped in the archive
#     • All official assets provide a static library (libwgpu_native.a)₁
#     • Some also ship a shared library (.so / .dylib), so we fall back to that
# ──────────────────────────────────────────────────────────────────────────────

if(WIN32)
  # Use the shared library: the static wgpu_native.lib would require manually
  # linking a long list of system libraries (d3dcompiler, ws2_32, userenv, ...)
  if(EXISTS "${WGPU_ROOT}/lib/wgpu_native.dll" AND EXISTS "${WGPU_ROOT}/lib/wgpu_native.dll.lib")
      add_library(wgpu_native SHARED IMPORTED GLOBAL)
      set_target_properties(wgpu_native PROPERTIES
          IMPORTED_LOCATION "${WGPU_ROOT}/lib/wgpu_native.dll"
          IMPORTED_IMPLIB   "${WGPU_ROOT}/lib/wgpu_native.dll.lib")
      # Make the DLL available next to the built binaries
      file(COPY "${WGPU_ROOT}/lib/wgpu_native.dll" DESTINATION "${CMAKE_BINARY_DIR}")
  else()
      message(FATAL_ERROR
          "Couldn't find wgpu_native.dll in ${WGPU_ROOT}/lib — "
          "did the download fail?")
  endif()
elseif(APPLE)
  if(EXISTS "${WGPU_ROOT}/lib/libwgpu_native${CMAKE_SHARED_LIBRARY_SUFFIX}")
      add_library(wgpu_native SHARED IMPORTED GLOBAL)
      set_target_properties(wgpu_native PROPERTIES
          IMPORTED_LOCATION
              "${WGPU_ROOT}/lib/libwgpu_native${CMAKE_SHARED_LIBRARY_SUFFIX}")
  else()
      message(FATAL_ERROR
          "Couldn't find libwgpu_native in ${WGPU_ROOT}/lib — "
          "did the download fail?")
  endif()
else()
  if(EXISTS "${WGPU_ROOT}/lib/libwgpu_native.a")
      add_library(wgpu_native STATIC IMPORTED GLOBAL)
      set_target_properties(wgpu_native PROPERTIES
          IMPORTED_LOCATION "${WGPU_ROOT}/lib/libwgpu_native.a")
  elseif(EXISTS "${WGPU_ROOT}/lib/libwgpu_native${CMAKE_SHARED_LIBRARY_SUFFIX}")
      add_library(wgpu_native SHARED IMPORTED GLOBAL)
      set_target_properties(wgpu_native PROPERTIES
          IMPORTED_LOCATION
              "${WGPU_ROOT}/lib/libwgpu_native${CMAKE_SHARED_LIBRARY_SUFFIX}")
  else()
      message(FATAL_ERROR
          "Couldn't find libwgpu_native in ${WGPU_ROOT}/lib — "
          "did the download fail?")
  endif()
endif()

# Headers are identical on all platforms
set_target_properties(wgpu_native PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${WGPU_ROOT}/include")

# Optionally expose the root dir via the cache (useful for other targets)
set(OIDN_WGPU_DIR "${WGPU_ROOT}" CACHE PATH "Path to unpacked wgpu-native")

