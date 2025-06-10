# CMake ≥3.18 recommended for file(ARCHIVE_EXTRACT)
function(oidn_download_wgpu out_dir)
  # ---------------------------------------------------------------------------
  # Download and unpack a pre-built wgpu-native binary for the current host
  #
  #  * Supported OS      : macOS 10.13+  |  Linux (glibc ≥ 2.28, “manylinux_2_28”)
  #  * Supported CPUs    : x86-64 | AArch64 (Apple-silicon, Raspberry Pi 64-bit, etc.)
  #  * Asset pattern     : wgpu-<os>-<arch>-release.zip
  #                         os   = macos | linux
  #                         arch = x86_64 | aarch64
  #  * Upstream release  : https://github.com/gfx-rs/wgpu-native/releases/tag/v25.0.2.1
  # ---------------------------------------------------------------------------
  set(_ver v25.0.2.1)

  # --- OS / architecture detection -------------------------------------------------
  if(APPLE)
    set(_os macos)
  elseif(UNIX AND (CMAKE_HOST_SYSTEM_NAME STREQUAL "Linux"))
    set(_os linux)
  else()
    message(FATAL_ERROR "oidn_download_wgpu(): unsupported host OS '${CMAKE_HOST_SYSTEM_NAME}'")
  endif()

  if(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "^(arm64|aarch64)$")
    set(_arch aarch64)
  elseif(CMAKE_HOST_SYSTEM_PROCESSOR STREQUAL "x86_64")
    set(_arch x86_64)
  else()
    message(FATAL_ERROR "oidn_download_wgpu(): unsupported CPU '${CMAKE_HOST_SYSTEM_PROCESSOR}'")
  endif()

  set(_url "https://github.com/gfx-rs/wgpu-native/releases/download/${_ver}/wgpu-${_os}-${_arch}-release.zip")

  # --- create output dir -----------------------------------------------------------
  file(MAKE_DIRECTORY "${out_dir}")

  # --- download (once) -------------------------------------------------------------
  set(_archive "${out_dir}/wgpu-native-${_os}-${_arch}.zip")
  if(NOT EXISTS "${_archive}")
    message(STATUS "Downloading wgpu-native ${_ver} (${_os}-${_arch})")
    file(DOWNLOAD "${_url}" "${_archive}" SHOW_PROGRESS STATUS _dl_status)
    list(GET _dl_status 0 _dl_code)
    if(NOT _dl_code EQUAL 0)
      message(FATAL_ERROR "Failed to download ${_url}")
    endif()
  endif()

  # --- unpack ----------------------------------------------------------------------
  file(ARCHIVE_EXTRACT INPUT "${_archive}" DESTINATION "${out_dir}")

  # --- expose result ---------------------------------------------------------------
  set(OIDN_WGPU_DIR "${out_dir}" CACHE INTERNAL "Path to downloaded wgpu-native")
endfunction()
