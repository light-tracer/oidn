// SPDX-License-Identifier: Apache-2.0
#include "webgpu_compat.h"

#if defined(OIDN_WEBGPU_LEGACY_BROWSER)
  #include <emscripten.h>

// Emscripten can create multiple C handles for one JavaScript GPU object.
// Share one identity space across both managers: buffer ID 1 and texture ID 1
// are unrelated resources, while two buffer handles may reference one object.
EM_JS(uintptr_t, oidnWebGPUResourceIdentity, (int texture, uintptr_t handle), {
  const object = texture ? WebGPU.mgrTexture.get(handle) : WebGPU.mgrBuffer.get(handle);
  let identities = WebGPU['oidnResourceIdentities'];
  if (!identities) identities = WebGPU['oidnResourceIdentities'] = { objects:new WeakMap(), next:1 };
  let id = identities.objects.get(object);
  if (id === undefined) {
    if (identities.next > 0xffffffff) throw new Error('OIDN resource identity space exhausted');
    id = identities.next++;
    identities.objects.set(object,id);
  }
  return id;
});
#endif

OIDN_NAMESPACE_BEGIN

  const void* oidnWGPUBufferIdentity(WGPUBuffer buffer)
  {
  #if defined(OIDN_WEBGPU_LEGACY_BROWSER)
    return buffer ? reinterpret_cast<const void*>(oidnWebGPUResourceIdentity(0,reinterpret_cast<uintptr_t>(buffer))) : nullptr;
  #else
    return buffer;
  #endif
  }

  const void* oidnWGPUTextureIdentity(WGPUTexture texture)
  {
  #if defined(OIDN_WEBGPU_LEGACY_BROWSER)
    return texture ? reinterpret_cast<const void*>(oidnWebGPUResourceIdentity(1,reinterpret_cast<uintptr_t>(texture))) : nullptr;
  #else
    return texture;
  #endif
  }

OIDN_NAMESPACE_END
