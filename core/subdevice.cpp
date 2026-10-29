// Copyright 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "subdevice.h"
#include "engine.h"

OIDN_NAMESPACE_BEGIN

  Subdevice::Subdevice(std::unique_ptr<Engine>&& engine)
    : engine(std::move(engine))
  {
    this->engine->setSubdevice(this);
  }

  Ref<Arena> Subdevice::newScratchArena(size_t byteSize, const std::string& name)
  {
    if (!scratchArenaManager)
      scratchArenaManager.reset(new ScratchArenaManager(engine.get()));
    return makeRef<ScratchArena>(scratchArenaManager.get(), byteSize, name);
  }

  void Subdevice::trimScratch()
  {
    if (scratchArenaManager)
      scratchArenaManager->trim();
  }

  std::shared_ptr<TensorMap> Subdevice::getCachedTensors(const void* key)
  {
    std::shared_ptr<TensorMap>& tensorMap = cachedTensors[key];
    if (!tensorMap)
      tensorMap = std::make_shared<TensorMap>();
    return tensorMap;
  }

  std::shared_ptr<TensorMap> Subdevice::getCachedUserTensors(const Data& weights)
  {
    // Key by bytes, not pointer identity: oidnUpdateFilterData allows callers
    // to change a shared blob in place. A cache hit therefore cannot reuse a
    // reordered tensor from an older version at the same address.
    std::string snapshot(static_cast<const char*>(weights.ptr), weights.size);
    auto& tensors = cachedUserTensors[std::move(snapshot)];
    if (!tensors) tensors = std::make_shared<TensorMap>();
    return tensors;
  }

OIDN_NAMESPACE_END
