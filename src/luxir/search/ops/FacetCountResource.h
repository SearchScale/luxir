// Copyright 2020-2026 Yonik Seeley and Luxir contributors
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <memory_resource>
#include <string>
#include "luxir/search/RequestMemTracker.h"

namespace luxir {
// Charge actual allocator requests, including rehash overlap and tuple tails.
// The resource outlives every container using it. Refused charges leave the
// tracker unchanged; allocation failures roll back before propagating.
class FacetCountResource : public std::pmr::memory_resource {
  RequestMemTracker& tracker;
  std::string detail;

  void* do_allocate(size_t bytes, size_t alignment) override {
    tracker.charge(bytes, "facet distinct state", detail);
    try {
      return std::pmr::new_delete_resource()->allocate(bytes, alignment);
    } catch (...) {
      tracker.release(bytes);
      throw;
    }
  }
  void do_deallocate(void* p, size_t bytes, size_t alignment) override {
    std::pmr::new_delete_resource()->deallocate(p, bytes, alignment);
    tracker.release(bytes);
  }
  bool do_is_equal(const std::pmr::memory_resource& other) const noexcept override {
    return this == &other;
  }

public:
  FacetCountResource(RequestMemTracker& tracker, std::string_view label)
      : tracker(tracker), detail(label) {}
};

} // namespace luxir
