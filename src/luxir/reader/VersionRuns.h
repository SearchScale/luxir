// Copyright 2020-2026 Yonik Seeley and Luxir contributors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <ranges>
#include <span>
#include <stdexcept>
#include "luxir/util/luxir_util.h"
#include "luxir/util/screaming.h"

namespace luxir {

LUXIR_UNALIGNED_START
struct VersionRun {
  uint32_t firstDocId;
  uint64_t version;
} LUXIR_UNALIGNED_END;
static_assert(sizeof(VersionRun) == 12);

class VersionRuns {
  std::span<const VersionRun> runs_;
  int32_t maxDoc_ = 0;

public:
  static constexpr uint32_t PACKED = 1;

  VersionRuns() = default;
  VersionRuns(std::span<const VersionRun> runs, int32_t maxDoc) : runs_(runs), maxDoc_(maxDoc) {
    if (maxDoc < 0 || (maxDoc == 0 ? !runs.empty() : runs.empty())) {
      throw std::runtime_error("Invalid request-version run coverage");
    }
    for (size_t i = 0; i < runs.size(); i++) {
      if (runs[i].firstDocId >= (uint32_t)maxDoc || (i == 0 ? runs[i].firstDocId != 0
          : runs[i].firstDocId <= runs[i - 1].firstDocId)) {
        throw std::runtime_error("Invalid request-version run coverage");
      }
    }
  }

  VersionRun run(size_t index) const { return runs_[index]; }
  auto runs() const {
    return std::views::iota((size_t)0, runs_.size())
        | std::views::transform([this](size_t i) { return run(i); });
  }

  class Cursor {
    const VersionRuns& source;
    int64_t next = 0;
    int32_t lastDoc = -1;

  public:
    explicit Cursor(const VersionRuns& source) : source(source) {}

    uint64_t birth(int32_t docId) {
      if (docId < 0 || docId >= source.maxDoc_) {
        throw std::runtime_error("Document has no request birth version");
      }
      if (docId < lastDoc) next = 0;
      next = screaming::gallopLowerBound(next, (int64_t)source.runs_.size(), (uint32_t)docId + 1,
          [&](int64_t i) { return source.run((size_t)i).firstDocId; });
      lastDoc = docId;
      return source.run((size_t)next - 1).version;
    }
  };
};

} // namespace luxir
