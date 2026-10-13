// Copyright 2020-2026 Yonik Seeley and Luxir contributors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <boost/sort/spreadsort/integer_sort.hpp>

namespace luxir {

// boost integer_sort ordered by an integer key alone. Its comparator must
// order exactly by the shifted key: below its size threshold (and for small
// bins) it falls back to pdqsort with the comparator, but large equal-key
// buckets are never compared, so a tie-break folded into the comparator is
// honored only sometimes. Callers resolve ties themselves, e.g. by re-sorting
// each equal-key run.
template <class It, class Key>
void integerSortByKey(It first, It last, Key key) {
  boost::sort::spreadsort::integer_sort(first, last,
      [&](const auto& x, unsigned shift) { return key(x) >> shift; },
      [&](const auto& x, const auto& y) { return key(x) < key(y); });
}

}  // namespace luxir
