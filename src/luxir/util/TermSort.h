// Copyright 2020-2026 Yonik Seeley and Luxir contributors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstring>
#include <memory>
#include <type_traits>

#include <boost/sort/pdqsort/pdqsort.hpp>

#include "luxir/util/IntegerSort.h"
#include "luxir/util/StrRef.h"

namespace luxir {

namespace detail {

template <class T>
struct KeyedTerm {
  uint64_t key;
  T term;
};

// Runs of tied digits shorter than this finish with a full term compare
// instead of another digit pass.
inline constexpr size_t TERM_SORT_MIN_RADIX = 32;

// How many entries ahead the digit gather prefetches. The terms are scattered
// across the pool, and the gather's size-dependent branches would otherwise
// serialize the misses (measured 1.2-1.6x on 0.6-1M terms; 8 and 32 are close).
inline constexpr size_t TERM_SORT_PREFETCH = 16;

// Bytes [off, off + 8) of the term as a big-endian integer, zero padded past
// its end, so integer order is byte order.
inline uint64_t termDigit(const PackedTerm& term, uint32_t off) noexcept {
  static_assert(std::endian::native == std::endian::little);
  uint64_t digit = 0;
  uint32_t size = term.size();
  if (size > off) std::memcpy(&digit, term.data() + off, std::min(size - off, 8u));
  return std::byteswap(digit);
}

// Sorts terms that all share the zero-padded bytes [0, off).
template <class T>
void sortTermDigits(KeyedTerm<T>* a, size_t n, uint32_t off) {
  uint32_t maxSize = 0;
  for (size_t i = 0; i < n; i++) {
    if (i + TERM_SORT_PREFETCH < n) {
      __builtin_prefetch(a[i + TERM_SORT_PREFETCH].term.data() + std::min(off, 56u));
    }
    a[i].key = termDigit(a[i].term, off);
    maxSize = std::max(maxSize, a[i].term.size());
  }
  if (maxSize <= off) {
    // Every term ends within the shared bytes: each is a prefix of the longer ones.
    boost::sort::pdqsort(a, a + n, [](const KeyedTerm<T>& x, const KeyedTerm<T>& y) {
      return x.term.size() < y.term.size();
    });
    return;
  }
  integerSortByKey(a, a + n, [](const KeyedTerm<T>& x) { return x.key; });
  for (size_t start = 0; start < n;) {
    size_t end = start + 1;
    while (end < n && a[end].key == a[start].key) end++;
    if (end - start >= TERM_SORT_MIN_RADIX) {
      sortTermDigits(a + start, end - start, off + 8);
    } else if (end - start > 1) {
      boost::sort::pdqsort(a + start, a + end, [](const KeyedTerm<T>& x, const KeyedTerm<T>& y) {
        return PackedTerm::lessthan()(x.term, y.term);
      });
    }
    start = end;
  }
}

}  // namespace detail

/// Bytes sortTerms allocates for n terms when `capacity` slots of T are
/// writable at the terms: none when they can hold its {digit, term} pairs.
template <class T>
size_t sortTermsScratchBytes(size_t n, size_t capacity) {
  if (n < detail::TERM_SORT_MIN_RADIX || capacity * sizeof(T) >= n * sizeof(detail::KeyedTerm<T>)) return 0;
  return n * sizeof(detail::KeyedTerm<T>);
}

/// Sorts terms (PackedTerm, or a TermValRef over one) into byte order.
/// MSD radix on 8-byte digits held in {digit, term} pairs: the pool is
/// visited once per digit level (plus compares within small tied runs),
/// where spreadsort::string_sort chases every term pointer on each one-byte
/// pass. That chase dominates once the terms outgrow cache (standalone, on
/// natural-language and id-like term sets: 2.5-4.5x at 0.6-1M terms).
/// The pairs take 16 bytes per term. They are built in place when the
/// `capacity` slots writable at `terms` (e.g. a compressed hash table, at
/// load <= 0.5) can hold them, else allocated (see sortTermsScratchBytes).
template <class T>
  requires std::is_base_of_v<PackedTerm, T>
void sortTerms(T* terms, size_t n, size_t capacity) {
  using Keyed = detail::KeyedTerm<T>;
  static_assert(std::is_trivially_copyable_v<T> && alignof(Keyed) == alignof(T));
  // In place, each move's source and destination are then disjoint.
  static_assert(sizeof(T) <= offsetof(Keyed, term));
  if (n < detail::TERM_SORT_MIN_RADIX) {
    boost::sort::pdqsort(terms, terms + n, PackedTerm::lessthan());
    return;
  }
  std::unique_ptr<Keyed[]> scratch;
  auto* keyed = reinterpret_cast<Keyed*>(terms);
  if (sortTermsScratchBytes<T>(n, capacity) != 0) {
    scratch = std::make_unique_for_overwrite<Keyed[]>(n);
    keyed = scratch.get();
  }
  // In place, pair i overlaps only terms at or past i, so spread back to
  // front and gather front to back. memcpy because the slots change type.
  for (size_t i = n; i-- > 0;) std::memcpy((void*)&keyed[i].term, &terms[i], sizeof(T));
  detail::sortTermDigits(keyed, n, 0);
  for (size_t i = 0; i < n; i++) std::memcpy((void*)&terms[i], &keyed[i].term, sizeof(T));
}

}  // namespace luxir
