// Copyright 2020-2026 Yonik Seeley and Luxir contributors
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <algorithm>
#include <random>
#include <set>
#include <string>
#include <vector>

#include "luxir/util/TermSort.h"
#include "luxir/util/TermValHash.h"

using namespace luxir;

namespace {

// Sorts the (unique) strings as pool terms in shuffled order and checks byte order.
void checkSort(const std::set<std::string>& unique) {
  MemPool pool;
  std::vector<PackedTerm> terms;
  for (auto& s : unique) terms.emplace_back(pool, s);
  std::shuffle(terms.begin(), terms.end(), std::mt19937_64(1));
  sortTerms(terms.data(), terms.size());
  auto it = unique.begin();
  for (size_t i = 0; i < terms.size(); i++, it++) {
    ASSERT_EQ(std::string_view(terms[i]), *it) << "at " << i;
  }
}

}  // namespace

// A fixed 11-byte head, then few distinct bytes (including 0 and 0xff): the
// first digit ties, the second radix partitions and recurses into tied runs,
// and terms that end early meet terms continuing with zero bytes at every
// depth up to 255.
TEST(TermSort, randomBinaryTerms) {
  std::mt19937_64 rng(7);
  const char alphabet[] = {'\0', '\x01', 'a', '\xff'};
  std::set<std::string> unique;
  while (unique.size() < 50'000) {
    std::string s(rng() % 245, '\0');
    for (auto& c : s) c = alphabet[rng() % 4];
    unique.insert("0123456789a" + s);
  }
  checkSort(unique);
}

// A shared prefix spanning several digits, at a size where integer_sort radix
// partitions. A comparator tie-break is never consulted in equal-key buckets,
// so this is the shape a single-pass prefix-key sort leaves unordered.
TEST(TermSort, longSharedPrefix) {
  std::set<std::string> unique;
  for (int i = 0; i < 100'000; i++) unique.insert("https://example.org/" + std::to_string(i * 7919 % 100'003));
  checkSort(unique);
}

// One digit-identical run longer than the radix threshold, distinguished only
// by length once every term is exhausted.
TEST(TermSort, trailingZeroChain) {
  std::set<std::string> unique;
  for (int k = 0; k < 100; k++) {
    unique.insert(std::string(16, 'x') + std::string(k, '\0'));
    unique.insert(std::string(3, 'y') + std::string(k, '\0'));
  }
  checkSort(unique);
}

TEST(TermSort, termValHashKeepsValues) {
  MemPool pool;
  TermValHash<int32_t> hash(pool, 4);
  std::vector<std::string> inserted;
  for (int i = 0; i < 20'000; i++) {
    inserted.push_back("term" + std::to_string(i * 37 % 20'011));
    hash.try_emplace(inserted.back(), i);
  }
  auto* terms = hash.destructiveSort();
  ASSERT_EQ(hash.size(), inserted.size());
  for (size_t i = 0; i < hash.size(); i++) {
    if (i > 0) {
      ASSERT_LT(terms[i - 1], terms[i]);
    }
    ASSERT_EQ(std::string_view(terms[i]), inserted[(size_t)terms[i].val()]);
  }
}
