// Copyright 2020-2026 Yonik Seeley and Luxir contributors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <boost/unordered/unordered_flat_map.hpp>
#include <boost/unordered/unordered_flat_set.hpp>
#include <optional>
#include <vector>

#include "luxir/api/build.h"
#include "luxir/search/IndexReader.h"
#include "luxir/reader/IntColReader.h"
#include "luxir/reader/OrdColReader.h"
#include "luxir/search/OrdMapStr.h"

namespace luxir {

// Types are fixed by the resolved key specification. Null is tagged separately
// from every value, including numeric extrema and string ordinal zero.
struct CollapseKey {
  struct Component {
    bool present = false;
    int64_t value = 0;
    auto operator<=>(const Component&) const = default;
  };
  // Ordinary tuples sort before expand singletons; null precedes a value.
  std::optional<uint64_t> singleton;
  std::vector<Component> components;
  auto operator<=>(const CollapseKey&) const = default;
};

struct CollapseKeyHash {
  size_t operator()(const CollapseKey& key) const {
    size_t hash = 0;
    boost::hash_combine(hash, key.singleton.has_value());
    if (key.singleton) boost::hash_combine(hash, *key.singleton);
    for (auto component : key.components) {
      boost::hash_combine(hash, component.present);
      boost::hash_combine(hash, component.value);
    }
    return hash;
  }
};

struct CollapsePlan {
  struct Field {
    std::string name;
    luxir::api::GroupComponent_::Type type;
    std::shared_ptr<OrdMap> ords;
    IndexReader::FieldSegments segments;
    bool ordinal() const { return type == luxir::api::GroupComponent_::Type::STRING
                               || type == luxir::api::GroupComponent_::Type::ID; }
  };
  std::vector<Field> fields;
  luxir::api::Collapse_::MissingPolicy missing;
  IndexReader& reader;
  bool getNumber;

  CollapsePlan(IndexReader& reader, luxir::api::Collapse_::MissingPolicy missing,
               bool getNumber) : missing(missing), reader(reader), getNumber(getNumber) {}

  void emit(luxir::api::GroupKey& out, const CollapseKey& key,
            std::pmr::memory_resource& mr, MemPool& pool) const {
    if (key.singleton) { out.singleton = key.singleton; return; }
    auto* components = luxir::api::build::allocArray(out.component, fields.size(), mr);
    for (size_t i = 0; i < fields.size(); i++) {
      components[i].type = fields[i].type;
      if (!key.components[i].present) {
        components[i].value = google::protobuf::NullValue::NULL_VALUE;
      } else if (fields[i].ordinal()) {
        OrdMapStr strings(pool, fields[i].ords.get(), reader, fields[i].segments);
        components[i].value = luxir::api::build::arenaStr(
            mr, strings.ordToStr(key.components[i].value));
      } else components[i].value = key.components[i].value;
    }
  }
};

// Segment-local decode state. Identities contain global ords and actual int64
// values, so independent workers compare and union keys without an interner lock.
class CollapseKeyReader {
  struct Column {
    std::optional<OrdColReader> ords;
    OrdMap::SegToGlobal mapping;
    std::optional<IntColReader> numbers;
    std::optional<IntColReader::Iterator> values;
  };
  const CollapsePlan& plan;
  std::vector<Column> columns;
  CollapseKey scratch;

public:
  explicit CollapseKeyReader(const CollapsePlan& plan) : plan(plan) {}

  void setSegment(int32_t segment) {
    columns.clear();
    columns.resize(plan.fields.size());
    scratch.components.resize(plan.fields.size());
    auto& postings = plan.reader.segments()[(size_t)segment].postingsReader();
    for (size_t i = 0; i < columns.size(); i++) {
      auto* info = plan.fields[i].segments.at((size_t)segment);
      if (!info) continue;
      if (plan.fields[i].ordinal()) {
        columns[i].ords.emplace(postings, *info);
        columns[i].mapping = plan.fields[i].ords->getSegToGlobal(segment);
      } else {
        columns[i].numbers.emplace(postings, *info);
        columns[i].values.emplace(*columns[i].numbers);
      }
    }
  }

  const CollapseKey* read(int32_t segment, int32_t doc) {
    scratch.singleton.reset();
    scratch.components.resize(columns.size());
    bool missing = false;
    for (size_t i = 0; i < columns.size(); i++) {
      auto& column = columns[i];
      auto& component = scratch.components[i];
      component = {};
      if (column.ords) {
        int32_t ord = column.ords->ordAt(doc);
        if (ord != 0) component = {true, column.mapping.globalOrd((int64_t)ord - 1)};
      } else if (column.values) {
        // A missing doc can leave the sparse iterator on a future value.
        // Its forward-seek contract requires target > current.
        if (column.values->docId() < doc) column.values->advance(doc);
        if (column.values->docId() == doc) component = {true, column.values->value()};
      }
      missing |= !component.present;
    }
    using Missing = luxir::api::Collapse_::MissingPolicy;
    if (missing && plan.missing == Missing::DROP) return nullptr;
    if (missing && plan.missing == Missing::EXPAND) {
      scratch.components.clear();
      scratch.singleton = ((uint64_t)(uint32_t)segment << 32) | (uint32_t)doc;
    }
    return &scratch;
  }
};

// Only retained identities are interned for ranking: state is O(topCount).
// Exhaustive totals have a separate bitmap (one ordinal key) or exact set.
class CollapseAdmission {
public:
  const CollapsePlan& plan;
  CollapseKeyReader reader;
  std::vector<CollapseKey> keys;
  boost::unordered_flat_map<CollapseKey, size_t, CollapseKeyHash> retained;
  std::vector<uint64_t> seenOrds;
  boost::unordered_flat_set<CollapseKey, CollapseKeyHash> seenKeys;
  int64_t singletons = 0;

  explicit CollapseAdmission(const CollapsePlan& plan) : plan(plan), reader(plan) {
    if (plan.getNumber && plan.fields.size() == 1 && plan.fields[0].ordinal()) {
      seenOrds.resize((size_t)(((plan.fields[0].ords ? plan.fields[0].ords->numOrds() : 0) + 1 + 63) / 64));
    }
  }

  void count(const CollapseKey& key) {
    if (!plan.getNumber) return;
    if (key.singleton) { singletons++; return; }
    if (!seenOrds.empty()) {
      // Bit zero is the null key; present ord zero occupies bit one.
      int64_t ord = key.components[0].present ? key.components[0].value + 1 : 0;
      seenOrds[(size_t)(ord >> 6)] |= 1ULL << (ord & 63);
    } else seenKeys.insert(key);
  }

  int64_t totalGroups() const {
    int64_t count = singletons + (int64_t)seenKeys.size();
    for (uint64_t word : seenOrds) count += std::popcount(word);
    return count;
  }

  void mergeCounts(const CollapseAdmission& other) {
    singletons += other.singletons;
    for (size_t i = 0; i < seenOrds.size(); i++) seenOrds[i] |= other.seenOrds[i];
    seenKeys.insert(other.seenKeys.begin(), other.seenKeys.end());
  }

  size_t retain(const CollapseKey& key, size_t slot) {
    if (slot == keys.size()) keys.push_back(key);
    else { retained.erase(keys[slot]); keys[slot] = key; }
    retained.emplace(key, slot);
    return slot;
  }
};

} // namespace luxir
