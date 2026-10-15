// Copyright 2020-2026 Yonik Seeley and Luxir contributors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <boost/container/small_vector.hpp>
#include <boost/unordered/unordered_flat_set.hpp>
#include <optional>
#include <vector>

#include "luxir/api/build.h"
#include "luxir/search/IndexReader.h"
#include "luxir/reader/IntColReader.h"
#include "luxir/reader/OrdColReader.h"
#include "luxir/search/OrdMapStr.h"
#include "luxir/search/ops/DomainIter.h"

namespace luxir {

// Types are fixed by the resolved key specification. Null is tagged separately
// from every value, including numeric extrema and string ordinal zero.
struct CollapseKey {
  struct Component {
    bool present = false;
    int64_t value = 0;
    bool operator==(const Component&) const = default;
  };
  // Singleton and component tags are identities, never result-order keys.
  std::optional<uint64_t> singleton;
  boost::container::small_vector<Component, 2> components;
  bool operator==(const CollapseKey& other) const {
    return singleton == other.singleton && components == other.components;
  }
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

// Tagged channels leave every int64 pattern available to ordinary keys.
struct GroupIdentity {
  enum Kind { VALUE, NULL_KEY, SINGLETON, DROP } kind;
  int64_t value = 0;
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

  GroupIdentity missingIdentity(int32_t segment, int32_t doc) const {
    using Missing = luxir::api::Collapse_::MissingPolicy;
    if (missing == Missing::GROUP) return {GroupIdentity::NULL_KEY};
    if (missing == Missing::DROP) return {GroupIdentity::DROP};
    return {GroupIdentity::SINGLETON,
        (int64_t)(((uint64_t)(uint32_t)segment << 32) | (uint32_t)doc)};
  }

  const CollapseKey* finishKey(CollapseKey& key, bool absent,
                                int32_t segment, int32_t doc) const {
    key.singleton.reset();
    if (absent) {
      auto identity = missingIdentity(segment, doc);
      if (identity.kind == GroupIdentity::DROP) return nullptr;
      if (identity.kind == GroupIdentity::SINGLETON) {
        key.components.clear();
        key.singleton = (uint64_t)identity.value;
      }
    }
    return &key;
  }

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

// Shared segment-local decode state. Facet walks use global ords and actual
// int64 values. Collapse can keep local ords until admission, then translate
// retained keys for output and cross-worker merge. No interner lock.
class CollapseKeyReader {
  struct Column {
    std::optional<OrdColReader> ords;
    OrdMap::SegToGlobal mapping;
    std::optional<IntColReader> numbers;
    std::optional<IntColReader::Iterator> values;
    bool dense = false;
  };
  const CollapsePlan& plan;
  std::vector<Column> columns;
  CollapseKey scratch;
  int32_t lastDoc = -1;

public:
  explicit CollapseKeyReader(const CollapsePlan& plan) : plan(plan) {}

  void setSegment(int32_t segment) {
    lastDoc = -1;
    columns.clear();
    columns.resize(plan.fields.size());
    scratch.components.resize(plan.fields.size());
    auto& postings = plan.reader.segments()[(size_t)segment].postingsReader();
    for (size_t i = 0; i < columns.size(); i++) {
      auto* info = plan.fields[i].segments.at((size_t)segment);
      if (!info) continue;
      columns[i].dense = info->docsWithField == postings.maxDoc();
      if (plan.fields[i].ordinal()) {
        columns[i].ords.emplace(postings, *info);
        columns[i].mapping = plan.fields[i].ords->getSegToGlobal(segment);
      } else {
        columns[i].numbers.emplace(postings, *info);
        columns[i].values.emplace(*columns[i].numbers);
      }
    }
  }

  const CollapseKey* read(int32_t segment, int32_t doc, bool localOrds = false) {
    // Bound-order and seeded field drivers may rewind within a segment.
    // Reset presence cursors only on an actual backward read; looking ahead
    // over a missing value must retain the future position.
    if (doc < lastDoc) {
      for (auto& column : columns) {
        if (column.values) column.values->reset();
      }
    }
    lastDoc = doc;
    scratch.singleton.reset();
    scratch.components.resize(columns.size());
    bool missing = false;
    for (size_t i = 0; i < columns.size(); i++) {
      auto& column = columns[i];
      auto& component = scratch.components[i];
      component = {};
      if (column.ords) {
        int32_t ord = column.ords->ordAt(doc);
        if (ord != 0) {
          int64_t value = (int64_t)ord - 1;
          if (!localOrds && column.mapping.bits != 0) {
            value = column.mapping.globalOrd(value);
          }
          component = {true, value};
        }
      } else if (column.values) {
        // A missing doc can leave the sparse iterator on a future value.
        // Its forward-seek contract requires target > current.
        if (column.values->docId() < doc) column.values->advance(doc);
        if (column.values->docId() == doc) component = {true, column.values->value()};
      }
      missing |= !component.present;
    }
    return plan.finishKey(scratch, missing, segment, doc);
  }

  // Invert only the retained ords, in global order. Nothing proportional to
  // maxDoc or the segment dictionary is allocated. An absent component means
  // that retained tuple cannot occur in this segment, not that it is null.
  void remapRetained(std::span<const CollapseKey> keys,
                     std::vector<CollapseKey>& local,
                     std::vector<uint8_t>& available) const {
    local.assign(keys.begin(), keys.end());
    available.assign(keys.size(), true);
    std::vector<size_t> order;
    order.reserve(keys.size());
    for (size_t component = 0; component < columns.size(); component++) {
      if (!plan.fields[component].ordinal()) continue;
      const auto& column = columns[component];
      order.clear();
      for (size_t slot = 0; slot < keys.size(); slot++) {
        if (available[slot] && !keys[slot].singleton
            && keys[slot].components[component].present) order.push_back(slot);
      }
      std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
        return keys[a].components[component].value < keys[b].components[component].value;
      });
      int64_t cursor = 0;
      const auto& mapping = column.mapping;
      auto globalAt = [&](int64_t ord) {
        skipCount(SkipStats::collapseRemapProbes);
        return mapping.globalOrd(ord);
      };
      for (size_t slot : order) {
        int64_t global = keys[slot].components[component].value;
        cursor = mapping.bits == 0 ? std::min(global, mapping.numOrds)
            : screaming::gallopLowerBound(cursor, mapping.numOrds, global, globalAt);
        if (cursor == mapping.numOrds || globalAt(cursor) != global) available[slot] = false;
        else local[slot].components[component].value = cursor;
      }
    }
  }

  // Price L inverse lookups against the documents this segment can supply.
  // Deep heaps use direct translation; identity maps need no remap at all.
  bool remapPays(int64_t depth, int64_t sourceCost) const {
    if (sourceCost <= 0) return false;
    int64_t work = 0;
    bool encoded = false;
    for (const auto& column : columns) {
      if (column.mapping.bits == 0) continue;
      encoded = true;
      int64_t probes = 1 + (int64_t)std::bit_width((uint64_t)depth)
          + (int64_t)std::bit_width((uint64_t)(column.mapping.numOrds
              / std::max<int64_t>(depth, 1)));
      // Include sorting the retained ords; cap before multiplying, including
      // unbounded limits whose depth spans the whole reader.
      if (depth > (sourceCost - work) / probes) return false;
      work += depth * probes;
    }
    return encoded && work < sourceCost;
  }

  void globalize(CollapseKey& key) const {
    if (key.singleton) return;
    for (size_t i = 0; i < columns.size(); i++) {
      auto& component = key.components[i];
      if (component.present && columns[i].mapping.bits != 0) {
        component.value = columns[i].mapping.globalOrd(component.value);
      }
    }
  }

private:
  // Bounded, forward-only cursors. The ordinary ordinal match is just a
  // two-pointer join; refill/seeks stay outside that per-document fast path.
  class OrdWalk {
    OrdValueBatches values;
    OrdValueBatches::Batch batch;
    int32_t position = 0;
    int32_t previous = -1;
    int64_t saved = -1;
    bool ended = false;

    int64_t seek(int32_t doc) {
      if (ended) return -1;
      while (position == batch.size || batch.docs[batch.size - 1] < doc) {
        if (!values.next(batch)) { ended = true; return -1; }
        position = 0;
      }
      position = (int32_t)(screaming::gallopLowerBound(
          batch.docs + position, batch.docs + batch.size, doc) - batch.docs);
      return batch.docs[position] == doc ? batch.ords[position++] : -1;
    }
  public:
    OrdWalk(Column& column, DocSet* domain, int32_t maxDoc)
        : values(domain, *column.ords, column.mapping, maxDoc, false, nullptr) {}
    int64_t read(int32_t doc) LUXIR_INLINE {
      assert(doc >= previous);
      if (doc == previous) return saved;
      previous = doc;
      if (position < batch.size && batch.docs[position] == doc) return saved = batch.ords[position++];
      return saved = seek(doc);
    }
  };

  template <typename Iterator>
  class NumberWalk {
    Iterator values;
    int32_t previous = -1;
    CollapseKey::Component saved;
  public:
    explicit NumberWalk(Column& column) : values(*column.numbers) {}
    CollapseKey::Component read(int32_t doc) LUXIR_INLINE {
      assert(doc >= previous);
      if (doc == previous) return saved;
      previous = doc;
      saved = {};
      if (values.docId() < doc) values.advance(doc);
      if (values.docId() == doc) saved = {true, values.value()};
      return saved;
    }
  };

  template <typename Iterator>
  class WalkColumn {
    std::optional<OrdWalk> ords;
    std::optional<NumberWalk<Iterator>> numbers;
  public:
    WalkColumn(Column& column, DocSet* domain, int32_t maxDoc) {
      if (column.ords) ords.emplace(column, domain, maxDoc);
      else if (column.numbers) numbers.emplace(column);
    }
    CollapseKey::Component read(int32_t doc) {
      if (ords) {
        int64_t value = ords->read(doc);
        return value >= 0 ? CollapseKey::Component{true, value} : CollapseKey::Component{};
      }
      return numbers ? numbers->read(doc) : CollapseKey::Component{};
    }
  };

public:
  // Facet value streams are in document order. One key cursor serves all of
  // a document's values. A second pass (missing or another query bucket) opens
  // a new walk, so numeric iterators never rewind within a walk.
  template <typename Visit>
  void withWalk(int32_t segment, DocSet* domain, Visit&& visit) {
    setSegment(segment);
    int32_t maxDoc = plan.reader.segments()[(size_t)segment].maxDoc();
    if (columns.size() == 1) {
      auto& info = columns[0];
      if (info.ords) {
        OrdWalk column(info, domain, maxDoc);
        if (info.dense) visit([&](int32_t doc) LUXIR_INLINE { return column.read(doc); });
        else visit([&](int32_t doc) LUXIR_INLINE -> GroupIdentity {
          int64_t value = column.read(doc);
          return value >= 0 ? GroupIdentity{GroupIdentity::VALUE, value} : plan.missingIdentity(segment, doc);
        });
      } else if (info.numbers) {
        withIntColIteratorType(domain, [&]<typename Iterator>() {
          NumberWalk<Iterator> column(info);
          if (info.dense) visit([&](int32_t doc) LUXIR_INLINE { return column.read(doc).value; });
          else visit([&](int32_t doc) LUXIR_INLINE -> GroupIdentity {
            auto value = column.read(doc);
            return value.present ? GroupIdentity{GroupIdentity::VALUE, value.value} : plan.missingIdentity(segment, doc);
          });
        });
      } else visit([&](int32_t doc) { return plan.missingIdentity(segment, doc); });
      return;
    }
    withIntColIteratorType(domain, [&]<typename Iterator>() {
      std::vector<WalkColumn<Iterator>> cursors;
      cursors.reserve(columns.size());
      for (auto& column : columns) cursors.emplace_back(column, domain, maxDoc);
      int32_t previous = -1;
      const CollapseKey* saved = nullptr;
      visit([&](int32_t doc) -> const CollapseKey* {
        if (doc == previous) return saved;
        previous = doc;
        scratch.components.resize(columns.size());
        bool absent = false;
        for (size_t i = 0; i < cursors.size(); i++) {
          scratch.components[i] = cursors[i].read(doc);
          absent |= !scratch.components[i].present;
        }
        return saved = plan.finishKey(scratch, absent, segment, doc);
      });
    });
  }

};

// Only retained identities are interned for ranking: state is O(topCount).
// Exhaustive totals have a separate bitmap (one ordinal key) or exact set.
class CollapseAdmission {
  // The lookup table owns only slot indexes. Hash/equality read the one key
  // owned by each stable slot, including after the vector's storage grows.
  struct SlotHash {
    using is_transparent = void;
    const std::vector<CollapseKey>* keys;
    size_t operator()(size_t slot) const { return (*this)((*keys)[slot]); }
    size_t operator()(const CollapseKey& key) const { return CollapseKeyHash{}(key); }
  };
  struct SlotEqual {
    using is_transparent = void;
    const std::vector<CollapseKey>* keys;
    bool operator()(size_t a, size_t b) const { return (*keys)[a] == (*keys)[b]; }
    bool operator()(size_t slot, const CollapseKey& key) const { return (*keys)[slot] == key; }
    bool operator()(const CollapseKey& key, size_t slot) const { return (*this)(slot, key); }
  };
  bool segmentLocal = false;
  std::vector<CollapseKey> localKeys;
  std::vector<uint8_t> localAvailable;
public:
  const CollapsePlan& plan;
  CollapseKeyReader reader;
  std::vector<CollapseKey> keys;
  boost::unordered_flat_set<size_t, SlotHash, SlotEqual> retained;
  std::vector<uint64_t> seenOrds;
  boost::unordered_flat_set<CollapseKey, CollapseKeyHash> seenKeys;
  int64_t singletons = 0;

  explicit CollapseAdmission(const CollapsePlan& plan)
      : plan(plan), reader(plan), retained(0, SlotHash{&keys}, SlotEqual{&keys}) {
    if (plan.getNumber && plan.fields.size() == 1 && plan.fields[0].ordinal()) {
      seenOrds.resize((size_t)(((plan.fields[0].ords ? plan.fields[0].ords->numOrds() : 0) + 1 + 63) / 64));
    }
  }

  // Hash/equality callbacks point to the key vectors, whose addresses are stable.
  CollapseAdmission(const CollapseAdmission&) = delete;
  CollapseAdmission& operator=(const CollapseAdmission&) = delete;

  void setSegment(int32_t segment, int64_t depth, int64_t sourceCost = -1) {
    reader.setSegment(segment);
    if (sourceCost < 0) sourceCost = plan.reader.segments()[(size_t)segment].maxDoc();
    // Exact totals need every global identity; remapping just retained groups
    // cannot replace that work. Facet walks also keep their global key space.
    bool nextLocal = !plan.getNumber && reader.remapPays(depth, sourceCost);
    if (!segmentLocal && !nextLocal) return;
    if (segmentLocal == nextLocal) retained.clear();
    else {
      // Rebind once on a key-space transition, keeping hash/equality as the
      // original direct slot-to-key lookup on every hit. Ordinary global
      // segments retain their table across switches, without a rebuild.
      auto* indexed = nextLocal ? &localKeys : &keys;
      retained = decltype(retained)(0, SlotHash{indexed}, SlotEqual{indexed});
    }
    segmentLocal = nextLocal;
    if (segmentLocal) reader.remapRetained(keys, localKeys, localAvailable);
    for (size_t slot = 0; slot < keys.size(); slot++) {
      if (!segmentLocal || localAvailable[slot]) retained.insert(slot);
    }
  }

  const CollapseKey* read(int32_t segment, int32_t doc) {
    return reader.read(segment, doc, segmentLocal);
  }

  void finishSegment() {
    if (!segmentLocal) return;
    retained = decltype(retained)(0, SlotHash{&keys}, SlotEqual{&keys});
    segmentLocal = false;
    for (size_t slot = 0; slot < keys.size(); slot++) retained.insert(slot);
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
    if (slot < keys.size() && (!segmentLocal || localAvailable[slot])) retained.erase(slot);
    if (slot == keys.size()) keys.push_back(key);
    else keys[slot] = key;
    if (segmentLocal) {
      if (slot == localKeys.size()) {
        localKeys.push_back(key);
        localAvailable.push_back(true);
      } else {
        localKeys[slot] = key;
        localAvailable[slot] = true;
      }
      // Local hits of retained groups never translate. Only admission of a
      // new group needs a global identity for output and cross-worker merge.
      reader.globalize(keys[slot]);
    }
    retained.insert(slot);
    return slot;
  }
};

} // namespace luxir
