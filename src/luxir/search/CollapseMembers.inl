// Copyright 2020-2026 Yonik Seeley and Luxir contributors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "luxir/reader/DocsEnum.h"
#include "luxir/reader/PointsReader.h"
#include "luxir/search/SearchOverrides.h"
#include <boost/unordered/unordered_flat_map.hpp>
#include <queue>

namespace luxir {
namespace collapse_members_detail {


// Postings stay compressed. Numeric exact-value lookups materialize just the
// touched value's point docs, in doc order, charged to the request.
struct KeyPosting {
  struct DeletePosting {
    std::pmr::memory_resource* resource;
    void operator()(DocsOnlyEnum* value) const {
      std::destroy_at(value);
      resource->deallocate(value, sizeof(DocsOnlyEnum), alignof(DocsOnlyEnum));
    }
  };
  std::unique_ptr<DocsOnlyEnum, DeletePosting> postings;
  std::pmr::vector<int32_t> docs;
  size_t cursor = 0;
  int32_t doc = -1;
  explicit KeyPosting(std::pmr::memory_resource* mr)
      : postings(nullptr, DeletePosting{mr}), docs(mr) {}
  void open(const TermsEnum::PostingsState& state) {
    auto* mr = docs.get_allocator().resource();
    auto* storage = (DocsOnlyEnum*)mr->allocate(sizeof(DocsOnlyEnum), alignof(DocsOnlyEnum));
    try { std::construct_at(storage, state); }
    catch (...) { mr->deallocate(storage, sizeof(DocsOnlyEnum), alignof(DocsOnlyEnum)); throw; }
    postings.reset(storage);
  }
  int32_t advance(int32_t target) {
    if (postings) return doc = postings->advance(target);
    cursor = (size_t)screaming::gallopLowerBound((int64_t)cursor,
        (int64_t)docs.size(), target, [&](int64_t i) { return docs[(size_t)i]; });
    return doc = cursor == docs.size() ? DocsOnlyEnum::END : docs[cursor];
  }
};
struct MemberPostings {
  size_t slot;
  std::pmr::vector<KeyPosting> components;
  explicit MemberPostings(size_t slot, std::pmr::memory_resource* mr)
      : slot(slot), components(mr) {}
  int32_t advance(int32_t target) {
    // Correlated tuple intersection, not a union of each component's values.
    for (size_t i = 0; i < components.size();) {
      auto& part = components[i];
      int32_t doc = part.doc >= target ? part.doc : part.advance(target);
      if (doc == DocsOnlyEnum::END) return doc;
      if (doc > target) { target = doc; i = 0; }
      else i++;
    }
    return target;
  }
};

// Both the postings merge and the domain walk are forward-only. In an array
// eligibility, gallop from the previous position rather than probing it anew.
struct Eligibility {
  DocSet* domain;
  std::span<int32_t> docs;
  int64_t cursor = 0;
  explicit Eligibility(DocSet* domain) : domain(domain) {
    if (domain && domain->type == DocSet::ARRAY) docs = ((ArrDocSet*)domain)->docs();
  }
  bool contains(int32_t doc) {
    if (domain == nullptr) return true;
    if (domain->type != DocSet::ARRAY) return domain->get(doc);
    cursor = screaming::gallopLowerBound(cursor, (int64_t)docs.size(), doc,
        [&](int64_t i) { return docs[(size_t)i]; });
    return cursor != (int64_t)docs.size() && docs[(size_t)cursor] == doc;
  }
};

} // namespace collapse_members_detail

template <typename Visit>
void CollapseMembers::forSegment(int32_t segment, DocSet* eligibility,
    Query::SegmentSource& source, bool scores, Visit&& visit) {
  using namespace collapse_members_detail;
  auto& seg = plan.reader.segments()[(size_t)segment];
  MemPool pool;
  auto* supplier = source.scorerSupplier(pool, seg);
  if (supplier == nullptr) return;
  int64_t domainCost = std::min<int64_t>(supplier->cost(),
      eligibility ? eligibility->card() : seg.maxDoc());
  if (domainCost == 0) return;
  auto demand = Query::Demand::fromLeadCost(domainCost);
  auto* scorerPlan = supplier->resolve(pool, supplier->makePlanContext(demand));
  // Exact next/advance is the public scorer protocol, including two-phase
  // verification. No min-competitive score is installed during replay.
  auto* scorer = scorerPlan->build(pool);
  Eligibility eligible(eligibility);
  auto matches = [&](int32_t doc) {
    if (!eligible.contains(doc)) return false;
    if (scorer->docId() < doc) scorer->advance(doc);
    return scorer->docId() == doc;
  };
  auto member = [&](size_t slot, int32_t doc, float score) {
    if (collapseMemberStats) collapseMemberStats->members++;
    visit(slot, doc, score);
  };

  bool indexed = true;
  for (const auto& key : keys) {
    if (key.singleton) continue;
    for (const auto& component : key.components) indexed &= component.present;
  }
  std::vector<const SegFieldInfo*> info(plan.fields.size());
  for (size_t c = 0; c < info.size(); c++) {
    info[c] = plan.fields[c].segments.at(seg);
    if (!info[c]) { indexed = false; break; }
    if (!plan.fields[c].ordinal() && info[c]->pointsMetaOff == 0) indexed = false;
  }

  // Estimate before opening streams. A page covering a large part of a
  // selective domain is cheaper as one key-column walk. The benchmark can
  // force either traversal to measure the crossover.
  bool useKeys = indexed && collapseMemberStrategy != CollapseMemberStrategy::DOMAIN;
  if (useKeys && collapseMemberStrategy == CollapseMemberStrategy::AUTO) {
    int64_t groups = 0;
    for (const auto& field : plan.fields) {
      if (field.ordinal()) groups = std::max(groups, field.ords->numOrds());
    }
    double estimated = groups ? (double)seg.maxDoc() * (double)keys.size() / (double)groups : 0.0;
    // Stream setup (term seek, cursor construction and inverse-ord lookup)
    // measured at about 32 domain-row visits per present key. Account for
    // absent keys on small segments; tuple setup and row costs both scale
    // with their component count. No dictionary or document scan is needed.
    // Measured on 20K products x 1/8/100 variants, 1/7 segments, pages
    // 10/100/1000, scalar/correlated-tuple keys and all/~1% queries. The
    // sparse-domain 0.5 discount below comes from the same release corpus.
    constexpr double STREAM_SETUP_ROWS = 32.0;
    double present = groups ? std::min((double)keys.size(), estimated) : (double)keys.size();
    estimated += STREAM_SETUP_ROWS * present;
    // Points have no ordinal cardinality. Leaf fences bound the exact-value
    // lookup work, including repeated tuple-component lookups. Decline an
    // expensive points plan before materializing any doc arrays.
    for (size_t c = 0; c < info.size() && estimated < (double)domainCost; c++) {
      if (plan.fields[c].ordinal()) continue;
      PointsReader points(seg.postingsReader(), *info[c]);
      for (const auto& key : keys) {
        if (key.singleton) continue;
        int64_t value = key.components[c].value;
        auto fence = points.fenceRange(value, value);
        if (!fence.empty) estimated += (double)(points.leafOrdinalEnd(fence.lastLeaf)
                                               - points.leafOrdinalStart(fence.firstLeaf));
        if (estimated >= (double)domainCost) break;
      }
    }
    // A sparse replay is sequential, while postings candidates must seek /
    // verify against it. After batching sparse domains, the measured crossover
    // prices that key-driven work at twice the sequential domain work.
    double domainWork = (double)domainCost;
    if (domainCost < DocSetBuilder::arrayLimitFor(seg.maxDoc())) domainWork *= 0.5;
    useKeys = estimated < domainWork;
  }
  if (useKeys) {
    CollapseKeyReader keyReader(plan);
    keyReader.setSegment(segment);
    std::vector<CollapseKey> local;
    std::vector<uint8_t> available;
    size_t remapBytes = keys.size() * (sizeof(CollapseKey) + sizeof(uint8_t) + sizeof(size_t));
    for (const auto& key : keys) {
      if (key.components.size() <= 2) continue;
      size_t tail = 2 * key.components.size() * sizeof(CollapseKey::Component);
      if (tail > std::numeric_limits<size_t>::max() - remapBytes) {
        tracker.chargeOverflow("collapse member state", "retained ord remap");
      }
      remapBytes += tail;
    }
    tracker.charge(remapBytes, "collapse member state", "retained ord remap");
    auto releaseRemap = scope_guard([&]() noexcept { tracker.release(remapBytes); });
    keyReader.remapRetained(keys, local, available);
    std::pmr::vector<MemberPostings> streams(&resource);
    streams.reserve(keys.size());
    for (size_t slot = 0; slot < keys.size(); slot++) {
      if (keys[slot].singleton || !available[slot]) continue;
      auto& stream = streams.emplace_back(slot, &resource);
      for (size_t c = 0; c < info.size(); c++) {
        auto& part = stream.components.emplace_back(&resource);
        int64_t value = local[slot].components[c].value;
        if (plan.fields[c].ordinal()) {
          TermsEnum terms(pool, seg.postingsReader(), *info[c]);
          terms.seekOrd(value);
          part.open(terms.postingsState());
        } else {
          PointsReader points(seg.postingsReader(), *info[c]);
          auto fence = points.fenceRange(value, value);
          if (!fence.empty) {
            for (uint32_t leaf = fence.firstLeaf; leaf <= fence.lastLeaf; leaf++) {
              for (const auto& point : points.decodeLeaf(leaf)) {
                if (point.value == value) part.docs.push_back(point.docid);
              }
            }
          }
          std::sort(part.docs.begin(), part.docs.end());
        }
      }
    }
    struct Entry { int32_t doc; size_t stream; };
    auto later = [](const Entry& a, const Entry& b) { return a.doc > b.doc; };
    std::priority_queue<Entry, std::pmr::vector<Entry>, decltype(later)> queue(
        later, std::pmr::vector<Entry>(&resource));
    for (size_t i = 0; i < streams.size(); i++) {
      int32_t doc = streams[i].advance(0);
      if (doc != DocsOnlyEnum::END) queue.push({doc, i});
    }
    if (collapseMemberStats) collapseMemberStats->keySegments++;
    while (!queue.empty()) {
      auto entry = queue.top(); queue.pop();
      auto& stream = streams[entry.stream];
      if (collapseMemberStats) collapseMemberStats->candidates++;
      if (matches(entry.doc)) member(stream.slot, entry.doc, scores ? scorer->score() : 0.0f);
      int32_t doc = stream.advance(entry.doc + 1);
      if (doc != DocsOnlyEnum::END) queue.push({doc, entry.stream});
    }
    return;
  }

  if (collapseMemberStats) collapseMemberStats->domainSegments++;
  struct Hash {
    size_t operator()(const CollapseKey* key) const { return CollapseKeyHash{}(*key); }
  };
  struct Equal {
    bool operator()(const CollapseKey* a, const CollapseKey* b) const { return *a == *b; }
  };
  using Slot = std::pair<const CollapseKey* const, size_t>;
  boost::unordered_flat_map<const CollapseKey*, size_t, Hash, Equal,
      std::pmr::polymorphic_allocator<Slot>> slots(&resource);
  for (size_t slot = 0; slot < keys.size(); slot++) {
    if (!keys[slot].singleton) slots.emplace(&keys[slot], slot);
  }
  if (slots.empty()) return;
  CollapseKeyReader keyReader(plan);
  if (domainCost < DocSetBuilder::arrayLimitFor(seg.maxDoc())) {
    // The replay scorer supplies the finalized domain, not merely eligibility.
    // Bound sparse key decodes to batches of those exact matches. This avoids
    // decoding every intervening key block without retaining a match-set or
    // identity array for the segment. Scores come from this same forward walk.
    constexpr size_t BATCH = OrdValueBatches::BATCH_SIZE;
    std::array<int32_t, BATCH> docs;
    std::array<float, BATCH> values;
    int32_t doc = scorer->next();
    while (doc != DocsOnlyEnum::END) {
      size_t size = 0;
      while (size < BATCH && doc != DocsOnlyEnum::END) {
        if (eligible.contains(doc)) {
          docs[size] = doc;
          values[size++] = scores ? scorer->score() : 0.0f;
        }
        doc = scorer->next();
      }
      if (size == 0) continue;
      size_t bytes = sizeof(ArrDocSet) + size * sizeof(int32_t);
      tracker.charge(bytes, "collapse member state", "replay batch");
      auto release = scope_guard([&]() noexcept { tracker.release(bytes); });
      ArrDocSet batch(std::vector<int32_t>(docs.data(), docs.data() + size));
      keyReader.withKeyWalk(segment, &batch, [&](auto keyAt) {
        for (size_t i = 0; i < size; i++) {
          if (collapseMemberStats) collapseMemberStats->candidates++;
          const auto* key = keyAt(docs[i]);
          if (key == nullptr) continue;
          auto found = slots.find(key);
          if (found != slots.end()) member(found->second, docs[i], values[i]);
        }
      });
    }
  } else {
    keyReader.withKeyWalk(segment, eligibility, [&](auto keyAt) {
      for (int32_t doc = scorer->next(); doc != DocsOnlyEnum::END; doc = scorer->next()) {
        if (!eligible.contains(doc)) continue;
        if (collapseMemberStats) collapseMemberStats->candidates++;
        const auto* key = keyAt(doc);
        if (key == nullptr) continue;
        auto found = slots.find(key);
        if (found != slots.end()) member(found->second, doc, scores ? scorer->score() : 0.0f);
      }
    });
  }
}

} // namespace luxir
