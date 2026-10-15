// Copyright 2020-2026 Yonik Seeley and Luxir contributors
// SPDX-License-Identifier: Apache-2.0

#include "bench/luxir_bench.h"
#include "test/CollectionHelper.h"
#include "test/LocalReq.h"
#include "luxir/search/SearchOverrides.h"
#include "luxir/reader/SkipStats.h"

using namespace luxir;
using namespace luxir::test;

// Product variants spread across segments. Every group has one shop component;
// the tuple is correlated and has the same membership as the scalar key.
// Args: page groups, variants/group, segments, selective, tuple.
static void BM_CollapseMembers(benchmark::State& state, int mode) {
  int32_t limit = (int32_t)state.range(0), variants = (int32_t)state.range(1);
  int32_t segments = (int32_t)state.range(2);
  bool selective = state.range(3), tuple = state.range(4);
  int32_t products = unit_tests ? 120 : (int32_t)LuxirTest::scaleTestWork(20'000);
  std::string name = "members_bm_" + std::to_string(products) + "_"
      + std::to_string(variants) + "_" + std::to_string(segments);
  CollectionHelper helper(name);
  int32_t total = products * variants;
  std::vector<int32_t> sizes;
  for (int32_t s = 0; s < segments; s++) sizes.push_back((s + 1) * total / segments - s * total / segments);
  if (!helper.indexMatchesShape(sizes)) {
    helper.clear();
    for (int32_t s = 0; s < segments; s++) {
      std::vector<Doc> docs;
      for (int32_t id = s * total / segments; id < (s + 1) * total / segments; id++) {
        int32_t product = id % products, variant = id / products;
        docs.push_back(flatdoc("id", std::to_string(id), "key_s", "p" + std::to_string(product),
            "shop_s", "s" + std::to_string(product % 17),
            "price_i", (int64_t)(100 + variants - variant), "rank_i", (int64_t)product,
            "eligible_s", (product + variant) % 101 == 0 ? "yes" : "no"));
      }
      ASSERT_TRUE(helper.indexAll(docs, UpdateMessage::COMMIT).success);
    }
  }
  SearchOverridesGuard guard(collapseMemberStrategy, collapseMemberStats);
  collapseMemberStrategy = mode == 1 ? CollapseMemberStrategy::KEYS
      : mode == 2 ? CollapseMemberStrategy::DOMAIN : CollapseMemberStrategy::AUTO;
  std::string query = selective ? "eligible_s:yes" : "*:*";
  std::string body = R"({"collection":")" + name + R"(","query":")" + query
      + R"(","sort":"rank_i desc","fields":["id"],"limit":)" + std::to_string(limit);
  std::string collapse = tuple ? R"(["key_s","shop_s"])" : R"("key_s")";
  auto json = body + R"(,"collapse":{"field":)" + collapse
      + (mode == 0 || mode == 4 ? "}}" : R"(,"get_number":true,"head_sort":"price_i asc"}})");
  auto execute = [&](const std::string& text) {
    auto req = localReq(helper.getSearchEngine());
    EXPECT_TRUE(api::read_json(req->rawRequest(), text, req->mr)) << text;
    req->execute(false);
    EXPECT_TRUE(req->ok()) << req->errorMsg();
    return req;
  };
  auto run = [&]() {
    auto page = execute(json);
    if (mode == 4) {
      // Emulated inner_hits: an independent filtered request for each page
      // group, including query preparation, count and top-1 head selection.
      for (auto* response : page->responses) {
        const auto& list = *response->proto.ops.at("q")->docList();
        for (const auto& key : list.groups->keys) {
          auto product = std::get<std::string_view>(key.component[0].value);
          std::string filter = "key_s:" + std::string(product);
          if (tuple) filter += " AND shop_s:" + std::string(std::get<std::string_view>(key.component[1].value));
          auto members = execute(R"({"collection":")" + name + R"(","query":")" + query
              + R"(","filter":[")" + filter
              + R"("],"sort":"price_i asc","limit":1,"get_number":true,"fields":["id"]})");
          int32_t rows = members->docList()->row_count;
          benchmark::DoNotOptimize(rows);
        }
      }
    }
    return page->memoryTracker.peak();
  };
  {
    CollapseMemberStats stats;
    collapseMemberStats = &stats;
    SkipStatsScope skip;
    auto peak = run();
    state.counters["peak_bytes"] = (double)peak;
    state.counters["candidates"] = (double)stats.candidates;
    state.counters["members"] = (double)stats.members;
    state.counters["key_segments"] = (double)stats.keySegments;
    state.counters["domain_segments"] = (double)stats.domainSegments;
    state.counters["doc_blocks"] = (double)SkipStats::docBlocksDecoded;
    collapseMemberStats = nullptr;
  }
  for (auto _ : state) benchmark::DoNotOptimize(run());
}

#define MEMBER_GRID(name, mode) \
  TUNING_BENCHMARK_CAPTURE(BM_CollapseMembers, name, mode)->UseRealTime() \
      ->ArgsProduct({{10, 100, 1000}, {1, 8, 100}, {1, 7}, {0, 1}, {0, 1}})
MEMBER_GRID(plain, 0);
MEMBER_GRID(keys, 1);
MEMBER_GRID(domain, 2);
MEMBER_GRID(auto, 3);
MEMBER_GRID(inner_hits, 4);
#undef MEMBER_GRID

// Small default-suite coverage; the complete measurement matrix is explicit.
LUXIR_BENCHMARK_CAPTURE(BM_CollapseMembers, page_keys, 1)->Args({10, 8, 3, 0, 1});
LUXIR_BENCHMARK_CAPTURE(BM_CollapseMembers, page_domain, 2)->Args({10, 8, 3, 1, 1});
