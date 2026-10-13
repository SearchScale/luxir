// Copyright 2020-2026 Yonik Seeley and Luxir contributors
// SPDX-License-Identifier: Apache-2.0

#include "bench/luxir_bench.h"
#include "test/CollectionHelper.h"
#include "test/LocalReq.h"
#include "luxir/reader/SkipStats.h"

using namespace luxir;
using namespace luxir::test;

// The uncollapsed cases use only the existing request API, so this file can
// also be copied into a HEAD checkout for a same-corpus release A/B.
static void BM_CollapseTopDocs(benchmark::State& state, int32_t mode) {
  int32_t nDocs = unit_tests ? (int32_t)LuxirTest::scaleTestWork(500) : 1'000'000;
  int32_t nGroups = unit_tests ? 23 : 50'000;
  std::vector<int32_t> sizes(5, nDocs / 5);
  nDocs = sizes[0] * 5;
  CollectionHelper helper("collapse_bm");
  if (!helper.indexMatchesShape(sizes)) {
    helper.clear();
    for (int32_t segment = 0; segment < 5; segment++) {
      std::vector<Doc> docs;
      docs.reserve((size_t)sizes[(size_t)segment]);
      for (int32_t doc = 0; doc < sizes[(size_t)segment]; doc++) {
        int32_t id = segment * sizes[0] + doc;
        // Impact bounds can prune the lower-score bands. Every segment
        // contains many groups and repeated members, in scattered key order.
        std::string body;
        int32_t freq = doc < sizes[0] / 10 ? 8 : doc < sizes[0] / 2 ? 3 : 1;
        for (int32_t word = 0; word < 16; word++) body += word < freq ? "alpha " : "beta ";
        docs.push_back(flatdoc("id", std::to_string(id), "body_w", body,
            "key_s", "g" + std::to_string(id % nGroups), "sort_i", (int64_t)(id * 37 % nDocs)));
      }
      ASSERT_TRUE(helper.indexAll(docs, UpdateMessage::COMMIT).success);
    }
  }

  std::string json = R"({"collection":"collapse_bm","query":"body_w:alpha","fields":["_score_"],"limit":)"
      + std::to_string(mode == 3 ? -1 : 100);
  if (mode == 1) json += R"(,"sort":"sort_i asc")";
  if (mode == 2) json += R"(,"collapse":"key_s")";
  json += '}';
  // Warm the reader/OrdMap and record decode work outside the timing loop.
  {
    SkipStatsScope stats;
    auto req = localReq(LuxirTest::luxirNode->getSearchEngine());
    if (!luxir::api::read_json(req->rawRequest(), json, req->mr)) {
      state.SkipWithMessage("collapse not implemented in this baseline");
      return;
    }
    req->execute(false);
    ASSERT_TRUE(req->ok()) << req->errorMsg();
    state.counters["doc_blocks"] = (double)SkipStats::docBlocksDecoded;
    state.counters["tfreq_blocks"] = (double)SkipStats::tfreqBlocksDecoded;
  }
  for (auto _ : state) {
    auto req = localReq(LuxirTest::luxirNode->getSearchEngine());
    ASSERT_TRUE(luxir::api::read_json(req->rawRequest(), json, req->mr));
    req->execute(false);
    ASSERT_TRUE(req->ok()) << req->errorMsg();
    int64_t rows = 0;
    for (auto* response : req->responses) rows += response->proto.ops.at("q")->docList()->row_count;
    benchmark::DoNotOptimize(rows);
    ASSERT_GT(rows, 0);
  }
  state.SetItemsProcessed(state.iterations() * nDocs);
}

LUXIR_BENCHMARK_CAPTURE(BM_CollapseTopDocs, docs_score, 0);
LUXIR_BENCHMARK_CAPTURE(BM_CollapseTopDocs, docs_field, 1);
LUXIR_BENCHMARK_CAPTURE(BM_CollapseTopDocs, collapse_score, 2);
LUXIR_BENCHMARK_CAPTURE(BM_CollapseTopDocs, exhaustive_docs, 3);
