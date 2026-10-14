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
            "key_s", "g" + std::to_string(id % nGroups), "sort_i", (int64_t)(id * 37 % nDocs), "seq_i", (int64_t)id + 1, "sort_s", "s" + std::to_string(id * 17 % 101)));
      }
      ASSERT_TRUE(helper.indexAll(docs, UpdateMessage::COMMIT).success);
    }
  }

  bool churn = mode == 4 || mode == 5;
  std::string query = churn ? "rescore(*:*, seq_i)" : "body_w:alpha";
  bool unscored = mode == 6 || mode == 7 || mode >= 9;
  std::string fields = unscored ? R"([])" : R"(["_score_"])";
  std::string json = R"({"collection":"collapse_bm","query":")" + query
      + R"(","fields":)" + fields + R"(,"limit":)" + std::to_string((mode == 3 || mode == 9 || mode == 10) ? -1 : churn ? 10 : 100);
  if (mode == 1 || mode == 6 || mode == 9 || mode == 11) json += R"(,"sort":"sort_i asc")";
  if (mode == 7 || mode == 10 || mode == 12) json += R"(,"sort":"sort_s desc")";
  if (mode == 8) json += R"(,"sort":["sort_i desc","score desc"])";
  if (mode == 2 || mode == 4 || (mode >= 6 && mode <= 8)) json += R"(,"collapse":"key_s")";
  if (mode == 5) json += R"(,"collapse":{"field":["key_s","sort_i"]})";
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

LUXIR_BENCHMARK_CAPTURE(BM_CollapseTopDocs, churn_score, 4);
LUXIR_BENCHMARK_CAPTURE(BM_CollapseTopDocs, churn_tuple, 5);
LUXIR_BENCHMARK_CAPTURE(BM_CollapseTopDocs, collapse_field_numeric, 6);
LUXIR_BENCHMARK_CAPTURE(BM_CollapseTopDocs, collapse_field_string, 7);
LUXIR_BENCHMARK_CAPTURE(BM_CollapseTopDocs, collapse_field_score_secondary, 8);
LUXIR_BENCHMARK_CAPTURE(BM_CollapseTopDocs, exhaustive_field_numeric, 9);
LUXIR_BENCHMARK_CAPTURE(BM_CollapseTopDocs, exhaustive_field_string, 10);
LUXIR_BENCHMARK_CAPTURE(BM_CollapseTopDocs, docs_field_unscored, 11);
LUXIR_BENCHMARK_CAPTURE(BM_CollapseTopDocs, docs_field_string, 12);
