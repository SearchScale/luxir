// Copyright 2020-2026 Yonik Seeley and Luxir contributors
// SPDX-License-Identifier: Apache-2.0

#include "test/LuxirTest.h"
#include "test/CollectionHelper.h"
#include "test/LocalReq.h"
#include "test/TestUtils.h"
#include "test/SchemaBuilder.h"
#include "luxir/query/KnnQuery.h"
#include "luxir/index/VectorIndexBuilder.h"
#include "luxir/search/SearchOverrides.h"

#include <random>
#include <set>

using namespace luxir;
using namespace luxir::test;

namespace {
using Part = std::variant<std::monostate, std::string, int64_t>;
using Key = std::vector<Part>;
using Row = std::pair<std::string, float>;

std::vector<Row> rows(const LocalReq& req) {
  std::vector<Row> result;
  for (auto* response : req.responses) {
    auto* value = response->proto.ops.find("q");
    if (value == nullptr) continue;
    for (const auto& doc : (*value)->docList()->docs) {
      result.emplace_back(std::string(doc.fields.at("id")->asString()),
          std::get<float>(doc.fields.at("_score_")->kind));
    }
  }
  return result;
}

std::vector<Row> oracle(const LocalReq& full, std::span<const std::string> fields,
                        std::string_view missing, size_t k) {
  std::set<Key> seen;
  std::vector<Row> result;
  for (auto* response : full.responses) for (const auto& doc : response->proto.ops.at("q")->docList()->docs) {
    Key key;
    bool absent = false;
    for (const auto& field : fields) {
      auto* part = doc.fields.find(field);
      if (part == nullptr || (*part)->isNull()) { absent = true; key.emplace_back(); }
      else if (auto* text = std::get_if<std::string_view>(&(*part)->kind)) key.emplace_back(std::string(*text));
      else key.emplace_back((*part)->asInt());
    }
    if (absent && missing == "drop") continue;
    if (absent && missing == "expand") key = {std::string("singleton"), std::string(doc.fields.at("id")->asString())};
    if (!seen.insert(key).second) continue;
    result.emplace_back(std::string(doc.fields.at("id")->asString()),
        std::get<float>(doc.fields.at("_score_")->kind));
    if (result.size() == k) return result;
  }
  return result;
}

std::string keySpec(std::span<const std::string> fields, std::string_view policy) {
  std::string s = R"({"field":[)";
  for (const auto& field : fields) { if (s.back() != '[') s += ','; s += '"' + field + '"'; }
  return s + R"(],"missing":")" + std::string(policy) + R"("})";
}

void schema(CollectionHelper& helper, api::VectorMetric metric = api::VectorMetric::L2) {
  SchemaBuilder b;
  auto& v = b.templ("_vs");
  v.type = api::FieldDef::FieldClass::VECTOR;
  v.column = true;
  v.multi = true;
  v.dims = 4;
  v.metric = metric;
  b.set(helper.collection());
}
}

class GroupKnnTest : public LuxirTest {
protected:
  auto run(std::string_view collection, std::string_view body, int32_t parallel = 1, bool ok = true) {
    auto request = localReq(luxirNode->getSearchEngine());
    std::string json = R"({"collection":")" + std::string(collection)
        + R"(","get_scores":true,"get_number":true,"document_format":"rows","batch_size":10000,)"
        + std::string(body) + '}';
    if (!api::read_json(request->rawRequest(), json, request->mr)) throw std::runtime_error(json);
    request->execute(parallel);
    EXPECT_EQ(ok, request->ok()) << request->errorMsg() << json;
    return request;
  }
  static std::string knn(int k, bool groups, bool exact = true, std::string_view extra = {},
                          std::string_view vector = "[0,0,0,0]") {
    return R"({"knn":{"field":"vec_vs","query":)" + std::string(vector) + R"(,"k":)" + std::to_string(k)
        + (groups ? R"(,"unit":"groups")" : "")
        + (exact ? R"(,"exact":true)" : "") + std::string(extra) + "}}";
  }
};

TEST_F(GroupKnnTest, exactMatchesExhaustiveDocumentOracle) {
  std::mt19937 rng(418);
  std::vector<Doc> docs;
  for (int64_t i = 0; i < 168; i++) {
    float x = (float)(rng() % 13), y = (float)(rng() % 11);
    auto doc = flatdoc("id", std::to_string(i), "eligible_s", i % 7 ? "yes" : "no");
    if (i % 17) doc.emplace_back("vec_vs", std::vector<std::vector<float>>{
        {x, y, 0, 0}, {y * 0.5f, x * 0.25f, 0, 0}});
    if (i % 5) doc.emplace_back("key_s", "g" + std::to_string(rng() % 13));
    if (i % 3) doc.emplace_back("key_i", i % 2 ? INT64_MIN + (int64_t)(rng() % 11) : INT64_MAX - (int64_t)(rng() % 11));
    if (i % 4) doc.emplace_back("when_dt", (int64_t)(rng() % 5) * 1000);
    docs.push_back(std::move(doc));
  }
  for (int segments : {1, 3, 7}) {
    std::string name = "knn_groups_" + std::to_string(segments);
    CollectionHelper helper(name);
    schema(helper);
    for (int segment = 0; segment < segments; segment++) {
      ASSERT_TRUE(helper.indexAll(std::span<const Doc>(docs).subspan(
          (size_t)(segment * 168 / segments), (size_t)(168 / segments)), UpdateMessage::COMMIT).success);
    }
    ASSERT_TRUE(helper.deleteByIds({"13", "100"}, UpdateMessage::COMMIT).success);
    for (bool filtered : {false, true}) {
      std::string filter = filtered ? R"(,"filter":["eligible_s:yes"] )" : "";
      auto full = run(name, R"("limit":-1,"query":)" + knn(1000, false) + filter);
      for (const auto& fields : std::vector<std::vector<std::string>>{
          {"key_s"}, {"key_i"}, {"when_dt"}, {"key_s", "key_i"}, {"key_i", "when_dt"}}) {
        for (auto policy : {"expand", "group", "drop"}) for (int k : {1, 9, 99, 1000}) {
          auto expected = oracle(*full, fields, policy, k);
          for (int parallel : {1, -1}) {
            auto actual = run(name, R"("limit":-1,"query":)" + knn(k, true)
                + R"(,"collapse":)" + keySpec(fields, policy) + filter, parallel);
            EXPECT_EQ(expected, rows(*actual)) << name << policy << k;
            EXPECT_EQ((int64_t)expected.size(), actual->docList()->found.value_or(-1));
          }
        }
      }
    }
  }
}

TEST_F(GroupKnnTest, adversarialGroupsPagingWrappersAndWitnessDomains) {
  CollectionHelper helper;
  schema(helper);
  for (int segment = 0; segment < 3; segment++) {
    std::vector<Doc> docs;
    for (int i = 0; i < 80; i++) docs.push_back(flatdoc("id", std::to_string(segment * 80 + i),
        "key_s", i < 70 ? "near" : "g" + std::to_string(i),
        "vec_vs", std::vector<std::vector<float>>{{i < 70 ? 0.0f : (float)i, 0, 0, 0}}));
    helper.indexAll(docs, UpdateMessage::COMMIT);
  }
  auto docs = run("main", R"("limit":-1,"collapse":"key_s","query":)" + knn(7, false));
  EXPECT_EQ(1u, rows(*docs).size());
  std::string group = knn(7, true);
  for (std::string query : {group, R"({"boolean":{"required":[)" + group + "]}}",
      R"({"rescore":{"query":)" + group + R"(,"expr":"score"}})"}) {
    auto actual = run("main", R"("limit":-1,"collapse":"key_s","query":)" + query
        + R"(,"ops":{"w":{"query_facet":{"count_unit":"docs","buckets":{"all":"*:*"}}}})", -1);
    EXPECT_EQ(7u, rows(*actual).size());
    EXPECT_EQ(7, actual->docList()->ops.at("w")->facetResult()->counts[0]);
    auto page = run("main", R"("limit":100,"offset":5,"collapse":"key_s","query":)" + query);
    auto expected = rows(*actual);
    ASSERT_GE(expected.size(), 5u);
    expected.erase(expected.begin(), expected.begin() + 5);
    EXPECT_EQ(expected, rows(*page)); // k is never enlarged by the page.
  }
}

TEST_F(GroupKnnTest, reportsExhaustionAndCapsAndRejectsMissingScope) {
  CollectionHelper helper;
  schema(helper);
  for (int i = 0; i < 8; i++) helper.index(flatdoc("id", std::to_string(i),
      "key_s", "g" + std::to_string(i), "vec_vs", std::vector<std::vector<float>>{{(float)i, 0, 0, 0}}));
  helper.commit();
  auto empty = run("main", R"("filter":["id:absent"],"collapse":"key_s","query":)" + knn(5, true));
  EXPECT_TRUE(rows(*empty).empty());
  ASSERT_EQ(1u, empty->responses.back()->proto.warnings.size());
  EXPECT_NE(std::string_view::npos, empty->responses.back()->proto.warnings[0].message.find("candidates_exhausted=true"));
  auto bad = run("main", R"("query":)" + knn(3, true), 1, false);
  EXPECT_NE(std::string::npos, bad->errorMsg().find("requires an enclosing collapse"));
  auto invalid = knn(3, false);
  invalid.insert(invalid.size() - 2, R"(,"unit":"bogus")");
  auto invalidUnit = run("main", R"("collapse":"key_s","query":)" + invalid, 1, false);
  EXPECT_NE(std::string::npos, invalidUnit->errorMsg().find("knn.unit must be docs or groups"));
  auto explicitDocs = knn(3, false);
  explicitDocs.insert(explicitDocs.size() - 2, R"(,"unit":"docs")");
  auto ordinary = run("main", R"("limit":-1,"query":)" + explicitDocs);
  EXPECT_EQ(3u, rows(*ordinary).size());
  auto exhausted = run("main", R"("limit":-1,"collapse":"key_s","query":)" + knn(20, true));
  ASSERT_EQ(1u, exhausted->responses.back()->proto.warnings.size());
  EXPECT_EQ("knn_group_underfill", exhausted->responses.back()->proto.warnings[0].code);
  EXPECT_NE(std::string_view::npos, exhausted->responses.back()->proto.warnings[0].message.find("candidates_exhausted=true"));
  SearchOverridesGuard guard(KnnQuery::maxKnnCandidates);
  KnnQuery::maxKnnCandidates = 2;
  auto capped = run("main", R"("limit":-1,"collapse":"key_s","query":)" + knn(5, true, false));
  EXPECT_EQ(2u, rows(*capped).size());
  ASSERT_EQ(1u, capped->responses.back()->proto.warnings.size());
  EXPECT_NE(std::string_view::npos, capped->responses.back()->proto.warnings[0].message.find("candidates_exhausted=false"));
  auto exact = run("main", R"("limit":-1,"collapse":"key_s","query":)" + knn(5, true));
  EXPECT_EQ(5u, rows(*exact).size()); // exact overrides the heuristic cap.
}

TEST_F(GroupKnnTest, actualIvfAndFlatSegmentsStillFindDistinctGroups) {
  CollectionHelper helper;
  schema(helper);
  for (int segment = 0; segment < 2; segment++) {
    std::vector<Doc> docs;
    for (int i = 0; i < 160; i++) {
      auto doc = flatdoc("id", std::to_string(segment * 160 + i),
          "vec_vs", std::vector<std::vector<float>>{{i < 128 ? 0.0f : (float)i, 0, 0, 0}});
      if (i % 7) doc.emplace_back("key_s", i < 128 ? "near" : "g" + std::to_string(i));
      if (i % 11) doc.emplace_back("key_i", (int64_t)(i < 128 ? 0 : i));
      docs.push_back(std::move(doc));
    }
    helper.indexAll(docs, UpdateMessage::COMMIT);
  }
  SearchOverridesGuard a(VectorIndexBuilder::ivfPqNList);
  SearchOverridesGuard b(VectorIndexBuilder::ivfPqM);
  SearchOverridesGuard c(VectorIndexBuilder::ivfPqBits);
  SearchOverridesGuard d(VectorIndexBuilder::ivfPqMinTrainingVectors);
  SearchOverridesGuard e(VectorIndexBuilder::ivfPqBuildThresholdScanCost);
  SearchOverridesGuard f(VectorIndexBuilder::buildFaissIvfPqAuxIndexes);
  VectorIndexBuilder::ivfPqNList = 4; VectorIndexBuilder::ivfPqM = 2;
  VectorIndexBuilder::ivfPqBits = 2; VectorIndexBuilder::ivfPqMinTrainingVectors = 16;
  VectorIndexBuilder::ivfPqBuildThresholdScanCost = 0; VectorIndexBuilder::buildFaissIvfPqAuxIndexes = true;
  helper.commit({"vec.vec_vs"});
  // A later segment has no aux, and the same near group straddles all three.
  helper.indexAll({flatdoc("id", "flat", "key_s", "flat", "key_i", (int64_t)1000,
      "vec_vs", std::vector<std::vector<float>>{{1, 0, 0, 0}})}, UpdateMessage::COMMIT);
  helper.deleteByIds({"0", "160"}, UpdateMessage::COMMIT);
  auto full = run("main", R"("limit":-1,"query":)" + knn(1000, false));
  for (auto fields : std::vector<std::vector<std::string>>{{"key_s"}, {"key_i"}, {"key_s", "key_i"}}) {
  for (auto policy : {"expand", "group", "drop"}) for (int parallel : {1, -1}) {
    KnnQuery::Stats stats;
    SearchOverridesGuard guard(KnnQuery::statsForTests);
    KnnQuery::statsForTests = &stats;
    auto actual = run("main", R"("limit":-1,"collapse":)" + keySpec(fields, policy)
        + R"(,"query":)" + knn(8, true, false, R"(,"refine_candidates":1000,"ivf":{"min_scan_fraction":1})"), parallel);
    EXPECT_EQ(oracle(*full, fields, policy, 8), rows(*actual));
    EXPECT_EQ(8u, rows(*actual).size());
    EXPECT_GT(stats.lists.load(), 0);
    EXPECT_GT(stats.bytes.load(), 0);
  }
  }
  auto pinned = run("main", R"("limit":-1,"collapse":"key_s","filter":["key_s:near"],"query":)"
      + knn(8, true, false, R"(,"ivf":{"nprobe":1})"));
  ASSERT_LT(rows(*pinned).size(), 8u);
  ASSERT_EQ(1u, pinned->responses.back()->proto.warnings.size());
  EXPECT_NE(std::string_view::npos, pinned->responses.back()->proto.warnings[0].message.find("candidates_exhausted=false"));

}

TEST_F(GroupKnnTest, identityIncludesResolvedKeyPolicyAndUnit) {
  CollectionHelper helper;
  schema(helper);
  helper.indexAll({flatdoc("id", "0", "key_s", "a", "other_s", "a", "vec_vs",
      std::vector<std::vector<float>>{{0, 0, 0, 0}})}, UpdateMessage::COMMIT);
  auto reader = helper.getIndexWriter()->snapshots.readers.getReader();
  const auto& type = (const VectorFieldType&)*reader->schema()->getFieldTypePtr("vec_vs");
  using Missing = api::Collapse_::MissingPolicy;
  CollapsePlan first(*reader, Missing::GROUP, false), equal(*reader, Missing::GROUP, false),
      policy(*reader, Missing::EXPAND, false), field(*reader, Missing::GROUP, false);
  first.fields.push_back({"key_s", api::GroupComponent_::Type::STRING, reader->getOrdMap("key_s"),
      reader->field("key_s")});
  equal.fields = policy.fields = first.fields;
  field.fields.push_back({"other_s", api::GroupComponent_::Type::STRING, reader->getOrdMap("other_s"),
      reader->field("other_s")});
  std::array<float, 4> vector{0, 0, 0, 0};
  KnnQuery a("vec_vs", type, vector, 2, 0, 0, 0, false, &first),
      b("vec_vs", type, vector, 2, 0, 0, 0, false, &equal),
      c("vec_vs", type, vector, 2, 0, 0, 0, false, &policy),
      d("vec_vs", type, vector, 2, 0, 0, 0, false, &field),
      docs("vec_vs", type, vector, 2);
  EXPECT_TRUE(a.equals(b));
  EXPECT_EQ(a.hash(), b.hash());
  for (const auto* other : {&c, &d, &docs}) {
    EXPECT_FALSE(a.equals(*other));
    EXPECT_NE(a.hash(), other->hash());
  }
  auto key = [](const KnnQuery& q) {
    FilterKeyBuilder builder;
    FilterKeyContext context;
    auto scope = q.appendFilterKey(builder, context);
    return std::move(builder).finishStructural(scope, context);
  };
  EXPECT_EQ(key(a), key(b));
  EXPECT_NE(key(a), key(c));
  EXPECT_NE(key(a), key(d));
  EXPECT_NE(key(a), key(docs));
}

TEST_F(GroupKnnTest, annAdmissionPrecedesTheVectorCutForIpAndCosine) {
  SearchOverridesGuard enabled(VectorIndexBuilder::buildFaissIvfPqAuxIndexes);
  SearchOverridesGuard threshold(VectorIndexBuilder::ivfPqBuildThresholdScanCost);
  SearchOverridesGuard lists(VectorIndexBuilder::ivfPqNList);
  SearchOverridesGuard m(VectorIndexBuilder::ivfPqM);
  SearchOverridesGuard bits(VectorIndexBuilder::ivfPqBits);
  SearchOverridesGuard training(VectorIndexBuilder::ivfPqMinTrainingVectors);
  VectorIndexBuilder::buildFaissIvfPqAuxIndexes = true;
  VectorIndexBuilder::ivfPqBuildThresholdScanCost = 0;
  VectorIndexBuilder::ivfPqNList = 4; VectorIndexBuilder::ivfPqM = 2;
  VectorIndexBuilder::ivfPqBits = 2; VectorIndexBuilder::ivfPqMinTrainingVectors = 16;
  for (auto metric : {api::VectorMetric::IP, api::VectorMetric::COSINE}) {
    std::string name = "group_metric_" + std::to_string((int)metric);
    CollectionHelper helper(name);
    schema(helper, metric);
    for (int group = 20; group > 0; group--) for (int member = 0; member < 8; member++) {
      helper.index(flatdoc("id", std::to_string(group * 8 + member), "key_s", "g" + std::to_string(group),
          "vec_vs", std::vector<std::vector<float>>{{(float)group, 1, 0, 0}, {(float)group * 0.75f, 0.5f, 0, 0}}));
    }
    helper.commit({"vec.vec_vs"});
    auto full = run(name, R"("limit":-1,"query":)" + knn(1000, false, true, {}, "[1,0,0,0]"));
    auto known = rows(*full);
    for (int parallel : {1, -1}) {
      KnnQuery::Stats stats;
      SearchOverridesGuard statsGuard(KnnQuery::statsForTests);
      KnnQuery::statsForTests = &stats;
      auto actual = run(name, R"("limit":-1,"collapse":"key_s","query":)"
          + knn(3, true, false, R"(,"refine_candidates":3,"ivf":{"min_scan_fraction":1})", "[1,0,0,0]"), parallel);
      EXPECT_GT(stats.lists.load(), 0);
      ASSERT_EQ(3u, rows(*actual).size());
      for (const auto& row : rows(*actual)) EXPECT_NE(known.end(), std::find(known.begin(), known.end(), row));
    }
  }
}

TEST_F(GroupKnnTest, cappedAdmissionMatchesGreedyAcrossPartitions) {
  CollectionHelper helper;
  helper.indexAll({flatdoc("id", "0", "key_s", "key")}, UpdateMessage::COMMIT);
  auto reader = helper.getIndexWriter()->snapshots.readers.getReader();
  CollapsePlan plan(*reader, api::Collapse_::MissingPolicy::GROUP, false);
  struct Entry { int score, doc, key; size_t slot = 0; };
  struct Better {
    bool operator()(const Entry& a, const Entry& b) const {
      return a.score != b.score ? a.score > b.score : a.doc < b.doc;
    }
  };
  std::mt19937 rng(903);
  for (int cap = 1; cap <= 4; cap++) for (int n : {0, 1, 7, 23, 150}) {
    std::vector<Entry> all;
    for (int i = 0; i < 150; i++) all.push_back({(int)(rng() % 13), i, (int)(rng() % 17)});
    auto select = [&](std::vector<Entry> input) {
      GroupAdmission<false> admission(plan, (size_t)cap);
      ExpandingPQ<Entry, Better, true> heap((size_t)n);
      for (const auto& row : input) {
        CollapseKey key; key.components.push_back({true, row.key});
        admission.admit(key, heap, row, [&](const Entry& old) { return Better{}(row, old); }, [](size_t, bool) {});
      }
      std::vector<Entry> result(heap.span().begin(), heap.span().end());
      std::sort(result.begin(), result.end(), Better{});
      return result;
    };
    auto expected = all;
    std::sort(expected.begin(), expected.end(), Better{});
    std::vector<int> counts(17);
    std::erase_if(expected, [&](const Entry& row) { return counts[(size_t)row.key]++ >= cap; });
    expected.resize(std::min(expected.size(), (size_t)n));
    auto verify = [&](const std::vector<Entry>& actual) {
      ASSERT_EQ(expected.size(), actual.size());
      for (size_t i = 0; i < expected.size(); i++) EXPECT_EQ(expected[i].doc, actual[i].doc) << cap << '/' << n;
    };
    std::shuffle(all.begin(), all.end(), rng);
    verify(select(all));
    std::vector<std::vector<Entry>> parts(7);
    for (auto row : all) parts[rng() % parts.size()].push_back(row);
    for (auto& part : parts) part = select(std::move(part));
    std::shuffle(parts.begin(), parts.end(), rng);
    std::vector<Entry> merged;
    for (const auto& part : parts) {
      merged.insert(merged.end(), part.begin(), part.end());
      merged = select(std::move(merged));
    }
    verify(merged);
  }
}

TEST_F(GroupKnnTest, approximateMemberCapRecoversExactRepresentative) {
  CollectionHelper helper;
  schema(helper);
  for (int group = 0; group < 48; group++) for (int member = 0; member < 2; member++) {
    helper.index(flatdoc("id", std::to_string(group * 2 + member), "key_s", "g" + std::to_string(group),
        "vec_vs", std::vector<std::vector<float>>{{(float)group + (float)member * 0.1f, 0, 0, 0}}));
  }
  SearchOverridesGuard enabled(VectorIndexBuilder::buildFaissIvfPqAuxIndexes);
  SearchOverridesGuard threshold(VectorIndexBuilder::ivfPqBuildThresholdScanCost);
  SearchOverridesGuard lists(VectorIndexBuilder::ivfPqNList);
  SearchOverridesGuard bits(VectorIndexBuilder::ivfPqBits);
  SearchOverridesGuard training(VectorIndexBuilder::ivfPqMinTrainingVectors);
  VectorIndexBuilder::buildFaissIvfPqAuxIndexes = true;
  VectorIndexBuilder::ivfPqBuildThresholdScanCost = 0;
  VectorIndexBuilder::ivfPqNList = 2;
  VectorIndexBuilder::ivfPqBits = 1;
  VectorIndexBuilder::ivfPqMinTrainingVectors = 8;
  helper.commit({"vec.vec_vs"});
  auto truth = rows(*run("main", R"("limit":-1,"collapse":"key_s","query":)" + knn(3, true)));
  auto reader = helper.getIndexWriter()->snapshots.readers.getReader();
  CollapsePlan plan(*reader, api::Collapse_::MissingPolicy::GROUP, false);
  plan.fields.push_back({"key_s", api::GroupComponent_::Type::STRING, reader->getOrdMap("key_s"),
      reader->field("key_s")});
  // Like QuantizingEngine in KnnQueryTest: transform a real exhaustive result,
  // retaining its document identities but deliberately reversing sibling scores.
  auto exhaustive = rows(*run("main", R"("limit":-1,"query":)" + knn(1000, false)));
  std::vector<VectorEngineHit> vectors;
  for (const auto& [id, score] : exhaustive) vectors.push_back({score, 0, std::stoll(id)});
  class ReversedSiblings final : public VectorEngine {
    const CollapsePlan& plan;
    std::vector<VectorEngineHit> vectors;
  public:
    ReversedSiblings(const CollapsePlan& plan, std::vector<VectorEngineHit> vectors) : plan(plan), vectors(std::move(vectors)) {}
    VectorSearchResult search(const VectorSearchRequest& request) override {
      struct Entry : VectorEngineHit { size_t slot = 0; };
      struct Better {
        bool operator()(const Entry& a, const Entry& b) const {
          return a.score != b.score ? a.score > b.score : a.valueRank < b.valueRank;
        }
      };
      ExpandingPQ<Entry, Better, true> heap((size_t)request.candidates);
      GroupAdmission<false> admission(plan, (size_t)request.perKey);
      admission.reader.setSegment(0);
      for (const auto& vector : vectors) {
        Entry row; static_cast<VectorEngineHit&>(row) = vector;
        row.score = 1.0f / (1.0f + (float)(row.valueRank / 2)) + (float)(row.valueRank % 2) * 0.001f;
        if (heap.size() == heap.capacity() && !Better{}(row, heap.top())) continue;
        const auto* key = admission.reader.read(0, (int32_t)row.valueRank);
        admission.admit(*key, heap, row, [&](const Entry& old) { return Better{}(row, old); }, [](size_t, bool) {});
      }
      VectorSearchResult result;
      for (const auto& row : heap.span()) result.hits.push_back(row);
      result.scoresAreExact = false;
      result.poolExhausted = true;
      return result;
    }
  };
  SearchOverridesGuard wrapper(KnnQuery::engineWrapperForTests);
  KnnQuery::engineWrapperForTests = [&](VectorEngine&, int64_t) { return std::make_unique<ReversedSiblings>(plan, vectors); };
  SearchOverridesGuard memberCap(KnnQuery::membersPerGroupForTests);
  for (int parallel : {1, -1}) {
    KnnQuery::membersPerGroupForTests = 1;
    auto lost = rows(*run("main", R"("limit":-1,"collapse":"key_s","query":)"
        + knn(3, true, false, R"(,"refine_candidates":12)"), parallel));
    EXPECT_NE(truth, lost);
    KnnQuery::membersPerGroupForTests = 0;
    auto recovered = rows(*run("main", R"("limit":-1,"collapse":"key_s","query":)"
        + knn(3, true, false, R"(,"refine_candidates":12)"), parallel));
    EXPECT_EQ(truth, recovered);
  }
}

TEST_F(GroupKnnTest, enrichmentReplaysOnlyAdmittedMembers) {
  CollectionHelper helper;
  schema(helper);
  ASSERT_TRUE(helper.indexAll({
      flatdoc("id", "a", "key_s", "x", "price_i", (int64_t)4, "vec_vs", std::vector<std::vector<float>>{{0.0f, 0, 0, 0}}),
      flatdoc("id", "b", "key_s", "x", "price_i", (int64_t)2, "vec_vs", std::vector<std::vector<float>>{{1.0f, 0, 0, 0}}),
      flatdoc("id", "c", "key_s", "x", "price_i", (int64_t)0, "vec_vs", std::vector<std::vector<float>>{{50.0f, 0, 0, 0}}),
      flatdoc("id", "d", "key_s", "y", "price_i", (int64_t)1, "vec_vs", std::vector<std::vector<float>>{{2.0f, 0, 0, 0}})},
      UpdateMessage::COMMIT).success);
  for (bool groups : {false, true}) for (int parallel : {1, -1}) {
    auto actual = run("main", R"("limit":-1,"fields":["id"],"query":)" + knn(3, groups)
        + R"(,"collapse":{"field":"key_s","get_number":true,"head_sort":"price_i asc","ops":{"members":{"top_docs":{"sort":"price_i asc","limit":-1,"fields":["id"],"document_format":"rows"}}}})", parallel);
    ASSERT_TRUE(actual->ok()) << actual->errorMsg();
    auto result = rows(*actual);
    ASSERT_EQ(2u, result.size());
    EXPECT_EQ(groups ? "a" : "b", result[0].first);
    const auto& table = *actual->docList()->groups;
    EXPECT_EQ(groups ? 1 : 2, table.counts[0]);
    auto& members = *std::get<api::ArrVal>(table.ops.at("members")->kind).v[0].docList();
    ASSERT_EQ(groups ? 1 : 2, members.row_count);
    EXPECT_EQ(groups ? "a" : "b", members.docs[0].fields.at("id")->asString());
    for (auto& doc : members.docs) EXPECT_NE("c", doc.fields.at("id")->asString());
    if (!groups) {
      ASSERT_EQ(1u, table.ranks[0].v.size());
      EXPECT_GT(std::get<float>(table.ranks[0].v[0].kind), result[0].second);
    }
  }
}

TEST_F(GroupKnnTest, groupChildNeedsItsOwnCollapse) {
  CollectionHelper helper;
  schema(helper);
  ASSERT_TRUE(helper.indexAll({flatdoc("id", "a", "key_s", "x", "vec_vs",
      std::vector<std::vector<float>>{{0.0f, 0, 0, 0}})}, UpdateMessage::COMMIT).success);
  auto bad = run("main", R"("collapse":{"field":"key_s","ops":{"child":{"top_docs":{"query":)"
      + knn(1, true) + "}}}}", 1, false);
  EXPECT_NE(std::string::npos, bad->errorMsg().find("enclosing collapse"));
  auto good = run("main", R"("collapse":{"field":"key_s","ops":{"child":{"top_docs":{"collapse":"key_s","query":)"
      + knn(1, true) + "}}}}", -1);
  ASSERT_TRUE(good->ok()) << good->errorMsg();
  const auto& table = *good->docList()->groups;
  EXPECT_EQ(1, std::get<api::ArrVal>(table.ops.at("child")->kind).v[0].docList()->row_count);
}
