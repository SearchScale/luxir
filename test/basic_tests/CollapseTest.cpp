// Copyright 2020-2026 Yonik Seeley and Luxir contributors
// SPDX-License-Identifier: Apache-2.0

#include "test/LuxirTest.h"
#include "test/CollectionHelper.h"
#include "test/LocalReq.h"
#include "test/TestUtils.h"
#include "test/SchemaBuilder.h"
#include "luxir/server/JsonResponse.h"
#include "luxir/search/FieldSortCollector.h"
#include "luxir/search/SearchOverrides.h"
#include "luxir/reader/SkipStats.h"
#include "luxir/query/QueryPrep.h"
#include <map>
#include <random>

using namespace luxir;
using namespace luxir::test;

namespace {
using Part = std::variant<std::monostate, std::string, int64_t>;
struct Key {
  bool singleton = false;
  std::vector<Part> parts;
  size_t docOrder = 0;
  auto operator<=>(const Key&) const = default;
};
struct Row {
  std::string id;
  float score;
  std::vector<Part> parts;
  bool singleton = false;
  bool operator==(const Row&) const = default;
};

std::string spec(const std::vector<std::string>& fields, std::string_view missing) {
  std::string out = R"({"field":[)";
  for (size_t i = 0; i < fields.size(); i++) {
    if (i) out += ',';
    out += '"' + fields[i] + '"';
  }
  return out + R"(],"missing":")" + std::string(missing) + R"("})";
}

Part part(const luxir::api::Val* value) {
  if (value == nullptr || value->isNull()) return {};
  if (auto* s = std::get_if<std::string_view>(&value->kind)) return std::string(*s);
  return value->asInt();
}

std::vector<Row> rows(LocalReq& request, bool collapsed) {
  std::vector<Row> result;
  for (auto* response : request.responses) {
    const auto& list = *response->proto.ops.at("q")->docList();
    if (collapsed) { EXPECT_EQ(list.groups.size(), (size_t)list.row_count); }
    for (size_t i = 0; i < list.docs.size(); i++) {
      const auto& fields = list.docs[i].fields;
      Row row{std::string(fields.at("id")->asString()),
              fields.find("_score_") ? std::get<float>(fields.at("_score_")->kind) : 0.0f, {}, false};
      if (collapsed) {
        row.singleton = list.groups[i].singleton.has_value();
        for (const auto& component : list.groups[i].component) {
          if (auto* s = std::get_if<std::string_view>(&component.value)) row.parts.emplace_back(std::string(*s));
          else if (auto* n = std::get_if<int64_t>(&component.value)) row.parts.emplace_back(*n);
          else row.parts.emplace_back();
        }
      }
      result.push_back(std::move(row));
    }
  }
  return result;
}

struct Oracle {
  std::vector<Row> groups;
  int64_t eligible = 0;
};

// G0: exhaustive ordinary ranked documents select the first member per key.
// Consecutive equal rank values form a band; only group identity reorders
// that band. This oracle neither constructs nor consults engine collectors.
Oracle fieldOracle(LocalReq& full, const std::vector<std::string>& keyFields,
                   std::string_view missing, const std::vector<std::string>& rankFields) {
  std::map<Key, std::pair<size_t, Row>> heads;
  std::vector<Part> previous;
  size_t band = 0;
  size_t docOrder = 0;
  Oracle result;
  for (auto* response : full.responses) {
    for (const auto& doc : response->proto.ops.at("q")->docList()->docs) {
      std::vector<Part> rank;
      for (const auto& field : rankFields) {
        auto* value = doc.fields.find(field);
        if (field == "_doc_") rank.emplace_back((int64_t)docOrder);
        else if (field == "score") rank.emplace_back((int64_t)std::bit_cast<uint32_t>(
            std::get<float>(doc.fields.at("_score_")->kind)));
        else rank.push_back(part(value ? &**value : nullptr));
      }
      if (docOrder != 0 && rank != previous) band++;
      previous = std::move(rank);
      Key key;
      bool absent = false;
      for (const auto& field : keyFields) {
        auto* value = doc.fields.find(field);
        key.parts.push_back(part(value ? &**value : nullptr));
        absent |= std::holds_alternative<std::monostate>(key.parts.back());
      }
      if (!absent || missing != "drop") {
        result.eligible++;
        if (absent && missing == "expand") {
          key.parts.clear(); key.singleton = true; key.docOrder = docOrder;
        }
        auto* score = doc.fields.find("_score_");
        heads.try_emplace(key, band, Row{std::string(doc.fields.at("id")->asString()),
            score ? std::get<float>((*score)->kind) : 0.0f, key.parts, key.singleton});
      }
      docOrder++;
    }
  }
  std::vector<std::pair<Key, std::pair<size_t, Row>>> ordered(heads.begin(), heads.end());
  std::sort(ordered.begin(), ordered.end(), [](const auto& a, const auto& b) {
    return a.second.first == b.second.first ? a.first < b.first : a.second.first < b.second.first;
  });
  for (auto& entry : ordered) result.groups.push_back(std::move(entry.second.second));
  return result;
}

}

class CollapseTest : public LuxirTest {
protected:
  auto run(std::string_view collection, std::string_view query, std::string_view collapse,
           int64_t limit = -1, int64_t offset = 0, int32_t parallel = 1,
           bool count = true, int32_t batch = 13,
           std::string_view sort = {}, bool scores = true) {
    auto req = localReq(luxirNode->getSearchEngine());
    std::string queryJson;
    EXPECT_TRUE(luxir::api::write_json(luxir::api::Val{.kind = query}, queryJson));
    std::string json = R"({"collection":")" + std::string(collection)
        + R"(","query":)" + queryJson
        + R"(,"fields":["id","key_s","key_i","when_dt","sort_i","sort_s"],"document_format":"rows","get_scores":)"
        + (scores ? "true" : "false") + R"(,"get_number":)"
        + (count ? "true" : "false") + R"(,"limit":)" + std::to_string(limit)
        + R"(,"offset":)" + std::to_string(offset)
        + R"(,"batch_size":)" + std::to_string(batch);
    if (!sort.empty()) json += R"(,"sort":)" + std::string(sort);
    if (!collapse.empty()) json += R"(,"collapse":)" + std::string(collapse);
    json += '}';
    EXPECT_TRUE(luxir::api::read_json(req->rawRequest(), json, req->mr)) << json;
    req->execute(parallel);
    EXPECT_TRUE(req->ok()) << req->errorMsg() << json;
    return req;
  }
};

TEST_F(CollapseTest, bruteForceOracle) {
  std::mt19937 random(937);
  std::vector<Doc> docs;
  for (int32_t i = 0; i < 210; i++) {
    auto doc = flatdoc("id", "d" + std::to_string(i), "body_w",
        i % 4 == 0 ? "alpha beta" : i % 4 == 1 ? "alpha alpha beta" : "beta gamma gamma");
    if (random() % 5 != 0) doc.emplace_back("key_s", "g" + std::to_string(random() % 23));
    if (random() % 4 != 0) doc.emplace_back("key_i", (int64_t)(random() % 7) - 3);
    if (random() % 3 != 0) doc.emplace_back("when_dt", (int64_t)(random() % 4) * 86400000);
    docs.push_back(std::move(doc));
  }
  for (int32_t segments : {1, 3, 7}) {
    std::string collection = "collapse" + std::to_string(segments);
    CollectionHelper helper(collection);
    for (int32_t i = 0; i < segments; i++) {
      ASSERT_TRUE(helper.indexAll(std::span<const Doc>(docs).subspan(
          (size_t)(i * 210 / segments), (size_t)(210 / segments)), UpdateMessage::COMMIT).success);
    }
    ASSERT_TRUE(helper.deleteByIds({"d13", "d100"}, UpdateMessage::COMMIT).success);
    for (std::string query : {"*:*", "body_w:alpha", "body_w:alpha OR body_w:beta", "body_w:alpha AND key_i:>=0"}) {
      auto full = run(collection, query, {});
      // The oracle starts with ordinary exhaustive ranked documents. First
      // witness per exact key is its best member, including document ties.
      for (const std::vector<std::string>& fields : {
          std::vector<std::string>{"key_s"}, {"key_i"}, {"when_dt"}, {"id"}, {"key_s", "key_i"}}) {
        for (std::string missing : {"expand", "group", "drop"}) {
          std::map<Key, Row> groups;
          size_t rank = 0;
          int64_t eligible = 0;
          for (auto* response : full->responses) {
            const auto& list = *response->proto.ops.at("q")->docList();
            for (const auto& doc : list.docs) {
              Key key;
              bool absent = false;
              for (const auto& field : fields) {
                auto* value = doc.fields.find(field);
                key.parts.push_back(part(value ? &**value : nullptr));
                absent |= std::holds_alternative<std::monostate>(key.parts.back());
              }
              if (absent && missing == "drop") { rank++; continue; }
              eligible++;
              if (absent && missing == "expand") { key.parts.clear(); key.singleton = true; key.docOrder = rank; }
              groups.try_emplace(key, Row{std::string(doc.fields.at("id")->asString()),
                  std::get<float>(doc.fields.at("_score_")->kind), key.parts, key.singleton});
              rank++;
            }
          }
          std::vector<std::pair<Key, Row>> expected(groups.begin(), groups.end());
          std::sort(expected.begin(), expected.end(), [](const auto& a, const auto& b) {
            return a.second.score == b.second.score ? a.first < b.first : a.second.score > b.second.score;
          });
          std::vector<Row> all;
          for (const auto& entry : expected) all.push_back(entry.second);
          auto collapse = spec(fields, missing);
          for (int32_t parallel : {1, -1}) {
            SCOPED_TRACE(collection + " " + query + " " + collapse + " parallel=" + std::to_string(parallel));
            auto actual = run(collection, query, collapse, -1, 0, parallel);
            EXPECT_EQ(all, rows(*actual, true));
            EXPECT_EQ((int64_t)all.size(), actual->docList()->found);
            EXPECT_EQ(eligible, actual->docList()->found_docs);
            for (bool count : {false, true}) {
              auto page = run(collection, query, collapse, 7, 4, parallel, count);
              auto begin = all.begin() + std::min<size_t>(4, all.size());
              auto end = all.begin() + std::min<size_t>(11, all.size());
              EXPECT_EQ(std::vector<Row>(begin, end), rows(*page, true));
              EXPECT_EQ(count, page->docList()->found.has_value());
              EXPECT_EQ(count, page->docList()->found_docs.has_value());
            }
            auto totals = run(collection, query, collapse, 0, 0, parallel);
            EXPECT_EQ((int64_t)all.size(), totals->docList()->found);
            EXPECT_EQ(eligible, totals->docList()->found_docs);
            EXPECT_EQ(0, totals->docList()->row_count);
          }
          if (fields.size() == 2 && missing == "group") {
            std::vector<Row> pages;
            for (size_t offset = 0; offset < all.size(); offset += 9) {
              auto page = run(collection, query, collapse, 9, (int64_t)offset, -1, false, 3);
              auto current = rows(*page, true);
              pages.insert(pages.end(), current.begin(), current.end());
            }
            EXPECT_EQ(all, pages);
          }
        }
      }
    }
  }
}

TEST_F(CollapseTest, wireAndEmptyPages) {
  CollectionHelper helper;
  ASSERT_TRUE(helper.indexAll({flatdoc("id", "a", "key_s", "x", "key_i", INT64_MIN),
      flatdoc("id", "b", "key_i", INT64_MAX), flatdoc("id", "c", "key_s", "x", "key_i", INT64_MIN)},
      UpdateMessage::COMMIT).success);
  auto req = run("main", "*:*", R"({"field":["key_s","key_i"],"missing":"group"})");
  std::vector<std::byte> binary;
  ASSERT_TRUE(luxir::api::encode(req->rawRequest(), binary));
  size_t size = binary.size();
  binary.resize(size + 16);
  auto copied = localReq(luxirNode->getSearchEngine());
  ASSERT_TRUE(luxir::api::decode(copied->rawRequest(), std::span(binary).first(size), copied->mr));
  copied->execute(-1);
  EXPECT_EQ(rows(*req, true), rows(*copied, true));
  const auto& response = req->responses[0]->proto;
  binary.clear();
  ASSERT_TRUE(luxir::api::encode(response, binary));
  size = binary.size();
  binary.resize(size + 16);
  luxir::api::SearchResponse decoded;
  std::pmr::monotonic_buffer_resource arena;
  ASSERT_TRUE(luxir::api::decode(decoded, std::span(binary).first(size), arena));
  EXPECT_EQ(2, decoded.ops.at("q")->docList()->groups.size());
  EXPECT_EQ(INT64_MAX, std::get<int64_t>(decoded.ops.at("q")->docList()->groups[0].component[1].value));
  std::string listJson;
  ASSERT_TRUE(luxir::api::write_json(*req->docList(), listJson));
  luxir::api::DocList jsonList;
  ASSERT_TRUE(luxir::api::read_json(jsonList, listJson, arena)) << listJson;
  EXPECT_EQ(2, jsonList.groups.size());
  EXPECT_EQ(3, jsonList.found_docs);
  EXPECT_EQ(INT64_MAX, std::get<int64_t>(jsonList.groups[0].component[1].value));
  auto empty = run("main", "*:*", R"("key_s")", 10, 500);
  EXPECT_EQ(0, empty->docList()->row_count);
  EXPECT_EQ(500, empty->docList()->offset);
  EXPECT_EQ(2, empty->docList()->found);
  EXPECT_EQ(3, empty->docList()->found_docs);
  std::string json;
  ASSERT_TRUE(luxir::api::write_json(req->rawRequest(), json));
  EXPECT_NE(std::string::npos, json.find(R"("collapse":{"field":["key_s","key_i"],"missing":"group"})"));
}

TEST_F(CollapseTest, fieldSortWire) {
  CollectionHelper helper;
  ASSERT_TRUE(helper.indexAll({flatdoc("id", "a", "key_s", "x", "sort_i", (int64_t)9),
      flatdoc("id", "b", "key_s", "x", "sort_i", (int64_t)3),
      flatdoc("id", "c", "key_s", "y", "sort_i", (int64_t)3)}, UpdateMessage::COMMIT).success);
  auto req = run("main", "*:*", R"("key_s")", -1, 0, -1, true, 1,
      R"(["sort_i asc","score desc"])");
  for (bool protobuf : {false, true}) {
    auto copy = localReq(luxirNode->getSearchEngine());
    std::vector<std::byte> binary;
    std::string json;
    if (protobuf) {
      ASSERT_TRUE(luxir::api::encode(req->rawRequest(), binary));
      size_t size = binary.size();
      binary.resize(size + 16);
      ASSERT_TRUE(luxir::api::decode(copy->rawRequest(), std::span(binary).first(size), copy->mr));
    } else {
      ASSERT_TRUE(luxir::api::write_json(req->rawRequest(), json));
      ASSERT_TRUE(luxir::api::read_json(copy->rawRequest(), json, copy->mr));
    }
    copy->execute(-1);
    ASSERT_TRUE(copy->ok()) << copy->errorMsg();
    EXPECT_EQ(rows(*req, true), rows(*copy, true));
    EXPECT_EQ(2, copy->docList()->found);
    EXPECT_EQ(3, copy->docList()->found_docs);
    for (auto* response : copy->responses) {
      std::vector<std::byte> binary;
      ASSERT_TRUE(luxir::api::encode(response->proto, binary));
      size_t size = binary.size();
      binary.resize(size + 16);
      luxir::api::SearchResponse decoded;
      std::pmr::monotonic_buffer_resource arena;
      ASSERT_TRUE(luxir::api::decode(decoded, std::span(binary).first(size), arena));
      auto& list = *decoded.ops.at("q")->docList();
      EXPECT_EQ((size_t)list.row_count, list.groups.size());
      EXPECT_EQ(2, list.found);
      EXPECT_EQ(3, list.found_docs);
    }
  }
}

TEST_F(CollapseTest, validationAndAbsentKeys) {
  CollectionHelper helper;
  ASSERT_TRUE(helper.index(flatdoc("id", "a"), UpdateMessage::COMMIT).success);
  for (std::string collapse : {R"({"field":[]})", R"({"field":["key_s","key_s"]})",
      R"("key_f")", R"("body_w")", R"("keys_ss")"}) {
    auto req = localReq(luxirNode->getSearchEngine());
    ASSERT_TRUE(luxir::api::read_json(req->rawRequest(),
        R"({"collection":"main","limit":0,"collapse":)" + collapse + "}", req->mr));
    req->execute();
    EXPECT_FALSE(req->ok()) << collapse;
  }
  for (std::string body : {
      R"({"ops":{"f":{"fusion":{"sources":{"s":{"collapse":"key_s"}},"rrf":{}}}}})",
      R"({"ops":{"f":{"fusion":{"sources":{"s":{}},"rrf":{},"ops":{"q":{"top_docs":{"collapse":"key_s"}}}}}}})",
      R"({"ops":{"f":{"field_facet":{"field":"key_s","ops":{"q":{"top_docs":{"collapse":"key_s"}}}}}}})"}) {
    auto req = localReq(luxirNode->getSearchEngine());
    ASSERT_TRUE(luxir::api::read_json(req->rawRequest(), body, req->mr));
    req->collection("main").execute();
    EXPECT_FALSE(req->ok()) << body;
  }
  for (std::string missing : {"expand", "group", "drop"}) {
    auto req = run("main", "*:*", spec({"key_s"}, missing));
    EXPECT_EQ(missing == "drop" ? 0 : 1, req->docList()->found);
    EXPECT_EQ(missing == "drop" ? 0 : 1, req->docList()->found_docs);
  }
  CollectionHelper empty("empty_collapse");
  auto req = run("empty_collapse", "*:*", R"("key_s")");
  EXPECT_EQ(0, req->docList()->found);
  EXPECT_EQ(0, req->docList()->found_docs);
}

TEST_F(CollapseTest, valueBindingAndCapabilities) {
  CollectionHelper helper;
  SchemaBuilder schema;
  auto& field = schema.field("product");
  field.type = luxir::api::FieldDef_::FieldClass::TEXT;
  field.stored = true;
  schema.analyzer(field, "unicode_word", {"nfkc_cf"});
  auto& value = schema.variant(field, "key");
  value.type = luxir::api::FieldDef_::FieldClass::STRING;
  value.column = true;
  schema.normalizer(value, {"nfkc_cf"});
  field.defaults.emplace().value = "key";
  auto& unindexed = schema.field("unindexed");
  unindexed.type = luxir::api::FieldDef_::FieldClass::STRING;
  unindexed.index = luxir::api::FieldDef_::IndexMode::NONE;
  unindexed.column = true;
  auto& uncolumned = schema.field("uncolumned");
  uncolumned.type = luxir::api::FieldDef_::FieldClass::STRING;
  uncolumned.column = false;
  schema.set(helper.collection());
  ASSERT_TRUE(helper.index(flatdoc("id", "a", "product", "AbC"), UpdateMessage::COMMIT).success);
  ASSERT_TRUE(helper.index(flatdoc("id", "b", "product", "abc"), UpdateMessage::COMMIT).success);
  auto req = run("main", "*:*", R"("product")");
  EXPECT_EQ(1, req->docList()->found);
  ASSERT_EQ(1, req->docList()->groups.size());
  EXPECT_EQ("abc", std::get<std::string_view>(req->docList()->groups[0].component[0].value));
  for (std::string collapse : {R"("unindexed")", R"("uncolumned")",
      R"({"field":["product","product__key"]})"}) {
    auto bad = localReq(luxirNode->getSearchEngine());
    ASSERT_TRUE(luxir::api::read_json(bad->rawRequest(),
        R"({"collection":"main","collapse":)" + collapse + "}", bad->mr));
    bad->execute();
    EXPECT_FALSE(bad->ok());
  }
}

TEST_F(CollapseTest, fieldSortOracle) {
  std::mt19937 random(1841);
  std::vector<Doc> docs;
  for (int32_t i = 0; i < 315; i++) {
    auto doc = flatdoc("id", "d" + std::to_string(i), "body_w",
        i % 3 == 0 ? "alpha alpha beta" : i % 3 == 1 ? "alpha beta beta" : "beta gamma");
    if (random() % 5 != 0) doc.emplace_back("key_s", "g" + std::to_string(random() % 29));
    if (random() % 4 != 0) doc.emplace_back("key_i", (int64_t)(random() % 9) - 4);
    if (random() % 3 != 0) doc.emplace_back("when_dt", (int64_t)(random() % 5) * 86400000);
    if (random() % 4 != 0) doc.emplace_back("sort_i", (int64_t)(random() % 7) - 3);
    if (random() % 5 != 0) doc.emplace_back("sort_s", "s" + std::to_string(random() % 5));
    docs.push_back(std::move(doc));
  }
  struct SortCase {
    std::string json;
    std::vector<std::string> fields;
  };
  for (int32_t segments : {1, 3, 7}) {
    auto savedMode = SortField::setStringSortModeForTests(
        segments == 3 ? StringSortMode::GLOBAL : StringSortMode::SEGMENT);
    auto restoreMode = scope_guard([&]() { SortField::setStringSortModeForTests(savedMode); });
    std::string collection = "collapse_field" + std::to_string(segments);
    CollectionHelper helper(collection);
    for (int32_t segment = 0; segment < segments; segment++) {
      ASSERT_TRUE(helper.indexAll(std::span<const Doc>(docs).subspan(
          (size_t)(segment * 315 / segments), (size_t)(315 / segments)), UpdateMessage::COMMIT).success);
    }
    ASSERT_TRUE(helper.deleteByIds({"d12", "d103"}, UpdateMessage::COMMIT).success);
    for (const SortCase& sort : std::vector<SortCase>{
        {R"("sort_i asc")", {"sort_i"}}, {R"("sort_i desc")", {"sort_i"}},
        {R"("sort_s asc")", {"sort_s"}}, {R"("sort_s desc")", {"sort_s"}},
        {R"(["sort_i asc","sort_s desc"])", {"sort_i", "sort_s"}},
        {R"(["sort_s desc","sort_i desc"])", {"sort_s", "sort_i"}},
        {R"(["sort_i asc","score desc"])", {"sort_i", "score"}},
        {R"(["score asc","sort_s asc"])", {"score", "sort_s"}},
        {R"("_docid_ desc")", {"_doc_"}},
        {R"(["score desc","_docid_ asc"])", {"score", "_doc_"}},
        {R"({"expr":"sort_i * 2","dir":"asc"})", {"sort_i"}}}) {
      for (std::string query : {"*:*", "body_w:alpha OR body_w:beta"}) {
        bool scores = std::find(sort.fields.begin(), sort.fields.end(), "score") != sort.fields.end();
        auto full = run(collection, query, {}, -1, 0, 1, true, 13, sort.json, scores);
        for (const std::vector<std::string>& fields : {
            std::vector<std::string>{"key_s"}, {"key_i"}, {"key_s", "key_i", "when_dt"}}) {
          for (std::string missing : {"expand", "group", "drop"}) {
            auto expected = fieldOracle(*full, fields, missing, sort.fields);
            auto collapse = spec(fields, missing);
            for (int32_t parallel : {1, -1}) {
              SCOPED_TRACE(collection + " " + query + " " + sort.json + " " + collapse
                  + " parallel=" + std::to_string(parallel));
              auto actual = run(collection, query, collapse, -1, 0, parallel, true, 11, sort.json, scores);
              EXPECT_EQ(expected.groups, rows(*actual, true));
              EXPECT_EQ((int64_t)expected.groups.size(), actual->docList()->found);
              EXPECT_EQ(expected.eligible, actual->docList()->found_docs);
              for (bool count : {false, true}) {
                auto page = run(collection, query, collapse, 7, 4, parallel, count, 3, sort.json, scores);
                auto begin = expected.groups.begin() + std::min<size_t>(4, expected.groups.size());
                auto end = expected.groups.begin() + std::min<size_t>(11, expected.groups.size());
                EXPECT_EQ(std::vector<Row>(begin, end), rows(*page, true));
                EXPECT_EQ(count, page->docList()->found.has_value());
                EXPECT_EQ(count, page->docList()->found_docs.has_value());
              }
              auto totals = run(collection, query, collapse, 0, 0, parallel, true, 13, sort.json, scores);
              EXPECT_EQ((int64_t)expected.groups.size(), totals->docList()->found);
              EXPECT_EQ(expected.eligible, totals->docList()->found_docs);
            }
            if (fields.size() == 3 && missing == "group") {
              std::vector<Row> pages;
              for (size_t offset = 0; offset < expected.groups.size(); offset += 17) {
                auto page = run(collection, query, collapse, 17, (int64_t)offset, -1, false, 5, sort.json, scores);
                auto current = rows(*page, true);
                pages.insert(pages.end(), current.begin(), current.end());
              }
              EXPECT_EQ(expected.groups, pages);
            }
          }
        }
      }
    }
  }
}

TEST_F(CollapseTest, fieldSortDrivers) {
  CollectionHelper helper;
  constexpr int32_t nDocs = 8192;
  std::vector<Doc> docs;
  for (int32_t i = 0; i < nDocs; i++) {
    auto doc = flatdoc("id", "d" + std::to_string(i), "body_w", i % 3 == 0 ? "alpha" : "beta",
        "sort_i", (int64_t)((nDocs - i - 1) / 256), "sort_s", "s" + std::to_string(i % 19),
        "key_s", "g" + std::to_string((nDocs - i) % 211));
    if (i % 7 != 0) doc.emplace_back("key_i", (int64_t)(i * 13 % 197));
    docs.push_back(std::move(doc));
  }
  ASSERT_TRUE(helper.indexAll(docs, UpdateMessage::COMMIT).success);
  for (std::string missing : {"group", "drop", "expand"}) {
    auto collapse = spec({"key_i", "key_s"}, missing);
    for (std::string sort : {R"("sort_i asc")", R"("sort_i desc")"}) {
      for (std::string query : {"*:*", "body_w:alpha OR body_w:beta", "body_w:alpha"}) {
        SCOPED_TRACE(missing + " " + sort + " " + query);
        auto full = run("main", query, {}, -1, 0, 1, true, 2000, sort, false);
        auto expected = fieldOracle(*full, {"key_i", "key_s"}, missing, {"sort_i"});
        expected.groups.resize(17);
        for (int32_t arm = 0; arm < 6; arm++) {
          auto savedBest = forceFieldSortBestFirst;
          auto savedSeed = forceFieldSortSeeding;
          auto savedCap = forceFieldSortWorkCapForTests;
          auto savedBulk = disableFieldSortBulk;
          auto savedPrune = disableFieldSortPruning;
          auto savedBudget = fieldSortSeedBudgetPerMilleForTests;
          auto savedWhole = QueryPrep::disableWholeMembershipPlanForTests;
          auto savedGather = FieldSortCollector::disableKeyGatherForTests;
          auto restore = scope_guard([&]() {
            forceFieldSortBestFirst = savedBest;
            forceFieldSortSeeding = savedSeed;
            forceFieldSortWorkCapForTests = savedCap;
            disableFieldSortBulk = savedBulk;
            disableFieldSortPruning = savedPrune;
            fieldSortSeedBudgetPerMilleForTests = savedBudget;
            QueryPrep::disableWholeMembershipPlanForTests = savedWhole;
            FieldSortCollector::disableKeyGatherForTests = savedGather;
          });
          forceFieldSortBestFirst = arm == 0 || arm == 1;
          forceFieldSortWorkCapForTests = arm == 1 ? 1 : 0;
          forceFieldSortSeeding = arm == 2;
          fieldSortSeedBudgetPerMilleForTests = arm == 2 ? 100 : 0;
          disableFieldSortBulk = arm == 3;
          disableFieldSortPruning = arm == 4;
          FieldSortCollector::disableKeyGatherForTests = arm == 5;
          QueryPrep::disableWholeMembershipPlanForTests = true;
          SkipStatsScope stats;
          auto actual = run("main", query, collapse, 17, 0, 1, false, 13, sort, false);
          EXPECT_EQ(expected.groups, rows(*actual, true)) << "arm=" << arm;
          if (query == "*:*" && arm == 0) { EXPECT_GT(SkipStats::fieldSortBestFirstActivations, 0); }
          if (query == "body_w:alpha OR body_w:beta" && arm == 2) { EXPECT_GT(SkipStats::fieldSortSeededActivations, 0); }
        }
      }
    }
  }
  // Candidate ord intervals must retain the bottom term at ties. Exercise
  // the single-segment global-ord comparator, including non-bottom updates.
  for (StringSortMode mode : {StringSortMode::SEGMENT, StringSortMode::GLOBAL}) {
    auto saved = SortField::setStringSortModeForTests(mode);
    auto restore = scope_guard([&]() { SortField::setStringSortModeForTests(saved); });
    for (std::string sort : {R"("sort_s asc")", R"("sort_s desc")"}) {
      auto full = run("main", "*:*", {}, -1, 0, 1, true, 2000, sort, false);
      auto expected = fieldOracle(*full, {"key_s"}, "group", {"sort_s"});
      expected.groups.resize(17);
      auto actual = run("main", "*:*", spec({"key_s"}, "group"), 17, 0, 1, false, 13, sort, false);
      EXPECT_EQ(expected.groups, rows(*actual, true));
    }
  }
}

TEST_F(CollapseTest, fieldSortCandidatesAndTwoPhase) {
  CollectionHelper helper;
  for (int32_t segment = 0; segment < 2; segment++) {
    std::vector<Doc> docs;
    for (int32_t i = 0; i < 12000; i++) {
      int32_t id = segment * 12000 + i;
      uint32_t value = (uint32_t)id * 2654435761u % 200000;
      auto doc = flatdoc("id", "d" + std::to_string(id),
          "body_w", id % 3 == 0 ? "alpha red fox" : "alpha red pad fox",
          "key_s", "g" + std::to_string(id % 211), "sort_i", (int64_t)(value % 31));
      if (id % 13 != 0) doc.emplace_back("sort_s", std::format("s{:06}", value));
      docs.push_back(std::move(doc));
    }
    ASSERT_TRUE(helper.indexAll(docs, UpdateMessage::COMMIT).success);
  }
  auto savedWhole = QueryPrep::disableWholeMembershipPlanForTests;
  auto restoreWhole = scope_guard([&]() { QueryPrep::disableWholeMembershipPlanForTests = savedWhole; });
  QueryPrep::disableWholeMembershipPlanForTests = true;
  for (StringSortMode mode : {StringSortMode::SEGMENT, StringSortMode::GLOBAL}) {
    auto savedMode = SortField::setStringSortModeForTests(mode);
    auto restoreMode = scope_guard([&]() { SortField::setStringSortModeForTests(savedMode); });
    for (std::string query : {"*:*", "body_w:alpha", "body_w:\"red fox\""}) {
      for (std::string field : {"sort_s", "sort_i"}) {
        for (std::string direction : {"asc", "desc"}) {
          std::string sort = '"' + field + ' ' + direction + '"';
          SCOPED_TRACE(query + " " + sort + " mode=" + std::to_string((int)mode));
          auto full = run("main", query, {}, -1, 0, 1, true, 2000, sort, false);
          auto expected = fieldOracle(*full, {"key_s"}, "group", {field});
          expected.groups.resize(9);
          for (bool pull : {false, true}) {
            auto savedBulk = disableFieldSortBulk;
            auto restoreBulk = scope_guard([&]() { disableFieldSortBulk = savedBulk; });
            disableFieldSortBulk = pull;
            SkipStatsScope stats;
            auto actual = run("main", query, R"("key_s")", 9, 0, 1, false, 4, sort, false);
            EXPECT_EQ(expected.groups, rows(*actual, true));
            // Multi-segment global-ord sorts have no inverse ord map;
            // their existing candidate arm is unsupported and falls back.
            if (mode == StringSortMode::SEGMENT && field == "sort_s"
                && query != "body_w:\"red fox\"") {
              EXPECT_GT(SkipStats::fieldSortCandidateActivations, 0);
            }
            if (pull && field == "sort_i" && query == "body_w:\"red fox\"") {
              EXPECT_GT(SkipStats::fieldSortExternalApproxActivations, 0);
              EXPECT_GT(SkipStats::fieldSortExternalApproxBoundRejects, 0);
            }
          }
        }
      }
    }
  }
}
