// Copyright 2020-2026 Yonik Seeley and Luxir contributors
// SPDX-License-Identifier: Apache-2.0

#include "test/LuxirTest.h"
#include "test/CollectionHelper.h"
#include "test/LocalReq.h"
#include "test/TestUtils.h"
#include "test/SchemaBuilder.h"
#include "luxir/server/JsonResponse.h"
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
              std::get<float>(fields.at("_score_")->kind), {}, false};
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
}

class CollapseTest : public LuxirTest {
protected:
  auto run(std::string_view collection, std::string_view query, std::string_view collapse,
           int64_t limit = -1, int64_t offset = 0, int32_t parallel = 1,
           bool count = true, int32_t batch = 13) {
    auto req = localReq(luxirNode->getSearchEngine());
    std::string json = R"({"collection":")" + std::string(collection)
        + R"(","query":")" + std::string(query)
        + R"(","fields":["id","key_s","key_i","when_dt"],"document_format":"rows","get_scores":true,"get_number":)"
        + (count ? "true" : "false") + R"(,"limit":)" + std::to_string(limit)
        + R"(,"offset":)" + std::to_string(offset)
        + R"(,"batch_size":)" + std::to_string(batch);
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
  for (std::string extra : {R"("sort":"key_i asc")",
      R"json("ops":{"metric":"sum(key_i)"})json"}) {
    auto req = localReq(luxirNode->getSearchEngine());
    ASSERT_TRUE(luxir::api::read_json(req->rawRequest(),
        R"({"collection":"main","collapse":"key_s",)" + extra + "}", req->mr));
    req->execute();
    EXPECT_FALSE(req->ok());
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
