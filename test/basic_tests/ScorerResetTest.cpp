// Copyright 2020-2026 Yonik Seeley and Luxir contributors
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <array>
#include <limits>
#include <random>
#include <vector>

#include "luxir/query/AllQuery.h"
#include "luxir/query/BooleanQuery.h"
#include "luxir/query/BoostQuery.h"
#include "luxir/query/ConstantScoreQuery.h"
#include "luxir/query/ExistsQuery.h"
#include "luxir/query/NumericPredicateQuery.h"
#include "luxir/query/PhraseQuery.h"
#include "luxir/query/PrefixQuery.h"
#include "luxir/query/QueryPrep.h"
#include "luxir/reader/FieldReader.h"
#include "test/CollectionHelper.h"
#include "test/LuxirTest.h"
#include "test/SchemaBuilder.h"
#include "test/TestUtils.h"

using namespace luxir;
using namespace luxir::test;

namespace {

using Use = Query::ExecutionUse;
using Answer = Query::ScorerSupplier::BulkAnswer;

struct Products {
  MemPool pool;
  Query::Context context;
  Query::ScorerSupplier* supplier;

  Products(IndexReader& reader, Query& query, int32_t flags = 0)
      : context(pool, reader),
        supplier(query.createWeight(context, flags)->scorerSupplier(
            pool, reader.segments()[0], Query::SupplierExecutionMode::RAW_MEMBERSHIP)) {}

  Query::ScorerPlan* pullPlan(int64_t leadCost, Use use = Use::MATCH_WINDOWS) {
    return supplier->resolve(pool, supplier->makePlanContext(
        Query::Demand::fromLeadCost(leadCost, use)));
  }
};

std::vector<std::pair<int32_t, int32_t>> windows(int32_t maxDoc) {
  const int32_t l0 = Postings::DOCS_BLOCK_SIZE;
  const int32_t l1 = DocsEnumMeta::L1_DOCS;
  std::vector<std::pair<int32_t, int32_t>> result{
      {0, maxDoc}, {maxDoc, maxDoc}, {0, 1}, {maxDoc - 1, maxDoc},
      {0, maxDoc}, {l1, l1}, {l0 - 1, l0 + 1}, {l1 - 1, l1 + 1}, {3 * l1 - 1, 3 * l1 + 1},
      {0, l0}, {l0, l1}, {l1, maxDoc}, {1, 1}, {0, maxDoc}};
  std::mt19937 random(0x58a301);
  for (int32_t i = 0; i < 40; i++) {
    int32_t min = (int32_t) (random() % (uint32_t) (maxDoc + 1));
    int32_t length = (int32_t) (random() % (uint32_t) (maxDoc - min + 1));
    result.emplace_back(min, min + length);
  }
  return result;
}

std::vector<int32_t> pullDocs(Query::Scorer& scorer, int32_t min,
                              int32_t max, bool approximation) {
  std::vector<int32_t> docs;
  if (min >= max) return docs;
  int32_t doc = approximation ? scorer.approximationAdvance(min) : scorer.advance(min);
  while (doc < max) {
    if (!approximation || scorer.matches()) docs.push_back(doc);
    doc = approximation ? scorer.approximationNext() : scorer.next();
  }
  return docs;
}

std::vector<int32_t> bulkDocs(BulkScorer& bulk, DocSet* domain,
                              int32_t min, int32_t max) {
  std::vector<int32_t> docs;
  // Call even for an empty range: it must leave no stale output behind.
  for (;;) {
    ScoreWindow window;
    int32_t next = bulk.matchNextWindow(window, domain, min, max);
    for (int32_t i = 0; i < window.size; i++) {
      int32_t doc = window.docs[(size_t) i];
      EXPECT_GE(doc, min);
      EXPECT_LT(doc, max);
      docs.push_back(doc);
    }
    if (next == PostingsReader::END || next >= max) return docs;
    EXPECT_GT(next, min);
    if (next <= min) return docs;
    min = next;
  }
}

int64_t bulkCount(BulkScorer& bulk, DocSet* domain, int32_t min, int32_t max,
                  DocSetBuilder* output = nullptr) {
  int64_t count = 0;
  for (;;) {
    int32_t next = bulk.countNextWindow(count, output, domain, min, max);
    if (next == PostingsReader::END || next >= max) return count;
    EXPECT_GT(next, min);
    if (next <= min) return count;
    min = next;
  }
}

void checkPull(IndexReader& reader, Query& query, int64_t leadCost) {
  Products reused(reader, query);
  ASSERT_NE(nullptr, reused.supplier);
  auto* plan = reused.pullPlan(leadCost);
  ASSERT_TRUE(plan->supportsReset());
  auto* scorer = plan->build(reused.pool);
  ASSERT_NE(nullptr, scorer);
  EXPECT_TRUE(plan->supportsReset());
  bool twoPhase = plan->shape().reportedTwoPhase == Query::ReportedTwoPhase::YES;
  int32_t epoch = 0;
  for (auto [min, max] : windows(reader.segments()[0].maxDoc())) {
    SCOPED_TRACE(::testing::Message() << "pull epoch=" << epoch << " range=" << min << ':' << max);
    Products fresh(reader, query);
    auto* expected = fresh.pullPlan(leadCost)->build(fresh.pool);
    ASSERT_NE(nullptr, expected);
    // Exhaustion hints belong to the old traversal, including the bounds
    // that constant scorers clamp in place.
    scorer->setMinCompetitiveScore(std::numeric_limits<float>::infinity());
    scorer->reset();
    EXPECT_EQ(-1, scorer->docId());
    EXPECT_EQ(-1, scorer->approximationDocId());
    bool approximation = twoPhase && (epoch++ & 1) != 0;
    EXPECT_EQ(pullDocs(*expected, min, max, approximation),
              pullDocs(*scorer, min, max, approximation));
    EXPECT_TRUE(plan->supportsReset());
  }
}

void checkBulk(IndexReader& reader, Query& query, bool expectedAvailable,
               DocSet* domain = nullptr, Query* attached = nullptr, bool probe = false) {
  int32_t maxDoc = reader.segments()[0].maxDoc();
  for (Use use : {Use::MATCH_WINDOWS, Use::COUNT_WINDOWS}) {
    Products reused(reader, query);
    ASSERT_NE(nullptr, reused.supplier);
    auto plan = reused.supplier->planBulk(use, {});
    if (!expectedAvailable) {
      EXPECT_EQ(Answer::NO, plan.supportsReposition());
      continue;
    }
    ASSERT_EQ(Answer::YES, plan.available);
    ASSERT_EQ(Answer::YES, plan.supportsReposition());
    auto* bulk = reused.supplier->buildBulk(reused.pool, plan);
    ASSERT_NE(nullptr, bulk);
    auto attach = [&](Products& product, BulkScorer* target) {
      auto* supplier = attached->createWeight(product.context, 0)->scorerSupplier(
          product.pool, reader.segments()[0], Query::SupplierExecutionMode::RAW_MEMBERSHIP);
      ASSERT_NE(nullptr, supplier);
      auto* childPlan = supplier->resolve(product.pool, supplier->makePlanContext(
          Query::Demand::fromLeadCost(maxDoc, use)));
      ASSERT_TRUE(childPlan->supportsReset());
      auto scorers = product.pool.make_span<Query::Scorer*>(1);
      scorers[0] = childPlan->build(product.pool);
      ASSERT_TRUE(target->attachWindowFilter(product.pool.make<WindowFilter>(
          product.pool, scorers, probe, supplier->cost())));
    };
    if (attached != nullptr) attach(reused, bulk);
    int32_t epoch = 0;
    for (auto [min, max] : windows(maxDoc)) {
      SCOPED_TRACE(::testing::Message() << "bulk epoch=" << epoch++ << " range=" << min << ':' << max);
      Products fresh(reader, query);
      auto freshPlan = fresh.supplier->planBulk(use, {});
      auto* expected = fresh.supplier->buildBulk(fresh.pool, freshPlan);
      ASSERT_NE(nullptr, expected);
      if (attached != nullptr) attach(fresh, expected);
      bulk->reposition(min);
      if (use == Use::MATCH_WINDOWS) {
        EXPECT_EQ(bulkDocs(*expected, domain, min, max), bulkDocs(*bulk, domain, min, max));
        // Cover both rewinding partial state and continuing past bounded END.
        if (epoch & 1) {
          EXPECT_EQ(bulkDocs(*expected, domain, max, maxDoc), bulkDocs(*bulk, domain, max, maxDoc));
        }
      } else {
        EXPECT_EQ(bulkCount(*expected, domain, min, max), bulkCount(*bulk, domain, min, max));
        if (epoch & 1) {
          EXPECT_EQ(bulkCount(*expected, domain, max, maxDoc), bulkCount(*bulk, domain, max, maxDoc));
        }
      }
      EXPECT_EQ(Answer::YES, plan.supportsReposition());
    }
    // Leaf consumers continue at the old bound, even when a sparse scorer
    // returned an out-of-range lookahead doc or bounded END. Compare the
    // concatenation against an independent, unbounded traversal.
    Products fresh(reader, query);
    auto freshPlan = fresh.supplier->planBulk(use, {});
    auto* expected = fresh.supplier->buildBulk(fresh.pool, freshPlan);
    ASSERT_NE(nullptr, expected);
    if (attached != nullptr) attach(fresh, expected);
    bulk->reposition(0);
    if (use == Use::MATCH_WINDOWS) {
      std::vector<int32_t> actual;
      for (int32_t min = 0; min < maxDoc; min += 512) {
        auto leaf = bulkDocs(*bulk, domain, min, std::min(min + 512, maxDoc));
        actual.insert(actual.end(), leaf.begin(), leaf.end());
      }
      EXPECT_EQ(bulkDocs(*expected, domain, 0, maxDoc), actual);
    } else {
      int64_t actual = 0;
      for (int32_t min = 0; min < maxDoc; min += 512) {
        actual += bulkCount(*bulk, domain, min, std::min(min + 512, maxDoc));
      }
      EXPECT_EQ(bulkCount(*expected, domain, 0, maxDoc), actual);
    }
  }
}

void checkChangingDomains(IndexReader& reader, Query& query, DocSet* array, DocSet* bits) {
  int32_t maxDoc = reader.segments()[0].maxDoc();
  for (Use use : {Use::MATCH_WINDOWS, Use::COUNT_WINDOWS}) {
    Products reused(reader, query);
    auto plan = reused.supplier->planBulk(use, {});
    ASSERT_EQ(Answer::YES, plan.supportsReposition());
    auto* bulk = reused.supplier->buildBulk(reused.pool, plan);
    int32_t epoch = 0;
    for (DocSet* domain : {array, (DocSet*) nullptr, bits, (DocSet*) nullptr,
                          array, bits, (DocSet*) nullptr, (DocSet*) nullptr}) {
      Products fresh(reader, query);
      auto freshPlan = fresh.supplier->planBulk(use, {});
      auto* expected = fresh.supplier->buildBulk(fresh.pool, freshPlan);
      int32_t min = epoch % 2 == 0 ? maxDoc / 2 : 0;
      int32_t max = maxDoc;
      epoch++;
      bulk->reposition(min);
      if (use == Use::MATCH_WINDOWS) {
        EXPECT_EQ(bulkDocs(*expected, domain, min, max), bulkDocs(*bulk, domain, min, max));
      } else if (epoch % 3 == 0) {
        DocSetBuilder want(maxDoc), actual(maxDoc);
        EXPECT_EQ(bulkCount(*expected, domain, min, max, &want),
                  bulkCount(*bulk, domain, min, max, &actual));
        auto wantDocs = want.build();
        auto actualDocs = actual.build();
        QueryPrep::DocSetScorer wantScorer(wantDocs.get(), maxDoc);
        QueryPrep::DocSetScorer actualScorer(actualDocs.get(), maxDoc);
        EXPECT_EQ(pullDocs(wantScorer, 0, maxDoc, false),
                  pullDocs(actualScorer, 0, maxDoc, false));
      } else {
        EXPECT_EQ(bulkCount(*expected, domain, min, max), bulkCount(*bulk, domain, min, max));
      }
    }
  }
}

template<class Enum>
void checkEnum(const TermsEnum::PostingsState& state, int32_t maxDoc) {
  Enum reused(state);
  for (auto [min, max] : windows(maxDoc)) {
    SCOPED_TRACE(::testing::Message() << "enum range=" << min << ':' << max);
    Enum fresh(state);
    reused.reset();
    ASSERT_EQ(-1, reused.docId());
    int32_t expected = fresh.advance(min);
    int32_t actual = reused.advance(min);
    for (;;) {
      ASSERT_EQ(expected, actual);
      if (actual == PostingsReader::END) break;
      if constexpr (Enum::TIER == DocsEnumTier::FREQS) {
        EXPECT_EQ(fresh.termFreq(), reused.termFreq());
      }
      if (actual >= max) break;
      expected = fresh.next();
      actual = reused.next();
    }
    // Crossing from bit production to frequency iteration in the next epoch
    // must clear docsOnlyConsumed as well as the ordinary doc cursor.
    reused.reset();
    Enum freshBits(state);
    std::vector<uint64_t> actualBits((size_t) (max - min + 63) / 64);
    auto expectedBits = actualBits;
    reused.intoBitSet(actualBits, min, max);
    freshBits.intoBitSet(expectedBits, min, max);
    EXPECT_EQ(expectedBits, actualBits);
  }
}

} // namespace

class ScorerResetTest : public LuxirTest {};

TEST_F(ScorerResetTest, freshProductsAcrossProtocolsAndEncodings) {
  constexpr int32_t N = 6 * DocsEnumMeta::L1_DOCS + 137;
  CollectionHelper helper;
  SchemaBuilder schema;
  for (std::string_view name : {"point", "column", "sparse", "scan", "multi", "rare"}) {
    auto& field = schema.field(name);
    field.type = api::FieldDef::FieldClass::INT;
    field.index = name == "point" ? api::FieldDef::IndexMode::RANGE : api::FieldDef::IndexMode::NONE;
    field.column = true;
    field.multi = name == "multi";
  }
  auto& tags = schema.field("tags");
  tags.type = api::FieldDef::FieldClass::STRING;
  tags.index = api::FieldDef::IndexMode::MATCH;
  tags.multi = true;
  schema.set(helper.collection());
  auto writer = helper.getIndexWriter();
  auto& inverter = writer->obtainInverter();
  auto& text = inverter.getIndexHandler("body_w");
  auto& tagValues = inverter.getIndexHandler("tags");
  auto& point = inverter.getIndexHandler("point");
  auto& column = inverter.getIndexHandler("column");
  auto& sparseValues = inverter.getIndexHandler("sparse");
  auto& scanValues = inverter.getIndexHandler("scan");
  auto& multiValues = inverter.getIndexHandler("multi");
  auto& rareValues = inverter.getIndexHandler("rare");
  std::mt19937 random(7919);
  for (int32_t doc = 0; doc < N; doc++) {
    inverter.startDoc();
    std::string body = "qdense";
    bool wordDoc = (random() & 3) != 0;
    if (wordDoc) body += " qword qword";
    if (doc % 3 == 0) body += " qpacked qpacked qpacked";
    if (doc < 43) body += " qshort";
    if (doc == 137) body += " qpulse";
    if (doc == 3 * DocsEnumMeta::L1_DOCS + 17
        || doc == 5 * DocsEnumMeta::L1_DOCS + 41) body += " qlate";
    text.index(inverter, std::string_view(body));
    std::vector<std::string_view> tagList{"qdense"};
    if (wordDoc) tagList.push_back("qword");
    if (doc % 3 == 0) tagList.push_back("qpacked");
    if (doc < 43) tagList.push_back("qshort");
    if (doc == 137) tagList.push_back("qpulse");
    tagValues.index(inverter, std::span<const std::string_view>(tagList));
    point.index(inverter, doc);
    column.index(inverter, doc);
    scanValues.index(inverter, doc % 73);
    if (doc % 5 != 0) {
      std::array<int64_t, 2> values{doc % 73, 100 + doc % 79};
      multiValues.index(inverter, std::span<const int64_t>(values));
    }
    if (doc % 7 == 0) sparseValues.index(inverter, doc);
    if (doc % 1000 == 0) rareValues.index(inverter, doc);
    inverter.finishDoc();
  }
  writer->releaseInverter(inverter);
  writer->commit();
  auto reader = writer->snapshots.readers.getReader();
  ASSERT_EQ(1, reader->segments().size());

  MemPool queries;
  auto term = [&](std::string_view name) { return queries.make<TermQuery>("body_w", name); };
  auto clauses = [&](std::initializer_list<Query*> values) {
    auto result = queries.make_span<Query*>(values.size());
    std::copy(values.begin(), values.end(), result.begin());
    return result;
  };
  auto boolean = [&](std::span<Query*> must, std::span<Query*> should,
                     std::span<Query*> prohibited = {}, std::span<Query*> filters = {}, int minMatch = 0) {
    return queries.make<BooleanQuery>(must, should, prohibited, filters, minMatch);
  };
  Query* dense = term("qdense");
  Query* word = term("qword");
  Query* packed = term("qpacked");
  Query* shortTerm = term("qshort");
  Query* pulse = term("qpulse");
  Query* conjunction = boolean(clauses({word, packed}), {});
  Query* lookahead = boolean(clauses({dense, term("qlate")}), {});
  Query* disjunction = boolean({}, clauses({shortTerm, pulse, packed}));
  Query* twoWay = boolean({}, clauses({shortTerm, packed}));
  Query* nested = boolean({}, clauses({conjunction, shortTerm}));
  Query* nestedFiltered = boolean(clauses({disjunction}), {}, {}, clauses({word}));
  Query* excluded = boolean(clauses({dense}), {}, clauses({disjunction}));
  Query* filtered = boolean({}, clauses({word, packed}), {}, clauses({dense}), 1);
  Query* filterOnly = boolean({}, {}, {}, clauses({word, packed}));
  Query* optional = boolean(clauses({packed}), clauses({word, shortTerm}));
  Query* all = queries.make<AllQuery>();
  Query* prefix = queries.make<PrefixQuery>("body_w", "q");
  Query* zone = queries.make<NumericPredicateQuery>("column", 100, N / 2);
  Query* pointArray = queries.make<NumericPredicateQuery>("point", 125, 140);
  Query* pointBits = queries.make<NumericPredicateQuery>("point", 100, N / 2);
  Query* pointComplement = queries.make<NumericPredicateQuery>("point", 0, N - 100);
  Query* sparse = queries.make<NumericPredicateQuery>("sparse", 100, N / 2);
  Query* scan = queries.make<NumericPredicateQuery>("scan", 20, 40);
  Query* multi = queries.make<NumericPredicateQuery>("multi", 120, 140);
  Query* exists = queries.make<ExistsQuery>("sparse");
  Query* existsDense = queries.make<ExistsQuery>("column");
  Query* existsRare = queries.make<ExistsQuery>("rare");
  Query* constant = queries.make<ConstantScoreQuery>(disjunction);
  Query* boosted = queries.make<BoostQuery>(disjunction, 3.0f);
  Query* minMatch = boolean({}, clauses({word, packed, shortTerm}), {}, {}, 2);
  Query* numericConjunction = boolean(clauses({word, scan}), {});
  Query* numericUnion = boolean({}, clauses({packed, sparse}));
  Query* numericNot = boolean(clauses({word}), {}, clauses({multi}));
  for (Query* query : {dense, word, packed, shortTerm, pulse, conjunction, lookahead, disjunction,
                      nested, nestedFiltered, excluded, filtered, filterOnly, optional, all, prefix,
                      zone, pointArray, pointBits, pointComplement, sparse, scan,
                      multi, exists, existsDense, existsRare, constant, boosted, twoWay, minMatch,
                      numericConjunction, numericUnion, numericNot}) {
    SCOPED_TRACE(::testing::Message() << "query=" << (int) query->getKind());
    checkPull(*reader, *query, N);
    checkPull(*reader, *query, 0);
  }
  for (Query* query : {dense, word, packed, shortTerm, pulse, conjunction, lookahead, disjunction,
                      nested, nestedFiltered, excluded, filtered, filterOnly, optional, all, prefix,
                      zone, pointArray, pointBits, pointComplement, boosted, twoWay}) {
    SCOPED_TRACE(::testing::Message() << "bulk query=" << (int) query->getKind());
    checkBulk(*reader, *query, true);
  }
  checkBulk(*reader, *sparse, false);
  checkBulk(*reader, *scan, false);
  checkBulk(*reader, *exists, false);
  checkBulk(*reader, *constant, false);
  {
    // Ordinary unscored normalization drops score-only optionals. Keep
    // them here to exercise the reset contracts of the MandOpt products.
    SearchOverridesGuard guard(BooleanQuery::disableUnscoredOptionalDropForTests);
    BooleanQuery::disableUnscoredOptionalDropForTests = true;
    checkPull(*reader, *optional, N);
    checkPull(*reader, *optional, 0);
    checkBulk(*reader, *optional, true);
  }
  checkBulk(*reader, *dense, true, nullptr, word);
  checkBulk(*reader, *dense, true, nullptr, scan);
  checkBulk(*reader, *dense, true, nullptr, scan, true);
  checkBulk(*reader, *disjunction, true, nullptr, dense);
  checkBulk(*reader, *conjunction, true, nullptr, dense, true);

  Query* exactConjunction = boolean(clauses({word, packed}), {}, {}, clauses({dense}));
  checkBulk(*reader, *exactConjunction, true);
  {
    SearchOverridesGuard guard(BooleanQuery::disableExactTermCountForTests);
    BooleanQuery::disableExactTermCountForTests = true;
    checkBulk(*reader, *filtered, true);
    checkBulk(*reader, *exactConjunction, true);
  }
  // A dominant term engages COUNT's union identity. A supplied domain
  // selects its fallback, whose state must also survive a new epoch.
  Query* dominantUnion = boolean({}, clauses({dense, shortTerm, pulse}));
  checkBulk(*reader, *dominantUnion, true);
  checkBulk(*reader, *dominantUnion, true, nullptr, word);
  {
    Products product(*reader, *dominantUnion);
    auto plan = product.supplier->planBulk(Use::COUNT_WINDOWS, {});
    ASSERT_EQ(Answer::YES, plan.supportsReposition());
    auto* bulk = product.supplier->buildBulk(product.pool, plan);
    Products filter(*reader, *word);
    auto* filterPlan = filter.pullPlan(N);
    auto scorers = filter.pool.make_span<Query::Scorer*>(1);
    scorers[0] = filterPlan->build(filter.pool);
    ASSERT_TRUE(bulk->attachWindowFilter(
        filter.pool.make<WindowFilter>(filter.pool, scorers)));
    // Fresh-product parity alone cannot catch a shortcut that ignores a
    // filter in both products. This union is all docs, so the filter is the
    // independent membership oracle, before and after a reset.
    Products oracle(*reader, *word);
    auto* oracleScorer = oracle.pullPlan(N)->build(oracle.pool);
    int64_t want = (int64_t) pullDocs(*oracleScorer, 0, N, false).size();
    EXPECT_EQ(want, bulkCount(*bulk, nullptr, 0, N));
    bulk->reposition(0);
    EXPECT_EQ(want, bulkCount(*bulk, nullptr, 0, N));
  }

  std::vector<int32_t> domainDocs;
  std::vector<uint64_t> domainWords(FixedBitSet::sizeInWords(N));
  for (int32_t doc = 0; doc < N; doc += 11) {
    domainDocs.push_back(doc);
    domainWords[(size_t) doc >> 6] |= 1ULL << (doc & 63);
  }
  ArrDocSet array(std::move(domainDocs));
  BitDocSet bits(FixedBitSet(domainWords.data(), N));
  for (Query* query : {dominantUnion, disjunction, all, filterOnly}) {
    checkChangingDomains(*reader, *query, &array, &bits);
  }
  for (DocSet* domain : std::array<DocSet*, 2>{&array, &bits}) {
    checkBulk(*reader, *dense, true, domain);
    checkBulk(*reader, *disjunction, true, domain);
    checkBulk(*reader, *dominantUnion, true, domain);
    checkBulk(*reader, *all, true, domain);
    MemPool pool;
    QueryPrep::DocSetSupplier supplier(domain, reader->segments()[0]);
    auto* plan = supplier.resolve(pool, Query::PlanContext::fromLeadCost(N));
    ASSERT_TRUE(plan->supportsReset());
    auto* pull = plan->build(pool);
    auto bulkPlan = supplier.planBulk(Use::MATCH_WINDOWS, {});
    ASSERT_EQ(Answer::YES, bulkPlan.supportsReposition());
    auto* bulk = supplier.buildBulk(pool, bulkPlan);
    for (auto [min, max] : windows(N)) {
      QueryPrep::DocSetScorer freshPull(domain, N);
      pull->reset();
      EXPECT_EQ(pullDocs(freshPull, min, max, false), pullDocs(*pull, min, max, false));
      freshPull.reset();
      pull->reset();
      std::vector<uint64_t> want((size_t) (max - min + 63) / 64);
      auto actual = want;
      freshPull.fillWindowBits(want, min, max);
      pull->fillWindowBits(actual, min, max);
      EXPECT_EQ(want, actual);
      MemPool freshPool;
      DocSetBulkScorer freshBulk(freshPool, domain, N);
      bulk->reposition(min);
      EXPECT_EQ(bulkDocs(freshBulk, nullptr, min, max), bulkDocs(*bulk, nullptr, min, max));
    }
  }

  std::array<std::string_view, 2> phraseTerms{"qdense", "qword"};
  std::array<int32_t, 2> positions{0, 1};
  PhraseQuery phrase("body_w", phraseTerms, positions);
  PhraseQuery sloppy("body_w", phraseTerms, positions, 3);
  Query* mixedMust = boolean(clauses({dense, &phrase}), {});
  Query* mixedShould = boolean({}, clauses({packed, &sloppy}));
  Query* mixedNot = boolean(clauses({dense}), {}, clauses({&phrase}));
  for (Query* query : std::array<Query*, 5>{&phrase, &sloppy, mixedMust, mixedShould, mixedNot}) {
    Products product(*reader, *query);
    ASSERT_NE(nullptr, product.supplier);
    EXPECT_FALSE(product.pullPlan(N)->supportsReset());
    EXPECT_EQ(Answer::NO, product.supplier->planBulk(Use::MATCH_WINDOWS, {}).supportsReposition());
  }
  for (Query* query : {dense, disjunction, all, exists, constant, prefix, zone, pointArray}) {
    for (int32_t flags : {Query::NEED_SCORES, Query::NEED_SCORES | Query::ALLOW_PRUNING}) {
      Products scored(*reader, *query, flags);
      EXPECT_FALSE(scored.pullPlan(N, Use::PULL)->supportsReset());
      // Membership requests must also reject scored weights. Without
      // pruning, prefix expansion reaches the eager bulk arm's score guard.
      for (Use use : {Use::MATCH_WINDOWS, Use::COUNT_WINDOWS, Use::SCORED_WINDOWS}) {
        auto plan = scored.supplier->planBulk(use, {});
        if (flags == Query::NEED_SCORES
            && (query == prefix || query == zone || query == pointArray)) {
          EXPECT_EQ(Answer::YES, plan.available);
        }
        EXPECT_EQ(Answer::NO, plan.supportsReposition());
      }
    }
  }
  for (Query* query : {dense, conjunction, disjunction, filtered, filterOnly, optional,
                      nestedFiltered, all, exists, constant, prefix, zone, pointArray}) {
    Products unscored(*reader, *query);
    for (Use use : {Use::PULL, Use::SCORED_WINDOWS, Use::EXACT_CANDIDATE_SCORING}) {
      EXPECT_EQ(Answer::NO, unscored.supplier->planBulk(use, {}).supportsReposition());
    }
  }
  for (auto mode : {Query::PlanContext::MultiTermScorerMode::FORCE_WINDOWED,
                    Query::PlanContext::MultiTermScorerMode::FORCE_HEAP}) {
    Products lazy(*reader, *prefix, Query::NEED_SCORES | Query::ALLOW_PRUNING);
    auto context = lazy.supplier->makePlanContext(Query::Demand::fromLeadCost(N));
    context.multiTermScorerModeForTests = mode;
    EXPECT_FALSE(lazy.supplier->resolve(lazy.pool, context)->supportsReset());
  }

  for (std::string_view field : {"body_w", "tags"}) {
    FieldReader fieldReader(reader->segments()[0].postingsReader());
    ASSERT_TRUE(fieldReader.seek(field));
    SegFieldInfo info;
    fieldReader.readFieldInfo(info);
    TermsEnum terms(queries, reader->segments()[0].postingsReader(), info);
    for (std::string_view name : {"qdense", "qword", "qpacked", "qshort", "qpulse"}) {
      SCOPED_TRACE(::testing::Message() << field << ':' << name);
      ASSERT_TRUE(terms.seek(name));
      auto state = terms.postingsState();
      if (name == "qpacked") {
        EXPECT_GT(state.packedBlockCount, DocsEnumMeta::L1_PERIOD);
      }
      if (name == "qpulse") {
        EXPECT_EQ(state.docsStart, state.docsEnd);
      }
      checkEnum<DocsOnlyEnum>(state, N);
      checkEnum<DocsFreqEnum>(state, N);
    }
  }
}
