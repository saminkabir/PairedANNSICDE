// Tests for building/searching a FlatNav graph whose vectors live in an
// external (shared) buffer, e.g. the storage of a faiss::IndexFlatL2.
#include <flatnav/distances/InnerProductDistance.h>
#include <flatnav/distances/SquaredL2Distance.h>
#include <flatnav/index/Index.h>
#include <algorithm>
#include <cstdio>
#include <numeric>
#include <random>
#include <unordered_set>
#include <vector>
#include "gtest/gtest.h"

using flatnav::ExternalVectorStorage;
using flatnav::Index;
using flatnav::distances::SquaredL2Distance;
using flatnav::util::DataType;

namespace flatnav::testing {

using L2Index = Index<SquaredL2Distance<DataType::float32>, int>;

static const uint32_t N = 5000;
static const uint32_t NQ = 200;
static const uint32_t DIM = 32;
static const uint32_t M = 16;
static const uint32_t EF_CONSTRUCTION = 100;
static const uint32_t EF_SEARCH = 64;
static const uint32_t K = 10;

static std::vector<float> randomVectors(uint32_t n, uint32_t dim, uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> u(0.f, 1.f);
  std::vector<float> v(static_cast<size_t>(n) * dim);
  for (auto& x : v) x = u(rng);
  return v;
}

static std::vector<std::vector<int>> bruteForce(const std::vector<float>& base, const std::vector<float>& queries,
                                                uint32_t dim, uint32_t k) {
  size_t n = base.size() / dim, nq = queries.size() / dim;
  std::vector<std::vector<int>> gt(nq);
  for (size_t q = 0; q < nq; q++) {
    std::vector<std::pair<float, int>> d(n);
    for (size_t i = 0; i < n; i++) {
      float s = 0;
      for (uint32_t j = 0; j < dim; j++) {
        float t = base[i * dim + j] - queries[q * dim + j];
        s += t * t;
      }
      d[i] = {s, static_cast<int>(i)};
    }
    std::partial_sort(d.begin(), d.begin() + k, d.end());
    for (uint32_t j = 0; j < k; j++) gt[q].push_back(d[j].second);
  }
  return gt;
}

template <typename IndexT>
static double recall(IndexT& index, const std::vector<float>& queries, const std::vector<std::vector<int>>& gt) {
  size_t hits = 0;
  for (size_t q = 0; q < gt.size(); q++) {
    auto res = index.search(queries.data() + q * DIM, K, EF_SEARCH);
    std::unordered_set<int> truth(gt[q].begin(), gt[q].end());
    for (auto& [dist, label] : res) hits += truth.count(label);
  }
  return static_cast<double>(hits) / (gt.size() * K);
}

static std::unique_ptr<L2Index> makeExternal(const std::vector<float>& base) {
  ExternalVectorStorage storage{base.data(), base.size() / DIM, DIM * sizeof(float)};
  return std::make_unique<L2Index>(SquaredL2Distance<DataType::float32>::create(DIM), N, M, storage);
}

TEST(FlatNavExternalStorage, ZeroCopyLayoutAndRecall) {
  auto base = randomVectors(N, DIM, 1);
  auto queries = randomVectors(NQ, DIM, 2);
  auto gt = bruteForce(base, queries, DIM, K);

  auto shared = makeExternal(base);
  ASSERT_TRUE(shared->usesExternalStorage());
  ASSERT_EQ(shared->externalStorageData(), static_cast<const void*>(base.data()));
  // Graph only: M links + 1 label, no vector bytes.
  ASSERT_EQ(shared->nodeSizeBytes(), 4 * M + 4);
  shared->addFromStorage(N, EF_CONSTRUCTION);
  ASSERT_EQ(shared->currentNumNodes(), N);

  // Reference: the normal FlatNav index that copies the vectors.
  L2Index owned(SquaredL2Distance<DataType::float32>::create(DIM), N, M);
  std::vector<int> labels(N);
  std::iota(labels.begin(), labels.end(), 0);
  owned.addBatch<float>(base.data(), labels, EF_CONSTRUCTION);
  ASSERT_EQ(owned.nodeSizeBytes(), DIM * sizeof(float) + 4 * M + 4);

  double r_shared = recall(*shared, queries, gt);
  double r_owned = recall(owned, queries, gt);
  // Single-threaded construction is deterministic and identical in both modes.
  EXPECT_DOUBLE_EQ(r_shared, r_owned);
  EXPECT_GT(r_shared, 0.9);
}

TEST(FlatNavExternalStorage, ParallelBuildKeepsNodeIdEqualToStorageRow) {
  auto base = randomVectors(N, DIM, 3);
  auto shared = makeExternal(base);
  shared->setNumThreads(std::max(2u, std::min(4u, std::thread::hardware_concurrency())));
  // Build in two chunks to also exercise incremental addFromStorage.
  shared->addFromStorage(N / 2, EF_CONSTRUCTION);
  shared->addFromStorage(N - N / 2, EF_CONSTRUCTION);
  ASSERT_EQ(shared->currentNumNodes(), N);

  // Searching with storage row i must return label i at distance 0.
  size_t self_hits = 0;
  for (uint32_t i = 0; i < N; i += 25) {
    auto res = shared->search(base.data() + static_cast<size_t>(i) * DIM, 1, EF_SEARCH);
    ASSERT_FALSE(res.empty());
    if (res[0].second == static_cast<int>(i)) {
      EXPECT_FLOAT_EQ(res[0].first, 0.f);
      self_hits++;
    }
  }
  EXPECT_GE(self_hits, (N / 25) * 0.98);
}

TEST(FlatNavExternalStorage, CustomLabels) {
  auto base = randomVectors(N, DIM, 4);
  auto shared = makeExternal(base);
  std::vector<int> labels(N);
  for (uint32_t i = 0; i < N; i++) labels[i] = 1000000 + static_cast<int>(i);
  shared->addFromStorage(N, EF_CONSTRUCTION, 100, labels);
  auto res = shared->search(base.data() + 123 * DIM, 1, EF_SEARCH);
  EXPECT_EQ(res[0].second, 1000123);
}

TEST(FlatNavExternalStorage, SaveLoadGraphOnlyAndReattach) {
  auto base = randomVectors(N, DIM, 5);
  auto queries = randomVectors(NQ, DIM, 6);
  auto shared = makeExternal(base);
  shared->addFromStorage(N, EF_CONSTRUCTION);

  const std::string file = "flatnav_external_storage_test.index";
  shared->saveIndex(file);
  auto loaded = L2Index::loadIndex(file);
  std::remove(file.c_str());

  ASSERT_TRUE(loaded->usesExternalStorage());
  ASSERT_EQ(loaded->nodeSizeBytes(), 4 * M + 4);
  ASSERT_EQ(loaded->currentNumNodes(), N);
  EXPECT_THROW(loaded->search(queries.data(), K, EF_SEARCH), std::logic_error);

  // Re-attach to a *different* buffer holding the same vectors (e.g. the
  // storage of a faiss index read back with faiss::read_index).
  std::vector<float> base_copy = base;
  loaded->attachExternalStorage({base_copy.data(), N, DIM * sizeof(float)});
  for (uint32_t q = 0; q < NQ; q++) {
    auto a = shared->search(queries.data() + q * DIM, K, EF_SEARCH);
    auto b = loaded->search(queries.data() + q * DIM, K, EF_SEARCH);
    ASSERT_EQ(a, b);
  }
}

TEST(FlatNavExternalStorage, OwnedIndexStillRoundTrips) {
  auto base = randomVectors(1000, DIM, 7);
  L2Index owned(SquaredL2Distance<DataType::float32>::create(DIM), 1000, M);
  std::vector<int> labels(1000);
  std::iota(labels.begin(), labels.end(), 0);
  owned.addBatch<float>(base.data(), labels, EF_CONSTRUCTION);
  const std::string file = "flatnav_owned_storage_test.index";
  owned.saveIndex(file);
  auto loaded = L2Index::loadIndex(file);
  std::remove(file.c_str());
  ASSERT_FALSE(loaded->usesExternalStorage());
  auto a = owned.search(base.data(), K, EF_SEARCH);
  auto b = loaded->search(base.data(), K, EF_SEARCH);
  ASSERT_EQ(a, b);
}

TEST(FlatNavExternalStorage, Guards) {
  auto base = randomVectors(N, DIM, 8);
  auto shared = makeExternal(base);
  std::vector<int> labels(10, 0);
  EXPECT_THROW(shared->addBatch<float>(base.data(), labels, EF_CONSTRUCTION), std::logic_error);
  EXPECT_THROW(shared->addFromStorage(N + 1, EF_CONSTRUCTION), std::invalid_argument);
  // Wrong row size (e.g. attaching a float buffer to a uint8 index).
  EXPECT_THROW(shared->attachExternalStorage({base.data(), N, DIM}), std::invalid_argument);
  shared->addFromStorage(100, EF_CONSTRUCTION);
  EXPECT_THROW(shared->reorderRCM(), std::logic_error);
  EXPECT_THROW(shared->attachExternalStorage({base.data(), 50, 0}), std::invalid_argument);

  L2Index owned(SquaredL2Distance<DataType::float32>::create(DIM), N, M);
  EXPECT_THROW(owned.attachExternalStorage({base.data(), N, 0}), std::logic_error);
  EXPECT_THROW(owned.addFromStorage(10, EF_CONSTRUCTION), std::logic_error);
}

}  // namespace flatnav::testing
