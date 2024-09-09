#include <algorithm>
#include <cmath>
#include <istream>
#include <iostream>
#include <fstream>
#include <utility>
#include <string>
#include <random>
#include <cstdio>
#include <set>
#include <chrono>
#include <span>
#include <array>
#include <unordered_map>
#include <bitset>

// Hash libraries
#include "contrib/boost/uuid/detail/sha1.hpp"
#include "contrib/xxHash/xxhash.h"
#include "contrib/ssdeep/fuzzy.h"

// Resemblance detection support
#include <deque>
#include <filesystem>
#include <ranges>
#include <unordered_set>

#include "contrib/bitstream.h"
#include "contrib/stream.h"
#include "contrib/embeddings.cpp/bert.h"
//#include <faiss/IndexFlat.h>
//#include <faiss/MetricType.h>
//#include <faiss/utils/distances.h>
//#include <faiss/IndexLSH.h>
//#include <faiss/IndexBinaryHash.h>

char get_char_with_echo() {
  return getchar();
}

std::string get_sha1_hash(boost::uuids::detail::sha1& s) {
  unsigned int hash[5];
  s.get_digest(hash);

  // Back to string
  char buf[41] = { 0 };

  for (uint64_t i = 0; i < 5; i++)
  {
    std::sprintf(buf + (i << 3), "%08x", hash[i]);
  }

  return { buf };
}

namespace constants {
  // Smallest acceptable value for the minimum chunk size.
  static constexpr uint32_t MINIMUM_MIN = 64;
  // Largest acceptable value for the minimum chunk size.
  static constexpr uint32_t MINIMUM_MAX = 67'108'864;
  // Smallest acceptable value for the average chunk size.
  static constexpr uint32_t AVERAGE_MIN = 256;
  // Largest acceptable value for the average chunk size.
  static constexpr uint32_t AVERAGE_MAX = 268'435'456;
  // Smallest acceptable value for the maximum chunk size.
  static constexpr uint32_t MAXIMUM_MIN = 1024;
  // Largest acceptable value for the maximum chunk size.
  static constexpr uint32_t MAXIMUM_MAX = 1'073'741'824;

  static constexpr uint32_t GEAR[256] = {
    0,          574654857,  759734804,  310648967,  1393527547, 1195718329,
    694400241,  1154184075, 1319583805, 1298164590, 122602963,  989043992,
    1918895050, 933636724,  1369634190, 1963341198, 1565176104, 1296753019,
    1105746212, 1191982839, 1195494369, 29065008,   1635524067, 722221599,
    1355059059, 564669751,  1620421856, 1100048288, 1018120624, 1087284781,
    1723604070, 1415454125, 737834957,  1854265892, 1605418437, 1697446953,
    973791659,  674750707,  1669838606, 320299026,  1130545851, 1725494449,
    939321396,  748475270,  554975894,  1651665064, 1695413559, 671470969,
    992078781,  1935142196, 1062778243, 1901125066, 1935811166, 1644847216,
    744420649,  2068980838, 1988851904, 1263854878, 1979320293, 111370182,
    817303588,  478553825,  694867320,  685227566,  345022554,  2095989693,
    1770739427, 165413158,  1322704750, 46251975,   710520147,  700507188,
    2104251000, 1350123687, 1593227923, 1756802846, 1179873910, 1629210470,
    358373501,  807118919,  751426983,  172199468,  174707988,  1951167187,
    1328704411, 2129871494, 1242495143, 1793093310, 1721521010, 306195915,
    1609230749, 1992815783, 1790818204, 234528824,  551692332,  1930351755,
    110996527,  378457918,  638641695,  743517326,  368806918,  1583529078,
    1767199029, 182158924,  1114175764, 882553770,  552467890,  1366456705,
    934589400,  1574008098, 1798094820, 1548210079, 821697741,  601807702,
    332526858,  1693310695, 136360183,  1189114632, 506273277,  397438002,
    620771032,  676183860,  1747529440, 909035644,  142389739,  1991534368,
    272707803,  1905681287, 1210958911, 596176677,  1380009185, 1153270606,
    1150188963, 1067903737, 1020928348, 978324723,  962376754,  1368724127,
    1133797255, 1367747748, 1458212849, 537933020,  1295159285, 2104731913,
    1647629177, 1691336604, 922114202,  170715530,  1608833393, 62657989,
    1140989235, 381784875,  928003604,  449509021,  1057208185, 1239816707,
    525522922,  476962140,  102897870,  132620570,  419788154,  2095057491,
    1240747817, 1271689397, 973007445,  1380110056, 1021668229, 12064370,
    1186917580, 1017163094, 597085928,  2018803520, 1795688603, 1722115921,
    2015264326, 506263638,  1002517905, 1229603330, 1376031959, 763839898,
    1970623926, 1109937345, 524780807,  1976131071, 905940439,  1313298413,
    772929676,  1578848328, 1108240025, 577439381,  1293318580, 1512203375,
    371003697,  308046041,  320070446,  1252546340, 568098497,  1341794814,
    1922466690, 480833267,  1060838440, 969079660,  1836468543, 2049091118,
    2023431210, 383830867,  2112679659, 231203270,  1551220541, 1377927987,
    275637462,  2110145570, 1700335604, 738389040,  1688841319, 1506456297,
    1243730675, 258043479,  599084776,  41093802,   792486733,  1897397356,
    28077829,   1520357900, 361516586,  1119263216, 209458355,  45979201,
    363681532,  477245280,  2107748241, 601938891,  244572459,  1689418013,
    1141711990, 1485744349, 1181066840, 1950794776, 410494836,  1445347454,
    2137242950, 852679640,  1014566730, 1999335993, 1871390758, 1736439305,
    231222289,  603972436,  783045542,  370384393,  184356284,  709706295,
    1453549767, 591603172,  768512391,  854125182
  };

  // Coeficients (coprime pairs) for N-Transform feature extraction, only have 16 as more than 4SF-4F is unlikely to yield good results
  static constexpr std::pair<uint16_t, uint16_t> N_Transform_Coefs[16] = {
    {2, 3},
    {3, 5},
    {5, 7},
    {7, 11},
    {11, 13},
    {13, 17},
    {17, 19},
    {19, 23},
    {23, 29},
    {29, 31},
    {31, 37},
    {37, 41},
    {41, 43},
    {43, 47},
    {47, 53},
    {53, 59},
  };

  // Sampling mask for Content Defined Sampling with 1/64 frequency and somewhat spread 1bits as used by ODESS paper
  static constexpr uint32_t CDS_SAMPLING_MASK = 0x40030341;
}

namespace utility {
  class CDChunk {
  public:
    uint64_t offset;
    uint32_t length;
    std::shared_ptr<uint8_t[]> data;
    std::bitset<64> hash;
    std::bitset<64> lsh;
    std::array<uint32_t, 4> super_features{};
    bool feature_sampling_failure = true;
    // Smaller chunks inside this chunk, used for delta compression
    std::shared_ptr<uint32_t[]> minichunks{};
    uint32_t minichunks_len = 0;

    explicit CDChunk(uint64_t _offset, uint32_t _length) : offset(_offset), length(_length), data(nullptr) {}
    explicit CDChunk(uint64_t _offset, uint32_t _length, uint8_t* _data)
      : offset(_offset), length(_length), data(std::make_shared_for_overwrite<uint8_t[]>(_length))
    {
      std::copy_n(_data, _length, data.get());
    }

    // This is used to set the chunk's data to a span of data owned by another (hopefully duplicate) chunk, so we can save space.
    // Set a chunk's data to a future chunk's data, as the previous chunks will be deleted first as the context window moves.
    void setData(CDChunk& otherChunk) {
      data = otherChunk.data;
    }

    void deleteData() {
      data = nullptr;
    }
  };

  uint32_t logarithm2(uint32_t value) {
    return std::lround(std::log2(value));
  }

  uint32_t mask(uint32_t bits) {
    return std::pow(2, bits) - 1;
  }
}

class IStreamLike {
public:
  virtual ~IStreamLike() = default;
  virtual void read(void* buf, std::streamsize count) = 0;
  virtual std::streamsize gcount() = 0;
};

class IStreamWrapper: public IStreamLike {
  std::istream* istream;
public:
  IStreamWrapper(std::istream* _istream): istream(_istream) {}
  ~IStreamWrapper() override = default;

  void read(void* buf, std::streamsize count) override {
    istream->read(static_cast<char*>(buf), count);
  }

  std::streamsize gcount() override {
    return istream->gcount();
  }
};

class IStreamMem : public IStreamLike {
  const uint8_t* membuf;
  uint32_t len;
  uint32_t pos = 0;
  std::streamsize _gcount = 0;
public:
  IStreamMem(const uint8_t* _buf, uint32_t _len) : membuf(_buf), len(_len) {}
  ~IStreamMem() override = default;

  void read(void* buf, std::streamsize count) override {
    if (pos >= len) {
      _gcount = 0;
      return;
    }

    auto to_read = std::min<std::streamsize>(len - pos, count);
    std::copy_n(membuf + pos, to_read, static_cast<uint8_t*>(buf));
    _gcount = to_read;
    pos += to_read;
  }

  std::streamsize gcount() override {
    return _gcount;
  }
};

namespace fastcdc {
  uint32_t cdc_offset(
    const std::span<uint8_t> data,
    uint32_t min_size,
    uint32_t avg_size,
    uint32_t max_size,
    uint32_t mask_s,
    uint32_t mask_l
  ) {
    uint32_t pattern = 0;
    uint32_t size = data.size();
    uint32_t barrier = std::min(avg_size, size);
    uint32_t i = std::min(barrier, min_size);

    // SuperCDC's even easier "backup mask" and backup result, if mask_l fails to find a cutoff point before the max_size we use the backup result
    // gotten with the easier to meet mask_b cutoff point. This should make it much more unlikely that we have to forcefully end chunks at max_size,
    // which helps better preserve the content defined nature of chunks and thus increase dedup ratios.
    std::optional<uint32_t> backup_i{};
    uint32_t mask_b = mask_l >> 1;

    // SuperCDC's Min-Max adjustment of the Gear Hash on jump to minimum chunk size, should improve deduplication ratios by better preserving
    // the content defined nature of the Hashes.
    // We backtrack a little to ensure when we get to the i we actually wanted we have the exact same hash as if we hadn't skipped anything.
    uint32_t remaining_minmax_adjustment = std::min<uint32_t>(i, 32);
    i -= remaining_minmax_adjustment;

    while (i < barrier) {
      pattern = (pattern >> 1) + constants::GEAR[data[i]];
      if (remaining_minmax_adjustment > 0) {
        remaining_minmax_adjustment--;
        i++;
        continue;
      }
      if (!(pattern & mask_s)) return i;
      i++;
    }
    barrier = std::min(max_size, size);
    while (i < barrier) {
      pattern = (pattern >> 1) + constants::GEAR[data[i]];
      if (!(pattern & mask_l)) return i;
      if (!backup_i.has_value() && !(pattern & mask_b)) backup_i = i;
      i++;
    }
    return backup_i.has_value() ? *backup_i : i;
  }

  std::pair<uint32_t, std::vector<uint32_t>> cdc_offset_with_features(
    const std::span<uint8_t> data,
    uint32_t min_size,
    uint32_t avg_size,
    uint32_t max_size,
    uint32_t mask_s,
    uint32_t mask_l
  ) {
    uint32_t pattern = 0;
    uint32_t size = data.size();
    uint32_t barrier = std::min(avg_size, size);

    std::pair<uint32_t, std::vector<uint32_t>> return_val{};
    uint32_t& i = std::get<0>(return_val);
    i = std::min(barrier, min_size);
    auto& features = std::get<1>(return_val);

    // SuperCDC's even easier "backup mask" and backup result, if mask_l fails to find a cutoff point before the max_size we use the backup result
    // gotten with the easier to meet mask_b cutoff point. This should make it much more unlikely that we have to forcefully end chunks at max_size,
    // which helps better preserve the content defined nature of chunks and thus increase dedup ratios.
    std::optional<uint32_t> backup_i{};
    uint32_t mask_b = mask_l >> 1;

    // SuperCDC's Min-Max adjustment of the Gear Hash on jump to minimum chunk size, should improve deduplication ratios by better preserving
    // the content defined nature of the Hashes.
    // We backtrack a little to ensure when we get to the i we actually wanted we have the exact same hash as if we hadn't skipped anything.
    uint32_t remaining_minmax_adjustment = std::min<uint32_t>(i, 32);
    i -= remaining_minmax_adjustment;

    while (i < barrier) {
      pattern = (pattern >> 1) + constants::GEAR[data[i]];
      if (remaining_minmax_adjustment > 0) {
        remaining_minmax_adjustment--;
        i++;
        continue;
      }
      if (!(pattern & mask_s)) return return_val;
      if (!(pattern & constants::CDS_SAMPLING_MASK)) {
        if (features.empty()) {
          features.resize(16);
        }
        for (int feature_i = 0; feature_i < 16; feature_i++) {
          const auto& [mi, ai] = constants::N_Transform_Coefs[feature_i];
          features[feature_i] = std::max<uint32_t>(features[feature_i], (mi * pattern + ai) % (1LL << 32));
        }
      }
      i += 1;
    }
    barrier = std::min(max_size, size);
    while (i < barrier) {
      pattern = (pattern >> 1) + constants::GEAR[data[i]];
      if (!(pattern & mask_l)) return return_val;
      if (!backup_i.has_value() && !(pattern & mask_b)) backup_i = i;
      if (!(pattern & constants::CDS_SAMPLING_MASK)) {
        if (features.empty()) {
          features.resize(16);
        }
        for (int feature_i = 0; feature_i < 16; feature_i++) {
          const auto& [mi, ai] = constants::N_Transform_Coefs[feature_i];
          features[feature_i] = std::max<uint32_t>(features[feature_i], (mi * pattern + ai) % (1LL << 32));
        }
      }
      i += 1;
    }
    if (backup_i.has_value()) {
      return { *backup_i, std::move(std::get<1>(return_val)) };
    }
    return return_val;
  }

  class ChunkGeneratorContext {
  public:
    IStreamLike* stream = nullptr;
    uint32_t min_size;
    uint32_t avg_size;
    uint32_t max_size;
    bool extract_features;

    uint32_t bits;
    uint32_t mask_s;
    uint32_t mask_l;
    uint32_t read_size;

    uint64_t offset = 0;
    std::vector<uint8_t> _blob{};
    std::span<uint8_t> blob{};
    uint32_t blob_len = 0;
    std::span<unsigned char>::iterator blob_it;

    ChunkGeneratorContext(IStreamLike* _stream, uint32_t _min_size, uint32_t _avg_size, uint32_t _max_size, bool _extract_features = false)
      : stream(_stream), min_size(_min_size), avg_size(_avg_size), max_size(_max_size), extract_features(_extract_features) {
      bits = utility::logarithm2(avg_size);
      mask_s = utility::mask(bits + 1);
      mask_l = utility::mask(bits - 1);
      read_size = std::max<uint32_t>(1024 * 64, max_size);

      _blob.resize(read_size);
      stream->read(_blob.data(), read_size);
      blob_len = stream->gcount();
      blob = std::span(_blob.data(), blob_len);
      blob_it = blob.begin();
    }
    ChunkGeneratorContext(std::span<uint8_t> _blob, uint32_t _min_size, uint32_t _avg_size, uint32_t _max_size, bool _fat, bool _extract_features = false)
      : min_size(_min_size), avg_size(_avg_size), max_size(_max_size), extract_features(_extract_features), blob(_blob) {
      bits = utility::logarithm2(avg_size);
      mask_s = utility::mask(bits + 1);
      mask_l = utility::mask(bits - 1);
      read_size = std::max<uint32_t>(1024 * 64, max_size);

      blob_len = blob.size();
      blob_it = blob.begin();
    }
  };

  std::optional<std::tuple<utility::CDChunk, std::span<uint8_t>>> chunk_generator(ChunkGeneratorContext& context) {
    std::optional<std::tuple<utility::CDChunk, std::span<uint8_t>>> result{};
    if (context.blob_len == 0) return result;

    if (context.blob_len <= context.max_size && context.stream != nullptr) {
      std::memmove(context.blob.data(), &*context.blob_it, context.blob_len);
      context.stream->read(reinterpret_cast<char*>(context.blob.data()) + context.blob_len, context.read_size - context.blob_len);
      context.blob_len += context.stream->gcount();
      context.blob_it = context.blob.begin();
    }
    uint32_t cp;
    std::vector<uint32_t> features{};
    if (context.extract_features) {
      std::tie(cp, features) = cdc_offset_with_features(std::span(context.blob_it, context.blob_len), context.min_size, context.avg_size, context.max_size, context.mask_s, context.mask_l);
    }
    else {
      cp = cdc_offset(std::span(context.blob_it, context.blob_len), context.min_size, context.avg_size, context.max_size, context.mask_s, context.mask_l);
    }
    result.emplace(utility::CDChunk(context.offset, cp), std::span(context.blob_it, cp));
    if (!features.empty()) {
      auto& chunk = std::get<0>(*result);
      for (uint64_t i = 0; i < 4; i++) {
        // Takes 4 features (32bit(4byte) fingerprints, so 4 of them is 16bytes) and hash them into a single SuperFeature (seed used arbitrarily just because it needed one)
        chunk.super_features[i] = XXH32(features.data() + static_cast<ptrdiff_t>(i * 4), 16, constants::CDS_SAMPLING_MASK);
      }
      chunk.feature_sampling_failure = false;
    }
    context.offset += cp;
    context.blob_it += cp;
    context.blob_len -= cp;
    return result;
  }
}

#include <immintrin.h>

__m256i movemask_inverse_epi8(const uint32_t mask) {
  __m256i vmask(_mm256_set1_epi32(mask));
  const __m256i shuffle(_mm256_setr_epi64x(0x0000000000000000, 0x0101010101010101, 0x0202020202020202, 0x0303030303030303));
  vmask = _mm256_shuffle_epi8(vmask, shuffle);
  const __m256i bit_mask(_mm256_set1_epi64x(0x7fbfdfeff7fbfdfe));
  vmask = _mm256_or_si256(vmask, bit_mask);
  return _mm256_cmpeq_epi8(vmask, _mm256_set1_epi64x(-1));
}

const __m256i all_1s = _mm256_set1_epi8(1);
const __m256i all_minus_1s = _mm256_set1_epi8(-1);

__m256i add_32bit_hash_to_simhash_counter(const uint32_t new_hash, __m256i counter_vector) {
  const auto new_hash_mask = movemask_inverse_epi8(new_hash);
  __m256i new_simhash_encoded = _mm256_blendv_epi8(all_1s, all_minus_1s, new_hash_mask);
  return _mm256_adds_epi8(new_simhash_encoded, counter_vector);
}

uint32_t finalize_32bit_simhash_from_counter(const __m256i counter_vector) {
  return _mm256_movemask_epi8(counter_vector);
}

std::tuple<std::bitset<64>, std::shared_ptr<uint32_t[]>, uint32_t> simhash_data_xxhash_shingling(uint8_t* data, uint32_t data_len, uint32_t chunk_size) {
  __m256i upper_counter_vector = _mm256_set1_epi8(0);
  __m256i lower_counter_vector = _mm256_set1_epi8(0);

  std::tuple<std::bitset<64>, std::shared_ptr<uint32_t[]>, uint32_t> return_val{};
  auto& simhash = std::get<0>(return_val);
  auto& minichunks_ptr = std::get<1>(return_val);
  auto& minichunks_len = std::get<2>(return_val);
  std::vector<uint32_t> minichunks_vec{};

  // Iterate over the data in chunks
  for (uint32_t i = 0; i < data_len; i += chunk_size) {
    // Calculate hash for current chunk
    const auto current_chunk_len = std::min(chunk_size, data_len - i);
    std::bitset<64> chunk_hash = XXH3_64bits(data + i, current_chunk_len);

    // Update SimHash vector with the hash of the chunk
    // Using AVX/2 we don't have enough vector size to handle the whole 64bit hash, we should be able to do it with AVX512,
    // but that is less readily available in desktop chips, so we split the hash into 2 and process it that way
    const std::bitset<64> upper_chunk_hash = chunk_hash >> 32;
    upper_counter_vector = add_32bit_hash_to_simhash_counter(upper_chunk_hash.to_ulong(), upper_counter_vector);
    const std::bitset<64> lower_chunk_hash = (chunk_hash << 32) >> 32;
    lower_counter_vector = add_32bit_hash_to_simhash_counter(lower_chunk_hash.to_ulong(), lower_counter_vector);

    minichunks_vec.emplace_back(current_chunk_len);
  }

  const uint32_t upper_chunk_hash = finalize_32bit_simhash_from_counter(upper_counter_vector);
  const uint32_t lower_chunk_hash = finalize_32bit_simhash_from_counter(lower_counter_vector);
  simhash = (static_cast<uint64_t>(upper_chunk_hash) << 32) | lower_chunk_hash;
  minichunks_ptr = std::make_shared_for_overwrite<uint32_t[]>(minichunks_vec.size());
  std::copy_n(minichunks_vec.data(), minichunks_vec.size(), minichunks_ptr.get());
  minichunks_len = minichunks_vec.size();
  return return_val;
}

std::tuple<std::bitset<64>, std::shared_ptr<uint32_t[]>, uint32_t> simhash_data_xxhash_cdc(uint8_t* data, uint32_t data_len, uint32_t chunk_size) {
  __m256i upper_counter_vector = _mm256_set1_epi8(0);
  __m256i lower_counter_vector = _mm256_set1_epi8(0);

  auto ctx = fastcdc::ChunkGeneratorContext(std::span(data, data_len), chunk_size / 2, chunk_size, chunk_size * 2, false);

  std::tuple<std::bitset<64>, std::shared_ptr<uint32_t[]>, uint32_t> return_val{};
  auto& simhash = std::get<0>(return_val);
  auto& minichunks_ptr = std::get<1>(return_val);
  auto& minichunks_len = std::get<2>(return_val);
  std::vector<uint32_t> minichunks_vec{};

  // Iterate over the data in chunks
  auto cdc_minichunk_result = fastcdc::chunk_generator(ctx);
  while (cdc_minichunk_result.has_value()) {
    auto& [chunk, chunk_data] = *cdc_minichunk_result;
    // Calculate hash for current chunk
    const std::bitset<64> chunk_hash = XXH3_64bits(data + chunk.offset, chunk_data.size());

    // Update SimHash vector with the hash of the chunk
    // Using AVX/2 we don't have enough vector size to handle the whole 64bit hash, we should be able to do it with AVX512,
    // but that is less readily available in desktop chips, so we split the hash into 2 and process it that way
    const std::bitset<64> upper_chunk_hash = chunk_hash >> 32;
    upper_counter_vector = add_32bit_hash_to_simhash_counter(upper_chunk_hash.to_ulong(), upper_counter_vector);
    const std::bitset<64> lower_chunk_hash = (chunk_hash << 32) >> 32;
    lower_counter_vector = add_32bit_hash_to_simhash_counter(lower_chunk_hash.to_ulong(), lower_counter_vector);

    minichunks_vec.emplace_back(chunk.length);
    cdc_minichunk_result = fastcdc::chunk_generator(ctx);
  }

  const uint32_t upper_chunk_hash = finalize_32bit_simhash_from_counter(upper_counter_vector);
  const uint32_t lower_chunk_hash = finalize_32bit_simhash_from_counter(lower_counter_vector);
  simhash = (static_cast<uint64_t>(upper_chunk_hash) << 32) | lower_chunk_hash;
  minichunks_ptr = std::make_shared_for_overwrite<uint32_t[]>(minichunks_vec.size());
  std::copy_n(minichunks_vec.data(), minichunks_vec.size(), minichunks_ptr.get());
  minichunks_len = minichunks_vec.size();
  return return_val;
}

template<uint8_t bit_size>
uint64_t hamming_distance(std::bitset<bit_size> data1, std::bitset<bit_size> data2) {
  const auto val = data1 ^ data2;
  return val.count();
}

template<uint8_t bit_size>
uint8_t hamming_syndrome(const std::bitset<bit_size>& data) {
  int result = 0;
  std::bitset<bit_size> mask {0b1};
  for (uint8_t i = 0; i < bit_size; i++) {
    auto bit = data & mask;
    if (bit != 0) result ^= i;
    mask <<= 1;
  }

  return result;
}

template<uint8_t bit_size>
std::bitset<bit_size> hamming_base(std::bitset<bit_size> data) {
  auto syndrome = hamming_syndrome(data);
  auto base = data.flip(syndrome);
  // The first bit doesn't really participate in non-extended hamming codes (and extended ones are not useful to us)
  // So we just collapse to them all to the version with 0 on the first bit, allows us to match some hamming distance 2 data
  base[0] = 0;
  return base;
}

enum LZInstructionType {
  COPY,
  INSERT,
  DELTA
};

struct LZInstruction {
  LZInstructionType type;
  uint64_t offset;  // For COPY: previous offset; For INSERT: offset on the original stream; For Delta: previous offset of original data
  uint64_t size;  // How much data to be copied or inserted, or size of the delta original data
};

struct DeltaEncodingResult {
  uint64_t estimated_savings;
  std::vector<LZInstruction> instructions;
};

uint64_t find_identical_prefix_byte_count(std::span<uint8_t> data1_span, std::span<uint8_t> data2_span) {
  // TODO: we just use std::memcmp and hope we have SIMD optimized versions, maybe explicitly handle SIMD?
  const auto cmp_size = std::min(data1_span.size(), data2_span.size());
  uint64_t matching_data_count = 0;
  uint64_t i = 0;
  while (i < cmp_size) {
    const bool can_sse_compare = cmp_size - i >= 16;
    if (can_sse_compare && std::memcmp(data1_span.data() + i, data2_span.data() + i, 16) == 0) {
      matching_data_count += 16;
      i += 16;
      continue;
    }

    const bool can_u64int_compare = cmp_size - i >= 8;
    if (can_u64int_compare && std::memcmp(data1_span.data() + i, data2_span.data() + i, 8) == 0) {
      matching_data_count += 8;
      i += 8;
      continue;
    }

    if (*(data1_span.data() + i) != *(data2_span.data() + i)) break;
    matching_data_count++;
    i++;
  }
  return matching_data_count;
}

uint64_t find_identical_suffix_byte_count(std::span<uint8_t> data1_span, std::span<uint8_t> data2_span) {
  // TODO: we just use std::memcmp and hope we have SIMD optimized versions, maybe explicitly handle SIMD?
  const auto cmp_size = std::min(data1_span.size(), data2_span.size());
  const auto data1_start = data1_span.data() + data1_span.size() - cmp_size;
  const auto data2_start = data2_span.data() + data2_span.size() - cmp_size;

  uint64_t matching_data_count = 0;
  uint64_t i = 0;
  while (i < cmp_size) {
    const bool can_sse_compare = cmp_size - i >= 16;
    if (
      can_sse_compare &&
      std::memcmp(data1_start + cmp_size - 16 - i, data2_start + cmp_size - 16 - i, 16) == 0
      ) {
      matching_data_count += 16;
      i += 16;
      continue;
    }

    const bool can_u64int_compare = cmp_size - i >= 8;
    if (
      can_u64int_compare &&
      std::memcmp(data1_start + cmp_size - 8 - i, data2_start + cmp_size - 8 - i, 8) == 0
      ) {
      matching_data_count += 8;
      i += 8;
      continue;
    }

    if (*(data1_start + cmp_size - 1 - i) != *(data2_start + cmp_size - 1 - i)) break;
    matching_data_count++;
    i++;
  }
  return matching_data_count;
}

DeltaEncodingResult simulate_delta_encoding_shingling(const utility::CDChunk& chunk, const utility::CDChunk& similar_chunk, uint32_t minichunk_size) {
  DeltaEncodingResult result;
  //DDelta style delta encoding, first start by greedily attempting to match data at the start and end on the chunks
  const uint64_t start_matching_data_len = find_identical_prefix_byte_count(std::span(chunk.data.get(), chunk.length), std::span(similar_chunk.data.get(), similar_chunk.length));
  uint64_t end_matching_data_len = find_identical_suffix_byte_count(std::span(chunk.data.get(), chunk.length), std::span(similar_chunk.data.get(), similar_chunk.length));

  uint64_t saved_size = start_matching_data_len + end_matching_data_len;

  // (offset, hash), it might be tempting to change this for a hashmap, don't, it will be slower as sequential search is really fast because there won't be many minuchunks
  std::vector<std::tuple<uint32_t, uint64_t>> similar_chunk_minichunk_hashes;

  // Iterate over the data in chunks
  uint32_t minichunk_offset = 0;
  for (uint64_t i = 0; i < similar_chunk.length; i += minichunk_size) {
    // Calculate hash for current chunk
    const auto minichunk_len = std::min<uint64_t>(minichunk_size, similar_chunk.length - i);
    uint64_t minichunk_hash = XXH3_64bits(similar_chunk.data.get() + i, minichunk_len);
    similar_chunk_minichunk_hashes.emplace_back(minichunk_offset, minichunk_hash);
    minichunk_offset += minichunk_len;
  }

  if (start_matching_data_len) {
    result.instructions.emplace_back(LZInstructionType::COPY, 0, start_matching_data_len);
  }

  uint64_t unaccounted_data_start_pos = start_matching_data_len;
  for (uint32_t minichunk_offset = 0; minichunk_offset < chunk.length; minichunk_offset += minichunk_size) {
    const auto minichunk_len = std::min<uint64_t>(minichunk_size, chunk.length - minichunk_offset);
    XXH64_hash_t minichunk_hash = 0;

    minichunk_hash = XXH3_64bits(chunk.data.get() + minichunk_offset, minichunk_len);

    const auto search_for_similar_minichunk = [&minichunk_hash](const std::tuple<uint32_t, uint64_t>& minichunk_data) { return std::get<1>(minichunk_data) == minichunk_hash; };
    auto similar_minichunk_iter = std::find_if(
      similar_chunk_minichunk_hashes.cbegin(),
      similar_chunk_minichunk_hashes.cend(),
      search_for_similar_minichunk
    );
    if (similar_minichunk_iter == similar_chunk_minichunk_hashes.cend()) continue;

    uint64_t extended_size = 0;
    uint64_t backwards_extended_size = 0;
    uint64_t best_similar_chunk_minichunk_offset = 0;  // we will select the minichunk that can be extended the most

    // We find the instance of the minichunk on the similar_chunk that can be extended the most
    while (similar_minichunk_iter != similar_chunk_minichunk_hashes.cend()) {
      auto& similar_minichunk_data = *similar_minichunk_iter;
      auto& candidate_similar_chunk_offset = std::get<0>(similar_minichunk_data);

      // We have a match, attempt to extend it, first backwards
      const uint64_t candidate_backwards_extended_size = find_identical_suffix_byte_count(
        std::span(chunk.data.get(), minichunk_offset),
        std::span(similar_chunk.data.get(), candidate_similar_chunk_offset)
      );

      // Then extend forwards
      const uint64_t candidate_extended_size = find_identical_prefix_byte_count(
        std::span(chunk.data.get() + minichunk_offset + minichunk_len, chunk.length - minichunk_offset - minichunk_len),
        std::span(similar_chunk.data.get() + candidate_similar_chunk_offset + minichunk_len, similar_chunk.length - candidate_similar_chunk_offset - minichunk_len)
      );

      if (candidate_backwards_extended_size + candidate_extended_size >= backwards_extended_size + extended_size) {
        extended_size = candidate_extended_size;
        backwards_extended_size = candidate_backwards_extended_size;
        best_similar_chunk_minichunk_offset = candidate_similar_chunk_offset;
      }

      similar_minichunk_iter = std::find_if(similar_minichunk_iter + 1, similar_chunk_minichunk_hashes.cend(), search_for_similar_minichunk);
    }

    // Any remaining unaccounted data needs to be INSERTed as it couldn't be matched from the similar_chunk
    const auto backward_extended_minichunk_start = minichunk_offset - backwards_extended_size;
    if (unaccounted_data_start_pos < backward_extended_minichunk_start) {
      result.instructions.emplace_back(
        LZInstructionType::INSERT,
        unaccounted_data_start_pos,
        minichunk_offset - backwards_extended_size - unaccounted_data_start_pos
      );
    }
    else {
      uint64_t to_backtrack_len = unaccounted_data_start_pos - backward_extended_minichunk_start;
      while (to_backtrack_len > 0) {
        auto& prevInstruction = result.instructions.back();
        const auto possible_backtrack = std::min(to_backtrack_len, prevInstruction.size);
        // Prevent double counting as this will all be added back with the current minichunk's COPY
        if (prevInstruction.type == LZInstructionType::COPY) saved_size -= possible_backtrack;
        if (prevInstruction.size <= to_backtrack_len) {
          result.instructions.pop_back();
        }
        else {
          prevInstruction.size -= to_backtrack_len;
        }
        to_backtrack_len -= possible_backtrack;
      }
    }
    const auto copy_instruction_size = backwards_extended_size + minichunk_len + extended_size;
    result.instructions.emplace_back(
      LZInstructionType::COPY,
      best_similar_chunk_minichunk_offset - backwards_extended_size,
      copy_instruction_size
    );
    saved_size += copy_instruction_size;
    unaccounted_data_start_pos = minichunk_offset + minichunk_len + extended_size;
  }
  // After the iteration there could remain some unaccounted data at the end (before the matched data), if so save it as an INSERT
  const auto end_matched_data_start_pos = std::max<uint64_t>(unaccounted_data_start_pos, chunk.length - end_matching_data_len);
  end_matching_data_len = chunk.length - end_matched_data_start_pos;
  if (unaccounted_data_start_pos < end_matched_data_start_pos) {
    result.instructions.emplace_back(
      LZInstructionType::INSERT,
      unaccounted_data_start_pos,
      end_matched_data_start_pos - unaccounted_data_start_pos
    );
  }
  // And finally any data matched at the end of the chunks
  if (end_matching_data_len) {
    result.instructions.emplace_back(
      LZInstructionType::COPY,
      similar_chunk.length - end_matching_data_len,
      end_matching_data_len
    );
  }

  result.estimated_savings = saved_size;
  return result;
}

DeltaEncodingResult simulate_delta_encoding_using_minichunks(const utility::CDChunk& chunk, const utility::CDChunk& similar_chunk, uint32_t minichunk_size) {
  DeltaEncodingResult result;
  //DDelta style delta encoding, first start by greedily attempting to match data at the start and end on the chunks
  const uint64_t start_matching_data_len = find_identical_prefix_byte_count(std::span(chunk.data.get(), chunk.length), std::span(similar_chunk.data.get(), similar_chunk.length));
  uint64_t end_matching_data_len = find_identical_suffix_byte_count(std::span(chunk.data.get(), chunk.length), std::span(similar_chunk.data.get(), similar_chunk.length));

  uint64_t saved_size = start_matching_data_len + end_matching_data_len;

  // (offset, hash), it might be tempting to change this for a hashmap, don't, it will be slower as sequential search is really fast because there won't be many minuchunks
  std::vector<std::tuple<uint32_t, uint64_t>> similar_chunk_minichunk_hashes;

  uint32_t minichunk_offset = 0;
  for (const auto& minichunk_len : std::span(similar_chunk.minichunks.get(), similar_chunk.minichunks_len)) {
    XXH64_hash_t minichunk_hash = XXH3_64bits(similar_chunk.data.get() + minichunk_offset, minichunk_len);
    similar_chunk_minichunk_hashes.emplace_back(minichunk_offset, minichunk_hash);
    minichunk_offset += minichunk_len;
  }

  if (start_matching_data_len) {
    result.instructions.emplace_back(LZInstructionType::COPY, 0, start_matching_data_len);
  }

  uint64_t unaccounted_data_start_pos = start_matching_data_len;
  minichunk_offset = 0;
  for (uint32_t minichunk_i = 0; minichunk_i < chunk.minichunks_len; minichunk_i++) {
    const auto& minichunk_len = chunk.minichunks[minichunk_i];
    XXH64_hash_t minichunk_hash = 0;

    minichunk_hash = XXH3_64bits(chunk.data.get() + minichunk_offset, minichunk_len);

    const auto search_for_similar_minichunk = [&minichunk_hash](const std::tuple<uint32_t, uint64_t>& minichunk_data) { return std::get<1>(minichunk_data) == minichunk_hash; };
    auto similar_minichunk_iter = std::find_if(
      similar_chunk_minichunk_hashes.cbegin(),
      similar_chunk_minichunk_hashes.cend(),
      search_for_similar_minichunk
    );
    if (similar_minichunk_iter == similar_chunk_minichunk_hashes.cend()) {
      minichunk_offset += minichunk_len;
      continue;
    }

    uint64_t extended_size = 0;
    uint64_t backwards_extended_size = 0;
    uint64_t best_similar_chunk_minichunk_offset = 0;  // we will select the minichunk that can be extended the most

    // We find the instance of the minichunk on the similar_chunk that can be extended the most
    while (similar_minichunk_iter != similar_chunk_minichunk_hashes.cend()) {
      auto& similar_minichunk_data = *similar_minichunk_iter;
      auto& candidate_similar_chunk_offset = std::get<0>(similar_minichunk_data);

      // We have a match, attempt to extend it, first backwards
      const uint64_t candidate_backwards_extended_size = find_identical_suffix_byte_count(
        std::span(chunk.data.get(), minichunk_offset),
        std::span(similar_chunk.data.get(), candidate_similar_chunk_offset)
      );

      // Then extend forwards
      const uint64_t candidate_extended_size = find_identical_prefix_byte_count(
        std::span(chunk.data.get() + minichunk_offset + minichunk_len, chunk.length - minichunk_offset - minichunk_len),
        std::span(similar_chunk.data.get() + candidate_similar_chunk_offset + minichunk_len, similar_chunk.length - candidate_similar_chunk_offset - minichunk_len)
      );

      if (candidate_backwards_extended_size + candidate_extended_size >= backwards_extended_size + extended_size) {
        extended_size = candidate_extended_size;
        backwards_extended_size = candidate_backwards_extended_size;
        best_similar_chunk_minichunk_offset = candidate_similar_chunk_offset;
      }

      similar_minichunk_iter = std::find_if(similar_minichunk_iter + 1, similar_chunk_minichunk_hashes.cend(), search_for_similar_minichunk);
    }

    // Any remaining unaccounted data needs to be INSERTed as it couldn't be matched from the similar_chunk
    const auto backward_extended_minichunk_start = minichunk_offset - backwards_extended_size;
    if (unaccounted_data_start_pos < backward_extended_minichunk_start) {
      result.instructions.emplace_back(
        LZInstructionType::INSERT,
        unaccounted_data_start_pos,
        minichunk_offset - backwards_extended_size - unaccounted_data_start_pos
      );
    }
    else {
      uint64_t to_backtrack_len = unaccounted_data_start_pos - backward_extended_minichunk_start;
      while (to_backtrack_len > 0) {
        auto& prevInstruction = result.instructions.back();
        const auto possible_backtrack = std::min(to_backtrack_len, prevInstruction.size);
        // Prevent double counting as this will all be added back with the current minichunk's COPY
        if (prevInstruction.type == LZInstructionType::COPY) saved_size -= possible_backtrack;
        if (prevInstruction.size <= to_backtrack_len) {
          result.instructions.pop_back();
        }
        else {
          prevInstruction.size -= to_backtrack_len;
        }
        to_backtrack_len -= possible_backtrack;
      }
    }
    const auto copy_instruction_size = backwards_extended_size + minichunk_len + extended_size;
    result.instructions.emplace_back(
      LZInstructionType::COPY,
      best_similar_chunk_minichunk_offset - backwards_extended_size,
      copy_instruction_size
    );
    saved_size += copy_instruction_size;
    unaccounted_data_start_pos = minichunk_offset + minichunk_len + extended_size;
    minichunk_offset += minichunk_len;
  }
  // After the iteration there could remain some unaccounted data at the end (before the matched data), if so save it as an INSERT
  const auto end_matched_data_start_pos = std::max<uint64_t>(unaccounted_data_start_pos, chunk.length - end_matching_data_len);
  end_matching_data_len = chunk.length - end_matched_data_start_pos;
  if (unaccounted_data_start_pos < end_matched_data_start_pos) {
    result.instructions.emplace_back(
      LZInstructionType::INSERT,
      unaccounted_data_start_pos,
      end_matched_data_start_pos - unaccounted_data_start_pos
    );
  }
  // And finally any data matched at the end of the chunks
  if (end_matching_data_len) {
    result.instructions.emplace_back(
      LZInstructionType::COPY,
      similar_chunk.length - end_matching_data_len,
      end_matching_data_len
    );
  }

  result.estimated_savings = saved_size;
  return result;
}

class WrappedIStreamInputStream: public InputStream {
private:
  std::istream* istream;

public:
  WrappedIStreamInputStream(std::istream* _istream): istream(_istream) {}

  bool eof() const override { return istream->eof(); }
  size_t read(unsigned char* buffer, const size_t size) override {
    istream->read(reinterpret_cast<char*>(buffer), size);
    return istream->gcount();
  }
};

class WrappedOStreamOutputStream: public OutputStream {
private:
  std::ostream* ostream;
  std::vector<uint8_t> output_buffer;
  uint64_t buffer_used_len = 0;

public:
  explicit WrappedOStreamOutputStream(std::ostream* _ostream, uint64_t buffer_size = 200 * 1024 * 1024): ostream(_ostream) {
    output_buffer.resize(buffer_size);
  }

  ~WrappedOStreamOutputStream() override {
    if (buffer_used_len > 0) {
      ostream->write(reinterpret_cast<const char*>(output_buffer.data()), buffer_used_len);
      buffer_used_len = 0;
    }
    ostream->flush();
  }

  size_t write(const unsigned char* buffer, const size_t size) override {
    if (size > output_buffer.size()) {
      if (buffer_used_len > 0) {
        ostream->write(reinterpret_cast<const char*>(output_buffer.data()), buffer_used_len);
        buffer_used_len = 0;
      }
      ostream->write(reinterpret_cast<const char*>(buffer), size);
      return size;
    }
    if (size + buffer_used_len > output_buffer.size()) {
      ostream->write(reinterpret_cast<const char*>(output_buffer.data()), buffer_used_len);
      buffer_used_len = 0;
    }
    std::copy_n(buffer, size, output_buffer.data() + buffer_used_len);
    buffer_used_len += size;
    return size;
  }
};

std::tuple<uint64_t, uint64_t> get_chunk_i_and_pos_for_offset(std::vector<utility::CDChunk>& chunks, uint64_t offset) {
  static constexpr auto compare_offset = [](const utility::CDChunk& x, const utility::CDChunk& y) { return x.offset < y.offset; };

  const auto search_chunk = utility::CDChunk(offset, 1);
  auto chunk_iter = std::ranges::lower_bound(std::as_const(chunks), search_chunk, compare_offset);
  uint64_t chunk_i = chunk_iter != chunks.cend() ? chunk_iter - chunks.cbegin() : chunks.size() - 1;
  if (chunks[chunk_i].offset > offset) chunk_i--;
  utility::CDChunk* chunk = &chunks[chunk_i];
  uint64_t chunk_pos = offset - chunk->offset;
  return { chunk_i, chunk_pos };
}

class LZInstructionManager {
  std::vector<utility::CDChunk>* chunks;
  std::deque<LZInstruction> instructions;

  std::ostream* ostream;
  WrappedOStreamOutputStream output_stream;
  BitOutputStream bit_output_stream;

  const bool use_match_extension_backwards;
  const bool use_match_extension;

  uint64_t accumulated_savings = 0;
  uint64_t accumulated_extended_backwards_savings = 0;
  uint64_t accumulated_extended_forwards_savings = 0;
  uint64_t omitted_small_match_size = 0;
  uint64_t outputted_up_to_offset = 0;
  uint64_t outputted_lz_instructions = 0;

public:
  explicit LZInstructionManager(std::vector<utility::CDChunk>* _chunks, bool _use_match_extension_backwards, bool _use_match_extension, std::ostream* _ostream)
    : chunks(_chunks), ostream(_ostream), output_stream(_ostream), bit_output_stream(output_stream),
      use_match_extension_backwards(_use_match_extension_backwards), use_match_extension(_use_match_extension) {}

  ~LZInstructionManager() {
    bit_output_stream.flush();
    ostream->flush();
  }

  uint64_t instructionCount() const {
    return outputted_lz_instructions;
  }

  uint64_t accumulatedSavings() const {
    return accumulated_savings;
  }
  uint64_t accumulatedExtendedBackwardsSavings() const {
    return accumulated_extended_backwards_savings;
  }
  uint64_t accumulatedExtendedForwardsSavings() const {
    return accumulated_extended_forwards_savings;
  }

  uint64_t omittedSmallMatchSize() const {
    return omitted_small_match_size;
  }

  void addInstruction(LZInstruction&& instruction, uint64_t current_offset, bool verify = true, std::optional<uint64_t> _earliest_allowed_offset = std::nullopt) {
    uint64_t earliest_allowed_offset = _earliest_allowed_offset.has_value() ? *_earliest_allowed_offset : 0;
#ifndef NDEBUG
    if (instruction.type == LZInstructionType::INSERT && instruction.offset != current_offset) {
      std::cout << "INSERT LZInstruction added is not at current offset!" << std::flush;
      exit(1);
    }
#endif
    if (instructions.empty()) {
      instructions.insert(instructions.end(), std::move(instruction));
      return;
    }

    // If same type of instruction, and it starts from the offset at the end of the previous instruction we just extend that one
    LZInstruction* prevInstruction = &instructions.back();
    if (
      prevInstruction->type == instruction.type &&
      prevInstruction->offset + prevInstruction->size == instruction.offset
    ) {
      prevInstruction->size += instruction.size;
      if (prevInstruction->type == LZInstructionType::COPY) {
        accumulated_savings += instruction.size;
      }
    }
    else {
      std::vector<uint8_t> verify_buffer_orig_data{};
      std::vector<uint8_t> verify_buffer_instruction_data{};
      uint64_t verify_end_offset = current_offset + instruction.size;
      if (verify) {
        verify_buffer_orig_data.resize(prevInstruction->size + instruction.size);
        verify_buffer_instruction_data.resize(prevInstruction->size + instruction.size);

        std::fstream verify_file{};
        verify_file.open("C:\\Users\\Administrator\\Desktop\\fastcdc_test\\sims.tar.pcf", std::ios_base::in | std::ios_base::binary);
        const auto data_count = prevInstruction->size + instruction.size;
        // Read original data
        verify_file.seekg(current_offset - prevInstruction->size);
        verify_file.read(reinterpret_cast<char*>(verify_buffer_orig_data.data()), data_count);
        // Read data according to the instructions
        verify_file.seekg(prevInstruction->offset);
        verify_file.read(reinterpret_cast<char*>(verify_buffer_instruction_data.data()), prevInstruction->size);
        verify_file.seekg(instruction.offset);
        verify_file.read(reinterpret_cast<char*>(verify_buffer_instruction_data.data()) + prevInstruction->size, instruction.size);
        // Ensure data matches
        if (std::memcmp(verify_buffer_orig_data.data(), verify_buffer_instruction_data.data(), data_count) != 0) {
          std::cout << "Error while verifying addInstruction at offset " + std::to_string(current_offset) + "\n" << std::flush;
          exit(1);
        }
      }

      const uint64_t original_instruction_size = instruction.size;
      uint64_t extended_backwards_savings = 0;
      const bool can_backwards_extend_instruction = instruction.offset > earliest_allowed_offset;
      if (use_match_extension_backwards && instruction.type == LZInstructionType::COPY && can_backwards_extend_instruction) {
        auto [instruction_chunk_i, instruction_chunk_pos] = get_chunk_i_and_pos_for_offset(*chunks, instruction.offset - 1);
        utility::CDChunk* instruction_chunk = &(*chunks)[instruction_chunk_i];
#ifndef NDEBUG
        if (instruction_chunk->offset + instruction_chunk_pos != instruction.offset - 1) {
          std::cout << "BACKWARD MATCH EXTENSION NEW INSTRUCTION OFFSET MISMATCH" << std::flush;
          exit(1);
        }
#endif

        uint64_t prevInstruction_end_offset = current_offset;
        const bool can_backwards_extend_prevInstruction = prevInstruction_end_offset > earliest_allowed_offset;
        while (!instructions.empty() && can_backwards_extend_prevInstruction) {
          // TODO: figure out why this happens, most likely some match extension is not properly cleaning up INSERTs that are shrunk into nothingness
          if (prevInstruction->size == 0) {
            instructions.pop_back();
            if (instructions.empty()) {
              break;
            }
            prevInstruction = &instructions.back();
            continue;
          }

          auto [prevInstruction_chunk_i, prevInstruction_chunk_pos] = get_chunk_i_and_pos_for_offset(*chunks, prevInstruction_end_offset - 1);
          utility::CDChunk* prevInstruction_chunk = &(*chunks)[prevInstruction_chunk_i];
#ifndef NDEBUG
          if (prevInstruction_chunk->offset + prevInstruction_chunk_pos != prevInstruction_end_offset - 1) {
            std::cout << "BACKWARD MATCH EXTENSION PREVIOUS INSTRUCTION OFFSET MISMATCH" << std::flush;
            exit(1);
          }
#endif

          bool stop_matching = false;
          while (true) {
            const auto bytes_remaining_for_prevInstruction_to_earliest_allowed_offset = prevInstruction_end_offset - earliest_allowed_offset;
            // bytes_remaining_for_prevInstruction_backtrack is including the current byte, which is why we +1 to bytes_remaining_for_prevInstruction_to_earliest_allowed_offset,
            // otherwise this would be 0 when we are on the actual earliest_allowed_offset
            const auto bytes_remaining_for_prevInstruction_backtrack = std::min(
              bytes_remaining_for_prevInstruction_to_earliest_allowed_offset + 1,
              std::min(prevInstruction->size, prevInstruction_chunk_pos + 1)
            );

            const auto prevInstruction_backtrack_data = std::span(
              prevInstruction_chunk->data.get() + prevInstruction_chunk_pos - (bytes_remaining_for_prevInstruction_backtrack - 1),
              bytes_remaining_for_prevInstruction_backtrack
            );
            const auto instruction_backtrack_data = std::span(instruction_chunk->data.get(),instruction_chunk_pos + 1);

            uint64_t matched_amt = find_identical_suffix_byte_count(prevInstruction_backtrack_data, instruction_backtrack_data);
            if (matched_amt == 0) {
              stop_matching = true;
              break;
            }

            prevInstruction_end_offset -= matched_amt;
            prevInstruction->size -= matched_amt;
            instruction.offset -= matched_amt;
            instruction.size += matched_amt;
            if (prevInstruction->type == LZInstructionType::INSERT) extended_backwards_savings += matched_amt;

            // Can't keep extending backwards, any previous data is disallowed (presumably to accomodate max matching distance)
            if (prevInstruction_end_offset < earliest_allowed_offset) {
              stop_matching = true;
              break;
            }
            if (instruction.offset < earliest_allowed_offset) {
              stop_matching = true;
              break;
            }

            if (instruction_chunk_pos >= matched_amt) {
              instruction_chunk_pos -= matched_amt;
            }
            else if (instruction_chunk_i == 0) {
              stop_matching = true;
              break;
            }
            else {
              instruction_chunk_i--;
              instruction_chunk = &(*chunks)[instruction_chunk_i];
              instruction_chunk_pos = instruction_chunk->length - 1;
            }

            if (prevInstruction->size == 0) {
              instructions.pop_back();
              if (instructions.empty()) {
                stop_matching = true;
                break;
              }
              prevInstruction = &instructions.back();
              break;
            }

            if (prevInstruction_chunk_pos >= matched_amt) {
              prevInstruction_chunk_pos -= matched_amt;
            }
            else if (prevInstruction_chunk_i == 0) {
              stop_matching = true;
              break;
            }
            else {
              prevInstruction_chunk_i--;
              prevInstruction_chunk = &(*chunks)[prevInstruction_chunk_i];
              prevInstruction_chunk_pos = prevInstruction_chunk->length - 1;
            }
          }
          if (stop_matching) break;
        }
      }
      uint64_t extended_forwards_savings = 0;
      if (use_match_extension && prevInstruction->type == LZInstructionType::COPY) {
        uint64_t prevInstruction_offset = prevInstruction->offset + prevInstruction->size;

        const bool can_forward_extend_prevInstruction = prevInstruction_offset >= earliest_allowed_offset;
        if (can_forward_extend_prevInstruction) {
          auto [prevInstruction_chunk_i, prevInstruction_chunk_pos] = get_chunk_i_and_pos_for_offset(*chunks, prevInstruction_offset);
          utility::CDChunk* prevInstruction_chunk = &(*chunks)[prevInstruction_chunk_i];

          auto [instruction_chunk_i, instruction_chunk_pos] = get_chunk_i_and_pos_for_offset(*chunks, instruction.offset);
          utility::CDChunk* instruction_chunk = &(*chunks)[instruction_chunk_i];

          while (instruction.size > 0) {
            const auto prevInstruction_extend_data = std::span(
              prevInstruction_chunk->data.get() + prevInstruction_chunk_pos,
              prevInstruction_chunk->length - prevInstruction_chunk_pos
            );
            const auto instruction_extend_data = std::span(
              instruction_chunk->data.get() + instruction_chunk_pos,
              std::min(instruction_chunk->length - instruction_chunk_pos, instruction.size)
            );

            uint64_t matched_amt = find_identical_prefix_byte_count(prevInstruction_extend_data, instruction_extend_data);
            if (matched_amt == 0) break;

            prevInstruction->size += matched_amt;
            instruction.offset += matched_amt;
            instruction.size -= matched_amt;
            if (instruction.type == LZInstructionType::INSERT) extended_forwards_savings += matched_amt;

            prevInstruction_chunk_pos += matched_amt;
            if (prevInstruction_chunk_pos == prevInstruction_chunk->length) {
              prevInstruction_chunk_i++;
              prevInstruction_chunk = &(*chunks)[prevInstruction_chunk_i];
              prevInstruction_chunk_pos = 0;
            }

            if (instruction.size == 0) {
              break;
            }

            instruction_chunk_pos += matched_amt;
            if (instruction_chunk_pos == instruction_chunk->length) {
              instruction_chunk_i++;
              instruction_chunk = &(*chunks)[instruction_chunk_i];
              instruction_chunk_pos = 0;
            }
          }
        }
      }

      prevInstruction = &instructions.back();
      if (verify) {
        std::fstream verify_file{};
        verify_file.open("C:\\Users\\Administrator\\Desktop\\fastcdc_test\\sims.tar.pcf", std::ios_base::in | std::ios_base::binary);
        const auto data_count = prevInstruction->size + instruction.size;

        verify_buffer_orig_data.resize(data_count);
        verify_buffer_instruction_data.resize(data_count);

        uint64_t orig_data_start = verify_end_offset - data_count;

        // Read original data
        verify_file.seekg(orig_data_start);
        verify_file.read(reinterpret_cast<char*>(verify_buffer_orig_data.data()), data_count);
        // Read data according to the instructions
        verify_file.seekg(prevInstruction->offset);
        verify_file.read(reinterpret_cast<char*>(verify_buffer_instruction_data.data()), prevInstruction->size);
        verify_file.seekg(instruction.offset);
        verify_file.read(reinterpret_cast<char*>(verify_buffer_instruction_data.data()) + prevInstruction->size, instruction.size);
        // Ensure data matches
        if (std::memcmp(verify_buffer_orig_data.data(), verify_buffer_instruction_data.data(), data_count) != 0) {
          std::cout << "Error while verifying addInstruction at offset " + std::to_string(current_offset) + "\n" << std::flush;
          exit(1);
        }
      }

      // If instruction is COPY then any forward extending is actually just retreading ground from what was extended backwards and
      // at most from the instruction itself, so it's all already counted there
      if (instruction.type == LZInstructionType::COPY) accumulated_savings += extended_backwards_savings + original_instruction_size;
      else accumulated_savings += extended_forwards_savings;
      accumulated_extended_backwards_savings += extended_backwards_savings;
      accumulated_extended_forwards_savings += extended_forwards_savings;

      // If the whole instruction is consumed by extending the previous COPY, then just quit, there is no instruction to add anymore
      if (instruction.size == 0) {
        return;
      }
      // If we are adding an INSERT, then the previous COPY was already extended backwards and/or forwards as much as it could.
      // If it's still so small that the overhead of outputting the extra instruction is larger than the deduplication we would get
      // we just extend this insert so that we save that overhead
      if (instruction.type == LZInstructionType::INSERT && prevInstruction->type == LZInstructionType::COPY && prevInstruction->size < 128) {
        accumulated_savings -= prevInstruction->size;
        omitted_small_match_size += prevInstruction->size;
        prevInstruction->type = LZInstructionType::INSERT;
        prevInstruction->offset = instruction.offset - prevInstruction->size;
        prevInstruction->size = instruction.size + prevInstruction->size;
        return;
      }
      instructions.insert(instructions.end(), std::move(instruction));
    }
  }

  void revertInstructionSize(uint64_t size) {
    LZInstruction* prevInstruction = &instructions.back();
    while (prevInstruction->size <= size) {
      if (prevInstruction->type == LZInstructionType::COPY) accumulated_savings -= size;
      size -= prevInstruction->size;
      instructions.pop_back();
      prevInstruction = &instructions.back();
    }
    prevInstruction->size -= size;
    if (prevInstruction->type == LZInstructionType::COPY) accumulated_savings -= size;
  }

  void dump(std::istream& istream, bool verify_copies = true, std::optional<uint64_t> up_to_offset = std::nullopt) {
    std::vector<char> buffer;
    std::vector<char> verify_buffer;

    auto prev_outputted_up_to_offset = outputted_up_to_offset;
    while (!instructions.empty()) {
      if (up_to_offset.has_value() && outputted_up_to_offset > *up_to_offset) break;
      auto instruction = std::move(instructions.front());
      instructions.pop_front();

      bit_output_stream.put(instruction.type, 8);
      bit_output_stream.putVLI(instruction.size);
      if (instruction.type == LZInstructionType::INSERT) {
        auto [instruction_chunk_i, instruction_chunk_pos] = get_chunk_i_and_pos_for_offset(*chunks, instruction.offset);
        buffer.resize(instruction.size);
        uint64_t written = 0;
        while (written < instruction.size) {
          auto& instruction_chunk = (*chunks)[instruction_chunk_i];
          uint64_t to_read_from_chunk = std::min(instruction.size - written, instruction_chunk.length - instruction_chunk_pos);
          std::copy_n(instruction_chunk.data.get() + instruction_chunk_pos, to_read_from_chunk, buffer.data() + written);
          written += to_read_from_chunk;

          instruction_chunk_i++;
          instruction_chunk_pos = 0;
        }
        bit_output_stream.putBytes(reinterpret_cast<const uint8_t*>(buffer.data()), instruction.size);
      }
      else {
        bit_output_stream.putVLI(outputted_up_to_offset - instruction.offset);
        if (verify_copies) {
          istream.seekg(instruction.offset);
          buffer.resize(instruction.size);
          istream.read(buffer.data(), instruction.size);

          istream.seekg(outputted_up_to_offset);
          verify_buffer.resize(instruction.size);
          istream.read(verify_buffer.data(), instruction.size);

          if (std::memcmp(verify_buffer.data(), buffer.data(), instruction.size) != 0) {
            std::cout << "Error while verifying outputted match at offset " + std::to_string(outputted_up_to_offset) + "\n" << std::flush;
            std::cout << "With prev offset " + std::to_string(prev_outputted_up_to_offset) + "\n" << std::flush;
            exit(1);
          }
        }
      }
      prev_outputted_up_to_offset = outputted_up_to_offset;
      outputted_up_to_offset += instruction.size;
      outputted_lz_instructions++;
    }
  }
};

  //get_char_with_echo();
int main(int argc, char* argv[]) {
  std::string file_path{ argv[1] };
  auto file_size = std::filesystem::file_size(file_path);
  if (argc > 3) {
    std::cout << "Invalid command line" << std::flush;
    return 1;
  }
  if (argc == 3) {  // decompress
    auto file_stream = std::fstream(file_path, std::ios::in | std::ios::binary);
    if (!file_stream.is_open()) {
      std::cout << "Can't read file\n";
      return 1;
    }
    auto decompress_start_time = std::chrono::high_resolution_clock::now();

    std::vector<char> buffer;
    std::string decomp_file_path{ argv[2] };
    auto decomp_file_stream = std::fstream(decomp_file_path, std::ios::in | std::ios::out | std::ios::binary | std::ios::trunc);
    auto wrapped_input_stream = WrappedIStreamInputStream(&file_stream);
    auto bit_input_stream = BitInputStream(wrapped_input_stream);

    uint64_t current_offset = 0;
    auto instruction = bit_input_stream.get(8);
    while (!bit_input_stream.eof()) {
      const auto eof = decomp_file_stream.eof();
      const auto fail = decomp_file_stream.fail();
      const auto bad = decomp_file_stream.bad();
      if (eof || fail || bad) {
        std::cout << "Something wrong bad during decompression\n" << std::flush;
        return 1;
      }

      uint64_t size = bit_input_stream.getVLI();

      const auto prev_write_pos = decomp_file_stream.tellp();
      if (instruction == LZInstructionType::INSERT) {
        buffer.resize(size);

        for (uint64_t i = 0; i < size; i++) {
          // pre-peak 64bits (or whatever amount of bits for the bytes left if at the end) so they are buffered and we don't read
          // byte-by-byte
          if (i % 8 == 0) {
            const auto to_peek = std::min<uint64_t>(64, (size - i) * 8);
            bit_input_stream.peek(static_cast<uint32_t>(to_peek));
          }
          const auto read = bit_input_stream.get(8);
          const char read_char = read & 0xFF;
          buffer[i] = read_char;
        }
        decomp_file_stream.write(buffer.data(), size);
      }
      else {  // LZInstructionType::COPY
        decomp_file_stream.flush();
        uint64_t relative_offset = bit_input_stream.getVLI();
        uint64_t offset = current_offset - relative_offset;

        // A COPY instruction might be overlapping with itself, which means we need to keep copying data already copied within the
        // same COPY instruction (usually because of repeating patterns in data)
        // In this case we have to only read again the non overlapping data
        const auto actual_read_size = std::min(size, static_cast<uint64_t>(prev_write_pos) - offset);
        buffer.resize(actual_read_size);

        decomp_file_stream.seekg(offset);
        decomp_file_stream.read(buffer.data(), actual_read_size);

        decomp_file_stream.seekp(prev_write_pos);
        auto remaining_size = size;
        while (remaining_size > 0) {
          const auto write_size = std::min(remaining_size, actual_read_size);
          decomp_file_stream.write(buffer.data(), write_size);
          remaining_size -= write_size;
        }
      }

      current_offset += size;
      instruction = bit_input_stream.get(8);
    }

    auto decompress_end_time = std::chrono::high_resolution_clock::now();
    std::printf("Decompression finished in %lld seconds!\n", std::chrono::duration_cast<std::chrono::seconds>(decompress_end_time - decompress_start_time).count());
    return 0;
  }

  uint32_t avg_size = 8192;
  uint32_t min_size = avg_size / 4;
  uint32_t max_size = avg_size * 8;

  if (constants::MINIMUM_MIN >= min_size || min_size >= constants::MINIMUM_MAX) throw std::runtime_error("Bad minimum size");
  if (constants::AVERAGE_MIN >= avg_size || avg_size >= constants::AVERAGE_MAX) throw std::runtime_error("Bad average size");
  if (constants::MAXIMUM_MIN >= max_size || max_size >= constants::MAXIMUM_MAX) throw std::runtime_error("Bad maximum size");

  min_size = 128;
  avg_size = 384;
  max_size = 1024;

  //bert_params params;
  //params.model = R"(C:\Users\Administrator\Desktop\fastcdc_test\paraphrase-MiniLM-L3-v2-GGML-q4_0.bin)";
  //params.n_threads = 20;
  //bert_ctx* bctx = bert_load_from_file(params.model);
  //const int bert_max_tokens_num = bert_n_max_tokens(bctx);
  //std::vector<bert_vocab_id> bert_feature_tokens(bert_max_tokens_num);
  int bert_tokens_num = 0;
  //const auto embeddings_dim = bert_n_embd(bctx);
  //std::vector<float> bert_embeddings(embeddings_dim);

  uint64_t batch_size = 100;

  //faiss::IndexLSH index(max_size / 4, 64, false);
  std::vector<float> search_results(5 * batch_size, 0);
  //std::vector<faiss::idx_t> search_result_labels(5 * batch_size, -1);
  //printf("is_trained = %s\n", index.is_trained ? "true" : "false");

  uint64_t total_size = 0;
  uint64_t deduped_size = 0;
  uint64_t delta_compressed_approx_size = 0;
  uint64_t delta_compressed_chunk_count = 0;

  std::unordered_map<std::bitset<64>, std::vector<uint64_t>> known_hashes{};  // Find chunk pos by hash

  uint64_t chunk_i = 0;
  uint64_t last_reduced_chunk_i = 0;
  std::vector<int32_t> pending_chunks_indexes(batch_size, 0);
  std::vector<uint8_t> pending_chunk_data(batch_size * max_size, 0);
  std::unordered_map<std::bitset<64>, uint64_t> simhashes_dict{};

  // Find chunk_i that has a given SuperFeature
  std::unordered_map<uint32_t, std::set<uint64_t>> superfeatures_dict{};

  const uint32_t simhash_chunk_size = 16;
  const uint32_t max_allowed_dist = 32;
  const uint32_t delta_mode = 2;
  const bool only_try_best_delta_match = false;
  const bool only_try_min_dist_delta_matches = false;
  const bool keep_first_delta_match = false;
  // Don't even bother saving chunk as delta chunk if savings are too little and overhead will probably negate them
  const uint64_t min_delta_saving = std::min(min_size * 2, avg_size);

  std::tuple<std::bitset<64>, std::shared_ptr<uint32_t[]>, uint32_t> (*simhash_func)(uint8_t * data, uint32_t data_len, uint32_t minichunk_size) = nullptr;
  DeltaEncodingResult(*simulate_delta_encoding_func)(const utility::CDChunk& chunk, const utility::CDChunk& similar_chunk, uint32_t minichunk_size) = nullptr;
  if (delta_mode == 0) {
    // Fastest delta mode, SimHash and delta encoding using Shingling
    simhash_func = &simhash_data_xxhash_shingling;
    simulate_delta_encoding_func = &simulate_delta_encoding_shingling;
  }
  else if (delta_mode == 1) {
    // Balanced delta mode, SimHashes with features using CDC but delta encoding using Shingling
    simhash_func = &simhash_data_xxhash_cdc;
    simulate_delta_encoding_func = &simulate_delta_encoding_shingling;
  }
  else {
    // Best? delta mode, both SimHashes and delta encoding using CDC
    simhash_func = &simhash_data_xxhash_cdc;
    simulate_delta_encoding_func = &simulate_delta_encoding_using_minichunks;
  }

  const bool output_disabled = false;

  const bool use_dupadj = true;
  const bool use_dupadj_backwards = true;
  const bool use_generalized_resemblance_detection = true;
  const bool use_feature_extraction = true;
  // Because of data locality, chunks close to the current one might be similar to it, we attempt to find the most similar out of the previous ones in this window,
  // and use it if it's good enough
  const int resemblance_attempt_window = 100;
  // if false, the first similar block found by any method will be used and other methods won't run, if true, all methods will run and the most similar block found will be used
  const bool attempt_multiple_delta_methods = true;
  const bool exhausive_delta_search = false;  // All delta matches with the best hamming distance will be tried
  const bool use_match_extension = true;
  const bool use_match_extension_backwards = true;
  const bool verify_delta_coding = false;

  const bool is_any_delta_on = use_dupadj || use_dupadj_backwards || use_feature_extraction || use_generalized_resemblance_detection || resemblance_attempt_window > 0;
  // If nothing is on that would require us to keep the data after we chunk and hash it, discard it
  const bool use_fat_chunks = !output_disabled || is_any_delta_on || use_match_extension || use_match_extension_backwards;

  static constexpr std::optional<uint64_t> max_match_distance = /*static_cast<uint64_t>(2) * 1024 * 1024 * 1024;*/std::nullopt;
  uint64_t first_non_out_of_range_chunk_i = 0;

  std::vector<utility::CDChunk> chunks{};

  auto file_stream = std::fstream(file_path, std::ios::in | std::ios::binary);
  if (!file_stream.is_open()) {
    std::cout << "Can't read file\n";
    return 1;
  }
  auto wrapped_file = IStreamWrapper(&file_stream);
  auto generator_ctx = fastcdc::ChunkGeneratorContext(&wrapped_file, min_size, avg_size, max_size, use_feature_extraction);

  std::optional<uint64_t> similarity_locality_anchor_i{};

  auto verify_file_stream = std::fstream(file_path, std::ios::in | std::ios::binary);
  std::vector<uint8_t> verify_buffer{};
  std::vector<uint8_t> verify_buffer_delta{};

  auto dump_file = std::fstream(file_path + ".ddp", std::ios::out | std::ios::binary | std::ios::trunc);
  LZInstructionManager lz_manager{ &chunks, use_match_extension_backwards, use_match_extension, &dump_file };

  static constexpr auto similar_chunk_tuple_cmp = [](const std::tuple<uint64_t, uint64_t>& a, const std::tuple<uint64_t, uint64_t>& b) {
    // 1st member is hamming dist, less hamming dist, better result, we want those tuple first, so we say they compare as lesser
    // 2nd member is chunk idx, more means more recent chunks, larger idx, to be prioritized because they are closer (locality principle)
    if (std::get<0>(a) < std::get<0>(b) || (std::get<0>(a) == std::get<0>(b) && std::get<1>(a) > std::get<1>(b))) return true;
    return false;
  };

  uint64_t chunk_generator_execution_time_ns = 0;
  uint64_t hashing_execution_time_ns = 0;
  uint64_t simhashing_execution_time_ns = 0;

  const auto calc_simhash_if_needed = [&simhash_func, &simhashing_execution_time_ns](utility::CDChunk& chunk) {
    if (chunk.minichunks != nullptr) return;
    const auto simhashing_start_time = std::chrono::high_resolution_clock::now();
    std::tie(chunk.lsh, chunk.minichunks, chunk.minichunks_len) = simhash_func(chunk.data.get(), chunk.length, simhash_chunk_size);
    const auto simhashing_end_time = std::chrono::high_resolution_clock::now();
    simhashing_execution_time_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(simhashing_end_time - simhashing_start_time).count();
  };

  auto total_runtime_start_time = std::chrono::high_resolution_clock::now();

  auto chunk_generator_start_time = std::chrono::high_resolution_clock::now();
  auto generated_chunk = fastcdc::chunk_generator(generator_ctx);
  auto chunk_generator_end_time = std::chrono::high_resolution_clock::now();
  chunk_generator_execution_time_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(chunk_generator_end_time - chunk_generator_start_time).count();

  while (generated_chunk.has_value()) {
    // TODO: BEWARE! lz_manager.estimatedSavings() might count some savings that might go away (because of instructions too small) and in that case
    // earliest_allowed_offset might change in such a way that chunks might not be out of range even though they should.
    const auto earliest_allowed_offset = max_match_distance.has_value() && total_size > *max_match_distance + lz_manager.accumulatedSavings()
      ? total_size - *max_match_distance - lz_manager.accumulatedSavings()
      : 0;
    if (!output_disabled) lz_manager.dump(verify_file_stream, false, earliest_allowed_offset);
    while (!chunks.empty()) {
      auto& previous_chunk = chunks[first_non_out_of_range_chunk_i];
      if (previous_chunk.offset + previous_chunk.length >= earliest_allowed_offset) break;

      // Delete its data
      previous_chunk.deleteData();

      // Remove the out of range chunks from the indexes so they can't be used for dedup or delta
      auto& hash_vec = known_hashes[previous_chunk.hash];
      if (hash_vec.size() == 1) {
        known_hashes.erase(previous_chunk.hash);
      }
      else {
        hash_vec.erase(std::ranges::find(hash_vec, first_non_out_of_range_chunk_i));
      }

      if (use_generalized_resemblance_detection) {
        const auto base = hamming_base(previous_chunk.lsh);
        if (simhashes_dict[base] == first_non_out_of_range_chunk_i) {
          simhashes_dict.erase(base);
        }
      }

      if (use_feature_extraction) {
        for (auto& sf : previous_chunk.super_features) {
          auto& chunk_idx_vec = superfeatures_dict[sf];
          if (chunk_idx_vec.size() == 1) {
            superfeatures_dict.erase(sf);
          }
          else {
            superfeatures_dict[sf].erase(first_non_out_of_range_chunk_i);
          }
        }
      }

      first_non_out_of_range_chunk_i++;
    }

    chunks.emplace_back(std::move(std::get<0>(*generated_chunk)));
    auto& chunk = chunks.back();

    std::optional<uint64_t> prev_similarity_locality_anchor_i = similarity_locality_anchor_i;
    similarity_locality_anchor_i = std::nullopt;
    if (chunk_i % 50000 == 0) std::cout << "\n%" + std::to_string((static_cast<float>(generator_ctx.offset) / file_size) * 100) + "\n" << std::flush;
    total_size += chunk.length;

    const auto& data_span = std::get<1>(*generated_chunk);

    // make hashes
    const auto hashing_start_time = std::chrono::high_resolution_clock::now();
    /*
    // SHA1 boost hash
    boost::uuids::detail::sha1 hasher;
    hasher.process_bytes(chunk.data.data(), chunk.data.size());
    chunk.hash = get_sha1_hash(hasher);
    */
    
    // xxHash
    chunk.hash = XXH3_64bits(data_span.data(), chunk.length);
    const auto hashing_end_time = std::chrono::high_resolution_clock::now();
    hashing_execution_time_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(hashing_end_time - hashing_start_time).count();

    const bool is_duplicate_chunk = known_hashes.contains(chunk.hash);

    if (use_fat_chunks && !is_duplicate_chunk) {
      chunk.data = std::make_shared_for_overwrite<uint8_t[]>(chunk.length);
      std::copy_n(data_span.data(), chunk.length, chunk.data.get());
    }

    std::vector<LZInstruction> new_instructions;
    if (!is_duplicate_chunk) {
      // Attempt resemblance detection and delta compression, first by using generalized resemblance detection
      std::set<std::tuple<uint64_t, uint64_t>, decltype(similar_chunk_tuple_cmp)> similar_chunks{};
      if (use_generalized_resemblance_detection) {
        calc_simhash_if_needed(chunk);
        const auto simhash_base = hamming_base(chunk.lsh);
        if (simhashes_dict.contains(simhash_base)) {
          similar_chunks.emplace(0, simhashes_dict[simhash_base]);
        }
        // If there is already a match via generalized resemblance detection, we still overwrite the index with this newer chunk.
        // Because of data locality, a more recent chunk on the data stream is more likely to yield good results
        simhashes_dict[simhash_base] = chunk_i;
      }

      // Resemblance detection on the resemblance attempt window, exploiting the principle of data locality, there is a high likelihood data chunks close
      // to the current chunk are fairly similar, just check distance for all chunks within the window
      if (resemblance_attempt_window > 0 && (similar_chunks.empty() || attempt_multiple_delta_methods)) {
        calc_simhash_if_needed(chunk);
        std::set<std::tuple<uint64_t, uint64_t>, decltype(similar_chunk_tuple_cmp)> resemblance_attempt_window_similar_chunks{};

        std::optional<uint64_t> best_candidate_dist{};
        uint64_t best_candidate_i = 0;
        for (uint64_t i = 1; i <= resemblance_attempt_window && i <= chunk_i; i++) {
          const auto candidate_i = chunk_i - i;
          auto& similar_chunk = chunks[candidate_i];
          if (similar_chunk.offset + similar_chunk.length < earliest_allowed_offset) break;
          // TODO: if the similar_chunk is just at the edge of the earliest_allowed_offset it might appear usable (and even good in
          // terms of the potential saved size) but all COPYs might end up replaced by INSERTs because part of the chunk is already unreachable.
          // Find out if it might be a better idea to skip them, penalize their distance, or anything of the sort so if there is some other
          // good suitable chunk that is not partially unreachable it is used instead

          calc_simhash_if_needed(similar_chunk);
          const auto new_dist = hamming_distance(chunk.lsh, similar_chunk.lsh);
          if (
            new_dist <= max_allowed_dist &&
            // If new_dist is the same as the best so far, privilege chunks with higher index as they are closer to the current chunk
            // and data locality suggests it should be a better match
            (exhausive_delta_search || (!best_candidate_dist.has_value() || *best_candidate_dist > new_dist || (*best_candidate_dist == new_dist && candidate_i > best_candidate_i)))
          ) {
            if (!exhausive_delta_search || new_dist < *best_candidate_dist)  {
              resemblance_attempt_window_similar_chunks.clear();
            }
            best_candidate_i = candidate_i;
            best_candidate_dist = new_dist;
            resemblance_attempt_window_similar_chunks.emplace(new_dist, candidate_i);
          }
        }
        if (!resemblance_attempt_window_similar_chunks.empty()) {
          similar_chunks.insert(resemblance_attempt_window_similar_chunks.begin(), resemblance_attempt_window_similar_chunks.end());
        }
      }

      // Resemblance detection via feature extraction and superfeature matching
      // If we couldn't sample the chunk's features we completely skip any attempt to match or register superfeatures, as we could not extract them for this chunk
      if (use_feature_extraction && !chunk.feature_sampling_failure) {
        // If we don't have a similar chunk yet, attempt to find one by SuperFeature matching
        if (similar_chunks.empty() || attempt_multiple_delta_methods) {
          calc_simhash_if_needed(chunk);
          std::set<std::tuple<uint64_t, uint64_t>, decltype(similar_chunk_tuple_cmp)> feature_extraction_similar_chunks{};

          std::optional<uint64_t> best_candidate_dist{};
          uint64_t best_candidate_i = 0;
          // We will rank potential similar chunks by their amount of matching SuperFeatures (so we select the most similar chunk possible)
          std::unordered_map<uint64_t, uint64_t> chunk_rank{};

          for (const auto& sf : chunk.super_features) {
            if (superfeatures_dict.contains(sf)) {
              for (const auto& candidate_i : superfeatures_dict[sf]) {
                auto& similar_chunk = chunks[candidate_i];
                if (similar_chunk.offset + similar_chunk.length < earliest_allowed_offset) {
                  continue;
                }
                // TODO: if the similar_chunk is just at the edge of the earliest_allowed_offset it might appear usable (and even good in
                // terms of the potential saved size) but all COPYs might end up replaced by INSERTs because part of the chunk is already unreachable.
                // Find out if it might be a better idea to skip them, penalize their distance, or anything of the sort so if there is some other
                // good suitable chunk that is not partially unreachable it is used instead
                chunk_rank[candidate_i] += 1;

                calc_simhash_if_needed(similar_chunk);
                const auto new_dist = hamming_distance(chunk.lsh, similar_chunk.lsh);
                if (
                  new_dist <= max_allowed_dist &&
                  // If new_dist is the same as the best so far, privilege chunks with higher index as they are closer to the current chunk
                  // and data locality suggests it should be a better match
                  (exhausive_delta_search || (!best_candidate_dist.has_value() || *best_candidate_dist > new_dist || (*best_candidate_dist == new_dist && candidate_i > best_candidate_i)))
                ) {
                  if (!exhausive_delta_search || new_dist < *best_candidate_dist) {
                    feature_extraction_similar_chunks.clear();
                  }
                  best_candidate_i = candidate_i;
                  best_candidate_dist = new_dist;
                  feature_extraction_similar_chunks.emplace(new_dist, candidate_i);
                }
              }
            }
          }
          if (!feature_extraction_similar_chunks.empty()) {
            similar_chunks.insert(feature_extraction_similar_chunks.begin(), feature_extraction_similar_chunks.end());
          }

          // If all SuperFeatures match, remove the previous chunk from the Index, prevents the index from getting pointlessly large, specially for some pathological cases
          for (const auto& [candidate_i, sf_count] : chunk_rank) {
            if (sf_count == 4) {
              for (const auto& sf : chunk.super_features) {
                superfeatures_dict[sf].erase(candidate_i);
              }
            }
          }

          // Register this chunk's SuperFeatures so that matches may be found with subsequent chunks
          for (uint64_t i = 0; i < 4; i++) {
            superfeatures_dict[chunk.super_features[i]].insert(chunk_i);
          }
        }
      }

      // If we don't have a similar chunk yet, attempt to find similar block via DARE's DupAdj,
      // checking if the next chunk from the similarity locality anchor is similar to this one
      if (use_dupadj && (similar_chunks.empty() || attempt_multiple_delta_methods)) {
        calc_simhash_if_needed(chunk);
        std::set<std::tuple<uint64_t, uint64_t>, decltype(similar_chunk_tuple_cmp)> dupadj_similar_chunks{};

        std::optional<uint64_t> best_candidate_dist{};
        uint64_t best_candidate_i = 0;
        if (prev_similarity_locality_anchor_i.has_value()) {
          for (const auto& anchor_instance_chunk_i : known_hashes[chunks[*prev_similarity_locality_anchor_i].hash]) {
            const auto similar_candidate_chunk_i = anchor_instance_chunk_i + 1;
            if (similar_candidate_chunk_i == chunk_i) continue;

            auto& similar_candidate = chunks[similar_candidate_chunk_i];
            calc_simhash_if_needed(similar_candidate);
            const auto new_dist = hamming_distance(chunk.lsh, similar_candidate.lsh);
            if (
              new_dist <= max_allowed_dist &&
              // If new_dist is the same as the best so far, privilege chunks with higher index as they are closer to the current chunk
              // and data locality suggests it should be a better match
              (exhausive_delta_search || (!best_candidate_dist.has_value() || *best_candidate_dist > new_dist || (*best_candidate_dist == new_dist && similar_candidate_chunk_i > best_candidate_i)))
            ) {
              if (!exhausive_delta_search || new_dist < *best_candidate_dist) {
                dupadj_similar_chunks.clear();
              }
              best_candidate_i = similar_candidate_chunk_i;
              best_candidate_dist = new_dist;
              dupadj_similar_chunks.emplace(new_dist, similar_candidate_chunk_i);
            }
          }
        }
        if (!dupadj_similar_chunks.empty()) {
          similar_chunks.insert(dupadj_similar_chunks.begin(), dupadj_similar_chunks.end());
        }
      }

      // If any of the methods detected chunks that appear to be similar, attempt delta encoding
      DeltaEncodingResult best_encoding_result;
      if (!similar_chunks.empty()) {
        uint64_t best_saved_size = 0;
        uint64_t best_similar_chunk_i;
        uint64_t best_similar_chunk_dist = 999;
        for (const auto& [similar_chunk_dist, similar_chunk_i] : similar_chunks) {
          if (only_try_min_dist_delta_matches && best_similar_chunk_dist < similar_chunk_dist) continue;

          const DeltaEncodingResult encoding_result = simulate_delta_encoding_func(chunk, chunks[similar_chunk_i], simhash_chunk_size);
          const auto& saved_size = encoding_result.estimated_savings;
          if (saved_size > best_saved_size) {
            best_saved_size = saved_size;
            best_encoding_result = encoding_result;
            best_similar_chunk_i = similar_chunk_i;
            best_similar_chunk_dist = similar_chunk_dist;
            if (keep_first_delta_match) break;
          }

          if (only_try_best_delta_match) break;
        }
        if (best_saved_size > 0 && best_saved_size > min_delta_saving) {
          delta_compressed_chunk_count++;
          similarity_locality_anchor_i = best_similar_chunk_i;
        }
      }

      if (similarity_locality_anchor_i.has_value()) {
        auto& similar_chunk = chunks[*similarity_locality_anchor_i];
        uint64_t chunk_offset_pos = 0;

        if (verify_delta_coding) {
          verify_buffer.resize(chunk.length);
          verify_buffer_delta.resize(chunk.length);

          verify_file_stream.seekg(chunk.offset);
          verify_file_stream.read(reinterpret_cast<char*>(verify_buffer.data()), chunk.length);
        }

        for (auto& instruction : best_encoding_result.instructions) {
          const auto offset = chunk.offset + chunk_offset_pos;
          // On delta instructions the offsets are relative to the reference chunk base offset
          const auto reference_offset = similar_chunk.offset + instruction.offset;
          if (instruction.type == LZInstructionType::INSERT) {
            new_instructions.emplace_back(LZInstructionType::INSERT, offset, instruction.size);
          }
          else {
            new_instructions.emplace_back(LZInstructionType::COPY, reference_offset, instruction.size);
          }
          if (verify_delta_coding) {
            verify_file_stream.seekg(instruction.type == LZInstructionType::INSERT ? offset : reference_offset);
            verify_file_stream.read(reinterpret_cast<char*>(verify_buffer_delta.data()) + chunk_offset_pos, instruction.size);

            if (std::memcmp(verify_buffer_delta.data() + chunk_offset_pos, verify_buffer.data() + chunk_offset_pos, instruction.size) != 0) {
              std::cout << "Delta coding data verification mismatch!\n" << std::flush;
              exit(1);
            }
          }
          chunk_offset_pos += instruction.size;
        }
        if (chunk_offset_pos != chunk.length) {
          std::cout << "Delta coding size mismatch: chunk_size/delta size " + std::to_string(chunk.length) + "/" + std::to_string(chunk_offset_pos) + "\n" << std::flush;
          exit(1);
        }
      }
    }
    else {
      const auto& previous_chunk_i = known_hashes[chunk.hash].back();
      similarity_locality_anchor_i = previous_chunk_i;
      const auto& previous_chunk_instance = chunks[previous_chunk_i];
      chunk.data = previous_chunk_instance.data;
      // Important to copy LSH in case this duplicate chunk ends up being candidate for Delta encoding via DupAdj or something
      chunk.lsh = previous_chunk_instance.lsh;
      // Get the minichunks from the first instance of the duplicate chunk as well
      chunk.minichunks = previous_chunk_instance.minichunks;
      chunk.minichunks_len = previous_chunk_instance.minichunks_len;

      if (use_generalized_resemblance_detection) {
        // Overwrite index for generalized resemblance detection with this newer chunk.
        // Because of data locality, a more recent chunk on the data stream is more likely to yield good results
        const auto simhash_base = hamming_base(chunk.lsh);
        simhashes_dict[simhash_base] = chunk_i;
      }
      // TODO: DO THE SAME FOR SUPERFEATURES
      // TODO: OR DON'T? ATTEMPTED IT AND IT HURT COMPRESSION A LOT

      new_instructions.emplace_back(LZInstructionType::COPY, previous_chunk_instance.offset, chunk.length);
    }
    known_hashes[chunk.hash].push_back(chunk_i);

    // if similarity_locality_anchor has a value it means we either deduplicated the chunk or delta compressed it
    if (similarity_locality_anchor_i.has_value()) {
      // set new last_reduced_chunk_i, if there are previous chunks that haven't been deduped/delta'd attempt to do so via backwards DupAdj
      uint64_t prev_last_reduced_chunk_i = last_reduced_chunk_i;
      last_reduced_chunk_i = chunk_i;

      // For us to be able to backtrack the previously last reduced chunk needs to not be the previous one but at least 2 before,
      // else we would be backtracking to that already reduced chunk
      if (chunk_i - prev_last_reduced_chunk_i >= 2 && use_dupadj_backwards) {
        uint64_t backtrack_similarity_anchor_i = *similarity_locality_anchor_i;
        auto backtrack_chunk_i = chunk_i - 1;

        std::vector<DeltaEncodingResult> dupadj_results{};

        for (; backtrack_chunk_i > prev_last_reduced_chunk_i; backtrack_chunk_i--) {
          auto& backtrack_chunk = chunks[backtrack_chunk_i];

          std::set<std::tuple<uint64_t, uint64_t>, decltype(similar_chunk_tuple_cmp)> backwards_dupadj_similar_chunks{};

          std::optional<uint64_t> best_candidate_dist{};
          uint64_t best_candidate_i = 0;
          for (const auto& anchor_instance_chunk_i : known_hashes[chunks[backtrack_similarity_anchor_i].hash]) {
            const auto similar_candidate_chunk_i = anchor_instance_chunk_i - 1;
            if (similar_candidate_chunk_i >= backtrack_chunk_i || anchor_instance_chunk_i == 0) continue;

            auto& similar_chunk = chunks[similar_candidate_chunk_i];
            if (similar_chunk.offset + similar_chunk.length < earliest_allowed_offset) continue;
            // TODO: if the similar_chunk is just at the edge of the earliest_allowed_offset it might appear usable (and even good in
            // terms of the potential saved size) but all COPYs might end up replaced by INSERTs because part of the chunk is already unreachable.
            // Find out if it might be a better idea to skip them, penalize their distance, or anything of the sort so if there is some other
            // good suitable chunk that is not partially unreachable it is used instead

            calc_simhash_if_needed(backtrack_chunk);
            calc_simhash_if_needed(similar_chunk);
            const auto new_dist = hamming_distance(backtrack_chunk.lsh, similar_chunk.lsh);
            if (
              new_dist <= max_allowed_dist &&
              // If new_dist is the same as the best so far, privilege chunks with higher index as they are closer to the current chunk
              // and data locality suggests it should be a better match
              (exhausive_delta_search || (!best_candidate_dist.has_value() || *best_candidate_dist > new_dist || (*best_candidate_dist == new_dist && similar_candidate_chunk_i > best_candidate_i)))
            ) {
              if (!exhausive_delta_search || new_dist < *best_candidate_dist) {
                backwards_dupadj_similar_chunks.clear();
              }
              best_candidate_i = similar_candidate_chunk_i;
              best_candidate_dist = new_dist;
              backwards_dupadj_similar_chunks.emplace(new_dist, similar_candidate_chunk_i);
            }
          }

          if (backwards_dupadj_similar_chunks.empty()) break;

          uint64_t best_saved_size = 0;
          std::optional<uint64_t> best_similar_chunk_i;
          uint64_t best_similar_chunk_dist = 999;
          DeltaEncodingResult best_encoding_result;
          for (const auto& [similar_chunk_dist, similar_chunk_i] : backwards_dupadj_similar_chunks) {
            if (only_try_min_dist_delta_matches && best_similar_chunk_dist < similar_chunk_dist) continue;
            auto& similar_chunk = chunks[similar_chunk_i];
            if (similar_chunk.offset + similar_chunk.length < earliest_allowed_offset) continue;

            const DeltaEncodingResult encoding_result = simulate_delta_encoding_func(backtrack_chunk, similar_chunk, simhash_chunk_size);
            const auto& saved_size = encoding_result.estimated_savings;
            if (saved_size > best_saved_size) {
              best_saved_size = saved_size;
              best_encoding_result = encoding_result;
              best_similar_chunk_i = similar_chunk_i;
              best_similar_chunk_dist = similar_chunk_dist;
              if (keep_first_delta_match) break;
            }

            if (only_try_best_delta_match) break;
          }

          if (!best_similar_chunk_i.has_value()) break;
          const auto& similar_chunk = chunks[*best_similar_chunk_i];

          if (best_saved_size > 0 && best_saved_size > min_delta_saving) {
            for (auto& instruction : best_encoding_result.instructions) {
              instruction.offset += instruction.type == LZInstructionType::INSERT ? backtrack_chunk.offset : similar_chunk.offset;
            }

            delta_compressed_chunk_count++;
            backtrack_similarity_anchor_i = best_candidate_i;

            dupadj_results.emplace_back(std::move(best_encoding_result));
          }
          else {
            break;
          }
        }

        if (!dupadj_results.empty()) {
          uint64_t revertSize = 0;
          for (auto& delta_result : dupadj_results) {
            for (auto& instruction : delta_result.instructions) {
              revertSize += instruction.size;
            }
          }

          // Iterating in reverse as the results are ordered from latest to earliest chunk, and they need to be added from earliest to latest
          lz_manager.revertInstructionSize(revertSize);
          auto lz_offset = chunk.offset - revertSize;
          for (auto& delta_result : std::ranges::reverse_view(dupadj_results)) {
            for (auto& instruction : delta_result.instructions) {
              if (instruction.type == LZInstructionType::COPY && instruction.offset < earliest_allowed_offset) {
                // We need to replace any data copy attempt from before the earliest_allowed_offset with an INSERT from current data
                // Otherwise we would be exceeding the maximum allowed match distance
                const auto shrink_size_needed = std::min<uint64_t>(earliest_allowed_offset - instruction.offset, instruction.size);
                instruction.offset += shrink_size_needed;
                instruction.size -= shrink_size_needed;
                lz_manager.addInstruction({ .type = LZInstructionType::INSERT, .offset = lz_offset, .size = shrink_size_needed }, lz_offset, false, earliest_allowed_offset);
                lz_offset += shrink_size_needed;

                // If COPY instruction was in its totality out of range then it was fully shrunk and there is nothing here to add anymore
                if (instruction.size == 0) continue;
              }

              const auto instruction_size = instruction.size;
              const auto prev_estimated_savings = lz_manager.accumulatedSavings();
              const auto instruction_type = instruction.type;
              lz_manager.addInstruction(std::move(instruction), lz_offset, false, earliest_allowed_offset);
              lz_offset += instruction_size;

              if (instruction_type == LZInstructionType::COPY) {
                // If the instruction was a COPY we need to adapt the data reduction counts
                // The LZManager might have extended or rejected the instruction, so we need to handle that
                // TODO: this is actually not accurate, some match extension is counted as regular dedup/delta reduction
                // but at least the count won't overflow anymore
                const auto instruction_savings = lz_manager.accumulatedSavings() - prev_estimated_savings;
                delta_compressed_approx_size += std::min(instruction_size, instruction_savings);
              }
            }
          }
        }
      }

      auto lz_offset = chunk.offset;
      for (auto& instruction : new_instructions) {
        if (instruction.type == LZInstructionType::COPY && instruction.offset < earliest_allowed_offset) {
          // We need to replace any data copy attempt from before the earliest_allowed_offset with an INSERT from current data
          // Otherwise we would be exceeding the maximum allowed match distance
          const auto shrink_size_needed = std::min<uint64_t>(earliest_allowed_offset - instruction.offset, instruction.size);
          instruction.offset += shrink_size_needed;
          instruction.size -= shrink_size_needed;
          lz_manager.addInstruction({ .type = LZInstructionType::INSERT, .offset = lz_offset, .size = shrink_size_needed }, lz_offset, false, earliest_allowed_offset);
          lz_offset += shrink_size_needed;

          // If COPY instruction was in its totality out of range then it was fully shrunk and there is nothing here to add anymore
          if (instruction.size == 0) continue;
        }
        
        const auto instruction_size = instruction.size;
        const auto prev_estimated_savings = lz_manager.accumulatedSavings();
        const auto instruction_type = instruction.type;
        lz_manager.addInstruction(std::move(instruction), lz_offset, false, earliest_allowed_offset);
        lz_offset += instruction_size;

        if (instruction_type == LZInstructionType::COPY) {
          // If the instruction was a COPY we need to adapt the data reduction counts
          // The LZManager might have extended or rejected the instruction, so we need to handle that
          // TODO: this is actually not accurate, some match extension is counted as regular dedup/delta reduction
          // but at least the count won't overflow anymore
          const auto instruction_savings = lz_manager.accumulatedSavings() - prev_estimated_savings;

          if (is_duplicate_chunk) {
            deduped_size += std::min(instruction_size, instruction_savings);
          }
          else {
            delta_compressed_approx_size += std::min(instruction_size, instruction_savings);
          }
        }
      }
    }
    else {
      lz_manager.addInstruction({ .type = LZInstructionType::INSERT, .offset = chunk.offset, .size = chunk.length }, chunk.offset, false, earliest_allowed_offset);
    }

    chunk_i++;

    chunk_generator_start_time = std::chrono::high_resolution_clock::now();
    generated_chunk = fastcdc::chunk_generator(generator_ctx);
    chunk_generator_end_time = std::chrono::high_resolution_clock::now();
    chunk_generator_execution_time_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(chunk_generator_end_time - chunk_generator_start_time).count();
  }

  auto total_dedup_end_time = std::chrono::high_resolution_clock::now();

  std::printf("Total dedup time:    %lld seconds\n", std::chrono::duration_cast<std::chrono::seconds>(total_dedup_end_time - total_runtime_start_time).count());

  // Dump any remaining data
  if (!output_disabled) lz_manager.dump(verify_file_stream);

  auto dump_end_time = std::chrono::high_resolution_clock::now();

  // Chunk stats
  std::cout << std::string("Chunk Sizes:    min ") + std::to_string(min_size) + " - avg " + std::to_string(avg_size) + " - max " + std::to_string(max_size) + "\n" << std::flush;
  std::cout << "Total chunk count: " + std::to_string(chunks.size()) + "\n" << std::flush;
  std::cout << "Real AVG chunk size: " + std::to_string(total_size / chunks.size()) + "\n" << std::flush;
  std::cout << "Total unique chunk count: " + std::to_string(known_hashes.size()) + "\n" << std::flush;
  std::cout << "Total delta compressed chunk count: " + std::to_string(delta_compressed_chunk_count) + "\n" << std::flush;

  const auto total_accumulated_savings = lz_manager.accumulatedSavings();
  const auto match_omitted_size = lz_manager.omittedSmallMatchSize();
  const auto match_extension_saved_size = lz_manager.accumulatedExtendedBackwardsSavings() + lz_manager.accumulatedExtendedForwardsSavings();
  // Results stats
  const auto total_size_mbs = total_size / (1024.0 * 1024);
  std::printf("Chunk data total size:    %.1f MB\n", total_size_mbs);
  const auto deduped_size_mbs = deduped_size / (1024.0 * 1024);
  std::printf("Chunk data deduped size:    %.1f MB\n", deduped_size_mbs);
  const auto deltaed_size_mbs = delta_compressed_approx_size / (1024.0 * 1024);
  std::printf("Chunk data delta compressed size:    %.1f MB\n", deltaed_size_mbs);
  const auto extension_size_mbs = match_extension_saved_size / (1024.0 * 1024);
  std::printf("Match extended size:    %.1f MB\n", extension_size_mbs);
  const auto omitted_size_mbs = match_omitted_size / (1024.0 * 1024);
  std::printf("Match omitted size (matches too small):    %.1f MB\n", omitted_size_mbs);
  const auto total_accummulated_savings_mbs = total_accumulated_savings / (1024.0 * 1024);
  std::printf("Total estimated reduced size:    %.1f MB\n", total_accummulated_savings_mbs);
  std::printf("Final size:    %.1f MB\n", total_size_mbs - total_accummulated_savings_mbs);

  std::printf("\n");

  // Throughput stats
  const auto chunking_mb_per_nanosecond = total_size_mbs / chunk_generator_execution_time_ns;
  std::printf("Chunking Throughput:    %.1f MB/s\n", chunking_mb_per_nanosecond * std::pow(10, 9));
  const auto hashing_mb_per_nanosecond = total_size_mbs / hashing_execution_time_ns;
  std::printf("Hashing Throughput:    %.1f MB/s\n", hashing_mb_per_nanosecond * std::pow(10, 9));
  const auto simhashing_mb_per_nanosecond = total_size_mbs / simhashing_execution_time_ns;
  std::printf("SimHashing Throughput:    %.1f MB/s\n", simhashing_mb_per_nanosecond* std::pow(10, 9));
  const auto total_elapsed_nanoseconds = std::chrono::duration_cast<std::chrono::nanoseconds>(total_dedup_end_time - total_runtime_start_time).count();
  const auto total_mb_per_nanosecond = total_size_mbs / total_elapsed_nanoseconds;
  std::printf("Total Throughput:    %.1f MB/s\n", total_mb_per_nanosecond * std::pow(10, 9));
  std::cout << "Total LZ instructions:    " + std::to_string(lz_manager.instructionCount()) + "\n" << std::flush;
  std::printf("Total dedup time:    %lld seconds\n", std::chrono::duration_cast<std::chrono::seconds>(total_dedup_end_time - total_runtime_start_time).count());
  std::printf("Dump time:    %lld seconds\n", std::chrono::duration_cast<std::chrono::seconds>(dump_end_time - total_dedup_end_time).count());
  std::printf("Total runtime:    %lld seconds\n", std::chrono::duration_cast<std::chrono::seconds>(dump_end_time - total_runtime_start_time).count());

  //get_char_with_echo();
  return 0;
}
