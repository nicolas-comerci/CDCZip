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
#include <filesystem>
#include <unordered_set>

#include "contrib/embeddings.cpp/bert.h"
#include <faiss/IndexFlat.h>
#include <faiss/MetricType.h>
#include <faiss/utils/distances.h>
#include <faiss/IndexLSH.h>
#include <faiss/IndexBinaryHash.h>

char get_char_with_echo() {
  return getchar();
}

std::string get_sha1_hash(boost::uuids::detail::sha1& s) {
  unsigned int hash[5];
  s.get_digest(hash);

  // Back to string
  char buf[41] = { 0 };

  for (int i = 0; i < 5; i++)
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
    1553318008, 574654857,  759734804,  310648967,  1393527547, 1195718329,
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
    unsigned long long offset;
    int length;
    std::vector<uint8_t> data;
    std::string hash;
    std::bitset<64> lsh;
    std::array<uint32_t, 4> super_features{};
    bool feature_sampling_failure = true;

    CDChunk() {}
    CDChunk(unsigned long long _offset, int _length, std::vector<uint8_t>&& _data, std::string _hash)
      : offset(_offset), length(_length), data(std::move(_data)), hash(_hash) {}
  };

  uint32_t logarithm2(uint32_t value) {
    return std::lround(std::log2(value));
  }

  uint32_t ceil_div(uint32_t x, uint32_t y) {
    return (x + y - 1) / y;
  }

  uint32_t mask(uint32_t bits) {
    return std::pow(2, bits) - 1;
  }

  uint32_t center_size(uint32_t average, uint32_t minimum, uint32_t source_size) {
    uint32_t offset = minimum + ceil_div(minimum, 2);
    if (offset > average)
      offset = average;
    uint32_t size = average - offset;
    if (size > source_size)
      return source_size;
    return size;
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
  int len;
  int pos = 0;
  std::streamsize _gcount = 0;
public:
  IStreamMem(const uint8_t* _buf, int _len) : membuf(_buf), len(_len) {}
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
    uint32_t center_size,
    uint32_t mask_s,
    uint32_t mask_l
  ) {
    uint32_t pattern = 0;
    uint32_t size = data.size();
    uint32_t barrier = std::min(center_size, size);
    uint32_t i = std::min(barrier, min_size);

    while (i < barrier) {
      pattern = (pattern >> 1) + constants::GEAR[data[i]];
      if (!(pattern & mask_s)) return i + 1;
      i += 1;
    }
    barrier = std::min(max_size, size);
    while (i < barrier) {
      pattern = (pattern >> 1) + constants::GEAR[data[i]];
      if (!(pattern & mask_l)) return i + 1;
      i += 1;
    }
    return i;
  }

  std::pair<uint32_t, std::optional<std::vector<uint32_t>>> cdc_offset_with_features(
    const std::span<uint8_t> data,
    uint32_t min_size,
    uint32_t avg_size,
    uint32_t max_size,
    uint32_t center_size,
    uint32_t mask_s,
    uint32_t mask_l
  ) {
    uint32_t pattern = 0;
    uint32_t size = data.size();
    uint32_t barrier = std::min(center_size, size);
    uint32_t i = std::min(barrier, min_size);

    std::optional<std::vector<uint32_t>> features{};

    while (i < barrier) {
      pattern = (pattern >> 1) + constants::GEAR[data[i]];
      if (!(pattern & mask_s)) return { i + 1, features };
      if (!(pattern & constants::CDS_SAMPLING_MASK)) {
        if (!features.has_value()) {
          features = std::vector<uint32_t>();
          features->resize(16);
        }
        for (int feature_i = 0; feature_i < 16; feature_i++) {
          const auto& [mi, ai] = constants::N_Transform_Coefs[feature_i];
          (*features)[feature_i] = std::max<uint32_t>((*features)[feature_i], (mi * pattern + ai) % (1LL << 32));
        }
      }
      i += 1;
    }
    barrier = std::min(max_size, size);
    while (i < barrier) {
      pattern = (pattern >> 1) + constants::GEAR[data[i]];
      if (!(pattern & mask_l)) return { i + 1, features };
      if (!(pattern & constants::CDS_SAMPLING_MASK)) {
        if (!features.has_value()) {
          features = std::vector<uint32_t>();
          features->resize(16);
        }
        for (int feature_i = 0; feature_i < 16; feature_i++) {
          const auto& [mi, ai] = constants::N_Transform_Coefs[feature_i];
          (*features)[feature_i] = std::max<uint32_t>((*features)[feature_i], (mi * pattern + ai) % (1LL << 32));
        }
      }
      i += 1;
    }
    return { i, features };
  }

  class ChunkGeneratorContext {
  public:
    IStreamLike* stream;
    uint32_t min_size;
    uint32_t avg_size;
    uint32_t max_size;
    bool fat;
    bool extract_features;

    uint32_t cs;
    uint32_t bits;
    uint32_t mask_s;
    uint32_t mask_l;
    uint32_t read_size;

    uintmax_t offset = 0;
    std::vector<uint8_t> blob{};
    uint32_t blob_len = 0;
    std::vector<unsigned char>::iterator blob_it;

    ChunkGeneratorContext(IStreamLike* _stream, uint32_t _min_size, uint32_t _avg_size, uint32_t _max_size, bool _fat, bool _extract_features = false)
      : stream(_stream), min_size(_min_size), avg_size(_avg_size), max_size(_max_size), fat(_fat), extract_features(_extract_features) {
      cs = utility::center_size(avg_size, min_size, max_size);
      bits = utility::logarithm2(avg_size);
      mask_s = utility::mask(bits + 1);
      mask_l = utility::mask(bits - 1);
      read_size = std::max<uint32_t>(1024 * 64, max_size);

      blob.resize(read_size);
      stream->read(blob.data(), read_size);
      uint32_t blob_len = stream->gcount();
      blob.resize(blob_len);
      blob_it = blob.begin();
    }
  };

  std::optional<utility::CDChunk> chunk_generator(ChunkGeneratorContext& context) {
    std::optional<utility::CDChunk> chunk{};
    if (context.blob_it == context.blob.end()) return chunk;

    context.blob_len = context.blob.end() - context.blob_it;
    if (context.blob_len <= context.max_size) {
      context.blob.erase(context.blob.begin(), context.blob_it);

      context.blob.resize(context.read_size);
      context.stream->read(reinterpret_cast<char*>(context.blob.data()) + context.blob_len, context.read_size - context.blob_len);
      context.blob_len += context.stream->gcount();
      context.blob.resize(context.blob_len);
      context.blob_it = context.blob.begin();  // iterators got invalidated
    }
    uint32_t cp;
    std::optional<std::vector<uint32_t>> features{};
    if (context.extract_features) {
      std::tie(cp, features) = cdc_offset_with_features(std::span(context.blob_it, context.blob.end()), context.min_size, context.avg_size, context.max_size, context.cs, context.mask_s, context.mask_l);
    }
    else {
      cp = cdc_offset(std::span(context.blob_it, context.blob.end()), context.min_size, context.avg_size, context.max_size, context.cs, context.mask_s, context.mask_l);
    }
    std::vector<uint8_t> raw{};
    if (context.fat) {
      raw = std::vector(context.blob_it, context.blob_it + cp);
    }
    chunk.emplace(context.offset, cp, std::move(raw), std::string());
    if (features.has_value()) {
      for (int i = 0; i < 4; i++) {
        // Takes 4 features (32bit(4byte) fingerprints, so 4 of them is 16bytes) and hash them into a single SuperFeature (seed used arbitrarily just because it needed one)
        chunk->super_features[i] = XXH32(features->data() + static_cast<ptrdiff_t>(i * 4), 16, constants::CDS_SAMPLING_MASK);
      }
      chunk->feature_sampling_failure = false;
    }
    context.offset += cp;
    context.blob_it += cp;
    return chunk;
  }
}

std::array<uint8_t, 4> simhash_data(uint8_t* data, int data_len, int chunk_size = 16) {
  // Initialize SimHash array
  std::array<uint8_t, 4> simhash = { 0 };

  // Iterate over the data in chunks
  for (int i = 0; i < data_len; i += chunk_size) {
    // Calculate hash for current chunk
    std::hash<std::string> hasher;
    uint32_t chunk_hash = hasher(reinterpret_cast<const char*>(data + i));

    // Update SimHash with the hash of the chunk
    for (int j = 0; j < 4; ++j) {
      // Update each byte of the SimHash with the corresponding byte of the chunk hash
      simhash[j] += (chunk_hash >> (j * 8)) & 0xFF;
    }
  }

  return simhash;
}

std::bitset<64> simhash_data_xxhash_shingling(uint8_t* data, int data_len, int chunk_size) {
  std::array<int, 64> simhash_vector{};

  XXH3_state_t* state = XXH3_createState();
  XXH3_64bits_reset(state);

  // Iterate over the data in chunks
  for (int i = 0; i < data_len; i += chunk_size) {
    // Calculate hash for current chunk
    XXH3_64bits_update(state, data + i, std::min(chunk_size, data_len - i));
    std::bitset<64> chunk_hash = XXH3_64bits_digest(state);
    XXH3_64bits_reset(state);

    // Update SimHash vector with the hash of the chunk
    for (int bit_i = 0; bit_i < 64; bit_i++) {
      simhash_vector[bit_i] += (chunk_hash.test(bit_i) ? 1 : -1);
    }
  }

  // Truncate simhash_vector into simhash, by keeping values larger than 0 as 1bit and values 0 or less to 0bit
  std::bitset<64> simhash{};
  for (int bit_i = 0; bit_i < 64; bit_i++) {
    simhash[bit_i] = simhash_vector[bit_i] > 0 ? 1 : 0;
  }
  XXH3_freeState(state);
  return simhash;
}

std::bitset<64> simhash_data_xxhash_cdc(uint8_t* data, int data_len, int chunk_size) {
  std::array<int, 64> simhash_vector{};

  XXH3_state_t* state = XXH3_createState();
  XXH3_64bits_reset(state);

  auto memstream = IStreamMem(data, data_len);
  auto ctx = fastcdc::ChunkGeneratorContext(&memstream, chunk_size / 2, chunk_size, chunk_size * 2, true);  

  // Iterate over the data in chunks
  auto cdc_minichunk = fastcdc::chunk_generator(ctx);
  while (cdc_minichunk.has_value()) {
    const auto& chunk = *cdc_minichunk;
    // Calculate hash for current chunk
    XXH3_64bits_update(state, chunk.data.data(), chunk.data.size());
    std::bitset<64> chunk_hash = XXH3_64bits_digest(state);
    XXH3_64bits_reset(state);

    // Update SimHash vector with the hash of the chunk
    for (int bit_i = 0; bit_i < 64; bit_i++) {
      simhash_vector[bit_i] += (chunk_hash.test(bit_i) ? 1 : -1);
    }

    cdc_minichunk = fastcdc::chunk_generator(ctx);
  }

  // Truncate simhash_vector into simhash, by keeping values larger than 0 as 1bit and values 0 or less to 0bit
  std::bitset<64> simhash{};
  for (int bit_i = 0; bit_i < 64; bit_i++) {
    simhash[bit_i] = simhash_vector[bit_i] > 0 ? 1 : 0;
  }
  XXH3_freeState(state);
  return simhash;
}

template<int bit_size>
int hamming_distance(std::bitset<bit_size> data1, std::bitset<bit_size> data2) {
  int dist = 0;
  /*
  for (int i = 0; i < bit_size; i++) {
    if (data1[i] != data2[i]) dist += 1;
  }
  */
  const auto val = data1 ^ data2;
  /*
  while (val != 0) {
    dist++;
    val &= val - 1;
  }
  return dist;
  */
  return val.count();
}

template<int bit_size>
int hamming_syndrome(const std::bitset<bit_size>& data) {
  int result = 0;
  std::bitset<bit_size> mask {0b1};
  for (int i = 0; i < bit_size; i++) {
    auto bit = data & mask;
    if (bit != 0) result ^= i;
    mask <<= 1;
  }

  return result;
}

template<int bit_size>
std::bitset<bit_size> hamming_base(std::bitset<bit_size> data) {
  auto syndrome = hamming_syndrome(data);
  auto base = data.flip(syndrome);
  // The first bit doesn't really participate in non-extended hamming codes (and extended ones are not useful to us)
  // So we just collapse to them all to the version with 0 on the first bit, allows us to match some hamming distance 2 data
  base[0] = 0;
  return data.flip(syndrome);
}

uint64_t simulate_delta_encoding_shingling(const utility::CDChunk& chunk, const utility::CDChunk& similar_chunk, uint32_t chunk_size) {
  //DDelta style delta encoding, first start by greedily attempting to match data at the start and end on the chunks
  uint64_t start_matching_data_len = 0;
  const auto cmp_size = std::min(chunk.data.size(), similar_chunk.data.size());
  std::span chunk_data_span{ chunk.data.data(), cmp_size };
  std::span similar_chunk_data_span{ similar_chunk.data.data(), cmp_size };
  for (uint64_t i = 0; i < cmp_size; i++) {
    if (chunk_data_span[i] != similar_chunk_data_span[i]) break;
    ++start_matching_data_len;
  }

  uint64_t end_matching_data_len = 0;
  chunk_data_span = std::span{ chunk.data.data() + chunk.data.size() - cmp_size, cmp_size };
  similar_chunk_data_span = std::span{ similar_chunk.data.data() + similar_chunk.data.size() - cmp_size, cmp_size };
  for (uint64_t i = 0; i < cmp_size; i++) {
    if (chunk_data_span[cmp_size - 1 - i] != similar_chunk_data_span[cmp_size - 1 - i]) break;
    ++end_matching_data_len;
  }

  uint64_t saved_size = start_matching_data_len + end_matching_data_len;

  // If the chunks are of different size, its possible we have matched the whole chunk inside the other, in that case we save all that size
  if (start_matching_data_len + end_matching_data_len >= chunk.data.size()) return chunk.data.size();

  // Simulate delta patching, really fucking stupid and slow, start by getting minichunks for the similar chunk
  XXH3_state_t* state = XXH3_createState();
  // Iterate over the data in chunks
  std::set<uint64_t> minichunk_hashes{};
  const int similar_chunk_len = similar_chunk.data.size();
  for (uint64_t i = 0; i < similar_chunk_len; i += chunk_size) {
    // Calculate hash for current chunk
    XXH3_64bits_reset(state);
    XXH3_64bits_update(state, similar_chunk.data.data() + i, std::min<uint64_t>(chunk_size, similar_chunk_len - i));
    uint64_t chunk_hash = XXH3_64bits_digest(state);
    minichunk_hashes.insert(chunk_hash);
  }

  // Clean the file and get minichunks for pending chunk, count filesize saved by omitting data on the similar chunk
  const uint64_t chunk_non_matched_len = chunk.data.size() - start_matching_data_len - end_matching_data_len;
  const uint8_t* chunk_non_matched_data_start = chunk.data.data() + start_matching_data_len;
  for (uint64_t i = 0; i < chunk_non_matched_len; i += chunk_size) {
    XXH3_64bits_reset(state);
    const auto to_read = std::min<uint64_t>(chunk_size, chunk_non_matched_len - i);
    XXH3_64bits_update(state, chunk_non_matched_data_start + i, to_read);
    XXH64_hash_t minichunk_hash = XXH3_64bits_digest(state);
    if (minichunk_hashes.contains(minichunk_hash)) {
      saved_size += to_read;
    }
  }
  XXH3_freeState(state);
  return saved_size;
}

uint64_t simulate_delta_encoding_cdc(const utility::CDChunk& chunk, const utility::CDChunk& similar_chunk, uint32_t chunk_size) {
  //DDelta style delta encoding, first start by greedily attempting to match data at the start and end on the chunks
  uint64_t start_matching_data_len = 0;
  const auto cmp_size = std::min(chunk.data.size(), similar_chunk.data.size());
  std::span chunk_data_span{ chunk.data.data(), cmp_size };
  std::span similar_chunk_data_span{ similar_chunk.data.data(), cmp_size };
  for (uint64_t i = 0; i < cmp_size; i++) {
    if (chunk_data_span[i] != similar_chunk_data_span[i]) break;
    ++start_matching_data_len;
  }

  uint64_t end_matching_data_len = 0;
  chunk_data_span = std::span{ chunk.data.data() + chunk.data.size() - cmp_size, cmp_size };
  similar_chunk_data_span = std::span{ similar_chunk.data.data() + similar_chunk.data.size() - cmp_size, cmp_size };
  for (uint64_t i = 0; i < cmp_size; i++) {
    if (chunk_data_span[cmp_size - 1 - i] != similar_chunk_data_span[cmp_size - 1 - i]) break;
    ++end_matching_data_len;
  }

  uint64_t saved_size = start_matching_data_len + end_matching_data_len;

  // If the chunks are of different size, its possible we have matched the whole chunk inside the other, in that case we save all that size
  if (start_matching_data_len + end_matching_data_len >= chunk.data.size()) return chunk.data.size();

  // Simulate delta patching, really fucking stupid and slow, start by getting minichunks for the similar chunk
  XXH3_state_t* state = XXH3_createState();
  auto similar_memstream = IStreamMem(similar_chunk.data.data(), similar_chunk.data.size());
  auto ctx = fastcdc::ChunkGeneratorContext(&similar_memstream, chunk_size / 2, chunk_size, chunk_size * 2, true);
  std::set<uint64_t> minichunk_hashes{};

  auto similar_minichunk = fastcdc::chunk_generator(ctx);
  while (similar_minichunk.has_value()) {
    const auto& minichunk = *similar_minichunk;
    XXH3_64bits_reset(state);
    XXH3_64bits_update(state, minichunk.data.data(), minichunk.data.size());
    XXH64_hash_t minichunk_hash = 0;
    minichunk_hash = XXH3_64bits_digest(state);
    minichunk_hashes.insert(minichunk_hash);

    similar_minichunk = fastcdc::chunk_generator(ctx);
  }

  // Clean the file and get minichunks for pending chunk, count filesize saved by omitting data on the similar chunk
  const uint64_t chunk_non_matched_len = chunk.data.size() - start_matching_data_len - end_matching_data_len;
  const uint8_t* chunk_non_matched_data_start = chunk.data.data() + start_matching_data_len;
  auto pending_memstream = IStreamMem(chunk_non_matched_data_start, chunk_non_matched_len);
  auto pending_ctx = fastcdc::ChunkGeneratorContext(&pending_memstream, chunk_size / 2, chunk_size, chunk_size * 2, true);

  auto pending_minichunk = fastcdc::chunk_generator(pending_ctx);
  while (pending_minichunk.has_value()) {
    const auto& minichunk = *pending_minichunk;
    XXH3_64bits_reset(state);
    XXH3_64bits_update(state, minichunk.data.data(), minichunk.data.size());
    XXH64_hash_t minichunk_hash = XXH3_64bits_digest(state);
    if (minichunk_hashes.contains(minichunk_hash)) {
      saved_size += minichunk.data.size();
    }

    pending_minichunk = fastcdc::chunk_generator(pending_ctx);
  }
  XXH3_freeState(state);
  return saved_size;
}

int main(int argc, char* argv[])
{
  //get_char_with_echo();
  std::string file_path{ argv[1] };
  uint32_t avg_size = 8192;
  uint32_t min_size = avg_size / 4;
  uint32_t max_size = avg_size * 8;

  if (constants::MINIMUM_MIN >= min_size || min_size >= constants::MINIMUM_MAX) throw std::runtime_error("Bad minimum size");
  if (constants::AVERAGE_MIN >= avg_size || avg_size >= constants::AVERAGE_MAX) throw std::runtime_error("Bad avarage size");
  if (constants::MAXIMUM_MIN >= max_size || max_size >= constants::MAXIMUM_MAX) throw std::runtime_error("Bad maximum size");

  min_size = 256;
  avg_size = 512;
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

  int batch_size = 100;

  faiss::IndexLSH index(max_size / 4, 64, false);
  std::vector<float> search_results(5 * batch_size, 0);
  std::vector<faiss::idx_t> search_result_labels(5 * batch_size, -1);
  //printf("is_trained = %s\n", index.is_trained ? "true" : "false");

  auto file_size = std::filesystem::file_size(file_path);
  auto file_stream = std::fstream(file_path, std::ios::in | std::ios::binary);
  if (!file_stream.is_open()) {
    std::cout << "Can't read file\n";
    return 1;
  }
  auto wrapped_file = IStreamWrapper(&file_stream);
  uint64_t total_size = 0;
  uint64_t deduped_size = 0;
  uint64_t delta_compressed_approx_size = 0;
  uint64_t delta_compressed_chunk_count = 0;
  
  std::vector<int> faiss_idx_to_chunk_id{};
  std::unordered_map<int, std::string> chunk_ssdeep_hashes{};

  std::unordered_map<std::string, std::vector<int>> known_hashes{};  // Find chunk pos by hash
  std::vector<std::string> chunk_hashes{};  // Find hash by chunk pos

  int chunk_i = 0;
  int last_reduced_chunk_i = 0;
  int pending_chunks = 0;
  std::vector<int32_t> pending_chunks_indexes(batch_size, 0);
  std::vector<uint8_t> pending_chunk_data(batch_size * max_size, 0);
  std::vector<std::bitset<64>> simhashes{};
  std::unordered_map<std::bitset<64>, int> simhashes_dict{};

  // Find chunk_i that has a given SuperFeature
  std::unordered_map<uint32_t, std::set<int>> superfeatures_dict{};

  const uint32_t simhash_chunk_size = 16;
  const uint32_t max_allowed_dist = 32;
  const bool best_delta = false;

  std::bitset<64> (*simhash_func)(uint8_t* data, int data_len, int chunk_size) = nullptr;
  uint64_t(*simulate_delta_encoding_func)(const utility::CDChunk & chunk, const utility::CDChunk & similar_chunk, uint32_t chunk_size) = nullptr;
  if (!best_delta) {
    simhash_func = &simhash_data_xxhash_shingling;
    simulate_delta_encoding_func = &simulate_delta_encoding_shingling;
  }
  else {
    simhash_func = &simhash_data_xxhash_cdc;
    simulate_delta_encoding_func = &simulate_delta_encoding_cdc;
  }

  const bool use_dupadj = true;
  const bool use_dupadj_backwards = true;
  const bool use_feature_extraction = true;
  // if false, the first similar block found by any method will be used and other methods won't run, if true, all methods will run and the most similar block found will be used
  const bool attempt_multiple_delta_methods = true;
  // if true and attempt_multiple_delta_methods is true as well, other similar blocks will be tried if one is attempted but the delta encoding failed to save size
  const bool attempt_multiple_blocks_on_failure = true;
  const bool use_match_extension = true;
  const bool use_match_extension_backwards = true;

  std::vector<utility::CDChunk> chunks{};
  uintmax_t chunk_generator_execution_time_ns = 0;
  uintmax_t hashing_execution_time_ns = 0;
  uintmax_t simhashing_execution_time_ns = 0;
  auto total_runtime_start_time = std::chrono::high_resolution_clock::now();  
  auto generator_ctx = fastcdc::ChunkGeneratorContext(&wrapped_file, min_size, avg_size, max_size, true, use_feature_extraction);

  std::optional<std::string> similarity_locality_anchor{};
  auto chunk_generator_start_time = std::chrono::high_resolution_clock::now();
  auto generated_chunk = fastcdc::chunk_generator(generator_ctx);
  auto chunk_generator_end_time = std::chrono::high_resolution_clock::now();
  chunk_generator_execution_time_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(chunk_generator_end_time - chunk_generator_start_time).count();
  while (generated_chunk.has_value()) {
    chunks.emplace_back(std::move(*generated_chunk));
    auto& chunk = chunks.back();

    std::optional<std::string> prev_similarity_locality_anchor = similarity_locality_anchor;
    if (chunk_i % 1000 == 0) std::cout << "\n%" + std::to_string((static_cast<float>(generator_ctx.offset) / file_size) * 100) + "\n" << std::flush;
    total_size += chunk.length;

    // make hashes
    const auto hashing_start_time = std::chrono::high_resolution_clock::now();
    /*
    // SHA1 boost hash
    boost::uuids::detail::sha1 hasher;
    hasher.process_bytes(chunk.data.data(), chunk.data.size());
    chunk.hash = get_sha1_hash(hasher);
    */
    
    // xxHash
    XXH3_state_t* state = XXH3_createState();
    XXH3_64bits_reset(state);
    XXH3_64bits_update(state, chunk.data.data(), chunk.data.size());
    XXH64_hash_t result = XXH3_64bits_digest(state);
    XXH3_freeState(state);
    chunk.hash = std::to_string(result);
    const auto hashing_end_time = std::chrono::high_resolution_clock::now();
    hashing_execution_time_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(hashing_end_time - hashing_start_time).count();

    if (!known_hashes.contains(chunk.hash)) {
      const auto simhashing_start_time = std::chrono::high_resolution_clock::now();
      chunk.lsh = simhash_func(chunk.data.data(), chunk.data.size(), simhash_chunk_size);
      const auto simhash_base = hamming_base(chunk.lsh);
      const auto simhashing_end_time = std::chrono::high_resolution_clock::now();
      simhashing_execution_time_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(simhashing_end_time - simhashing_start_time).count();

      std::forward_list<std::pair<utility::CDChunk*, int>> similar_chunks{};  // pairs are (similar chunk, hamming dist to chunk)
      if (!simhashes_dict.contains(simhash_base)) {
        faiss_idx_to_chunk_id.push_back(chunk_i);
        simhashes_dict[simhash_base] = chunk_i;
        simhashes.emplace_back(simhash_base);
      }
      else {
        similar_chunks.emplace_front(&chunks[simhashes_dict[simhash_base]], hamming_distance(chunks[simhashes_dict[simhash_base]].lsh, chunk.lsh));
      }

      // If we couldn't sample the chunk's features we completely skip any attempt to match or register superfeatures, as we could not extract them for this chunk
      if (use_feature_extraction && !chunk.feature_sampling_failure) {
        // If we don't have a similar chunk yet, attempt to find one by SuperFeature matching
        if (similar_chunks.empty() || attempt_multiple_delta_methods) {
          std::optional<int> best_candidate_dist{};
          int best_candidate = 0;
          for (const auto& sf : chunk.super_features) {
            if (superfeatures_dict.contains(sf)) {
              for (const auto& candidate_i : superfeatures_dict[sf]) {
                const auto new_dist = hamming_distance(chunk.lsh, chunks[candidate_i].lsh);
                if (new_dist <= max_allowed_dist && (!best_candidate_dist.has_value() || *best_candidate_dist > new_dist)) {
                  best_candidate = candidate_i;
                  best_candidate_dist = new_dist;
                }
              }
            }
          }
          if (best_candidate_dist.has_value()) {
            if (!similar_chunks.empty()) {
              bool inserted = false;
              auto iter = similar_chunks.begin();
              auto prev_iter = similar_chunks.before_begin();
              while (iter != similar_chunks.end()) {
                const auto& [existing_similar_chunk, existing_similar_chunk_dist] = *iter;
                if (existing_similar_chunk == &chunks[best_candidate]) {
                  // candidate already on list, exit
                  inserted = true;
                  break;
                }
                if (*best_candidate_dist < existing_similar_chunk_dist) {
                  similar_chunks.emplace_after(prev_iter, &chunks[best_candidate], *best_candidate_dist);
                  inserted = true;
                  break;
                }
                prev_iter = iter;
                ++iter;
              }
              // If we iterated the whole list and still haven't inserted we insert at the end
              if (!inserted) similar_chunks.emplace_after(prev_iter, &chunks[best_candidate], *best_candidate_dist);
            }
            else {
              similar_chunks.emplace_front(&chunks[best_candidate], *best_candidate_dist);
            }
          }

          // We will rank potential similar chunks by their amount of matching SuperFeatures (so we select the most similar chunk possible)
          std::unordered_map<int, int> chunk_rank{};
          for (const auto& sf : chunk.super_features) {
            if (superfeatures_dict.contains(sf)) {
              for (const auto& candidate_i : superfeatures_dict[sf]) {
                chunk_rank[candidate_i] += 1;
              }
            }
          }
          int best_candidate_i = 0;
          int matching_sfs = 0;
          for (const auto& [candidate_i, sf_count] : chunk_rank) {
            if (sf_count > matching_sfs) {
              best_candidate_i = candidate_i;
              matching_sfs = sf_count;
            }
            if (matching_sfs == 4) break;
          }
          // WARNING: We just use the SuperFeatures to index and find candidates by using the SimHashes, which gives us better results
          // so actually selecting a similar chunk is disabled here, we just keep all this so we can remove old chunks with the same SuperFeature set
          /*
          if (matching_sfs > 0) {
            similar_chunk = &chunks[best_candidate_i];
          }
          */
          // If all SuperFeatures match, remove the previous chunk from the Index, prevents the index from getting pointlessly large, specially for some pathological cases
          if (matching_sfs == 4) {
            for (const auto& sf : chunk.super_features) {
              superfeatures_dict[sf].erase(best_candidate_i);
            }
          }

          // Register this chunk's SuperFeatures so that matches may be found with subsequent chunks
          for (int i = 0; i < 4; i++) {
            superfeatures_dict[chunk.super_features[i]].insert(chunk_i);
          }
        }
      }

      // If we don't have a similar chunk yet, attempt to find similar block via DARE's DupAdj,
      // checking if the next chunk from the similarity locality anchor is similar to this one
      if (similar_chunks.empty() || attempt_multiple_delta_methods) {
        std::optional<int> best_candidate_dist{};
        int best_candidate = 0;
        if (use_dupadj && similarity_locality_anchor.has_value()) {
          for (const auto& anchor_instance_chunk_i : known_hashes[*similarity_locality_anchor]) {
            const auto similar_candidate_chunk = anchor_instance_chunk_i + 1;
            if (similar_candidate_chunk == chunk_i) continue;

            const auto& similar_candidate = chunks[similar_candidate_chunk];
            const auto new_dist = hamming_distance(chunk.lsh, similar_candidate.lsh);
            if (new_dist <= max_allowed_dist && (!best_candidate_dist.has_value() || *best_candidate_dist > new_dist)) {
              best_candidate_dist = new_dist;
              best_candidate = similar_candidate_chunk;
            }
          }
        }
        if (best_candidate_dist.has_value()) {
          if (!similar_chunks.empty()) {
            bool inserted = false;
            auto iter = similar_chunks.begin();
            auto prev_iter = similar_chunks.before_begin();
            while (iter != similar_chunks.end()) {
              const auto& [existing_similar_chunk, existing_similar_chunk_dist] = *iter;
              if (existing_similar_chunk == &chunks[best_candidate]) {
                // candidate already on list, exit
                inserted = true;
                break;
              }
              if (*best_candidate_dist < existing_similar_chunk_dist) {
                similar_chunks.emplace_after(prev_iter, &chunks[best_candidate], *best_candidate_dist);
                inserted = true;
                break;
              }
              prev_iter = iter;
              ++iter;
            }
            // If we iterated the whole list and still haven't inserted we insert at the end
            if (!inserted) similar_chunks.emplace_after(prev_iter, &chunks[best_candidate], *best_candidate_dist);
          }
          else {
            similar_chunks.emplace_front(&chunks[best_candidate], *best_candidate_dist);
          }
        }
      }

      if (!similar_chunks.empty()) {
        for (const auto& [similar_chunk, similar_chunk_dist] : similar_chunks) {
          auto saved_size = simulate_delta_encoding_func(chunk, *similar_chunk, simhash_chunk_size);

          if (saved_size > 0) {
            delta_compressed_approx_size += saved_size;
            delta_compressed_chunk_count++;
            similarity_locality_anchor = similar_chunk->hash;
            break;
          }
          if (!attempt_multiple_blocks_on_failure) break;
        }
      }
      else {
        similarity_locality_anchor = std::nullopt;
      }
      /*
      int32_t brute_dist = 99;
      int best_simhash_index = 0;
      for (int i = 0; i < simhashes.size() - 1; ++i) {
        const auto& existing_simhash_base = simhashes[i];
        const auto new_brute_dist = hamming_distance(simhash_base, existing_simhash_base);
        if (new_brute_dist < brute_dist) {
          brute_dist = new_brute_dist;
          best_simhash_index = i;
          if (brute_dist == 0) break;
        }

        //for (int byte = 0; byte < 8; ++byte) {
        //  brute_dist += hamming_distance(new_simhash[byte], existing_simhash[byte]);
        //}

      }
      if (brute_dist <= 6) {
        std::cout << "Yuppi! posible similar chunk, distance: " + std::to_string(brute_dist) + "\n" << std::flush;
        const auto& similar_chunk = chunks[faiss_idx_to_chunk_id[best_simhash_index]];

        auto saved_size = simulate_delta_encoding_cdc(chunk, similar_chunk);

        delta_compressed_approx_size += saved_size;
        delta_compressed_chunk_count++;
      }
      */
    }
    /*
    if (!known_hashes.contains(chunk.hash)) {
      std::copy_n(chunk.data.data(), chunk.data.size(), pending_chunk_data.data() + (max_size * pending_chunks));
      // Faiss IndexLSH requires all chunks to be of the same size so zero pad if needed
      if (chunk.data.size() < max_size) {
        std::memset(pending_chunk_data.data() + (max_size * pending_chunks) + chunk.data.size(), 0, max_size - chunk.data.size());
      }
      faiss_idx_to_chunk_id.push_back(chunk_i);
      pending_chunks_indexes[pending_chunks] = chunk_i;
      pending_chunks++;
      known_hashes.insert(chunk.hash);
    */
    else {
      deduped_size += chunk.length;
      similarity_locality_anchor = chunk.hash;
    }
    known_hashes[chunk.hash].push_back(chunk_i);
    chunk_hashes.push_back(chunk.hash);

    // if similarity_locality_anchor has a value it means we either deduplicated the chunk or delta compressed it
    if (similarity_locality_anchor.has_value()) {
      // set new last_reduced_chunk_i, if there are previous chunks that haven't been deduped/delta'd attempt to do so via backwards DupAdj
      int prev_last_reduced_chunk_i = last_reduced_chunk_i;
      last_reduced_chunk_i = chunk_i;

      std::string backtrack_similarity_anchor = *similarity_locality_anchor;
      int maximum_backtrack = chunk_i - prev_last_reduced_chunk_i;
      // for loop starts at 1 because if the last reduced chunk is the previous one then maximum_backtrack=1, and we don't backtrack at all
      int current_backtrack = 1;
      if (use_dupadj_backwards) {
        for (; current_backtrack < maximum_backtrack; current_backtrack++) {
          const auto backtrack_chunk_i = chunk_i - current_backtrack;
          const auto& backtrack_chunk = chunks[backtrack_chunk_i];

          std::optional<int> best_candidate_dist{};
          int best_candidate = 0;
          for (const auto& anchor_instance_chunk_i : known_hashes[backtrack_similarity_anchor]) {
            const auto similar_candidate_chunk = anchor_instance_chunk_i - 1;
            if (similar_candidate_chunk >= backtrack_chunk_i || similar_candidate_chunk < 0) continue;

            const auto& similar_chunk = chunks[similar_candidate_chunk];
            const auto new_dist = hamming_distance(backtrack_chunk.lsh, similar_chunk.lsh);
            if (new_dist <= max_allowed_dist && (!best_candidate_dist.has_value() || *best_candidate_dist > new_dist)) {
              best_candidate_dist = new_dist;
              best_candidate = similar_candidate_chunk;
            }
          }

          if (!best_candidate_dist.has_value()) break;
          const auto& similar_chunk = chunks[best_candidate];
          auto saved_size = simulate_delta_encoding_func(backtrack_chunk, similar_chunk, simhash_chunk_size);

          if (saved_size > 0) {
            delta_compressed_approx_size += saved_size;
            delta_compressed_chunk_count++;
            backtrack_similarity_anchor = similar_chunk.hash;
          }
        }
      }

      // If Backwards DupAdj is off or a mismatch caused it to stop while backtracking is still possible, we attempt to extend the last match backwards
      if (use_match_extension_backwards && current_backtrack < maximum_backtrack) {
        const auto backtrack_chunk_i = chunk_i - current_backtrack;
        const auto& backtrack_chunk = chunks[backtrack_chunk_i];

        uint64_t saved_size = 0;
        for (const auto& anchor_instance_chunk_i : known_hashes[backtrack_similarity_anchor]) {
          const auto anchor_prev_chunk_i = anchor_instance_chunk_i - 1;
          if (anchor_prev_chunk_i >= backtrack_chunk_i || anchor_prev_chunk_i < 0) continue;
          const auto& anchor_prev_chunk = chunks[anchor_prev_chunk_i];

          const auto cmp_size = std::min(anchor_prev_chunk.data.size(), backtrack_chunk.data.size());
          std::span anchor_prev_chunk_data{ anchor_prev_chunk.data.data(), cmp_size };
          std::span backtrack_chunk_data{ backtrack_chunk.data.data(), cmp_size };
          uint64_t current_attempt_saved_size = 0;
          for (uint64_t i = 0; i < cmp_size; i++) {
            if (anchor_prev_chunk_data[cmp_size - 1 - i] != backtrack_chunk_data[cmp_size - 1 - i]) break;
            ++current_attempt_saved_size;
          }
          saved_size = std::max(saved_size, current_attempt_saved_size);
        }
        deduped_size += saved_size;
      }
    }
    // If the last chunk was deduped or delta'd and this one hasn't, we attempt to "extend the match" as much as possible
    // This should actually run just before outputting the data/LZ match, there is technically a risk of double counting savings here with backwards DupAdj
    else if (use_match_extension && last_reduced_chunk_i == (chunk_i - 1)){
      uint64_t saved_size = 0;
      for (const auto& anchor_instance_chunk_i : known_hashes[*prev_similarity_locality_anchor]) {
        const auto anchor_next_chunk_i = anchor_instance_chunk_i + 1;
        if (anchor_next_chunk_i == chunk_i) continue;
        const auto& anchor_next_chunk = chunks[anchor_next_chunk_i];

        const auto cmp_size = std::min(anchor_next_chunk.data.size(), chunk.data.size());
        std::span anchor_next_chunk_data{ anchor_next_chunk.data.data(), cmp_size };
        std::span chunk_data{ chunk.data.data(), cmp_size };
        uint64_t current_attempt_saved_size = 0;
        for (uint64_t i = 0; i < cmp_size; i++) {
          if (anchor_next_chunk_data[i] != chunk_data[i]) break;
          ++current_attempt_saved_size;
        }
        saved_size = std::max(saved_size, current_attempt_saved_size);
      }
      deduped_size += saved_size;
    }

    /*
    if (pending_chunks < batch_size && chunk_i != chunks.size() - 1) {
      chunk_i++;
      continue;
    }

    // We add all the pending chunk hashes, and we will search for the 2nd most similar search result, as the top search result will be the very same hash
    index.add(pending_chunks, reinterpret_cast<float*>(pending_chunk_data.data()));
    index.search(pending_chunks, reinterpret_cast<float*>(pending_chunk_data.data()), 5, search_results.data(), search_result_labels.data());

    for (int i = 0; i < pending_chunks; ++i) {
      const int result_index = i * 5;
      faiss::idx_t& estimated_most_similar_block_idx = search_result_labels[result_index];
      auto& hamming_distance = search_results[result_index];
      // As expected (though not always) the most similar chunk according to FAISS is itself, picking the second most similar one
      if (faiss_idx_to_chunk_id[estimated_most_similar_block_idx] == pending_chunks_indexes[i]) {
        estimated_most_similar_block_idx = search_result_labels[result_index + 1];
        hamming_distance = search_results[result_index + 1];
      }
      float similarity = 1 - (hamming_distance / 64);
      if (hamming_distance <= 1) {
        //std::cout << "delta compression candidate, chunk " + std::to_string(pending_chunks_indexes[i]) + ": IndexLSH distance=" + std::to_string(hamming_distance) + "\n" << std::flush;
        const auto& pending_chunk = chunks[pending_chunks_indexes[i]];
        const auto& similar_chunk = chunks[faiss_idx_to_chunk_id[estimated_most_similar_block_idx]];

        // Simulate delta patching, really fucking stupid and slow, start by getting minichunks for the similar chunk
        auto chunk_tmp_file = std::fstream("tmpfile", std::ios_base::in | std::ios_base::out | std::ios_base::binary | std::ios_base::trunc);
        chunk_tmp_file.write(reinterpret_cast<const char*>(similar_chunk.data.data()), similar_chunk.data.size());
        chunk_tmp_file.flush();
        chunk_tmp_file.seekg(0);
        auto similar_minichunks = fastcdc::chunk_generator(chunk_tmp_file, 16, 32, 64, true);
        std::set<uint64_t> minichunk_hashes{};
        for (const auto& minichunk : similar_minichunks) {
          XXH3_state_t* state2 = XXH3_createState();
          XXH3_64bits_reset(state2);
          XXH3_64bits_update(state2, minichunk.data.data(), minichunk.data.size());
          XXH64_hash_t minichunk_hash = 0;
          minichunk_hash = XXH3_64bits_digest(state2);
          XXH3_freeState(state2);
          minichunk_hashes.insert(minichunk_hash);
        }

        // Clean the file and get minichunks for pending chunk, count filesize saved by omitting data on the similar chunk
        uint64_t saved_size = 0;
        chunk_tmp_file.close();
        std::filesystem::remove("tmpfile");
        chunk_tmp_file = std::fstream("tmpfile", std::ios_base::in | std::ios_base::out | std::ios_base::binary | std::ios_base::trunc);
        chunk_tmp_file.write(reinterpret_cast<const char*>(pending_chunk.data.data()), pending_chunk.data.size());
        chunk_tmp_file.flush();
        chunk_tmp_file.seekg(0);
        auto pending_minichunks = fastcdc::chunk_generator(chunk_tmp_file, 16, 32, 64, true);
        for (const auto& minichunk : pending_minichunks) {
          XXH3_state_t* state2 = XXH3_createState();
          XXH3_64bits_reset(state2);
          XXH3_64bits_update(state2, minichunk.data.data(), minichunk.data.size());
          XXH64_hash_t minichunk_hash = XXH3_64bits_digest(state2);
          XXH3_freeState(state2);
          if (minichunk_hashes.contains(minichunk_hash)) {
            saved_size += minichunk.data.size();
          }
        }
        chunk_tmp_file.close();
        std::filesystem::remove("tmpfile");

        delta_compressed_approx_size += saved_size;
        delta_compressed_chunk_count++;
      }
    }
    */

    pending_chunks = 0;
    chunk_i++;

    chunk_generator_start_time = std::chrono::high_resolution_clock::now();
    generated_chunk = fastcdc::chunk_generator(generator_ctx);
    chunk_generator_end_time = std::chrono::high_resolution_clock::now();
    chunk_generator_execution_time_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(chunk_generator_end_time - chunk_generator_start_time).count();
  }
  auto total_runtime_end_time = std::chrono::high_resolution_clock::now();

  // Chunk stats
  std::cout << std::string("Chunk Sizes:    min ") + std::to_string(min_size) + " - avg " + std::to_string(avg_size) + " - max " + std::to_string(max_size) + "\n" << std::flush;
  std::cout << "Total chunk count: " + std::to_string(chunks.size()) + "\n" << std::flush;
  std::cout << "Total unique chunk count: " + std::to_string(known_hashes.size()) + "\n" << std::flush;
  std::cout << "Total delta compressed chunk count: " + std::to_string(delta_compressed_chunk_count) + "\n" << std::flush;

  // Results stats
  const auto total_size_mbs = total_size / (1024.0 * 1024);
  std::printf("Chunk data total size:    %.1f MB\n", total_size_mbs);
  const auto deduped_size_mbs = deduped_size / (1024.0 * 1024);
  std::printf("Chunk data deduped size:    %.1f MB\n", deduped_size_mbs);
  const auto deltaed_size_mbs = delta_compressed_approx_size / (1024.0 * 1024);
  std::printf("Chunk data delta compressed size:    %.1f MB\n", deltaed_size_mbs);
  std::printf("Final size:    %.1f MB\n", total_size_mbs - deduped_size_mbs - deltaed_size_mbs);

  std::printf("\n");

  // Throughput stats
  const auto chunking_mb_per_nanosecond = total_size_mbs / chunk_generator_execution_time_ns;
  std::printf("Chunking Throughput:    %.1f MB/s\n", chunking_mb_per_nanosecond * std::pow(10, 9));
  const auto hashing_mb_per_nanosecond = total_size_mbs / hashing_execution_time_ns;
  std::printf("Hashing Throughput:    %.1f MB/s\n", hashing_mb_per_nanosecond * std::pow(10, 9));
  const auto simhashing_mb_per_nanosecond = total_size_mbs / simhashing_execution_time_ns;
  std::printf("SimHashing Throughput:    %.1f MB/s\n", simhashing_mb_per_nanosecond* std::pow(10, 9));
  const auto total_elapsed_nanoseconds = std::chrono::duration_cast<std::chrono::nanoseconds>(total_runtime_end_time - total_runtime_start_time).count();
  const auto total_mb_per_nanosecond = total_size_mbs / total_elapsed_nanoseconds;
  std::printf("Total Throughput:    %.1f MB/s\n", total_mb_per_nanosecond * std::pow(10, 9));
  std::printf("Total runtime:    %lld seconds\n", std::chrono::duration_cast<std::chrono::seconds>(total_runtime_end_time - total_runtime_start_time).count());

  return 0;
}
