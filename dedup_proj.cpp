#include <algorithm>
#include <cmath>
#include <istream>
#include <iostream>
#include <fstream>
#include <utility>
#include <string>
#include <random>
#include <cstdio>
#include <chrono>
#include <span>
#include <array>
#include <list>
#include <unordered_map>
#include <bitset>
#include <cassert>
#include <cstdarg>

#ifndef __unix
//#include <windows.h>
#include <io.h>
#include <fcntl.h>
#include <conio.h>
#endif

// Hash libraries
#include "contrib/boost/uuid/detail/sha1.hpp"
#include "contrib/xxHash/xxhash.h"
#include "contrib/ssdeep/fuzzy.h"

// Resemblance detection support
#include <deque>
#include <filesystem>
#include <functional>
#include <ranges>
#include <stack>
#include <unordered_set>
#include <thread>

#include <immintrin.h>

#include "contrib/bitstream.h"
#include "contrib/stream.h"
#include "contrib/task_pool.h"
#include "contrib/embeddings.cpp/bert.h"
//#include <faiss/IndexFlat.h>
//#include <faiss/MetricType.h>
//#include <faiss/utils/distances.h>
//#include <faiss/IndexLSH.h>
//#include <faiss/IndexBinaryHash.h>

typedef enum
{
  STDIN_HANDLE = 0,
  STDOUT_HANDLE = 1,
  STDERR_HANDLE = 2,
} StdHandles;
#ifndef __unix
void set_std_handle_binary_mode(StdHandles handle) { std::ignore = _setmode(handle, O_BINARY); }
#else
void set_std_handle_binary_mode(StdHandles handle) {}
#endif

#ifndef _WIN32
int ttyfd = -1;
#endif

void print_to_console(const std::string& format) {
#ifdef _WIN32
  for (char chr : format) {
    putch(chr);
  }
#else
  if (ttyfd < 0)
    ttyfd = open("/dev/tty", O_RDWR);
  write(ttyfd, format.c_str(), format.length());
#endif
}

void print_to_console(const char* fmt, ...) {
  va_list args;
  va_start(args, fmt);
  va_list args_copy;
  va_copy(args_copy, args);
  int length = std::vsnprintf(nullptr, 0, fmt, args);
  va_end(args);
  assert(length >= 0);

  char* buf = new char[length + 1];
  std::vsnprintf(buf, length + 1, fmt, args_copy);
  va_end(args_copy);

  std::string str(buf);
  delete[] buf;
  print_to_console(str);
}

char get_char_with_echo() {
#ifndef __unix
  return getche();
#else
  return fgetc(stdin);
#endif
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

  static constexpr alignas(16) uint32_t GEAR[256] = {
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
  class ChunkData {
  public:
    std::bitset<64> hash;
    std::bitset<64> lsh;

    std::vector<uint8_t> data = std::vector<uint8_t>(0);
    // Smaller chunks inside this chunk, used for delta compression
    std::vector<uint32_t> minichunks = std::vector<uint32_t>(0);

    std::array<uint32_t, 4> super_features{};
    bool feature_sampling_failure = true;

    explicit ChunkData() = default;
  };
  class ChunkEntry {
  public:
    uint64_t offset = 0;
    std::shared_ptr<ChunkData> chunk_data;
    explicit ChunkEntry() = default;
    explicit ChunkEntry(uint64_t _offset) : offset(_offset), chunk_data(std::make_shared<ChunkData>()) {}
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

enum CutPointCandidateType : uint8_t {
  HARD_CUT_MASK,  // Satisfied harder mask before average size
  EASY_CUT_MASK,  // Satisfied easier mask after average size
  SUPERCDC_BACKUP_MASK,  // Satisfied SuperCDC backup mask because no other mask worked
  EOF_CUT  // Forcibly cut because the data span reached its EOF
};

struct CutPointCandidate {
  CutPointCandidateType type;
  uint32_t offset;
};

template<
  bool compute_features,
  typename ReturnType = std::conditional_t<
    compute_features,
    std::tuple<CutPointCandidate, uint32_t, std::vector<uint32_t>>,
    std::tuple<CutPointCandidate, uint32_t>
  >
>
ReturnType cdc_next_offset(
  const std::span<uint8_t> data,
  uint32_t min_size,
  uint32_t avg_size,
  uint32_t max_size,
  uint32_t mask_s,
  uint32_t mask_l,
  uint32_t initial_pattern = 0
) {
  uint32_t pattern = initial_pattern;
  uint32_t size = data.size();
  uint32_t barrier = std::min(avg_size, size);
  uint32_t i = std::min(barrier, min_size);

  std::conditional_t<compute_features, std::vector<uint32_t>, std::monostate> features;

  // SuperCDC's even easier "backup mask" and backup result, if mask_l fails to find a cutoff point before the max_size we use the backup result
  // gotten with the easier to meet mask_b cutoff point. This should make it much more unlikely that we have to forcefully end chunks at max_size,
  // which helps better preserve the content defined nature of chunks and thus increase dedup ratios.
  std::optional<uint32_t> backup_i{};
  uint32_t mask_b = mask_l >> 1;

  // SuperCDC's Min-Max adjustment of the Gear Hash on jump to minimum chunk size, should improve deduplication ratios by better preserving
  // the content defined nature of the Hashes.
  // We backtrack a little to ensure when we get to the i we actually wanted we have the exact same hash as if we hadn't skipped anything.
  uint32_t remaining_minmax_adjustment = std::min<uint32_t>(i, 31);
  i -= remaining_minmax_adjustment;

  while (i < barrier) {
    pattern = (pattern >> 1) + constants::GEAR[data[i]];
    if (remaining_minmax_adjustment > 0) {
      remaining_minmax_adjustment--;
      i++;
      continue;
    }
    if (!(pattern & mask_s)) {
      if constexpr (compute_features) {
        return { {.type = CutPointCandidateType::HARD_CUT_MASK, .offset = i }, pattern, features };
      }
      else {
        return { {.type = CutPointCandidateType::HARD_CUT_MASK, .offset = i }, pattern };
      }
    }
    if constexpr (compute_features) {
      if (!(pattern & constants::CDS_SAMPLING_MASK)) {
        if (features.empty()) {
          features.resize(16);
          features.shrink_to_fit();
        }
        for (int feature_i = 0; feature_i < 16; feature_i++) {
          const auto& [mi, ai] = constants::N_Transform_Coefs[feature_i];
          features[feature_i] = std::max<uint32_t>(features[feature_i], (mi * pattern + ai) % (1LL << 32));
        }
      }
    }
    i++;
  }
  barrier = std::min(max_size, size);
  while (i < barrier) {
    pattern = (pattern >> 1) + constants::GEAR[data[i]];
    if (!(pattern & mask_l)) {
      if constexpr (compute_features) {
        return { {.type = CutPointCandidateType::EASY_CUT_MASK, .offset = i }, pattern, features };
      }
      else {
        return { {.type = CutPointCandidateType::EASY_CUT_MASK, .offset = i }, pattern };
      }
    }
    if (!backup_i.has_value() && !(pattern & mask_b)) backup_i = i;
    if constexpr (compute_features) {
      if (!(pattern & constants::CDS_SAMPLING_MASK)) {
        if (features.empty()) {
          features.resize(16);
          features.shrink_to_fit();
        }
        for (int feature_i = 0; feature_i < 16; feature_i++) {
          const auto& [mi, ai] = constants::N_Transform_Coefs[feature_i];
          features[feature_i] = std::max<uint32_t>(features[feature_i], (mi * pattern + ai) % (1LL << 32));
        }
      }
    }
    i++;
  }
  CutPointCandidate result_i;
  if (backup_i.has_value()) {
    result_i = { .type = CutPointCandidateType::SUPERCDC_BACKUP_MASK, .offset = *backup_i };
  }
  else {
    result_i = { .type = CutPointCandidateType::EOF_CUT, .offset = i };
  }
  if constexpr (compute_features) {
    return { result_i, pattern, features };
  }
  else {
    return { result_i, pattern };
  }
}

// Precondition: Chunk invariance condition satisfied
template<
  bool compute_features,
  typename CandidateFeaturesResult = std::conditional_t<compute_features, std::vector<std::vector<uint32_t>>, std::monostate>
>
void cdc_find_cut_points_with_invariance(
  std::vector<CutPointCandidate>& candidates,
  CandidateFeaturesResult& candidate_features,
  std::span<uint8_t> data,
  uint64_t base_offset,
  uint32_t min_size,
  uint32_t avg_size,
  uint32_t max_size,
  uint32_t mask_s,
  uint32_t mask_l,
  uint32_t initial_pattern = 0
) {
  std::array<std::vector<uint32_t>, 8> lane_results{};
  std::array<CandidateFeaturesResult, 8> lane_features_results{};
  std::array<std::vector<uint32_t>, 8> lane_current_features{};

  std::array<bool, 8> lane_achieved_chunk_invariance { true, false, false, false, false, false, false, false};
  __m256i mask_s_vec = _mm256_set1_epi32(static_cast<int>(mask_s));
  __m256i mask_l_vec = _mm256_set1_epi32(static_cast<int>(mask_l));
  __m256i cmask = _mm256_set1_epi32(0xff);
  __m256i hash = _mm256_set1_epi32(0);
  const __m256i zero_vec = _mm256_set1_epi32(0);
  const __m256i ones_vec = _mm256_set1_epi32(1);
  const __m256i window_size_minus_one_vec = _mm256_set1_epi32(31);

  uint32_t bytes_per_lane = data.size() / 8;
  __m256i vindex = _mm256_setr_epi32(0, bytes_per_lane, 2 * bytes_per_lane, 3 * bytes_per_lane, 4 * bytes_per_lane, 5 * bytes_per_lane, 6 * bytes_per_lane, 7 * bytes_per_lane);

  // This vector has the max allowed size for each lane's current chunk, we start at the maximum int to essentially disable max size chunks, they are set as soon
  // as chunk invariant condition is satisfied, and from then on after starting a new chunk
  __m256i max_size_vec = _mm256_set1_epi32(std::numeric_limits<int>::max());
  __m256i avg_size_vec = _mm256_set1_epi32(std::numeric_limits<int>::max());

  vindex.m256i_i32[0] = min_size;

  // For each lane, the last index they are allowed to access
  __m256i vindex_max = _mm256_add_epi32(vindex, _mm256_set1_epi32(bytes_per_lane));
  vindex_max.m256i_i32[7] = data.size();
  // Because we read 4bytes at a time we need to ensure we are not reading past the data end
  __m256i vindex_max_avx2 = vindex_max;
  vindex_max_avx2.m256i_i32[7] -= 4;

  // SuperCDC's even easier "backup mask" and backup result, if mask_l fails to find a cutoff point before the max_size we use the backup result
  // gotten with the easier to meet mask_b cutoff point. This should make it much more unlikely that we have to forcefully end chunks at max_size,
  // which helps better preserve the content defined nature of chunks and thus increase dedup ratios.
  __m256i mask_b_vec = _mm256_set1_epi32(static_cast<int>(mask_l >> 1));
  __m256i backup_cut_vec = max_size_vec;

  // SuperCDC's Min-Max adjustment of the Gear Hash on jump to minimum chunk size, should improve deduplication ratios by better preserving
  // the content defined nature of the Hashes.
  // We backtrack a little to ensure when we get to the i we actually wanted we have the exact same hash as if we hadn't skipped anything.
  __m256i minmax_adjustment_vec = window_size_minus_one_vec;
  vindex = _mm256_sub_epi32(vindex, window_size_minus_one_vec);
  // HACK FOR REALLY LOW min_size
  if (vindex.m256i_i32[0] < 0) {
    vindex.m256i_i32[0] = 0;
    minmax_adjustment_vec.m256i_i32[0] = 0;
  }

  __m256i cds_mask_vec = _mm256_set1_epi32(constants::CDS_SAMPLING_MASK);

  uint32_t pattern = 0;

  int lane_not_marked_for_jump = 0b11111111;
  __m256i jump_vec = zero_vec;

  auto process_lane = [&lane_results, &lane_achieved_chunk_invariance, &min_size, &avg_size, &max_size, &bytes_per_lane, &vindex, &vindex_max,
  &lane_not_marked_for_jump, &jump_vec, &avg_size_vec, &max_size_vec, &backup_cut_vec, &lane_features_results, &lane_current_features]
  (uint32_t lane_i, uint32_t pos) {
    // Lane already finished, and we are on a pos for another lane (or after data end)!
    if (pos >= static_cast<uint32_t>(vindex_max.m256i_i32[lane_i])) return;
      
    if (lane_achieved_chunk_invariance[lane_i]) {
      if (lane_results[lane_i].empty()) {
        if (pos >= lane_i * bytes_per_lane + min_size) {
          if (pos == max_size_vec.m256i_i32[lane_i] && pos > backup_cut_vec.m256i_i32[lane_i]) {
            pos = backup_cut_vec.m256i_i32[lane_i];
            vindex.m256i_i32[lane_i] = 0;  // We jump to a prior position so we set a lower idx such that _mm256_max_epi32 is able to use it for jump
          }
          lane_results[lane_i].emplace_back(pos);
          if constexpr (compute_features) {
            lane_features_results[lane_i].emplace_back(std::move(lane_current_features[lane_i]));
            lane_current_features[lane_i] = std::vector<uint32_t>();
          }
          max_size_vec.m256i_i32[lane_i] = pos + max_size;
          avg_size_vec.m256i_i32[lane_i] = pos + avg_size;
          lane_not_marked_for_jump ^= (0b1 << lane_i);
          jump_vec.m256i_i32[lane_i] = pos + min_size - 31;
        }
      }
      else {
        const auto dist_with_prev = pos - lane_results[lane_i].back();
        if (dist_with_prev >= min_size) {
          if (pos == max_size_vec.m256i_i32[lane_i] && pos > backup_cut_vec.m256i_i32[lane_i]) {
            pos = backup_cut_vec.m256i_i32[lane_i];
            vindex.m256i_i32[lane_i] = 0;  // We jump to a prior position so we set a lower idx such that _mm256_max_epi32 is able to use it for jump
          }
          lane_results[lane_i].emplace_back(pos);
          if constexpr (compute_features) {
            lane_features_results[lane_i].emplace_back(std::move(lane_current_features[lane_i]));
            lane_current_features[lane_i] = std::vector<uint32_t>();
          }
          max_size_vec.m256i_i32[lane_i] = pos + max_size;
          avg_size_vec.m256i_i32[lane_i] = pos + avg_size;
          lane_not_marked_for_jump ^= (0b1 << lane_i);
          jump_vec.m256i_i32[lane_i] = pos + min_size - 31;
        }
      }
    }
    else {
      if (!lane_results[lane_i].empty()) {
        // if this happens this lane is back in sync with non-segmented processing!
        const auto dist_with_prev = pos - lane_results[lane_i].back();
        if (dist_with_prev >= min_size && (dist_with_prev + min_size <= max_size)) {
          lane_achieved_chunk_invariance[lane_i] = true;
          max_size_vec.m256i_i32[lane_i] = pos + max_size;
          avg_size_vec.m256i_i32[lane_i] = pos + avg_size;
          lane_not_marked_for_jump ^= (0b1 << lane_i);
          jump_vec.m256i_i32[lane_i] = pos + min_size - 31;
        }
      }
      lane_results[lane_i].emplace_back(pos);
      if constexpr (compute_features) {
        lane_features_results[lane_i].emplace_back(std::move(lane_current_features[lane_i]));
        lane_current_features[lane_i] = std::vector<uint32_t>();
      }
    }
  };

  auto sample_feature_value = [&lane_current_features](uint32_t lane_i, uint32_t pattern) {
    if (lane_current_features[lane_i].empty()) {
      lane_current_features[lane_i].resize(16);
      lane_current_features[lane_i].shrink_to_fit();
    }
    for (int feature_i = 0; feature_i < 16; feature_i++) {
      const auto& [mi, ai] = constants::N_Transform_Coefs[feature_i];
      lane_current_features[lane_i][feature_i] = std::max<uint32_t>(lane_current_features[lane_i][feature_i], (mi * pattern + ai) % (1LL << 32));
    }
  };

  while (true) {
    vindex = _mm256_min_epi32(vindex, vindex_max_avx2);
    __m256i is_finish_vec = _mm256_cmpgt_epi32(vindex_max_avx2, vindex);
    __m256 is_finish_vec_ps = _mm256_castsi256_ps(is_finish_vec);
    int is_lane_not_finished = _mm256_movemask_ps(is_finish_vec_ps);
    // If all lanes are finished we break, else we continue and lanes that are already finished will ignore results
    if (is_lane_not_finished == 0) break;

    __m256i cbytes = _mm256_i32gather_epi32(reinterpret_cast<int const*>(data.data()), vindex, 1);

    uint32_t j = 0;
    while (j < 4) {
      hash = _mm256_srli_epi32(hash, 1);  // Shift all the hash values for each lane at the same time
      __m256i idx = _mm256_and_epi32(cbytes, cmask);  // Get byte on the lower bits of the packed 32bit lanes
      cbytes = _mm256_srli_epi32(cbytes, 8);  // We already got the byte on the lower bits, we can shift right to later get the next byte
      // This gives us the GEAR hash values for each of the bytes we just got, scale by 4 because 32bits=4bytes
      __m256i tentry = _mm256_i32gather_epi32(reinterpret_cast<int const*>(constants::GEAR), idx, 4);
      hash = _mm256_add_epi32(hash, tentry);  // Add the values we got from the GEAR hash values to the values on the hash

      // For each lane if we are at avg_size or higher we use the easier mask_l condition
      __m256i avg_size_hit = _mm256_cmpgt_epi32(vindex, avg_size_vec);
      __m256 mask_vec = _mm256_blendv_ps(_mm256_castsi256_ps(mask_s_vec), _mm256_castsi256_ps(mask_l_vec), _mm256_castsi256_ps(avg_size_hit));

      // Compare each packed int by bitwise AND with the mask and checking that its 0
      __m256i hash_masked = _mm256_and_epi32(hash, _mm256_castps_si256(mask_vec));
      __m256i hash_eq_mask = _mm256_cmpeq_epi32(hash_masked, zero_vec);
      __m256 hash_eq_mask_ps = _mm256_castsi256_ps(hash_eq_mask);

      // For the lanes that the backup cut condition is satisfied, update it if there is not a prior backup already
      __m256i hash_backup_masked = _mm256_and_epi32(hash, mask_b_vec);
      __m256i hash_backup_eq_mask = _mm256_cmpeq_epi32(hash_backup_masked, zero_vec);
      // If the lane is marked for jump then the backup cut point has been updated for the new maximum after the jump, don't touch it
      auto lane_not_marked_for_jump_vec = _mm256_setr_epi32(
        lane_not_marked_for_jump & 0b00000001 ? 0xFFFFFFFF : 0,
        lane_not_marked_for_jump & 0b00000010 ? 0xFFFFFFFF : 0,
        lane_not_marked_for_jump & 0b00000100 ? 0xFFFFFFFF : 0,
        lane_not_marked_for_jump & 0b00001000 ? 0xFFFFFFFF : 0,
        lane_not_marked_for_jump & 0b00010000 ? 0xFFFFFFFF : 0,
        lane_not_marked_for_jump & 0b00100000 ? 0xFFFFFFFF : 0,
        lane_not_marked_for_jump & 0b01000000 ? 0xFFFFFFFF : 0,
        lane_not_marked_for_jump & 0b10000000 ? 0xFFFFFFFF : 0
      );
      hash_backup_eq_mask = _mm256_and_epi32(hash_backup_eq_mask, lane_not_marked_for_jump_vec);
      // Backup mask should not be used until after avg_size is hit
      hash_backup_eq_mask = _mm256_and_epi32(hash_backup_eq_mask, avg_size_hit);
      __m256 new_backup_cut_vec_ps = _mm256_blendv_ps(_mm256_castsi256_ps(backup_cut_vec), _mm256_castsi256_ps(vindex), _mm256_castsi256_ps(hash_backup_eq_mask));
      backup_cut_vec = _mm256_min_epi32(backup_cut_vec, _mm256_castps_si256(new_backup_cut_vec_ps));

      __m256i minmax_adjustment_ready = _mm256_cmpgt_epi32(ones_vec, minmax_adjustment_vec);
      __m256 minmax_adjustment_ready_mask_ps = _mm256_castsi256_ps(minmax_adjustment_ready);

      __m256i max_size_hit = _mm256_cmpeq_epi32(max_size_vec, vindex);
      __m256 max_size_hit_mask_ps = _mm256_castsi256_ps(max_size_hit);

      if constexpr (compute_features) {
        // Check if content defined sampling condition is satisfied and we should sample feature values for some lane
        __m256i hash_cds_masked = _mm256_and_epi32(hash, cds_mask_vec);
        __m256i hash_cds_eq_vmask = _mm256_cmpeq_epi32(hash_cds_masked, zero_vec);
        // Ensure lane is not ready for jump or still doing min-max adjustment, in any of those cases we shouldn't sample
        hash_cds_eq_vmask = _mm256_and_epi32(hash_cds_eq_vmask, lane_not_marked_for_jump_vec);
        hash_cds_eq_vmask = _mm256_and_epi32(hash_cds_eq_vmask, minmax_adjustment_ready);
        int hash_cds_eq_mask = _mm256_movemask_ps(_mm256_castsi256_ps(hash_cds_eq_vmask));

        if (hash_cds_eq_mask != 0) {
          if (hash_cds_eq_mask & 0b00000001) sample_feature_value(0, hash.m256i_i32[0]);
          if (hash_cds_eq_mask & 0b00000010) sample_feature_value(1, hash.m256i_i32[1]);
          if (hash_cds_eq_mask & 0b00000100) sample_feature_value(2, hash.m256i_i32[2]);
          if (hash_cds_eq_mask & 0b00001000) sample_feature_value(3, hash.m256i_i32[3]);
          if (hash_cds_eq_mask & 0b00010000) sample_feature_value(4, hash.m256i_i32[4]);
          if (hash_cds_eq_mask & 0b00100000) sample_feature_value(5, hash.m256i_i32[5]);
          if (hash_cds_eq_mask & 0b01000000) sample_feature_value(6, hash.m256i_i32[6]);
          if (hash_cds_eq_mask & 0b10000000) sample_feature_value(7, hash.m256i_i32[7]);
        }
      }

      int lane_has_result = is_lane_not_finished &
        lane_not_marked_for_jump &
        _mm256_movemask_ps(minmax_adjustment_ready_mask_ps) &
        (_mm256_movemask_ps(hash_eq_mask_ps) | _mm256_movemask_ps(max_size_hit_mask_ps));

      if (lane_has_result != 0) {
        if (lane_has_result & 0b00000001) process_lane(0, vindex.m256i_i32[0]);
        if (lane_has_result & 0b00000010) process_lane(1, vindex.m256i_i32[1]);
        if (lane_has_result & 0b00000100) process_lane(2, vindex.m256i_i32[2]);
        if (lane_has_result & 0b00001000) process_lane(3, vindex.m256i_i32[3]);
        if (lane_has_result & 0b00010000) process_lane(4, vindex.m256i_i32[4]);
        if (lane_has_result & 0b00100000) process_lane(5, vindex.m256i_i32[5]);
        if (lane_has_result & 0b01000000) process_lane(6, vindex.m256i_i32[6]);
        if (lane_has_result & 0b10000000) process_lane(7, vindex.m256i_i32[7]);
      }

      minmax_adjustment_vec = _mm256_sub_epi32(minmax_adjustment_vec, ones_vec);

      ++j;
      vindex = _mm256_add_epi32(vindex, ones_vec);  // advance 1 byte in the data for each lane
    }

    if (lane_not_marked_for_jump != 0b11111111) {
      vindex = _mm256_max_epi32(vindex, jump_vec);

      if (!(lane_not_marked_for_jump & 0b00000001)) { minmax_adjustment_vec.m256i_i32[0] = 31; backup_cut_vec.m256i_i32[0] = max_size_vec.m256i_i32[0]; }
      if (!(lane_not_marked_for_jump & 0b00000010)) { minmax_adjustment_vec.m256i_i32[1] = 31; backup_cut_vec.m256i_i32[1] = max_size_vec.m256i_i32[1]; }
      if (!(lane_not_marked_for_jump & 0b00000100)) { minmax_adjustment_vec.m256i_i32[2] = 31; backup_cut_vec.m256i_i32[2] = max_size_vec.m256i_i32[2]; }
      if (!(lane_not_marked_for_jump & 0b00001000)) { minmax_adjustment_vec.m256i_i32[3] = 31; backup_cut_vec.m256i_i32[3] = max_size_vec.m256i_i32[3]; }
      if (!(lane_not_marked_for_jump & 0b00010000)) { minmax_adjustment_vec.m256i_i32[4] = 31; backup_cut_vec.m256i_i32[4] = max_size_vec.m256i_i32[4]; }
      if (!(lane_not_marked_for_jump & 0b00100000)) { minmax_adjustment_vec.m256i_i32[5] = 31; backup_cut_vec.m256i_i32[5] = max_size_vec.m256i_i32[5]; }
      if (!(lane_not_marked_for_jump & 0b01000000)) { minmax_adjustment_vec.m256i_i32[6] = 31; backup_cut_vec.m256i_i32[6] = max_size_vec.m256i_i32[6]; }
      if (!(lane_not_marked_for_jump & 0b10000000)) { minmax_adjustment_vec.m256i_i32[7] = 31; backup_cut_vec.m256i_i32[7] = max_size_vec.m256i_i32[7]; }

      jump_vec = zero_vec;
      lane_not_marked_for_jump = 0b11111111;
    }
  }

  auto i = vindex.m256i_i32[7];
  pattern = hash.m256i_i32[7];  // Recover hash value from last lane
  // Deal with any trailing data sequentially
  while (i < data.size()) {
    pattern = (pattern >> 1) + constants::GEAR[data[i]];
    if (!(pattern & mask_l)) process_lane(7, i);
    ++i;
  }

  for (uint64_t lane = 0; lane < 8; lane++) {
    auto& cut_points_list = lane_results[lane];
    for (uint64_t n = 0; n < cut_points_list.size(); n++) {
      auto& cut_point_offset = cut_points_list[n];
      candidates.emplace_back(CutPointCandidateType::EASY_CUT_MASK, base_offset + cut_point_offset);
      if constexpr (compute_features) {
        candidate_features.emplace_back(std::move(lane_features_results[lane][n]));
      }
    }
  }

  if (candidates.empty() || candidates.back().offset != base_offset + data.size()) {
    // TODO: EOF_CUT and MAX_SIZE_CUT ARE THE SAME? SHOULDNT THEY BE DIFFERENT?
    candidates.emplace_back(CutPointCandidateType::EOF_CUT, base_offset + data.size());
    if constexpr (compute_features) {
      candidate_features.emplace_back();
    }
  }
}

template<
  bool compute_features,
  typename CandidateFeaturesResult = std::conditional_t<compute_features, std::vector<std::vector<uint32_t>>, std::monostate>,
  typename CdcCandidatesResult = std::tuple<std::vector<CutPointCandidate>, CandidateFeaturesResult>
>
CdcCandidatesResult find_cdc_cut_candidates(std::span<uint8_t> data, uint32_t min_size, uint32_t avg_size, uint32_t max_size, bool is_first_segment = true) {
  CdcCandidatesResult result;
  std::vector<CutPointCandidate>& candidates = std::get<0>(result);
  CandidateFeaturesResult& candidate_features = std::get<1>(result);
  if (data.empty()) return result;

  const auto bits = utility::logarithm2(avg_size);
  const auto mask_s = utility::mask(bits + 1);
  const auto mask_l = utility::mask(bits - 1);

  using cdc_offset_return_type = typename decltype(std::function{ cdc_next_offset<compute_features> })::result_type;

  cdc_offset_return_type cdc_return{};
  uint32_t base_offset = 0;
  CutPointCandidate& cp = std::get<0>(cdc_return);
  uint32_t& pattern = std::get<1>(cdc_return);

  // If this is not the first segment then we need to deal with the previous segment extended data and attempt to recover chunk invariance
  if (!is_first_segment) {
    std::tie(cp, pattern) = cdc_next_offset<false>(data, 0, avg_size - 1, 4294967295, mask_s, mask_l, pattern);

    base_offset += cp.offset;
    candidates.emplace_back(cp.type, base_offset);
    if constexpr (compute_features) {
      candidate_features.emplace_back();
    }
    data = std::span(data.data() + cp.offset, data.size() - cp.offset);

    // And now we need to recover chunk invariance in accordance to the chunk invariance recovery condition.
    // We are guaranteed to be back in sync with what non-segmented CDC would have done as soon as we find a cut candidate with
    // distance_w_prev_cut_candidate >= min_size and distance_w_prev_cut_candidate + min_size <= max_size.
    // As soon as we find that we can break from here and keep processing as if we were processing without segments,
    // which in particular means we can exploit jumps to min_size again.
    while (!data.empty()) {
      std::tie(cp, pattern) = cdc_next_offset<false>(std::span(data.data() + 1, data.size() - 1), 0, avg_size - 1, 4294967295, mask_s, mask_l, pattern);
      cp.offset = cp.offset + 1;

      base_offset += cp.offset;
      candidates.emplace_back(cp.type, base_offset);
      if constexpr (compute_features) {
        candidate_features.emplace_back();
      }
      data = std::span(data.data() + cp.offset, data.size() - cp.offset);
      if (cp.offset >= min_size && (cp.offset + min_size <= max_size)) break;  // back in sync with non-segmented processing!
    }
  }

  if (data.size() < 1024) {
    while (!data.empty()) {
      if (base_offset == 0) {
        cdc_return = cdc_next_offset<compute_features>(data, min_size, avg_size, max_size, mask_s, mask_l, pattern);
      }
      else {
        cdc_return = cdc_next_offset<compute_features>(std::span(data.data() + 1, data.size() - 1), min_size - 1, avg_size - 1, max_size - 1, mask_s, mask_l, pattern);
        cp.offset = cp.offset + 1;
      }

      base_offset += cp.offset;
      candidates.emplace_back(cp.type, base_offset);
      if constexpr (compute_features) {
        candidate_features.emplace_back(std::move(std::get<2>(cdc_return)));
      }
      data = std::span(data.data() + cp.offset, data.size() - cp.offset);
    }
  }
  else {
    cdc_find_cut_points_with_invariance<compute_features>(candidates, candidate_features, data, base_offset, min_size, avg_size, max_size, mask_s, mask_l, pattern);
  }

  if constexpr (compute_features) {
    candidate_features.shrink_to_fit();
  }
  candidates.shrink_to_fit();
  return result;
}

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

std::tuple<std::bitset<64>, std::vector<uint32_t>> simhash_data_xxhash_shingling(uint8_t* data, uint32_t data_len, uint32_t chunk_size) {
  __m256i upper_counter_vector = _mm256_set1_epi8(0);
  __m256i lower_counter_vector = _mm256_set1_epi8(0);

  std::tuple<std::bitset<64>, std::vector<uint32_t>> return_val{};
  auto* simhash = &std::get<0>(return_val);
  auto& minichunks_vec = std::get<1>(return_val);

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
  *simhash = (static_cast<uint64_t>(upper_chunk_hash) << 32) | lower_chunk_hash;
  minichunks_vec.shrink_to_fit();
  return return_val;
}

std::tuple<std::bitset<64>, std::vector<uint32_t>> simhash_data_xxhash_cdc(uint8_t* data, uint32_t data_len, uint32_t chunk_size) {
  __m256i upper_counter_vector = _mm256_set1_epi8(0);
  __m256i lower_counter_vector = _mm256_set1_epi8(0);

  std::tuple<std::bitset<64>, std::vector<uint32_t>> return_val{};
  auto* simhash = &std::get<0>(return_val);
  auto& minichunks_vec = std::get<1>(return_val);

  // Find the CDC minichunks and update the SimHash with their data
  const auto min_chunk_size = chunk_size / 2;
  auto [cut_offsets, cut_offsets_features] = find_cdc_cut_candidates<false>(std::span(data, data_len), min_chunk_size, chunk_size, chunk_size * 2);
  uint32_t previous_offset = 0;
  for (const auto& cut_point_candidate : cut_offsets) {
    if (cut_point_candidate.offset <= previous_offset) continue;
    if (cut_point_candidate.offset < previous_offset + min_chunk_size && cut_point_candidate.offset != data_len) continue;
    const auto minichunk_len = cut_point_candidate.offset - previous_offset;
    // Calculate hash for current chunk
    const std::bitset<64> chunk_hash = XXH3_64bits(data + previous_offset, minichunk_len);

    // Update SimHash vector with the hash of the chunk
    // Using AVX/2 we don't have enough vector size to handle the whole 64bit hash, we should be able to do it with AVX512,
    // but that is less readily available in desktop chips, so we split the hash into 2 and process it that way
    const std::bitset<64> upper_chunk_hash = chunk_hash >> 32;
    upper_counter_vector = add_32bit_hash_to_simhash_counter(upper_chunk_hash.to_ulong(), upper_counter_vector);
    const std::bitset<64> lower_chunk_hash = (chunk_hash << 32) >> 32;
    lower_counter_vector = add_32bit_hash_to_simhash_counter(lower_chunk_hash.to_ulong(), lower_counter_vector);

    minichunks_vec.emplace_back(minichunk_len);
    previous_offset = cut_point_candidate.offset;
  }

  const uint32_t upper_chunk_hash = finalize_32bit_simhash_from_counter(upper_counter_vector);
  const uint32_t lower_chunk_hash = finalize_32bit_simhash_from_counter(lower_counter_vector);
  *simhash = (static_cast<uint64_t>(upper_chunk_hash) << 32) | lower_chunk_hash;
  minichunks_vec.shrink_to_fit();
  return return_val;
}

template<std::size_t bit_size>
uint64_t hamming_distance(const std::bitset<bit_size>& data1, const std::bitset<bit_size>& data2) {
  const auto val = data1 ^ data2;
  return val.count();
}

template<std::size_t bit_size>
std::size_t hamming_syndrome(const std::bitset<bit_size>& data) {
  int result = 0;
  std::bitset<bit_size> mask {0b1};
  for (std::size_t i = 0; i < bit_size; i++) {
    auto bit = data & mask;
    if (bit != 0) result ^= i;
    mask <<= 1;
  }

  return result;
}

template<std::size_t bit_size>
std::bitset<bit_size> hamming_base(const std::bitset<bit_size>& data) {
  auto syndrome = hamming_syndrome(data);
  std::bitset<bit_size> base = data;
  base = base.flip(syndrome);
  // The first bit doesn't really participate in non-extended hamming codes (and extended ones are not useful to us)
  // So we just collapse to them all to the version with 0 on the first bit, allows us to match some hamming distance 2 data
  base[0] = 0;
  return base;
}

enum LZInstructionType : uint8_t {
  COPY,
  INSERT,
  DELTA
};

struct LZInstruction {
  LZInstructionType type;
  uint64_t offset;  // For COPY: previous offset; For INSERT: offset on the original stream; For Delta: previous offset of original data
  uint64_t size;  // How much data to be copied or inserted, or size of the delta original data

  auto operator<=>(const LZInstruction&) const = default;
};

struct DeltaEncodingResult {
  uint64_t estimated_savings = 0;
  std::vector<LZInstruction> instructions;
};

inline unsigned long trailingZeroesCount32(uint32_t mask) {
#ifndef __GNUC__
  unsigned long result;
  result = _BitScanForward(&result, mask) ? result : 0;
  return result;
#else
  // TODO: This requires BMI instruction set, is there not a better way to do it?
  return _tzcnt_u32(mask);
#endif
}

inline unsigned long leadingZeroesCount32(uint32_t mask) {
#ifndef __GNUC__
  unsigned long result;
  // _BitScanReverse gets us the bit INDEX of the highest set bit, so we do 31 - result as 31 is the highest possible bit in a 32bit mask
  result = _BitScanReverse(&result, mask) ? 31 - result : 0;
  return result;
#else
  // TODO: This requires BMI instruction set, is there not a better way to do it?
  return _lzcnt_u32(mask);
#endif
}

uint64_t find_identical_prefix_byte_count(const std::span<const uint8_t> data1_span, const std::span<const uint8_t> data2_span) {
  const auto cmp_size = std::min(data1_span.size(), data2_span.size());
  uint64_t matching_data_count = 0;
  uint64_t i = 0;

  const uint64_t avx2_batches = cmp_size / 32;
  if (avx2_batches > 0) {
    uint64_t avx2_batches_size = avx2_batches * 32;

    const auto alignment_mask = ~31;
    const uintptr_t data1_ptr = reinterpret_cast<uintptr_t>(data1_span.data() + i);
    const auto data1_misalignment = data1_ptr ^ (data1_ptr & alignment_mask);
    const uintptr_t data2_ptr = reinterpret_cast<uintptr_t>(data2_span.data() + i);
    const auto data2_misalignment = data2_ptr ^ (data2_ptr & alignment_mask);
    __m256i data1_avx2_vector;
    __m256i data2_avx2_vector;
    bool data_aligned = data1_misalignment == 0 && data2_misalignment == 0;
    if (
      // If data is not aligned but aligning it might be worth it we align it
      // (if aligning it means sse can no longer run we run misaligned and be done with it)
      !data_aligned &&
      (data1_misalignment == data2_misalignment && avx2_batches_size > 1)
      ) {
      // To align the data we need to skip the misaligned bytes, but also adjust the sse_batches_size so it's still a multiple of 16
      for (uint64_t j = 0; j < data1_misalignment; j++) {
        if (*(data1_span.data() + i) != *(data2_span.data() + i)) return matching_data_count;
        matching_data_count++;
        i++;
      }
      avx2_batches_size -= 32 - data1_misalignment;
      data_aligned = true;
    }

    while (i < avx2_batches_size) {
      if (!data_aligned) {
        data1_avx2_vector = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data1_span.data() + i));
        data2_avx2_vector = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data2_span.data() + i));
      }
      else {
        data1_avx2_vector = _mm256_load_si256(reinterpret_cast<const __m256i*>(data1_span.data() + i));
        data2_avx2_vector = _mm256_load_si256(reinterpret_cast<const __m256i*>(data2_span.data() + i));
      }
      const __m256i avx2_cmp_result = _mm256_cmpeq_epi8(data1_avx2_vector, data2_avx2_vector);
      const uint32_t result_mask = _mm256_movemask_epi8(avx2_cmp_result);

      unsigned long avx2_matching_data_count;
      switch (result_mask) {
      case 0:
        avx2_matching_data_count = 0;
        break;
      case 0b11111111111111111111111111111111:
        avx2_matching_data_count = 32;
        break;
      default:
        const uint32_t inverted_mask = result_mask ^ 0b11111111111111111111111111111111;
        avx2_matching_data_count = trailingZeroesCount32(inverted_mask);
      }
      matching_data_count += avx2_matching_data_count;
      if (avx2_matching_data_count < 32) return matching_data_count;
      i += avx2_matching_data_count;
    }
  }
  
  const uint64_t sse_batches = (cmp_size - i) / 16;
  if (sse_batches > 0) {
    uint64_t sse_batches_size = sse_batches * 16;

    const auto alignment_mask = ~15;
    const uintptr_t data1_ptr = reinterpret_cast<uintptr_t>(data1_span.data() + i);
    const auto data1_misalignment = data1_ptr ^ (data1_ptr & alignment_mask);
    const uintptr_t data2_ptr = reinterpret_cast<uintptr_t>(data2_span.data() + i);
    const auto data2_misalignment = data2_ptr ^ (data2_ptr & alignment_mask);
    __m128i data1_sse_vector;
    __m128i data2_sse_vector;
    bool data_aligned = data1_misalignment == 0 && data2_misalignment == 0;
    if (
      // If data is not aligned but aligning it might be worth it we align it
      // (if aligning it means sse can no longer run we run misaligned and be done with it)
      !data_aligned &&
      (data1_misalignment == data2_misalignment && sse_batches > 1)
      ) {
      // To align the data we need to skip the misaligned bytes, but also adjust the sse_batches_size so it's still a multiple of 16
      for (uint64_t j = 0; j < data1_misalignment; j++) {
        if (*(data1_span.data() + i) != *(data2_span.data() + i)) return matching_data_count;
        matching_data_count++;
        i++;
      }
      sse_batches_size -= 16 - data1_misalignment;
      data_aligned = true;
    }

    while (i < sse_batches_size) {
      if (!data_aligned) {
        data1_sse_vector = _mm_loadu_si128(reinterpret_cast<const __m128i*>(data1_span.data() + i));
        data2_sse_vector = _mm_loadu_si128(reinterpret_cast<const __m128i*>(data2_span.data() + i));
      }
      else {
        data1_sse_vector = _mm_load_si128(reinterpret_cast<const __m128i*>(data1_span.data() + i));
        data2_sse_vector = _mm_load_si128(reinterpret_cast<const __m128i*>(data2_span.data() + i));
      }
      const __m128i sse_cmp_result = _mm_cmpeq_epi8(data1_sse_vector, data2_sse_vector);
      const uint32_t result_mask = _mm_movemask_epi8(sse_cmp_result);

      unsigned long sse_matching_data_count;
      switch (result_mask) {
      case 0:
        sse_matching_data_count = 0;
        break;
      case 0b1111111111111111:
        sse_matching_data_count = 16;
        break;
      default:
        const uint32_t inverted_mask = result_mask ^ 0b1111111111111111;
        sse_matching_data_count = trailingZeroesCount32(inverted_mask);
      }
      matching_data_count += sse_matching_data_count;
      if (sse_matching_data_count < 16) return matching_data_count;
      i += sse_matching_data_count;
    }
  }

  while (i < cmp_size) {
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

uint64_t find_identical_suffix_byte_count(const std::span<const uint8_t> data1_span, const std::span<const uint8_t> data2_span) {
  const auto cmp_size = std::min(data1_span.size(), data2_span.size());
  const auto data1_start = data1_span.data() + data1_span.size() - cmp_size;
  const auto data2_start = data2_span.data() + data2_span.size() - cmp_size;

  uint64_t matching_data_count = 0;
  uint64_t i = 0;

  const uint64_t avx2_batches = cmp_size / 32;
  if (avx2_batches > 0) {
    uint64_t avx2_batches_size = avx2_batches * 32;

    const auto alignment_mask = ~31;
    const uintptr_t data1_ptr = reinterpret_cast<uintptr_t>(data1_start + cmp_size);
    const auto data1_misalignment = data1_ptr ^ (data1_ptr & alignment_mask);
    const uintptr_t data2_ptr = reinterpret_cast<uintptr_t>(data2_start + cmp_size);
    const auto data2_misalignment = data2_ptr ^ (data2_ptr & alignment_mask);
    __m256i data1_avx2_vector;
    __m256i data2_avx2_vector;
    bool data_aligned = data1_misalignment == 0 && data2_misalignment == 0;
    if (
      // If data is not aligned but aligning it might be worth it we align it
      // (if aligning it means sse can no longer run we run misaligned and be done with it)
      !data_aligned &&
      (data1_misalignment == data2_misalignment && avx2_batches_size > 1)
      ) {
      // To align the data we need to skip the misaligned bytes, but also adjust the sse_batches_size so it's still a multiple of 16
      for (uint64_t j = 0; j < data1_misalignment; j++) {
        if (*(data1_start + cmp_size - 1 - i) != *(data2_start + cmp_size - 1 - i)) return matching_data_count;
        matching_data_count++;
        i++;
      }
      avx2_batches_size -= 32 - data1_misalignment;
      data_aligned = true;
    }

    while (i < avx2_batches_size) {
      if (!data_aligned) {
        data1_avx2_vector = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data1_start + cmp_size - 32 - i));
        data2_avx2_vector = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data2_start + cmp_size - 32 - i));
      }
      else {
        data1_avx2_vector = _mm256_load_si256(reinterpret_cast<const __m256i*>(data1_start + cmp_size - 32 - i));
        data2_avx2_vector = _mm256_load_si256(reinterpret_cast<const __m256i*>(data2_start + cmp_size - 32 - i));
      }
      const __m256i avx2_cmp_result = _mm256_cmpeq_epi8(data1_avx2_vector, data2_avx2_vector);
      const uint32_t result_mask = _mm256_movemask_epi8(avx2_cmp_result);

      unsigned long avx2_matching_data_count;
      switch (result_mask) {
      case 0:
        avx2_matching_data_count = 0;
        break;
      case 0b11111111111111111111111111111111:
        avx2_matching_data_count = 32;
        break;
      default:
        const uint32_t inverted_mask = result_mask ^ 0b11111111111111111111111111111111;
        avx2_matching_data_count = leadingZeroesCount32(inverted_mask);
      }
      matching_data_count += avx2_matching_data_count;
      if (avx2_matching_data_count < 32) return matching_data_count;
      i += avx2_matching_data_count;
    }
  }

  const uint64_t sse_batches = (cmp_size - i) / 16;
  if (sse_batches > 0) {
    uint64_t sse_batches_size = sse_batches * 16;

    const auto alignment_mask = ~15;
    const uintptr_t data1_ptr = reinterpret_cast<uintptr_t>(data1_start + cmp_size - i);
    const auto data1_misalignment = data1_ptr ^ (data1_ptr & alignment_mask);
    const uintptr_t data2_ptr = reinterpret_cast<uintptr_t>(data2_start + cmp_size - i);
    const auto data2_misalignment = data2_ptr ^ (data2_ptr & alignment_mask);
    __m128i data1_sse_vector;
    __m128i data2_sse_vector;
    bool data_aligned = data1_misalignment == 0 && data2_misalignment == 0;
    if (
      // If data is not aligned but aligning it might be worth it we align it
      // (if aligning it means sse can no longer run we run misaligned and be done with it)
      !data_aligned &&
      (data1_misalignment == data2_misalignment && sse_batches > 1)
      ) {
      // To align the data we need to skip the misaligned bytes, but also adjust the sse_batches_size so it's still a multiple of 16
      for (uint64_t j = 0; j < data1_misalignment; j++) {
        if (*(data1_start + cmp_size - 1 - i) != *(data2_start + cmp_size - 1 - i)) return matching_data_count;
        matching_data_count++;
        i++;
      }
      sse_batches_size -= 16 - data1_misalignment;
      data_aligned = true;
    }

    while (i < sse_batches_size) {
      if (!data_aligned) {
        data1_sse_vector = _mm_loadu_si128(reinterpret_cast<const __m128i*>(data1_start + cmp_size - 16 - i));
        data2_sse_vector = _mm_loadu_si128(reinterpret_cast<const __m128i*>(data2_start + cmp_size - 16 - i));
      }
      else {
        data1_sse_vector = _mm_load_si128(reinterpret_cast<const __m128i*>(data1_start + cmp_size - 16 - i));
        data2_sse_vector = _mm_load_si128(reinterpret_cast<const __m128i*>(data2_start + cmp_size - 16 - i));
      }
      const __m128i sse_cmp_result = _mm_cmpeq_epi8(data1_sse_vector, data2_sse_vector);
      const uint32_t result_mask = _mm_movemask_epi8(sse_cmp_result);

      unsigned long sse_matching_data_count;
      switch (result_mask) {
      case 0:
        sse_matching_data_count = 0;
        break;
      case 0b1111111111111111:
        sse_matching_data_count = 16;
        break;
      default:
        const uint32_t inverted_mask = result_mask ^ 0b1111111111111111;
        // despite the mask being 32bit, SSE has 16bytes so only the 16 lower bits could possibly be set, so we will always
        // get 16 extra leading zeroes we don't actually care about, so we subtract them from the result
        sse_matching_data_count = leadingZeroesCount32(inverted_mask) - 16;
      }
      matching_data_count += sse_matching_data_count;
      if (sse_matching_data_count < 16) return matching_data_count;
      i += sse_matching_data_count;
    }
  }

  while (i < cmp_size) {
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

DeltaEncodingResult simulate_delta_encoding_shingling(const utility::ChunkData& chunk, const utility::ChunkData& similar_chunk, uint32_t minichunk_size) {
  DeltaEncodingResult result;
  //DDelta style delta encoding, first start by greedily attempting to match data at the start and end on the chunks
  const uint64_t start_matching_data_len = find_identical_prefix_byte_count(chunk.data, similar_chunk.data);
  uint64_t end_matching_data_len = find_identical_suffix_byte_count(chunk.data, similar_chunk.data);

  uint64_t saved_size = start_matching_data_len + end_matching_data_len;

  // (offset, hash), it might be tempting to change this for a hashmap, don't, it will be slower as sequential search is really fast because there won't be many minuchunks
  std::vector<std::tuple<uint32_t, uint64_t>> similar_chunk_minichunk_hashes;

  // Iterate over the data in chunks
  uint32_t minichunk_offset = 0;
  for (uint64_t i = 0; i < similar_chunk.data.size(); i += minichunk_size) {
    // Calculate hash for current chunk
    const auto minichunk_len = std::min<uint64_t>(minichunk_size, similar_chunk.data.size() - i);
    uint64_t minichunk_hash = XXH3_64bits(similar_chunk.data.data() + i, minichunk_len);
    similar_chunk_minichunk_hashes.emplace_back(minichunk_offset, minichunk_hash);
    minichunk_offset += minichunk_len;
  }

  if (start_matching_data_len) {
    result.instructions.emplace_back(LZInstructionType::COPY, 0, start_matching_data_len);
  }

  uint64_t unaccounted_data_start_pos = start_matching_data_len;
  for (uint32_t minichunk_offset = 0; minichunk_offset < chunk.data.size(); minichunk_offset += minichunk_size) {
    const auto minichunk_len = std::min<uint64_t>(minichunk_size, chunk.data.size() - minichunk_offset);
    XXH64_hash_t minichunk_hash = 0;

    minichunk_hash = XXH3_64bits(chunk.data.data() + minichunk_offset, minichunk_len);

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
        std::span(chunk.data.data(), minichunk_offset),
        std::span(similar_chunk.data.data(), candidate_similar_chunk_offset)
      );

      // Then extend forwards
      const uint64_t candidate_extended_size = find_identical_prefix_byte_count(
        std::span(chunk.data.data() + minichunk_offset + minichunk_len, chunk.data.size() - minichunk_offset - minichunk_len),
        std::span(similar_chunk.data.data() + candidate_similar_chunk_offset + minichunk_len, similar_chunk.data.size() - candidate_similar_chunk_offset - minichunk_len)
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
  const auto end_matched_data_start_pos = std::max<uint64_t>(unaccounted_data_start_pos, chunk.data.size() - end_matching_data_len);
  end_matching_data_len = chunk.data.size() - end_matched_data_start_pos;
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
      similar_chunk.data.size() - end_matching_data_len,
      end_matching_data_len
    );
  }

  result.estimated_savings = saved_size;
  return result;
}

DeltaEncodingResult simulate_delta_encoding_using_minichunks(const utility::ChunkData& chunk, const utility::ChunkData& similar_chunk, uint32_t minichunk_size) {
  DeltaEncodingResult result;
  //DDelta style delta encoding, first start by greedily attempting to match data at the start and end on the chunks
  const uint64_t start_matching_data_len = find_identical_prefix_byte_count(chunk.data, similar_chunk.data);
  uint64_t end_matching_data_len = find_identical_suffix_byte_count(chunk.data, similar_chunk.data);

  uint64_t saved_size = start_matching_data_len + end_matching_data_len;

  // (offset, hash), it might be tempting to change this for a hashmap, don't, it will be slower as sequential search is really fast because there won't be many minuchunks
  std::vector<std::tuple<uint32_t, uint64_t>> similar_chunk_minichunk_hashes;

  uint32_t minichunk_offset = 0;
  for (const auto& minichunk_len : similar_chunk.minichunks) {
    XXH64_hash_t minichunk_hash = XXH3_64bits(similar_chunk.data.data() + minichunk_offset, minichunk_len);
    similar_chunk_minichunk_hashes.emplace_back(minichunk_offset, minichunk_hash);
    minichunk_offset += minichunk_len;
  }

  if (start_matching_data_len) {
    result.instructions.emplace_back(LZInstructionType::COPY, 0, start_matching_data_len);
  }

  uint64_t unaccounted_data_start_pos = start_matching_data_len;
  minichunk_offset = 0;
  for (uint32_t minichunk_i = 0; minichunk_i < chunk.minichunks.size(); minichunk_i++) {
    const auto& minichunk_len = chunk.minichunks[minichunk_i];
    XXH64_hash_t minichunk_hash = 0;

    minichunk_hash = XXH3_64bits(chunk.data.data() + minichunk_offset, minichunk_len);

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
        std::span(chunk.data.data(), minichunk_offset),
        std::span(similar_chunk.data.data(), candidate_similar_chunk_offset)
      );

      // Then extend forwards
      const uint64_t candidate_extended_size = find_identical_prefix_byte_count(
        std::span(chunk.data.data() + minichunk_offset + minichunk_len, chunk.data.size() - minichunk_offset - minichunk_len),
        std::span(similar_chunk.data.data() + candidate_similar_chunk_offset + minichunk_len, similar_chunk.data.size() - candidate_similar_chunk_offset - minichunk_len)
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
  const auto end_matched_data_start_pos = std::max<uint64_t>(unaccounted_data_start_pos, chunk.data.size() - end_matching_data_len);
  end_matching_data_len = chunk.data.size() - end_matched_data_start_pos;
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
      similar_chunk.data.size() - end_matching_data_len,
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

void dump_vector_to_ostream_with_guard(std::vector<uint8_t>&& _output_buffer, uint64_t buffer_used_len, std::ostream* ostream) {
  std::vector<uint8_t> output_buffer = std::move(_output_buffer);
  ostream->write(reinterpret_cast<const char*>(output_buffer.data()), buffer_used_len);
}

class WrappedOStreamOutputStream: public OutputStream {
private:
  std::ostream* ostream;
  uint64_t buffer_size;
  std::vector<uint8_t> output_buffer;
  uint64_t buffer_used_len = 0;
  std::thread dump_thread;

  void write_with_thread() {
    if (dump_thread.joinable()) dump_thread.join();
    dump_thread = std::thread(dump_vector_to_ostream_with_guard, std::move(output_buffer), buffer_used_len, ostream);
    buffer_used_len = 0;
    output_buffer.resize(buffer_size);
    output_buffer.shrink_to_fit();
  }

public:
  explicit WrappedOStreamOutputStream(std::ostream* _ostream, uint64_t _buffer_size = 200ull * 1024 * 1024): ostream(_ostream), buffer_size(_buffer_size) {
    output_buffer.resize(buffer_size);
    output_buffer.shrink_to_fit();
  }

  ~WrappedOStreamOutputStream() override {
    flush();
  }

  void flush() {
    if (buffer_used_len > 0) {
      write_with_thread();
      dump_thread.join();
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
      write_with_thread();
    }
    std::copy_n(buffer, size, output_buffer.data() + buffer_used_len);
    buffer_used_len += size;
    return size;
  }
};

template <class T>
class circular_vector {
  std::vector<T> vec{};
  // The first index for the circular vector in vec
  uint64_t first_index_vec = 0;
  // The first index overall num, like if this had been a regular expanding vector, what would the first item's index be
  uint64_t first_index_num = 0;
  // The last index for the circular vector in vec
  std::optional<uint64_t> last_index_vec = std::nullopt;

  uint64_t reclaimable_slots = 0;

  void realloc_vec(uint64_t new_capacity) {
    const auto used_size = size();
    if (new_capacity < used_size) return;
    std::vector<T> new_vec{};

    if (used_size > 0) {
      new_vec.reserve(new_capacity);
      new_vec.resize(used_size);
      auto begin_iter = begin();
      auto end_iter = end();
      std::move(begin_iter, end_iter, new_vec.begin());
    }

    vec = std::move(new_vec);
    // Reset everything so the new vec is now accessed in a non-circular way, at least until needed again
    last_index_vec = std::nullopt;
    first_index_vec = 0;
    reclaimable_slots = 0;
  }

public:
  using size_type = typename std::vector<T>::size_type;

  explicit circular_vector() = default;

  template <class U>
  class const_iterator {
  public:
    using difference_type = std::ptrdiff_t;
    using value_type = U;

  private:
    const std::vector<value_type>* vec = nullptr;
    uint64_t index = 0;  // vec->size() on the index means end/cend
    const uint64_t* first_index = nullptr;
    const std::optional<uint64_t>* last_index = nullptr;

    uint64_t add_to_index(difference_type add) const {
      const auto added_index = index + add;
      const auto vec_size = vec->size();
      const bool has_last_index = last_index != nullptr && last_index->has_value();
      // We are already on an index that has wrapped around the circular vector
      if (has_last_index && index < *first_index) {
        // Given that we are wrapping around already, we don't allow wrapping around again,
        // if we reach the end or exceed it, just point to the end
        return added_index > **last_index ? vec_size : added_index;
      }
      // We have yet to wrap around (if that's even possible)
      else {
        // Wrapping around is not even needed
        if (added_index < vec_size) {
          // Return the added index, unless we have a last_index not before the first_index, and we would be going past that last_index,
          // in which case we return the vec_size which will result in an end iterator
          return (has_last_index && **last_index >= *first_index && added_index > **last_index) ? vec_size : added_index;
        }
        
        if (
          // If the vector doesn't have circular wrapping around behavior then just return index for end of vec
          !has_last_index ||
          // If the last index we have is larger than the first_index then wrapping around is not even possible
          **last_index >= *first_index ||
          // If we were to do a whole loop, just return index for end/cend, as we need to stop at some point.
          add >= vec_size
          ) {
          return vec_size;
        }

        const auto wrapped_index = added_index % vec_size;
        // If even wrapping around we went too far we stop at the last_index + 1 as well
        return wrapped_index > **last_index ? vec_size : wrapped_index;
      }
    }

    uint64_t remove_from_index(difference_type substract) const {
      if (substract == 0) return index;

      const auto vec_size = vec->size();
      const bool has_last_index = last_index != nullptr && last_index->has_value();
      const auto subtracted_index = static_cast<difference_type>(has_last_index && index == vec_size ? **last_index + 1 : index) + substract;
      // We are already on an index that has wrapped around the circular vector
      if (has_last_index && index < *first_index) {
        // If it's not enough to wrap around in reverse, then just return the new index
        if (subtracted_index >= 0) return subtracted_index;
        // Wrap around in reverse, remember that subtracted_index is negative here
        const auto wrapped_index = vec_size + subtracted_index;
        // If wrapping in reverse gets us before the *first_index, we reversed back to the beginning, stop there
        return wrapped_index >= *first_index ? wrapped_index : *first_index;
      }
      else if (index == vec_size) {
        return has_last_index ? **last_index : vec_size - 1;
      }
      // We have yet to wrap around (if that's even possible)
      else {
        // As we can't wrap around in reverse, we simply reverse as far as the first_index if set, 0 otherwise
        if (first_index != nullptr) {
          return std::max<difference_type>(subtracted_index, *first_index);
        }
        else {
          return std::max<difference_type>(subtracted_index, 0);
        }
      }
    }

    uint64_t shift_index(difference_type diff) const {
      return diff >= 0 ? add_to_index(diff) : remove_from_index(diff);
    }

    bool index_pos_larger_than(difference_type other_index) const {
      // Checks if this iterator's index position is larger than a given index, accounting for circular behavior
      // Prerequisite: this->index != other_index

      // If we enter here then it means this iterator's index is NOT from an element that has wrapped around the circular_vector
      if (first_index == nullptr || index >= *first_index) {
        const bool has_last_index = last_index != nullptr && !last_index->has_value();
        // If the vector is not circular yet, the index must be numerically larger than the other one and that's it
        if (!has_last_index && index > other_index) return true;

        // If there is circular behavior, then other_index might be numerically smaller but refer to a later element.
        // If the other_index is larger than the first_index but still smaller than this->index, then it's for a prior position,
        // if it's larger than first_index and also larger than this->index then the other one is for a later position,
        // otherwise despite other_index being numerically smaller its actually for a later position on the circular_index
        // as it refers to an element after wrapping around circularly
        if (other_index >= *first_index && index > other_index) return true;
        return false;
      }
      // conversely, if we are here it's that this iterator's index is from an element that IS after wrapping around the circular_vector

      // Now it's quite simple, if the other_index is from before wrapping around then it must be from a prior position,
      // else then we just need to compare them numerically.
      if (other_index >= *first_index) return true;
      return index > other_index;
    }

  public:
    const_iterator(const std::vector<value_type>* _vec, uint64_t _index, const uint64_t* _first_index, const std::optional<uint64_t>* _last_index)
      : vec(_vec), index(_index), first_index(_first_index), last_index(_last_index) {
      if ((first_index != nullptr || last_index != nullptr) && (first_index == nullptr || last_index == nullptr)) {
        throw std::runtime_error("circular_vector::iterator: first_index and last_index need to be both set or unset, no mixing");
      }
    }
    const_iterator() = default;

    uint64_t get_index() const { return index; }

    // Forward iterator requirements
    const value_type& operator*() const { return (*vec)[index]; }
    bool operator==(const const_iterator& other) const { return other.index == this->index && other.vec == this->vec; }

    const_iterator& operator++() {
      index = shift_index(1);
      return *this;
    }
    const_iterator operator++(int) {
      auto tmp = *this;
      ++*this;
      return tmp;
    }

    // Bidirectional iterator requirements
    const_iterator& operator--() {
      index = shift_index(-1);
      return *this;
    }
    const_iterator operator--(int) {
      auto tmp = *this;
      --*this;
      return tmp;
    }

    // Random access iterator requirements
    const value_type& operator[](difference_type rhs) const { return (*vec)[shift_index(rhs)]; }

    const_iterator operator+(difference_type rhs) const { return const_iterator(this->vec, shift_index(rhs), first_index, last_index); }
    friend const_iterator operator+(difference_type lhs, const const_iterator& rhs) { return const_iterator(rhs.vec, rhs.shift_index(lhs), rhs.first_index, rhs.last_index); }
    const_iterator& operator+=(difference_type rhs) { index = shift_index(rhs); return *this; }

    difference_type operator-(const const_iterator& rhs) const { return shift_index(-rhs.index); }
    const_iterator operator-(difference_type rhs) const { return const_iterator(this->vec, shift_index(-rhs), first_index, last_index); }
    friend const_iterator operator-(difference_type lhs, const const_iterator& rhs) { return const_iterator(rhs.vec, rhs.shift_index(-lhs), rhs.first_index, rhs.last_index); }
    const_iterator& operator-=(difference_type rhs) { index = shift_index(-rhs); return *this; }

    bool operator>(const const_iterator& rhs) const { return this->index != rhs.index && index_pos_larger_than(rhs.index); }
    bool operator>=(const const_iterator& rhs) const { return this->index == rhs.index || index_pos_larger_than(rhs.index); }
    bool operator<(const const_iterator& rhs) const { return this->index != rhs.index && !index_pos_larger_than(rhs.index); }
    bool operator<=(const const_iterator& rhs) const { return this->index == rhs.index || !index_pos_larger_than(rhs.index); }
  };
  static_assert(std::random_access_iterator<const_iterator<utility::ChunkEntry>>);

  size_type get_last_index_vec() const { return last_index_vec.has_value() ? *last_index_vec : vec.size() - 1; }
  size_type get_last_index_num() const { return first_index_num + this->size() - 1; }
  uint64_t get_index(const const_iterator<T>& iter) {
    const auto iter_index = iter.get_index();
    if (iter_index >= first_index_vec) {
      const auto diff = iter_index - first_index_vec;
      return first_index_num + diff;
    }
    const auto non_wrapped_element_count = vec.size() - first_index_vec;
    return first_index_num + non_wrapped_element_count + iter_index;
  }

  T& operator[](size_type pos) {
    // Check we are not trying to access out of bounds
    const auto last_allowed_pos = get_last_index_num();
    if (pos < first_index_num || pos > last_allowed_pos) {
      throw std::runtime_error("Can't access out of bounds index on circular_vector");
    }

    // Finally, get the index, wrapping around if necessary, and return the element reference
    const auto relative_pos = pos - first_index_num;
    const auto in_vec_index = (first_index_vec + relative_pos) % vec.size();
    return vec[in_vec_index];
  }

  const_iterator<T> begin() const { return const_iterator<T>(&vec, first_index_vec, &first_index_vec, &last_index_vec); }
  const_iterator<T> end() const { return const_iterator<T>(&vec, vec.size(), &first_index_vec, &last_index_vec); }
  const_iterator<T> cbegin() const { return const_iterator<T>(&vec, first_index_vec, &first_index_vec, &last_index_vec); }
  const_iterator<T> cend() const { return const_iterator<T>(&vec, vec.size(), &first_index_vec, &last_index_vec); }

  // The size including reclaimed/removed items, as it would have been in a regular vector
  size_type fullSize() const {
    return first_index_num + size();
  }
  size_type innerVecSize() const { return vec.size(); }
  size_type size() const {
    if (!last_index_vec.has_value() || *last_index_vec < first_index_vec) {
      return vec.size() - reclaimable_slots;
    }
    else {
      return *last_index_vec - first_index_vec + 1;
    }
  }
  bool empty() const { return size() == 0; }
  void clear() {
    first_index_vec = 0;
    last_index_vec = std::nullopt;
    reclaimable_slots = 0;
    vec = std::vector<T>();
  }

  void pop_front() {
    if (last_index_vec.has_value() && *last_index_vec == first_index_vec) {
      // Popped the last element! reset all circular behavior stuff and quit
      clear();
      return;
    }
    first_index_vec++;
    if (first_index_vec == vec.size()) {
      first_index_vec = 0;
    }
    first_index_num++;
    reclaimable_slots++;
  }
  void pop_back() {
    if (last_index_vec.has_value()) {
      if (*last_index_vec == first_index_vec) {
        // Popped the last element! reset all circular behavior stuff and quit
        clear();
        return;
      }
      if (*last_index_vec == 0) {
        last_index_vec = std::nullopt;
      }
      else {
        (*last_index_vec)--;
      }
      reclaimable_slots++;
    }
    else {
      vec.pop_back();
    }
  }
  void emplace_back(T&& chunk) {
    // If we can still expand without realloc just do it
    if (vec.empty() || vec.size() < vec.capacity()) {
      vec.emplace_back(std::move(chunk));
    }
    // If vec is full but some slots are reclaimable, we do circular buffer style usage
    else if (reclaimable_slots > 0) {
      if (last_index_vec.has_value()) {
        (*last_index_vec)++;
        if (*last_index_vec == vec.size()) {
          *last_index_vec = 0;
        }
      }
      else {
        last_index_vec = 0;
      }
      vec[*last_index_vec] = std::move(chunk);
      reclaimable_slots--;
    }
    // max capacity and no reclaimable_slots, realloc unavoidable
    else {
      realloc_vec(static_cast<uint64_t>(std::ceil(static_cast<double>(vec.capacity()) * 1.5)));
      vec.emplace_back(std::move(chunk));
    }
  }
  void emplace_back(T& chunk) {
    T chunk_copy = chunk;
    emplace_back(std::move(chunk_copy));
  }

  void shrink_to_fit() {
    // we check that shrinking is worth it, at the very least we check that we shouldn't need to realloc again
    // if a few elements are added
    const auto current_capacity = vec.capacity();
    const auto target_capacity = static_cast<uint64_t>(std::ceil(static_cast<double>(current_capacity) / 1.5));
    if (target_capacity > size()) {
      realloc_vec(target_capacity);
    }
  }

  T& front() { return vec[first_index_vec]; }
  const T& front() const { return vec[first_index_vec]; }
  T& back() { return vec[get_last_index_vec()]; }
  const T& back() const { return vec[get_last_index_vec()]; }
};
static_assert(std::ranges::range<circular_vector<utility::ChunkEntry>>);

template <class T>
class circular_vector_debug {
  std::deque<T> instructions_deque;
  circular_vector<T*> instructions_vec;

  void check_instructions_equal(const T& instruction1, const T& instruction2) const {
    if (instruction1 != instruction2) {
      print_to_console("CHORI\n");
      throw std::runtime_error("La chorificacion");
    }
  }

  void check_iterators_equal(auto& deque_iter1, auto& vec_iter2) const {
    auto deque_begin = instructions_deque.begin();
    auto deque_end = instructions_deque.end();
    auto deque_cbegin = instructions_deque.cbegin();
    auto deque_cend = instructions_deque.cend();
    auto vec_begin = instructions_vec.begin();
    auto vec_end = instructions_vec.end();
    auto vec_cbegin = instructions_vec.cbegin();
    auto vec_cend = instructions_vec.cend();

    const bool deque_iter1_is_begin = deque_iter1 == deque_begin;
    const bool deque_iter1_is_cbegin = deque_iter1 == deque_cbegin;
    const bool deque_iter1_is_end = deque_iter1 == deque_end;
    const bool deque_iter1_is_cend = deque_iter1 == deque_cend;
    const bool vec_iter1_is_begin = vec_iter2 == vec_begin;
    const bool vec_iter1_is_cbegin = vec_iter2 == vec_cbegin;
    const bool vec_iter1_is_end = vec_iter2 == vec_end;
    const bool vec_iter1_is_cend = vec_iter2 == vec_cend;
    if (
      deque_iter1_is_begin && !vec_iter1_is_begin ||
      !deque_iter1_is_begin && vec_iter1_is_begin ||
      deque_iter1_is_cbegin && !vec_iter1_is_cbegin ||
      !deque_iter1_is_cbegin && vec_iter1_is_cbegin ||
      deque_iter1_is_end && !vec_iter1_is_end ||
      !deque_iter1_is_end && vec_iter1_is_end ||
      deque_iter1_is_cend && !vec_iter1_is_cend ||
      !deque_iter1_is_cend && vec_iter1_is_cend
      ) {
      print_to_console("CHORI\n");
      throw std::runtime_error("La chorificacion");
    }
    if (deque_iter1 == deque_end || deque_iter1 == deque_cend) return;
    auto& instruction1 = *deque_iter1;
    auto& instruction2 = **vec_iter2;
    check_instructions_equal(instruction1, instruction2);
  }

  void paranoid_check() const {
    auto deque_size = instructions_deque.size();
    auto vec_size = instructions_vec.size();
    if (deque_size != vec_size) {
      print_to_console("CHORI\n");
      throw std::runtime_error("La chorificacion");
    }

    if (deque_size == 0) return;

    {
      auto& instruction1 = instructions_deque.front();
      auto& instruction2 = *instructions_vec.front();
      check_instructions_equal(instruction1, instruction2);
    }
    {
      auto& instruction1 = instructions_deque.back();
      auto& instruction2 = *instructions_vec.back();
      check_instructions_equal(instruction1, instruction2);
    }
    {
      auto deque_iter_begin = instructions_deque.begin();
      auto vec_iter_begin = instructions_vec.begin();
      check_iterators_equal(deque_iter_begin, vec_iter_begin);
      check_instructions_equal(**vec_iter_begin, *instructions_vec.front());
    }
    {
      auto deque_iter_cbegin = instructions_deque.cbegin();
      auto vec_iter_cbegin = instructions_vec.cbegin();
      check_iterators_equal(deque_iter_cbegin, vec_iter_cbegin);
      check_instructions_equal(**vec_iter_cbegin, *instructions_vec.front());
    }
    {
      auto deque_iter_end = instructions_deque.end();
      auto vec_iter_end = instructions_vec.end();
      check_iterators_equal(deque_iter_end, vec_iter_end);
      check_instructions_equal(**(vec_iter_end - 1), *instructions_vec.back());
    }
    {
      auto deque_iter_cend = instructions_deque.cend();
      auto vec_iter_cend = instructions_vec.cend();
      check_iterators_equal(deque_iter_cend, vec_iter_cend);
      check_instructions_equal(**(vec_iter_cend - 1), *instructions_vec.back());
    }
  }

public:
  using size_type = typename std::vector<T>::size_type;

  explicit circular_vector_debug() = default;

  T& operator[](size_type pos) {
    paranoid_check();
    auto& instruction1 = instructions_deque[pos];
    auto& instruction2 = *instructions_vec[pos];
    check_instructions_equal(instruction1, instruction2);
    paranoid_check();
    return instruction1;
  }

  typename std::deque<T>::iterator begin() {
    paranoid_check();
    typename std::deque<T>::iterator deque_iter = instructions_deque.begin();
    auto vec_iter = instructions_vec.begin();
    check_iterators_equal(deque_iter, vec_iter);
    paranoid_check();
    return deque_iter;
  }
  typename std::deque<T>::iterator end() {
    paranoid_check();
    typename std::deque<T>::iterator deque_iter = instructions_deque.end();
    auto vec_iter = instructions_vec.end();
    check_iterators_equal(deque_iter, vec_iter);
    paranoid_check();
    return deque_iter;
  }
  typename std::deque<T>::const_iterator cbegin() const {
    paranoid_check();
    typename std::deque<T>::const_iterator deque_iter = instructions_deque.cbegin();
    auto vec_iter = instructions_vec.cbegin();
    check_iterators_equal(deque_iter, vec_iter);
    paranoid_check();
    return deque_iter;
  }
  typename std::deque<T>::const_iterator cend() const {
    paranoid_check();
    typename std::deque<T>::const_iterator deque_iter = instructions_deque.cend();
    auto vec_iter = instructions_vec.cend();
    check_iterators_equal(deque_iter, vec_iter);
    paranoid_check();
    return deque_iter;
  }

  size_type size() const {
    paranoid_check();
    auto deque_size = instructions_deque.size();
    auto vec_size = instructions_vec.size();
    if (deque_size != vec_size) {
      print_to_console("CHORI\n");
      throw std::runtime_error("La chorificacion");
    }
    paranoid_check();
    return deque_size;
  }
  void pop_front() {
    paranoid_check();
    size();
    auto& instruction1 = instructions_deque.front();
    auto& instruction2 = *instructions_vec.front();
    check_instructions_equal(instruction1, instruction2);
    instructions_deque.pop_front();
    instructions_vec.pop_front();
    size();
    paranoid_check();
  }
  void pop_back() {
    paranoid_check();
    size();
    auto& instruction1 = instructions_deque.back();
    auto& instruction2 = *instructions_vec.back();
    check_instructions_equal(instruction1, instruction2);
    instructions_deque.pop_back();
    instructions_vec.pop_back();
    size();
    paranoid_check();
  }
  void emplace_back(T&& instruction) {
    paranoid_check();
    size();
    instructions_deque.emplace_back(std::move(instruction));
    instructions_vec.emplace_back(&instructions_deque.back());
    size();

    back();
    paranoid_check();
  }
  void emplace_back(T& instruction) {
    T copy = instruction;
    emplace_back(std::move(copy));
  }

  void shrink_to_fit() {
    paranoid_check();
    size();
    instructions_deque.shrink_to_fit();
    instructions_vec.shrink_to_fit();
    size();
    paranoid_check();
  }

  T& front() {
    paranoid_check();
    auto& instruction1 = instructions_deque.front();
    auto& instruction2 = *instructions_vec.front();
    check_instructions_equal(instruction1, instruction2);
    paranoid_check();
    return instruction1;
  }
  const T& front() const {
    paranoid_check();
    auto& instruction1 = instructions_deque.front();
    auto& instruction2 = *instructions_vec.front();
    check_instructions_equal(instruction1, instruction2);
    paranoid_check();
    return instruction1;
  }
  T& back() {
    paranoid_check();
    auto& instruction1 = instructions_deque.back();
    auto& instruction2 = instructions_vec.back();
    check_instructions_equal(instruction1, *instruction2);
    paranoid_check();
    return instruction1;
  }
  const T& back() const {
    paranoid_check();
    auto& instruction1 = instructions_deque.back();
    auto& instruction2 = instructions_vec.back();
    check_instructions_equal(instruction1, *instruction2);
    paranoid_check();
    return instruction1;
  }
};

std::tuple<uint64_t, uint64_t> get_chunk_i_and_pos_for_offset(circular_vector<utility::ChunkEntry>& chunks, uint64_t offset) {
  static constexpr auto compare_offset = [](const utility::ChunkEntry& x, const utility::ChunkEntry& y) { return x.offset < y.offset; };

  const auto search_chunk = utility::ChunkEntry(offset);
  auto chunk_iter = std::ranges::lower_bound(std::as_const(chunks), search_chunk, compare_offset);
  if (chunk_iter == chunks.cend() || (*chunk_iter).offset > offset) --chunk_iter;

  uint64_t chunk_i = chunks.get_index(chunk_iter);
  const utility::ChunkEntry* chunk = &chunks[chunk_i];
  const uint64_t chunk_pos = offset - chunk->offset;

#ifndef NDEBUG
  if (chunk->offset + chunk_pos != offset || chunk_pos >= chunk->chunk_data->data.size()) {
    print_to_console("get_chunk_i_and_pos_for_offset() found incorrect chunk_i and pos at offset: " + std::to_string(offset) + "\n");
    throw std::runtime_error("La chorificacion");
  }
#endif

  return { chunk_i, chunk_pos };
}

class LZInstructionManager {
  circular_vector<utility::ChunkEntry>* chunks;
  circular_vector<LZInstruction> instructions;

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

  uint64_t check_backwards_extend_size(const LZInstruction& instruction, uint64_t current_offset, uint64_t earliest_allowed_offset) const {
    uint64_t extended_backwards_size = 0;
    const bool can_backwards_extend_instruction = instruction.offset > earliest_allowed_offset;
    if (!use_match_extension_backwards || instruction.type != LZInstructionType::COPY || instruction.size == 0 || !can_backwards_extend_instruction)
      return extended_backwards_size;

    auto prevInstruction_iter = instructions.cend();
    prevInstruction_iter = prevInstruction_iter - 1;
    const LZInstruction* prevInstruction = &*prevInstruction_iter;

    auto [instruction_chunk_i, instruction_chunk_pos] = get_chunk_i_and_pos_for_offset(*chunks, instruction.offset - 1);
    utility::ChunkEntry* instruction_chunk = &(*chunks)[instruction_chunk_i];
#ifndef NDEBUG
    if (instruction_chunk->offset + instruction_chunk_pos != instruction.offset - 1) {
      print_to_console("BACKWARD MATCH EXTENSION NEW INSTRUCTION OFFSET MISMATCH\n");
      throw std::runtime_error("La chorificacion");
    }
#endif

    uint64_t extended_instruction_offset = current_offset;
    uint64_t prevInstruction_eaten_size = 0;
    const bool can_backwards_extend_prevInstruction = extended_instruction_offset > earliest_allowed_offset;
    while (can_backwards_extend_prevInstruction) {
      // TODO: figure out why this happens, most likely some match extension is not properly cleaning up INSERTs that are shrunk into nothingness
      if (prevInstruction->size == 0) {
        if (prevInstruction_iter == instructions.cbegin()) {
          break;
        }
        --prevInstruction_iter;
        prevInstruction_eaten_size = 0;
        prevInstruction = &*prevInstruction_iter;
        continue;
      }

      auto [prevInstruction_chunk_i, prevInstruction_chunk_pos] = get_chunk_i_and_pos_for_offset(*chunks, extended_instruction_offset - 1);
      utility::ChunkEntry* prevInstruction_chunk = &(*chunks)[prevInstruction_chunk_i];
#ifndef NDEBUG
      if (prevInstruction_chunk->offset + prevInstruction_chunk_pos != extended_instruction_offset - 1) {
        print_to_console("BACKWARD MATCH EXTENSION PREVIOUS INSTRUCTION OFFSET MISMATCH\n");
        throw std::runtime_error("La chorificacion");
      }
#endif

      bool stop_matching = false;
      while (true) {
        const auto bytes_remaining_for_prevInstruction_to_earliest_allowed_offset = extended_instruction_offset - earliest_allowed_offset;
        // bytes_remaining_for_prevInstruction_backtrack is including the current byte, which is why we +1 to bytes_remaining_for_prevInstruction_to_earliest_allowed_offset,
        // otherwise this would be 0 when we are on the actual earliest_allowed_offset
        const auto bytes_remaining_for_prevInstruction_backtrack = std::min(
          bytes_remaining_for_prevInstruction_to_earliest_allowed_offset + 1,
          std::min(prevInstruction->size - prevInstruction_eaten_size, prevInstruction_chunk_pos + 1)
        );

        const auto prevInstruction_backtrack_data = std::span(
          prevInstruction_chunk->chunk_data->data.data() + prevInstruction_chunk_pos - (bytes_remaining_for_prevInstruction_backtrack - 1),
          bytes_remaining_for_prevInstruction_backtrack
        );
        const auto instruction_backtrack_data = std::span(instruction_chunk->chunk_data->data.data(), instruction_chunk_pos + 1);

        uint64_t matched_amt = find_identical_suffix_byte_count(prevInstruction_backtrack_data, instruction_backtrack_data);
        if (matched_amt == 0) {
          stop_matching = true;
          break;
        }

        extended_instruction_offset -= matched_amt;
        prevInstruction_eaten_size += matched_amt;
        extended_backwards_size += matched_amt;

        // Can't keep extending backwards, any previous data is disallowed (presumably to accomodate max matching distance)
        if (extended_instruction_offset < earliest_allowed_offset) {
          stop_matching = true;
          break;
        }
        if (extended_instruction_offset < earliest_allowed_offset) {
          stop_matching = true;
          break;
        }

        if (instruction_chunk_pos >= matched_amt) {
          instruction_chunk_pos -= matched_amt;
        }
        else if (instruction_chunk_i == 0 || instruction_chunk->offset == earliest_allowed_offset) {
          stop_matching = true;
          break;
        }
        else {
          instruction_chunk_i--;
          instruction_chunk = &(*chunks)[instruction_chunk_i];
          instruction_chunk_pos = instruction_chunk->chunk_data->data.size() - 1;
        }

        if (prevInstruction->size == prevInstruction_eaten_size) {
          if (prevInstruction_iter == instructions.cbegin()) {
            stop_matching = true;
            break;
          }
          --prevInstruction_iter;
          prevInstruction_eaten_size = 0;
          prevInstruction = &*prevInstruction_iter;
          break;
        }

        if (prevInstruction_chunk_pos >= matched_amt) {
          prevInstruction_chunk_pos -= matched_amt;
        }
        else if (prevInstruction_chunk_i == 0 || prevInstruction_chunk->offset == earliest_allowed_offset) {
          stop_matching = true;
          break;
        }
        else {
          prevInstruction_chunk_i--;
          prevInstruction_chunk = &(*chunks)[prevInstruction_chunk_i];
          prevInstruction_chunk_pos = prevInstruction_chunk->chunk_data->data.size() - 1;
        }
      }
      if (stop_matching) break;
    }

    return extended_backwards_size;
  }

  uint64_t check_forwards_extend_size(const LZInstruction& instruction, uint64_t earliest_allowed_offset) const {
    uint64_t extended_forwards_savings = 0;
    uint64_t extended_forwards_size = 0;
    const LZInstruction& prevInstruction = instructions.back();
    if (!use_match_extension || prevInstruction.type != LZInstructionType::COPY)
      return extended_forwards_size;

    uint64_t prevInstruction_offset = prevInstruction.offset + prevInstruction.size;

    const bool can_forward_extend_prevInstruction = prevInstruction_offset >= earliest_allowed_offset;
    if (can_forward_extend_prevInstruction) {
      auto [prevInstruction_chunk_i, prevInstruction_chunk_pos] = get_chunk_i_and_pos_for_offset(*chunks, prevInstruction_offset);
      utility::ChunkEntry* prevInstruction_chunk = &(*chunks)[prevInstruction_chunk_i];

      auto [instruction_chunk_i, instruction_chunk_pos] = get_chunk_i_and_pos_for_offset(*chunks, instruction.offset);
      utility::ChunkEntry* instruction_chunk = &(*chunks)[instruction_chunk_i];
#ifndef NDEBUG
      if (instruction_chunk->offset + instruction_chunk_pos != instruction.offset || instruction_chunk_pos >= instruction_chunk->chunk_data->data.size()) {
        print_to_console("HERE WE GO LA CAGAMOU!\n");
        throw std::runtime_error("La chorificacion");
      }
#endif

      while (extended_forwards_size < instruction.size) {
        const auto prevInstruction_extend_data = std::span(
          prevInstruction_chunk->chunk_data->data.data() + prevInstruction_chunk_pos,
          prevInstruction_chunk->chunk_data->data.size() - prevInstruction_chunk_pos
        );
        const auto instruction_extend_data = std::span(
          instruction_chunk->chunk_data->data.data() + instruction_chunk_pos,
          std::min(instruction_chunk->chunk_data->data.size() - instruction_chunk_pos, instruction.size - extended_forwards_size)
        );

        uint64_t matched_amt = find_identical_prefix_byte_count(prevInstruction_extend_data, instruction_extend_data);
        if (matched_amt == 0) break;

        extended_forwards_size += matched_amt;
        //prevInstruction.size += matched_amt;
        //instruction.offset += matched_amt;
        //instruction.size -= matched_amt;
        if (instruction.type == LZInstructionType::INSERT) extended_forwards_savings += matched_amt;

        prevInstruction_chunk_pos += matched_amt;
        if (prevInstruction_chunk_pos == prevInstruction_chunk->chunk_data->data.size()) {
          prevInstruction_chunk_i++;
          prevInstruction_chunk = &(*chunks)[prevInstruction_chunk_i];
          prevInstruction_chunk_pos = 0;
        }

        if (instruction.size == extended_forwards_size) {
          break;
        }

        instruction_chunk_pos += matched_amt;
        if (instruction_chunk_pos == instruction_chunk->chunk_data->data.size()) {
          instruction_chunk_i++;
          instruction_chunk = &(*chunks)[instruction_chunk_i];
          instruction_chunk_pos = 0;
        }
      }
    }
    return extended_forwards_size;
  }

public:
  explicit LZInstructionManager(circular_vector<utility::ChunkEntry>* _chunks, bool _use_match_extension_backwards, bool _use_match_extension, std::ostream* _ostream)
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

  void addInstruction(LZInstruction&& instruction, uint64_t current_offset, bool verify, std::optional<uint64_t> _earliest_allowed_offset = std::nullopt) {
    uint64_t earliest_allowed_offset = _earliest_allowed_offset.has_value() ? *_earliest_allowed_offset : 0;
#ifndef NDEBUG
    if (instruction.type == LZInstructionType::INSERT && instruction.offset != current_offset) {
      print_to_console("INSERT LZInstruction added is not at current offset!");
      throw std::runtime_error("La chorificacion");
    }
#endif
    if (instructions.size() == 0) {
      instructions.emplace_back(std::move(instruction));
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
        std::fstream verify_file{};
        verify_file.open(R"(C:\Users\Administrator\Documents\dedup_proj\Datasets\LNX-IMG\LNX-IMG.tar)", std::ios_base::in | std::ios_base::binary);

        verify_buffer_orig_data.resize(prevInstruction->size);
        verify_buffer_instruction_data.resize(prevInstruction->size);
        // Read prevInstruction original data
        verify_file.seekg(current_offset - prevInstruction->size);
        verify_file.read(reinterpret_cast<char*>(verify_buffer_orig_data.data()), prevInstruction->size);
        // Read data according to prevInstruction
        verify_file.seekg(prevInstruction->offset);
        verify_file.read(reinterpret_cast<char*>(verify_buffer_instruction_data.data()), prevInstruction->size);
        // Ensure data matches
        if (std::memcmp(verify_buffer_orig_data.data(), verify_buffer_instruction_data.data(), prevInstruction->size) != 0) {
          print_to_console("Error while verifying addInstruction prevInstruction at offset " + std::to_string(current_offset) + "\n");
          throw std::runtime_error("La chorificacion");
        }

        verify_buffer_orig_data.resize(instruction.size);
        verify_buffer_instruction_data.resize(instruction.size);
        // Read instruction original data
        verify_file.seekg(current_offset);
        verify_file.read(reinterpret_cast<char*>(verify_buffer_orig_data.data()), instruction.size);
        // Read data according to instruction
        verify_file.seekg(instruction.offset);
        verify_file.read(reinterpret_cast<char*>(verify_buffer_instruction_data.data()), instruction.size);
        // Ensure data matches
        if (std::memcmp(verify_buffer_orig_data.data(), verify_buffer_instruction_data.data(), instruction.size) != 0) {
          print_to_console("Error while verifying addInstruction instruction at offset " + std::to_string(current_offset) + "\n");
          throw std::runtime_error("La chorificacion");
        }
      }

      const uint64_t original_instruction_size = instruction.size;
      uint64_t extended_forwards_savings = 0;
      uint64_t extended_backwards_savings = 0;

      uint64_t forwards_extend_possible_size = check_forwards_extend_size(instruction, earliest_allowed_offset);
      const bool is_forwards_extend_eats_prevInstruction = forwards_extend_possible_size == instruction.size;
      if (is_forwards_extend_eats_prevInstruction) {
        prevInstruction->size += forwards_extend_possible_size;
        instruction.offset += forwards_extend_possible_size;
        instruction.size -= forwards_extend_possible_size;
        if (instruction.type == LZInstructionType::INSERT) {
          extended_forwards_savings += forwards_extend_possible_size;
        }
      }
      else {
        uint64_t backwards_extend_possible_size = check_backwards_extend_size(instruction, current_offset, earliest_allowed_offset);
        const bool is_backwards_extend_eats_prevInstruction = backwards_extend_possible_size >= prevInstruction->size;

        if (is_backwards_extend_eats_prevInstruction || prevInstruction->type == LZInstructionType::INSERT) {
          while (backwards_extend_possible_size > 0) {
            prevInstruction = &instructions.back();

            uint64_t prevInstruction_reduced_size = 0;
            const bool is_prevInstruction_INSERT = prevInstruction->type == LZInstructionType::INSERT;
            if (backwards_extend_possible_size >= prevInstruction->size) {
              prevInstruction_reduced_size = prevInstruction->size;
              instructions.pop_back();
            }
            else if (!is_prevInstruction_INSERT) {
              // If prevInstruction is a COPY, but we can't eat it completely, we skip it.
              // It's mostly pointless to reduce it, and that prior instruction is likely for data that is
              // more distant, which might make it more compressible.
              break;
            }
            else {
              prevInstruction_reduced_size = backwards_extend_possible_size;
              prevInstruction->size -= backwards_extend_possible_size;
            }

            backwards_extend_possible_size -= prevInstruction_reduced_size;
            if (is_prevInstruction_INSERT) {
              extended_backwards_savings += prevInstruction_reduced_size;
            }

            instruction.offset -= prevInstruction_reduced_size;
            instruction.size += prevInstruction_reduced_size;
          }
          prevInstruction = &instructions.back();
        }
        else if (forwards_extend_possible_size > 0) {
          prevInstruction->size += forwards_extend_possible_size;
          instruction.offset += forwards_extend_possible_size;
          instruction.size -= forwards_extend_possible_size;
          if (instruction.type == LZInstructionType::INSERT) {
            extended_forwards_savings += forwards_extend_possible_size;
          }
        }
      }

      if (verify) {
        std::fstream verify_file{};
        verify_file.open(R"(C:\Users\Administrator\Documents\dedup_proj\Datasets\LNX-IMG\LNX-IMG.tar)", std::ios_base::in | std::ios_base::binary);
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
          print_to_console("Error while verifying addInstruction at offset " + std::to_string(current_offset) + "\n");
          throw std::runtime_error("La chorificacion");
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
      instructions.emplace_back(std::move(instruction));
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

  void dump(std::istream& istream, bool verify_copies, std::optional<uint64_t> up_to_offset = std::nullopt, bool flush = false) {
    std::vector<char> buffer;
    std::vector<char> verify_buffer;

    auto prev_outputted_up_to_offset = outputted_up_to_offset;
    while (instructions.size() != 0) {
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
          uint64_t to_read_from_chunk = std::min(instruction.size - written, instruction_chunk.chunk_data->data.size() - instruction_chunk_pos);
          std::copy_n(instruction_chunk.chunk_data->data.data() + instruction_chunk_pos, to_read_from_chunk, buffer.data() + written);
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
            print_to_console("Error while verifying outputted match at offset " + std::to_string(outputted_up_to_offset) + "\n");
            print_to_console("With prev offset " + std::to_string(prev_outputted_up_to_offset) + "\n");
            throw std::runtime_error("La chorificacion");
          }

          /*
          auto* buffer_ptr = buffer.data();
          auto* verify_buffer_ptr = verify_buffer.data();

          auto [copy_instruction_chunk_i, copy_instruction_chunk_pos] = get_chunk_i_and_pos_for_offset(*chunks, instruction.offset);
          auto [curr_instruction_chunk_i, curr_instruction_chunk_pos] = get_chunk_i_and_pos_for_offset(*chunks, outputted_up_to_offset);
          auto* copy_instruction_chunk = &(*chunks)[copy_instruction_chunk_i];
          auto* curr_instruction_chunk = &(*chunks)[curr_instruction_chunk_i];

          uint64_t remaining_size = instruction.size;
          while (remaining_size > 0) {
            if (copy_instruction_chunk->chunk_data->data.size() == copy_instruction_chunk_pos) {
              copy_instruction_chunk_i++;
              copy_instruction_chunk_pos = 0;
              copy_instruction_chunk = &(*chunks)[copy_instruction_chunk_i];
            }
            if (curr_instruction_chunk->chunk_data->data.size() == curr_instruction_chunk_pos) {
              curr_instruction_chunk_i++;
              curr_instruction_chunk_pos = 0;
              curr_instruction_chunk = &(*chunks)[curr_instruction_chunk_i];
            }
            auto cmp_size = std::min<uint64_t>(
              copy_instruction_chunk->chunk_data->data.size() - copy_instruction_chunk_pos,
              curr_instruction_chunk->chunk_data->data.size() - curr_instruction_chunk_pos
            );
            cmp_size = std::min(cmp_size, remaining_size);
            const auto copy_chunk_data = copy_instruction_chunk->chunk_data->data.data() + copy_instruction_chunk_pos;
            const auto curr_chunk_data = curr_instruction_chunk->chunk_data->data.data() + curr_instruction_chunk_pos;

            if (std::memcmp(curr_chunk_data, verify_buffer_ptr, cmp_size) != 0) {
              print_to_console("ERROR ON CURR CHUNK DATA!\n");
              print_to_console("With prev offset " + std::to_string(prev_outputted_up_to_offset) + "\n");
              throw std::runtime_error("La chorificacion");
            }
            verify_buffer_ptr += cmp_size;
            if (std::memcmp(copy_chunk_data, buffer_ptr, cmp_size) != 0) {
              print_to_console("ERROR ON COPY CHUNK DATA!\n");
              print_to_console("With prev offset " + std::to_string(prev_outputted_up_to_offset) + "\n");
              throw std::runtime_error("La chorificacion");
            }
            buffer_ptr += cmp_size;

            if (std::memcmp(copy_chunk_data, curr_chunk_data, cmp_size) != 0) {
              print_to_console("Error while verifying outputted match with chunk data at offset " + std::to_string(outputted_up_to_offset) + "\n");
              print_to_console("With prev offset " + std::to_string(prev_outputted_up_to_offset) + "\n");
              throw std::runtime_error("La chorificacion");
            }
            remaining_size -= cmp_size;
            copy_instruction_chunk_pos += cmp_size;
            curr_instruction_chunk_pos += cmp_size;
          }
          */
        }
      }
      prev_outputted_up_to_offset = outputted_up_to_offset;
      outputted_up_to_offset += instruction.size;
      outputted_lz_instructions++;
    }

    instructions.shrink_to_fit();
    if (flush) {
      bit_output_stream.flush();
      output_stream.flush();
    }
  }
};

void select_cut_point_candidates(
  std::vector<CutPointCandidate>& new_cut_point_candidates,
  std::vector<std::vector<uint32_t>>& new_cut_point_candidates_features,
  std::deque<utility::ChunkEntry>& process_pending_chunks,
  uint64_t last_used_cut_point,
  uint64_t segment_start_offset,
  std::vector<uint8_t>& segment_data,
  std::vector<uint8_t>& prev_segment_remaining_data,
  uint32_t min_size,
  uint32_t max_size,
  bool segments_eof,
  bool use_feature_extraction
) {
  if (!segments_eof && !new_cut_point_candidates.empty() && new_cut_point_candidates.back().type == CutPointCandidateType::EOF_CUT) {
    new_cut_point_candidates.pop_back();
    if (use_feature_extraction) {
      new_cut_point_candidates_features.pop_back();
    }
  }

  auto make_chunk = [&] (uint64_t candidate_pos, uint64_t prev_point_offset, uint64_t segment_data_pos, uint64_t cut_point_pos) {
    process_pending_chunks.emplace_back(utility::ChunkEntry(prev_point_offset));
    auto& new_pending_chunk = process_pending_chunks.back();

    if (!prev_segment_remaining_data.empty()) {
      const auto chunk_size = prev_segment_remaining_data.size() + cut_point_pos - segment_data_pos;
      new_pending_chunk.chunk_data->data.resize(chunk_size);
      std::copy_n(prev_segment_remaining_data.data(), prev_segment_remaining_data.size(), new_pending_chunk.chunk_data->data.data());
      std::copy_n(segment_data.data() + segment_data_pos, cut_point_pos - segment_data_pos, new_pending_chunk.chunk_data->data.data() + prev_segment_remaining_data.size());
      prev_segment_remaining_data.clear();
    }
    else {
      new_pending_chunk.chunk_data->data.resize(cut_point_pos - segment_data_pos);
      std::copy_n(segment_data.data() + segment_data_pos, cut_point_pos - segment_data_pos, new_pending_chunk.chunk_data->data.data());
    }
    new_pending_chunk.chunk_data->data.shrink_to_fit();

    if (use_feature_extraction) {
      if (new_cut_point_candidates_features.size() >= candidate_pos + 1) {
        auto& cut_point_candidate_features = new_cut_point_candidates_features[candidate_pos];
        if (!cut_point_candidate_features.empty()) {
          // Takes 4 features (32bit(4byte) fingerprints, so 4 of them is 16bytes) and hash them into a single SuperFeature (seed used arbitrarily just because it needed one)
          new_pending_chunk.chunk_data->super_features = {
            XXH32(cut_point_candidate_features.data(), 16, constants::CDS_SAMPLING_MASK),
            XXH32(cut_point_candidate_features.data() + 4, 16, constants::CDS_SAMPLING_MASK),
            XXH32(cut_point_candidate_features.data() + 8, 16, constants::CDS_SAMPLING_MASK),
            XXH32(cut_point_candidate_features.data() + 12, 16, constants::CDS_SAMPLING_MASK)
          };
          new_pending_chunk.chunk_data->feature_sampling_failure = false;
        }
      }
    }
  };

  uint64_t segment_data_pos = 0;
  auto last_cut_point = process_pending_chunks.empty() ? last_used_cut_point : process_pending_chunks.back().offset + process_pending_chunks.back().chunk_data->data.size();
  // If the last used cut point is on data already on this segment (should be within the 31/window_size - 1 bytes extension of the previous segment)
  // we start from that pos on this segment
  if (last_cut_point > segment_start_offset) {
    segment_data_pos = last_cut_point - segment_start_offset;
  }

  for (uint64_t i = 0; i < new_cut_point_candidates.size(); i++) {
    auto& cut_point_candidate = new_cut_point_candidates[i];
    last_cut_point = process_pending_chunks.empty() ? last_used_cut_point : process_pending_chunks.back().offset + process_pending_chunks.back().chunk_data->data.size();

    const auto adjusted_cut_point_candidate = segment_start_offset + cut_point_candidate.offset;
    while (adjusted_cut_point_candidate > last_cut_point + max_size) {
      // if the last_cut_point is on the previous segment some data might have come from the prev_segment_remaining_data so it's
      // not counting towards our segment data position
      const auto chunk_size_in_segment = last_cut_point < segment_start_offset ? max_size - (segment_start_offset - last_cut_point) : max_size;
      // TODO: these chunks will share features, which is not right, need to figure out a solution
      make_chunk(i, last_cut_point, segment_data_pos, segment_data_pos + chunk_size_in_segment);
      last_cut_point = last_cut_point + max_size;
      segment_data_pos += chunk_size_in_segment;
    }

    // TODO: we are discarding the features (if we are computing them) along with the rejected cut point candidate, we need to roll those over
    if (
      adjusted_cut_point_candidate < last_cut_point + min_size &&
      // If this is the segment at EOF and also the last candidate, we can't skip it, as there won't be any future cut point to complete the chunk
      !(segments_eof && i == new_cut_point_candidates.size() - 1)
    ) {
      continue;
    }

    make_chunk(i, last_cut_point, segment_data_pos, cut_point_candidate.offset);
    segment_data_pos = cut_point_candidate.offset;
  }

  // We might have reached the end of the cut point candidates but there is enough data at the end of the segment that we need to enforce
  // the max_size for chunks, as we did between cut point candidates, just between the actual last candidate and the end of the segment this time
  last_cut_point = process_pending_chunks.empty() ? last_used_cut_point : process_pending_chunks.back().offset + process_pending_chunks.back().chunk_data->data.size();
  const auto segment_end_pos = segment_start_offset + segment_data.size();
  while (segment_end_pos > last_cut_point + max_size) {
    // if the last_cut_point is on the previous segment some data might have come from the prev_segment_remaining_data so it's
    // not counting towards our segment data position
    const auto chunk_size_in_segment = last_cut_point < segment_start_offset ? max_size - (segment_start_offset - last_cut_point) : max_size;
    // TODO: these chunks will share features, which is not right, need to figure out a solution
    make_chunk(0, last_cut_point, segment_data_pos, segment_data_pos + chunk_size_in_segment);
    last_cut_point = last_cut_point + max_size;
    segment_data_pos += chunk_size_in_segment;
  }

  // If there is more unused data at the end of the segment than the window_size - 1 bytes of the extension, we save that data
  // as it will need to be used for the next chunk
  {
    const auto in_segment_pos = process_pending_chunks.back().offset + process_pending_chunks.back().chunk_data->data.size() - segment_start_offset;
    const auto segment_data_tail_len = segment_data.size() - in_segment_pos;
    if (segment_data_tail_len > 31) {
      prev_segment_remaining_data.resize(segment_data_tail_len - 31);
      prev_segment_remaining_data.shrink_to_fit();
      std::copy_n(segment_data.data() + in_segment_pos, segment_data_tail_len - 31, prev_segment_remaining_data.data());
    }
  }
}

int main(int argc, char* argv[]) {
#ifndef NDEBUG
  get_char_with_echo();
#endif
  std::string file_path{ argv[1] };
  auto file_size = file_path == "-" ? 0 : std::filesystem::file_size(file_path);
  if (argc > 3) {
    print_to_console("Invalid command line\n");
    return 1;
  }
  bool do_decompression = false;
  if (argc == 3) {
    std::string third_arg{ argv[2] };
    if (!third_arg.starts_with("-d=")) {
      do_decompression = true;
    }
  }
  if (do_decompression) {  // decompress
    auto file_stream = std::fstream(file_path, std::ios::in | std::ios::binary);
    if (!file_stream.is_open()) {
      print_to_console("Can't read file\n");
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
        print_to_console("Something wrong bad during decompression\n");
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
    print_to_console("Decompression finished in " + std::to_string(std::chrono::duration_cast<std::chrono::seconds>(decompress_end_time - decompress_start_time).count()) + " seconds!\n");
    return 0;
  }

  uint32_t avg_size = 512;
  uint32_t min_size = avg_size / 4;
  uint32_t max_size = avg_size * 8;

  if (constants::MINIMUM_MIN > min_size || min_size > constants::MINIMUM_MAX) throw std::runtime_error("Bad minimum size");
  if (constants::AVERAGE_MIN > avg_size || avg_size > constants::AVERAGE_MAX) throw std::runtime_error("Bad average size");
  if (constants::MAXIMUM_MIN > max_size || max_size > constants::MAXIMUM_MAX) throw std::runtime_error("Bad maximum size");

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

  std::unordered_map<std::bitset<64>, circular_vector<uint64_t>> known_hashes{};  // Find chunk pos by hash

  uint64_t chunk_i = 0;
  uint64_t last_reduced_chunk_i = 0;
  std::vector<int32_t> pending_chunks_indexes(batch_size, 0);
  std::vector<uint8_t> pending_chunk_data(batch_size * max_size, 0);
  std::unordered_map<std::bitset<64>, uint64_t> simhashes_dict{};

  // Find chunk_i that has a given SuperFeature
  std::unordered_map<uint32_t, std::list<uint64_t>> superfeatures_dict{};

  const auto cdc_thread_count = std::thread::hardware_concurrency();

  const uint32_t simhash_chunk_size = std::max<uint32_t>(16, min_size / 8);
  const uint32_t max_allowed_dist = 32;
  const uint32_t delta_mode = 2;
  // Don't even bother saving chunk as delta chunk if savings are too little and overhead will probably negate them
  const uint64_t min_delta_saving = std::min(min_size * 2, avg_size);

  std::tuple<std::bitset<64>, std::vector<uint32_t>> (*simhash_func)(uint8_t * data, uint32_t data_len, uint32_t minichunk_size) = nullptr;
  DeltaEncodingResult(*simulate_delta_encoding_func)(const utility::ChunkData& chunk, const utility::ChunkData& similar_chunk, uint32_t minichunk_size) = nullptr;
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
  // If enabled then DupAdj will attempt other chunks with the same hashes as the adjacent ones in an attempt to find other possible similar chunks
  const bool exhaustive_dupadj = false;
  const bool use_generalized_resemblance_detection = true;
  const bool use_feature_extraction = true;
  // Because of data locality, chunks close to the current one might be similar to it, we attempt to find the most similar out of the previous ones in this window,
  // and use it if it's good enough
  const int resemblance_attempt_window = 100;
  const bool use_match_extension = true;
  const bool use_match_extension_backwards = true;

  // if false, the first similar block found by any method will be used and other methods won't run, if true, all methods will run and the most similar block found will be used
  const bool attempt_multiple_delta_methods = true;
  // Only try to delta encode against the candidate chunk with the lesser hamming distance and closest to the current chunk
  // (because of data locality it's likely to be the best)
  const bool only_try_best_delta_match = false;
  const bool only_try_min_dist_delta_matches = false;
  const bool keep_first_delta_match = true;
  const bool is_any_delta_on = use_dupadj || use_dupadj_backwards || use_feature_extraction || use_generalized_resemblance_detection || resemblance_attempt_window > 0;

  const bool verify_delta_coding = false;
  const bool verify_dumps = false;
  const bool verify_addInstruction = false;
  const bool verify_chunk_offsets = false;

  const std::optional<uint64_t> dictionary_size_limit = argc == 3
    ? std::optional(static_cast<uint64_t>(std::stoi(std::string(argv[2] + 3))) * 1024 * 1024)
    : std::nullopt;
  uint64_t dictionary_size_used = 0;
  uint64_t first_non_out_of_range_chunk_i = 0;

  circular_vector<utility::ChunkEntry> chunks{};

  auto file_stream = std::ifstream();
  if (file_path == "-") {
    set_std_handle_binary_mode(StdHandles::STDIN_HANDLE);
    reinterpret_cast<std::istream*>(&file_stream)->rdbuf(std::cin.rdbuf());
  }
  else {
    file_stream.open(file_path, std::ios::in | std::ios::binary);
    if (!file_stream.is_open()) {
      print_to_console("Can't read file\n");
      return 1;
    }
  }
  
  auto wrapped_file = IStreamWrapper(&file_stream);

  std::optional<uint64_t> similarity_locality_anchor_i{};

  auto verify_file_stream = std::fstream();
  if (file_path != "-") {
    verify_file_stream.open(file_path, std::ios::in | std::ios::binary);
  }
  std::vector<uint8_t> verify_buffer{};
  std::vector<uint8_t> verify_buffer_delta{};

  auto dump_file = std::ofstream(/*file_path + ".ddp", std::ios::out | std::ios::binary | std::ios::trunc*/);
  set_std_handle_binary_mode(StdHandles::STDOUT_HANDLE);
  reinterpret_cast<std::ostream*>(&dump_file)->rdbuf(std::cout.rdbuf());
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

  const auto calc_simhash_if_needed = [&simhash_func, &simhashing_execution_time_ns, &simhash_chunk_size](utility::ChunkEntry& chunk) {
    if (!chunk.chunk_data->minichunks.empty()) return;
    const auto simhashing_start_time = std::chrono::high_resolution_clock::now();
    std::tie(chunk.chunk_data->lsh, chunk.chunk_data->minichunks) = simhash_func(chunk.chunk_data->data.data(), chunk.chunk_data->data.size(), simhash_chunk_size);
    const auto simhashing_end_time = std::chrono::high_resolution_clock::now();
    simhashing_execution_time_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(simhashing_end_time - simhashing_start_time).count();
  };

  // SS-CDC style data segmenting to enable chunk invariant multithreading
  const uint64_t segment_size = 10 * 1024 * 1024;
  const auto segment_batch_size = segment_size * cdc_thread_count;
  std::vector<uint8_t> prev_segment_remaining_data{};
  uint64_t segment_start_offset = 0;
  std::deque<utility::ChunkEntry> process_pending_chunks{};

  uint64_t current_offset = 0;

  auto find_cdc_cut_candidates_in_thread = [min_size, avg_size, max_size]
  (std::vector<uint8_t>&& segment_data, uint64_t segment_start_offset, bool is_eof_segment, bool is_first_segment) {
    std::vector<CutPointCandidate> cut_point_candidates;
    std::vector<std::vector<uint32_t>> cut_point_candidates_features;
    if (use_feature_extraction) {
      std::tie(cut_point_candidates, cut_point_candidates_features) = find_cdc_cut_candidates<true>(segment_data, min_size, avg_size, max_size, is_first_segment);
    }
    else {
      std::tie(cut_point_candidates, std::ignore) = find_cdc_cut_candidates<false>(segment_data, min_size, avg_size, max_size, is_first_segment);
    }
    return std::tuple(std::move(cut_point_candidates), std::move(cut_point_candidates_features), std::move(segment_data), segment_start_offset, is_eof_segment);
  };

  std::deque<std::future<decltype(std::function{find_cdc_cut_candidates_in_thread})::result_type>> cdc_candidates_futures;
  bool is_first_segment = true;

  std::future<std::vector<uint8_t>> load_next_segment_batch_future;
  auto load_next_segment_batch = [&wrapped_file, segment_batch_size]
  (std::array<uint8_t, 31>&& _prev_segment_extend_data) -> std::vector<uint8_t> {
    std::array<uint8_t, 31> prev_segment_extend_data = std::move(_prev_segment_extend_data);
    std::vector<uint8_t> new_segment_batch_data;
    new_segment_batch_data.resize(segment_batch_size + 31);  // +31 bytes (our GEAR window size) at the end so it overlaps with next segment as described in SS-CDC

    // We get the 31byte extension to be at the start of the new segment data
    std::copy_n(prev_segment_extend_data.data(), 31, new_segment_batch_data.data());

    // And attempt to load remaining data for the new segments including next segment extension
    wrapped_file.read(new_segment_batch_data.data() + 31, segment_batch_size);
    new_segment_batch_data.resize(31 + wrapped_file.gcount());

    new_segment_batch_data.shrink_to_fit();
    return new_segment_batch_data;
  };

  auto launch_cdc_threads =
  [&cdc_candidates_futures, &find_cdc_cut_candidates_in_thread, &load_next_segment_batch_future, &load_next_segment_batch,
  cdc_thread_count, segment_size, segment_batch_size, &segment_start_offset, &is_first_segment, &chunk_generator_execution_time_ns]
  (const std::vector<uint8_t>& segment_batch_data) {
    auto chunk_generator_start_time = std::chrono::high_resolution_clock::now();
    auto segment_batch_data_span = std::span(segment_batch_data);
    for (uint64_t i = 0; i < cdc_thread_count; i++) {
      if (segment_batch_data_span.empty()) break;
      auto current_segment_data = std::vector<uint8_t>();
      current_segment_data.resize(std::min(segment_batch_data_span.size(), segment_size + 31));
      current_segment_data.shrink_to_fit();
      std::copy_n(segment_batch_data_span.data(), current_segment_data.size(), current_segment_data.data());
      bool segments_eof = current_segment_data.size() != segment_size + 31;

      const auto span_advance_size = std::min(current_segment_data.size(), segment_size);

      cdc_candidates_futures.emplace_back(
        globalTaskPool.addTask(
          [&find_cdc_cut_candidates_in_thread, current_segment_data = std::move(current_segment_data), segments_eof, segment_start_offset, &is_first_segment]() mutable {
            return find_cdc_cut_candidates_in_thread(std::move(current_segment_data), segment_start_offset, segments_eof, is_first_segment);
          }
        )
      );
      segment_start_offset += span_advance_size;
      segment_batch_data_span = std::span(segment_batch_data_span.data() + span_advance_size, segment_batch_data_span.size() - span_advance_size);
      is_first_segment = false;
    }
    auto chunk_generator_end_time = std::chrono::high_resolution_clock::now();
    chunk_generator_execution_time_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(chunk_generator_end_time - chunk_generator_start_time).count();

    // If not at EOF we already start loading the next segment batch data
    if (segment_batch_data.size() == segment_batch_size + 31) {
      std::array<uint8_t, 31> segment_extend_data;
      std::copy_n(segment_batch_data.data() + segment_batch_size, 31, segment_extend_data.data());
      load_next_segment_batch_future = globalTaskPool.addTask(
        [segment_extend_data = std::move(segment_extend_data), &load_next_segment_batch]() mutable {
          return load_next_segment_batch(std::move(segment_extend_data));
        }
      );
    }
  };

  auto get_more_pending_chunks_from_cdc_futures =
  [&cdc_candidates_futures, &process_pending_chunks, &prev_segment_remaining_data, &current_offset, min_size, max_size]
  (bool force_wait) {
    if (cdc_candidates_futures.size() > 0) {
      auto candidate_future_status = cdc_candidates_futures.front().wait_for(std::chrono::nanoseconds::zero());
      if (force_wait || candidate_future_status == std::future_status::ready || process_pending_chunks.empty()) {
        auto future = std::move(cdc_candidates_futures.front());
        auto [
          new_cut_point_candidates,
          new_cut_point_candidates_features,
          segment_data,
          curr_segment_start_offset,
          segments_eof
        ] = future.get();
        cdc_candidates_futures.pop_front();

        select_cut_point_candidates(
          new_cut_point_candidates,
          new_cut_point_candidates_features,
          process_pending_chunks,
          current_offset,
          curr_segment_start_offset,
          segment_data,
          prev_segment_remaining_data,
          min_size,
          max_size,
          segments_eof && cdc_candidates_futures.size() == 0,
          use_feature_extraction
        );
      }
    }
  };

  auto total_runtime_start_time = std::chrono::high_resolution_clock::now();

  {
    std::vector<uint8_t> segment_batch_data;
    segment_batch_data.resize(segment_size * cdc_thread_count + 31);  // +31 bytes (our GEAR window size) at the end so it overlaps with next segment as described in SS-CDC
    wrapped_file.read(segment_batch_data.data(), segment_batch_size + 31);
    segment_batch_data.resize(wrapped_file.gcount());
    segment_batch_data.shrink_to_fit();

    launch_cdc_threads(segment_batch_data);
    get_more_pending_chunks_from_cdc_futures(true);
  }

  while (!process_pending_chunks.empty()) {
    if (chunk_i > 0 && dictionary_size_limit.has_value() && dictionary_size_used > *dictionary_size_limit) {
      const auto prev_first_non_out_of_range_chunk_i = first_non_out_of_range_chunk_i;
    
      while (dictionary_size_used > *dictionary_size_limit) {
        auto& previous_chunk = chunks[first_non_out_of_range_chunk_i];

        // Remove the out of range chunks from the indexes so they can't be used for dedup or delta
        auto& hash_list = known_hashes[previous_chunk.chunk_data->hash];
        if (hash_list.size() == 1) {
          known_hashes.erase(previous_chunk.chunk_data->hash);

          // The chunk has no duplicate left so by removing it we are effectively taking it out of the dictionary.
          dictionary_size_used -= previous_chunk.chunk_data->data.size();
        }
        else {
          hash_list.pop_front();
          hash_list.shrink_to_fit();
        }

        if (use_generalized_resemblance_detection) {
          const auto base = hamming_base(previous_chunk.chunk_data->lsh);
          if (simhashes_dict[base] <= first_non_out_of_range_chunk_i) {
            simhashes_dict.erase(base);
          }
        }

        if (use_feature_extraction) {
          for (auto& sf : previous_chunk.chunk_data->super_features) {
            // SuperFeature index might have already been deleted, so we need to do proper check here
            auto sf_list_iter = superfeatures_dict.find(sf);
            if (sf_list_iter != superfeatures_dict.end()) {
              auto& sf_list = sf_list_iter->second;
              if (sf_list.front() == first_non_out_of_range_chunk_i) {
                if (sf_list.size() == 1) {
                  superfeatures_dict.erase(sf);
                }
                else {
                  sf_list.pop_front();
                }
              }
            }
          }
        }

        first_non_out_of_range_chunk_i++;
      }

      uint64_t earliest_allowed_offset = chunks[first_non_out_of_range_chunk_i].offset;
      if (!output_disabled) lz_manager.dump(verify_file_stream, verify_dumps, earliest_allowed_offset);

      // Actually remove the chunk's data, we had to wait until after dump to do this because dumping required the data to still be there
      for (uint64_t i = prev_first_non_out_of_range_chunk_i; i < first_non_out_of_range_chunk_i; i++) {
        // Delete its data
        chunks[i].chunk_data = nullptr;
        chunks.pop_front();
      }

      chunks.shrink_to_fit();
    }

    chunks.emplace_back(std::move(process_pending_chunks.front()));
    process_pending_chunks.pop_front();
    auto& chunk = chunks.back();

    std::optional<uint64_t> prev_similarity_locality_anchor_i = similarity_locality_anchor_i;
    similarity_locality_anchor_i = std::nullopt;
    if (chunk_i % 50000 == 0) print_to_console("\n%" + std::to_string((static_cast<float>(current_offset) / file_size) * 100) + "\n");

    std::span<uint8_t> data_span = chunk.chunk_data->data;
    total_size += data_span.size();

    if (verify_chunk_offsets) {
      std::vector<uint8_t> verify_buffer_orig_data{};
      std::fstream verify_file{};
      verify_file.open(R"(C:\Users\Administrator\Documents\dedup_proj\Datasets\LNX-IMG\LNX-IMG.tar)", std::ios_base::in | std::ios_base::binary);

      verify_buffer_orig_data.resize(data_span.size());
      // Read original data
      verify_file.seekg(current_offset);
      verify_file.read(reinterpret_cast<char*>(verify_buffer_orig_data.data()), data_span.size());
      // Ensure data matches
      if (std::memcmp(verify_buffer_orig_data.data(), data_span.data(), data_span.size()) != 0) {
        print_to_console("Error while verifying current_offset at offset " + std::to_string(current_offset) + "\n");
        throw std::runtime_error("La chorificacion");
      }
    }

    // make hashes
    const auto hashing_start_time = std::chrono::high_resolution_clock::now();
    /*
    // SHA1 boost hash
    boost::uuids::detail::sha1 hasher;
    hasher.process_bytes(chunk.data.data(), chunk.data.size());
    chunk.hash = get_sha1_hash(hasher);
    */
    
    // xxHash
    chunk.chunk_data->hash = XXH3_64bits(data_span.data(), data_span.size());
    const auto hashing_end_time = std::chrono::high_resolution_clock::now();
    hashing_execution_time_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(hashing_end_time - hashing_start_time).count();

    std::optional<uint64_t> duplicate_chunk_i = std::nullopt;
    // If last chunk was a duplicate or similar chunk, it's possible the next chunk from there is a duplicate of the current chunk, and if it is,
    // it's likely it will allow for better reduction (or at least less LZInstructions and match extension attempts)
    if (prev_similarity_locality_anchor_i.has_value()) {
      const auto prev_similarity_anchor_next_chunk_i = *prev_similarity_locality_anchor_i + 1;
      if (prev_similarity_anchor_next_chunk_i >= first_non_out_of_range_chunk_i) {
        auto& prev_similarity_anchor_next_chunk = chunks[prev_similarity_anchor_next_chunk_i];
        if (prev_similarity_anchor_next_chunk.chunk_data->hash == chunk.chunk_data->hash) {
          duplicate_chunk_i = prev_similarity_anchor_next_chunk_i;
        }
      }
    }
    if (!duplicate_chunk_i.has_value() && known_hashes.contains(chunk.chunk_data->hash)) {
      const auto duplicate_chunk_i_candidate = known_hashes[chunk.chunk_data->hash].back();
      duplicate_chunk_i = duplicate_chunk_i_candidate >= first_non_out_of_range_chunk_i ? std::optional{duplicate_chunk_i_candidate} : std::nullopt;
    }
    const bool is_duplicate_chunk = duplicate_chunk_i.has_value();

    std::vector<LZInstruction> new_instructions;
    if (!is_duplicate_chunk) {
      // Attempt resemblance detection and delta compression, first by using generalized resemblance detection
      std::vector<std::tuple<uint64_t, uint64_t>> similar_chunks{};
      if (use_generalized_resemblance_detection) {
        calc_simhash_if_needed(chunk);
        const auto simhash_base = hamming_base(chunk.chunk_data->lsh);
        if (simhashes_dict.contains(simhash_base)) {
          similar_chunks.emplace_back(0, simhashes_dict[simhash_base]);
        }
        // If there is already a match via generalized resemblance detection, we still overwrite the index with this newer chunk.
        // Because of data locality, a more recent chunk on the data stream is more likely to yield good results
        simhashes_dict[simhash_base] = chunk_i;
      }

      // Resemblance detection on the resemblance attempt window, exploiting the principle of data locality, there is a high likelihood data chunks close
      // to the current chunk are fairly similar, just check distance for all chunks within the window
      if (resemblance_attempt_window > 0 && (similar_chunks.empty() || attempt_multiple_delta_methods)) {
        calc_simhash_if_needed(chunk);
        std::vector<std::tuple<uint64_t, uint64_t>> resemblance_attempt_window_similar_chunks{};

        std::optional<uint64_t> best_candidate_dist{};
        uint64_t best_candidate_i = 0;
        for (uint64_t i = 1; i <= resemblance_attempt_window && i <= chunk_i; i++) {
          const auto candidate_i = chunk_i - i;
          if (candidate_i < first_non_out_of_range_chunk_i) break;
          auto& similar_chunk = chunks[candidate_i];

          calc_simhash_if_needed(similar_chunk);
          const auto new_dist = hamming_distance(chunk.chunk_data->lsh, similar_chunk.chunk_data->lsh);
          if (
            new_dist <= max_allowed_dist &&
            // If new_dist is the same as the best so far, privilege chunks with higher index as they are closer to the current chunk
            // and data locality suggests it should be a better match
            (!only_try_best_delta_match || (!best_candidate_dist.has_value() || *best_candidate_dist > new_dist || (*best_candidate_dist == new_dist && candidate_i > best_candidate_i)))
          ) {
            if (only_try_best_delta_match || best_candidate_dist.has_value() && new_dist < *best_candidate_dist)  {
              resemblance_attempt_window_similar_chunks.clear();
            }
            best_candidate_i = candidate_i;
            best_candidate_dist = new_dist;
            resemblance_attempt_window_similar_chunks.emplace_back(new_dist, candidate_i);
          }
        }
        if (!resemblance_attempt_window_similar_chunks.empty()) {
          similar_chunks.insert(similar_chunks.end(), resemblance_attempt_window_similar_chunks.begin(), resemblance_attempt_window_similar_chunks.end());
        }
      }

      // Resemblance detection via feature extraction and superfeature matching
      // If we couldn't sample the chunk's features we completely skip any attempt to match or register superfeatures, as we could not extract them for this chunk
      if (use_feature_extraction && !chunk.chunk_data->feature_sampling_failure) {
        // If we don't have a similar chunk yet, attempt to find one by SuperFeature matching
        if (similar_chunks.empty() || attempt_multiple_delta_methods) {
          calc_simhash_if_needed(chunk);
          std::vector<std::tuple<uint64_t, uint64_t>> feature_extraction_similar_chunks{};

          std::optional<uint64_t> best_candidate_dist{};
          uint64_t best_candidate_i = 0;
          // We will rank potential similar chunks by their amount of matching SuperFeatures (so we select the most similar chunk possible)
          std::unordered_map<uint64_t, uint64_t> chunk_rank{};

          for (const auto& sf : chunk.chunk_data->super_features) {
            if (superfeatures_dict.contains(sf)) {
              for (const auto& candidate_i : superfeatures_dict[sf]) {
                if (candidate_i < first_non_out_of_range_chunk_i) continue;
                chunk_rank[candidate_i] += 1;
                auto& similar_chunk = chunks[candidate_i];
                calc_simhash_if_needed(similar_chunk);
                const auto new_dist = hamming_distance(chunk.chunk_data->lsh, similar_chunk.chunk_data->lsh);
                if (
                  new_dist <= max_allowed_dist &&
                  // If new_dist is the same as the best so far, privilege chunks with higher index as they are closer to the current chunk
                  // and data locality suggests it should be a better match
                  (!only_try_best_delta_match || (!best_candidate_dist.has_value() || *best_candidate_dist > new_dist || (*best_candidate_dist == new_dist && candidate_i > best_candidate_i)))
                ) {
                  if (only_try_best_delta_match || best_candidate_dist.has_value() && new_dist < *best_candidate_dist) {
                    feature_extraction_similar_chunks.clear();
                  }
                  best_candidate_i = candidate_i;
                  best_candidate_dist = new_dist;
                  feature_extraction_similar_chunks.emplace_back(new_dist, candidate_i);
                }
              }
            }
          }
          if (!feature_extraction_similar_chunks.empty()) {
            similar_chunks.insert(similar_chunks.end(), feature_extraction_similar_chunks.begin(), feature_extraction_similar_chunks.end());
          }

          // If all SuperFeatures match, remove the previous chunk from the Index, prevents the index from getting pointlessly large, specially for some pathological cases
          for (const auto& [candidate_i, sf_count] : chunk_rank) {
            if (sf_count == 4) {
              for (const auto& sf : chunk.chunk_data->super_features) {
                auto& sf_list = superfeatures_dict[sf];
                if (sf_list.size() == 1) {
                  superfeatures_dict.erase(sf);
                }
                else {
                  auto sf_iter = std::find(sf_list.begin(), sf_list.end(), candidate_i);
                  if (sf_iter != sf_list.end()) {
                    sf_list.erase(sf_iter);
                  }
                }
              }
            }
          }

          // Register this chunk's SuperFeatures so that matches may be found with subsequent chunks
          for (const auto& sf : chunk.chunk_data->super_features) {
            superfeatures_dict[sf].push_back(chunk_i);
          }
        }
      }

      // If we don't have a similar chunk yet, attempt to find similar block via DARE's DupAdj,
      // checking if the next chunk from the similarity locality anchor is similar to this one
      if (use_dupadj && (similar_chunks.empty() || attempt_multiple_delta_methods)) {
        calc_simhash_if_needed(chunk);
        std::vector<std::tuple<uint64_t, uint64_t>> dupadj_similar_chunks{};

        std::optional<uint64_t> best_candidate_dist{};
        uint64_t best_candidate_i = 0;
        if (prev_similarity_locality_anchor_i.has_value()) {
          circular_vector<uint64_t> similarity_anchor_i_list;
          circular_vector<uint64_t>* chunks_with_similarity_anchor_hash;
          if (exhaustive_dupadj) {
            chunks_with_similarity_anchor_hash = &known_hashes[chunks[*prev_similarity_locality_anchor_i].chunk_data->hash];
          }
          else {
            similarity_anchor_i_list.emplace_back(*prev_similarity_locality_anchor_i);
            chunks_with_similarity_anchor_hash = &similarity_anchor_i_list;
          }
          for (const auto& anchor_instance_chunk_i : *chunks_with_similarity_anchor_hash) {
            const auto similar_candidate_chunk_i = anchor_instance_chunk_i + 1;
            if (similar_candidate_chunk_i == chunk_i) continue;

            auto& similar_candidate = chunks[similar_candidate_chunk_i];
            calc_simhash_if_needed(similar_candidate);
            const auto new_dist = hamming_distance(chunk.chunk_data->lsh, similar_candidate.chunk_data->lsh);
            if (
              new_dist <= max_allowed_dist &&
              // If new_dist is the same as the best so far, privilege chunks with higher index as they are closer to the current chunk
              // and data locality suggests it should be a better match
              (!only_try_best_delta_match || (!best_candidate_dist.has_value() || *best_candidate_dist > new_dist || (*best_candidate_dist == new_dist && similar_candidate_chunk_i > best_candidate_i)))
            ) {
              if (only_try_best_delta_match || best_candidate_dist.has_value() && new_dist < *best_candidate_dist) {
                dupadj_similar_chunks.clear();
              }
              best_candidate_i = similar_candidate_chunk_i;
              best_candidate_dist = new_dist;
              dupadj_similar_chunks.emplace_back(new_dist, similar_candidate_chunk_i);
            }
          }
        }
        if (!dupadj_similar_chunks.empty()) {
          similar_chunks.insert(similar_chunks.end(), dupadj_similar_chunks.begin(), dupadj_similar_chunks.end());
        }
      }

      // If any of the methods detected chunks that appear to be similar, attempt delta encoding
      DeltaEncodingResult best_encoding_result;
      if (!similar_chunks.empty()) {
        std::ranges::sort(similar_chunks, similar_chunk_tuple_cmp);

        uint64_t best_saved_size = 0;
        uint64_t best_similar_chunk_i;
        uint64_t best_similar_chunk_dist = 999;
        for (const auto& [similar_chunk_dist, similar_chunk_i] : similar_chunks) {
          if (only_try_min_dist_delta_matches && best_similar_chunk_dist < similar_chunk_dist) continue;

          const DeltaEncodingResult encoding_result = simulate_delta_encoding_func(*chunk.chunk_data, *chunks[similar_chunk_i].chunk_data, simhash_chunk_size);
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
          verify_buffer.resize(chunk.chunk_data->data.size());
          verify_buffer_delta.resize(chunk.chunk_data->data.size());

          verify_file_stream.seekg(chunk.offset);
          verify_file_stream.read(reinterpret_cast<char*>(verify_buffer.data()), chunk.chunk_data->data.size());
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
              print_to_console("Delta coding data verification mismatch!\n");
              throw std::runtime_error("La chorificacion");
            }
          }
          chunk_offset_pos += instruction.size;
        }
        if (chunk_offset_pos != chunk.chunk_data->data.size()) {
          print_to_console("Delta coding size mismatch: chunk_size/delta size " + std::to_string(chunk.chunk_data->data.size()) + "/" + std::to_string(chunk_offset_pos) + "\n");
          throw std::runtime_error("La chorificacion");
        }
      }
    }
    else {
      similarity_locality_anchor_i = duplicate_chunk_i;
      const auto& previous_chunk_instance = chunks[*duplicate_chunk_i];

      // Important to copy LSH in case this duplicate chunk ends up being candidate for Delta encoding via DupAdj or something
      chunk.chunk_data = previous_chunk_instance.chunk_data;

      if (use_generalized_resemblance_detection) {
        // Overwrite index for generalized resemblance detection with this newer chunk.
        // Because of data locality, a more recent chunk on the data stream is more likely to yield good results
        const auto simhash_base = hamming_base(chunk.chunk_data->lsh);
        simhashes_dict[simhash_base] = chunk_i;
      }
      // TODO: DO THE SAME FOR SUPERFEATURES
      // TODO: OR DON'T? ATTEMPTED IT AND IT HURT COMPRESSION A LOT

      new_instructions.emplace_back(LZInstructionType::COPY, previous_chunk_instance.offset, chunk.chunk_data->data.size());
    }
    known_hashes[chunk.chunk_data->hash].emplace_back(chunk_i);

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
          std::vector<std::tuple<uint64_t, uint64_t>> backwards_dupadj_similar_chunks{};
          std::optional<uint64_t> best_candidate_dist{};
          uint64_t best_candidate_i = 0;

          circular_vector<uint64_t> backtrack_anchor_i_list;
          circular_vector<uint64_t>* chunks_with_backtrack_anchor_hash;
          if (exhaustive_dupadj) {
            chunks_with_backtrack_anchor_hash = &known_hashes[chunks[backtrack_similarity_anchor_i].chunk_data->hash];
          }
          else {
            backtrack_anchor_i_list.emplace_back(backtrack_similarity_anchor_i);
            chunks_with_backtrack_anchor_hash = &backtrack_anchor_i_list;
          }
          for (const auto& anchor_instance_chunk_i : *chunks_with_backtrack_anchor_hash) {
            // if anchor_instance_chunk_i is <= first_non_out_of_range_chunk_i then when we look at the previous one we are out of range
            if (anchor_instance_chunk_i <= first_non_out_of_range_chunk_i /*|| anchor_instance_chunk_i == 0*/) continue;
            // if anchor_instance_chunk_i is larger than backtrack_chunk_i then that chunk is in the future, so not really backtracking
            if (anchor_instance_chunk_i > backtrack_chunk_i) break;
            const auto similar_candidate_chunk_i = anchor_instance_chunk_i - 1;
            auto& similar_chunk = chunks[similar_candidate_chunk_i];

            calc_simhash_if_needed(backtrack_chunk);
            calc_simhash_if_needed(similar_chunk);
            const auto new_dist = hamming_distance(backtrack_chunk.chunk_data->lsh, similar_chunk.chunk_data->lsh);
            if (
              new_dist <= max_allowed_dist &&
              // If new_dist is the same as the best so far, privilege chunks with higher index as they are closer to the current chunk
              // and data locality suggests it should be a better match
              (!only_try_best_delta_match || (!best_candidate_dist.has_value() || *best_candidate_dist > new_dist || (*best_candidate_dist == new_dist && similar_candidate_chunk_i > best_candidate_i)))
            ) {
              if (only_try_best_delta_match || best_candidate_dist.has_value() && new_dist < *best_candidate_dist) {
                backwards_dupadj_similar_chunks.clear();
              }
              best_candidate_i = similar_candidate_chunk_i;
              best_candidate_dist = new_dist;
              backwards_dupadj_similar_chunks.emplace_back(new_dist, similar_candidate_chunk_i);
            }
          }

          if (backwards_dupadj_similar_chunks.empty()) break;
          std::ranges::sort(backwards_dupadj_similar_chunks, similar_chunk_tuple_cmp);

          uint64_t best_saved_size = 0;
          std::optional<uint64_t> best_similar_chunk_i;
          uint64_t best_similar_chunk_dist = 999;
          DeltaEncodingResult best_encoding_result;
          for (const auto& [similar_chunk_dist, similar_chunk_i] : backwards_dupadj_similar_chunks) {
            if (only_try_min_dist_delta_matches && best_similar_chunk_dist < similar_chunk_dist) continue;
            if (similar_chunk_i < first_non_out_of_range_chunk_i) continue;
            auto& similar_chunk = chunks[similar_chunk_i];

            const DeltaEncodingResult encoding_result = simulate_delta_encoding_func(*backtrack_chunk.chunk_data, *similar_chunk.chunk_data, simhash_chunk_size);
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
              const auto instruction_size = instruction.size;
              const auto prev_estimated_savings = lz_manager.accumulatedSavings();
              const auto instruction_type = instruction.type;
              lz_manager.addInstruction(std::move(instruction), lz_offset, verify_addInstruction, chunks[first_non_out_of_range_chunk_i].offset);
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
        const auto instruction_size = instruction.size;
        const auto prev_estimated_savings = lz_manager.accumulatedSavings();
        const auto instruction_type = instruction.type;
        lz_manager.addInstruction(std::move(instruction), lz_offset, verify_addInstruction, chunks[first_non_out_of_range_chunk_i].offset);
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
      lz_manager.addInstruction({ .type = LZInstructionType::INSERT, .offset = chunk.offset, .size = chunk.chunk_data->data.size() }, chunk.offset, verify_addInstruction, chunks[first_non_out_of_range_chunk_i].offset);
    }

    dictionary_size_used += is_duplicate_chunk ? 0 : chunk.chunk_data->data.size();
    chunk_i++;
    current_offset += chunk.chunk_data->data.size();

    get_more_pending_chunks_from_cdc_futures(process_pending_chunks.empty());
    // If we have a single batch or less worth of cdc futures and the future for loading the next batch is ready we already launch the loading of the new next batch
    if (
      cdc_candidates_futures.size() <= cdc_thread_count &&
      process_pending_chunks.size() <= 1000 &&
      load_next_segment_batch_future.valid() &&
      load_next_segment_batch_future.wait_for(std::chrono::nanoseconds::zero()) == std::future_status::ready
    ) {
      auto segment_extend_data = load_next_segment_batch_future.get();
      //process_pending_chunks.shrink_to_fit();
      //cdc_candidates_futures.shrink_to_fit();
      launch_cdc_threads(segment_extend_data);
    }
    // If we were to run out of pending chunks but the next segment batch is not ready yet, we wait for it and get some pending chunks so we can continue
    if (process_pending_chunks.empty() && load_next_segment_batch_future.valid()) {
      auto segment_extend_data = load_next_segment_batch_future.get();
      //process_pending_chunks.shrink_to_fit();
      //cdc_candidates_futures.shrink_to_fit();
      launch_cdc_threads(segment_extend_data);
      get_more_pending_chunks_from_cdc_futures(true);
    }
  }

  print_to_console("Final offset: " + std::to_string(current_offset) + "\n");

  auto total_dedup_end_time = std::chrono::high_resolution_clock::now();

  print_to_console("Total dedup time:    " + std::to_string(std::chrono::duration_cast<std::chrono::seconds>(total_dedup_end_time - total_runtime_start_time).count()) + " seconds\n");

  // Dump any remaining data
  if (!output_disabled) lz_manager.dump(verify_file_stream, verify_dumps, std::nullopt, true);

  auto dump_end_time = std::chrono::high_resolution_clock::now();

  // Chunk stats
  print_to_console(std::string("Chunk Sizes:    min ") + std::to_string(min_size) + " - avg " + std::to_string(avg_size) + " - max " + std::to_string(max_size) + "\n");
  print_to_console("Total chunk count: " + std::to_string(chunks.fullSize()) + "\n");
  print_to_console("In dictionary chunk count: " + std::to_string(chunks.size()) + "\n");
  print_to_console("In memory chunk count: " + std::to_string(chunks.innerVecSize()) + "\n");
  print_to_console("Real AVG chunk size: " + std::to_string(total_size / chunks.fullSize()) + "\n");
  print_to_console("Total unique chunk count: " + std::to_string(known_hashes.size()) + "\n");
  print_to_console("Total delta compressed chunk count: " + std::to_string(delta_compressed_chunk_count) + "\n");

  const auto total_accumulated_savings = lz_manager.accumulatedSavings();
  const auto match_omitted_size = lz_manager.omittedSmallMatchSize();
  const auto match_extension_saved_size = lz_manager.accumulatedExtendedBackwardsSavings() + lz_manager.accumulatedExtendedForwardsSavings();
  // Results stats
  const auto total_size_mbs = total_size / (1024.0 * 1024);
  print_to_console("Chunk data total size:    %.1f MB\n", total_size_mbs);
  const auto deduped_size_mbs = deduped_size / (1024.0 * 1024);
  print_to_console("Chunk data deduped size:    %.1f MB\n", deduped_size_mbs);
  const auto deltaed_size_mbs = delta_compressed_approx_size / (1024.0 * 1024);
  print_to_console("Chunk data delta compressed size:    %.1f MB\n", deltaed_size_mbs);
  const auto extension_size_mbs = match_extension_saved_size / (1024.0 * 1024);
  print_to_console("Match extended size:    %.1f MB\n", extension_size_mbs);
  const auto omitted_size_mbs = match_omitted_size / (1024.0 * 1024);
  print_to_console("Match omitted size (matches too small):    %.1f MB\n", omitted_size_mbs);
  const auto total_accummulated_savings_mbs = total_accumulated_savings / (1024.0 * 1024);
  print_to_console("Total estimated reduced size:    %.1f MB\n", total_accummulated_savings_mbs);
  print_to_console("Final size:    %.1f MB\n", total_size_mbs - total_accummulated_savings_mbs);

  print_to_console("\n");

  // Throughput stats
  const auto chunking_mb_per_nanosecond = total_size_mbs / chunk_generator_execution_time_ns;
  print_to_console("Chunking Throughput:    %.1f MB/s\n", chunking_mb_per_nanosecond * std::pow(10, 9));
  const auto hashing_mb_per_nanosecond = total_size_mbs / hashing_execution_time_ns;
  print_to_console("Hashing Throughput:    %.1f MB/s\n", hashing_mb_per_nanosecond * std::pow(10, 9));
  const auto simhashing_mb_per_nanosecond = total_size_mbs / simhashing_execution_time_ns;
  print_to_console("SimHashing Throughput:    %.1f MB/s\n", simhashing_mb_per_nanosecond* std::pow(10, 9));
  const auto total_elapsed_nanoseconds = std::chrono::duration_cast<std::chrono::nanoseconds>(total_dedup_end_time - total_runtime_start_time).count();
  const auto total_mb_per_nanosecond = total_size_mbs / total_elapsed_nanoseconds;
  print_to_console("Total Throughput:    %.1f MB/s\n", total_mb_per_nanosecond * std::pow(10, 9));
  print_to_console("Total LZ instructions:    " + std::to_string(lz_manager.instructionCount()) + "\n");
  print_to_console("Total dedup time:    " + std::to_string(std::chrono::duration_cast<std::chrono::seconds>(total_dedup_end_time - total_runtime_start_time).count()) + " seconds\n");
  print_to_console("Dump time:    " + std::to_string(std::chrono::duration_cast<std::chrono::seconds>(dump_end_time - total_dedup_end_time).count()) + " seconds\n");
  print_to_console("Total runtime:    " + std::to_string(std::chrono::duration_cast<std::chrono::seconds>(dump_end_time - total_runtime_start_time).count()) + " seconds\n");

  print_to_console("Processing finished, press enter to quit.\n");
  get_char_with_echo();
  exit(0);  // Dirty, dirty, dirty, but should be fine as long all threads have finished, for exiting quickly until I refactor the codebase a little
  return 0;
}
