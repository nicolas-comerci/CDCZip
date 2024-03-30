#include <algorithm>
#include <cmath>
#include <istream>
#include <iostream>
#include <fstream>
#include <utility>
#include <string>
#include <random>
#include <cstdio>

#include "contrib/boost/uuid/detail/sha1.hpp"

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
/*
std::string calculate_sha1(IStreamLike& file1, unsigned int pos1) {
  std::vector<unsigned char> input_bytes1;
  input_bytes1.resize(CHUNK);
  long long size1;
  boost::uuids::detail::sha1 s;

  file1.seekg(pos1, std::ios_base::beg);

  do {
    file1.read(reinterpret_cast<char*>(input_bytes1.data()), CHUNK);
    size1 = file1.gcount();
    s.process_bytes(input_bytes1.data(), size1);
  } while (size1 == CHUNK);

  return get_sha1_hash(s);
}
*/

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
}

namespace utility {
  class CDChunk {
  public:
    unsigned long long offset;
    int length;
    std::vector<uint8_t> data;
    std::string hash;

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

namespace fastcdc {
  uint32_t cdc_offset(
    const std::vector<uint8_t>& data,
    uint32_t min_size,
    uint32_t avg_size,
    uint32_t max_size,
    uint32_t center_size,
    uint32_t mask_s,
    uint32_t mask_l
  ) {
    uint32_t pattern = 0;
    uint32_t i = min_size;
    uint32_t size = data.size();
    uint32_t barrier = std::min(center_size, size);
    while (i < barrier) {
      pattern = (pattern >> 1) + constants::GEAR[data[i]];
      if (!(pattern & mask_s))
        return i + 1;
      i += 1;
    }
    barrier = std::min(max_size, size);
    while (i < barrier) {
      pattern = (pattern >> 1) + constants::GEAR[data[i]];
      if (!(pattern & mask_l))
        return i + 1;
      i += 1;
    }
    return i;
  }

  std::vector<utility::CDChunk> chunk_generator(std::istream& stream, uint32_t min_size, uint32_t avg_size, uint32_t max_size, bool fat) {
    uint32_t cs = utility::center_size(avg_size, min_size, max_size);
    uint32_t bits = utility::logarithm2(avg_size);
    uint32_t mask_s = utility::mask(bits + 1);
    uint32_t mask_l = utility::mask(bits - 1);
    uint32_t read_size = std::max<uint32_t>(1024 * 64, max_size);

    std::vector<uint8_t> blob{};
    blob.resize(read_size);
    stream.read(reinterpret_cast<char*>(blob.data()), read_size);
    uint32_t blob_len = stream.gcount();
    blob.resize(blob_len);

    uint32_t offset = 0;
    std::vector<utility::CDChunk> chunks{};
    while (!blob.empty()) {
      blob_len = blob.size();
      if (blob_len <= max_size) {
        blob.resize(read_size);
        stream.read(reinterpret_cast<char*>(blob.data()) + blob_len, read_size - blob_len);
        blob_len += stream.gcount();
        blob.resize(blob_len);
      }
      uint32_t cp = cdc_offset(blob, min_size, avg_size, max_size, cs, mask_s, mask_l);
      std::vector<uint8_t> raw{};
      if (fat) {
        raw = std::vector<uint8_t>(blob.begin(), blob.begin() + cp);
      }
      chunks.emplace_back(offset, cp, std::move(raw), std::string());
      blob.erase(blob.begin(), blob.begin() + cp);
      offset += cp;
    }
    return chunks;
  }
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

  auto file_stream = std::fstream(file_path, std::ios::in | std::ios::binary);
  if (!file_stream.is_open()) {
    std::cout << "Can't read file\n";
    return 1;
  }
  uint64_t total_size = 0;
  auto chunks = fastcdc::chunk_generator(file_stream, min_size, avg_size, max_size, true);
  for (const auto& chunk : chunks) {
    total_size += chunk.length;
    // make hashes
  }
  std::cout << std::string("Chunk Sizes:    min ") + std::to_string(min_size) + " - avg " + std::to_string(avg_size) + " - max " + std::to_string(max_size) + "\n" << std::flush;
  std::cout << "Total chunk count: " + std::to_string(chunks.size()) + "\n" << std::flush;
  auto total_size_double = total_size / (1024.0 * 1024);
  std::printf("Chunk data total size:    %.1f MB\n", total_size_double);

  return 0;
}
