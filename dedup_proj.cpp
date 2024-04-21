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
}

namespace utility {
  class CDChunk {
  public:
    unsigned long long offset;
    int length;
    std::vector<uint8_t> data;
    std::string hash;
    std::bitset<64> lsh;

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
    auto blob_it = blob.begin();
    while (blob_it != blob.end()) {
      blob_len = blob.end() - blob_it;
      if (blob_len <= max_size) {
        blob.erase(blob.begin(), blob_it);

        blob.resize(read_size);
        stream.read(reinterpret_cast<char*>(blob.data()) + blob_len, read_size - blob_len);
        blob_len += stream.gcount();
        blob.resize(blob_len);
        blob_it = blob.begin();  // iterators got invalidated
      }
      uint32_t cp = cdc_offset(std::span(blob_it, blob.end()), min_size, avg_size, max_size, cs, mask_s, mask_l);
      std::vector<uint8_t> raw{};
      if (fat) {
        raw = std::vector(blob_it, blob_it + cp);
      }
      chunks.emplace_back(offset, cp, std::move(raw), std::string());
      offset += cp;
      blob_it += cp;
    }
    return chunks;
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

std::bitset<64> simhash_data_xxhash(uint8_t* data, int data_len, int chunk_size = 16) {
  // Initialize SimHash array
  std::array<uint8_t, 8> simhash = { 0 };

  XXH3_state_t* state = XXH3_createState();
  XXH3_64bits_reset(state);

  // Iterate over the data in chunks
  for (int i = 0; i < data_len; i += chunk_size) {
    // Calculate hash for current chunk
    XXH3_64bits_update(state, data + i, std::min(chunk_size, data_len - i));
    XXH64_hash_t chunk_hash = XXH3_64bits_digest(state);
    XXH3_64bits_reset(state);

    // Update SimHash with the hash of the chunk
    for (int j = 0; j < 8; ++j) {
      // Update each byte of the SimHash with the corresponding byte of the chunk hash
      simhash[j] += (chunk_hash >> (j * 8)) & 0xFF;
    }
  }

  XXH3_freeState(state);
  return { *reinterpret_cast<uint64_t*>(simhash.data())};
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

uint64_t simulate_delta_encoding(const utility::CDChunk& chunk, const utility::CDChunk& similar_chunk) {
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
  chunk_tmp_file.write(reinterpret_cast<const char*>(chunk.data.data()), chunk.data.size());
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
  return saved_size;
}

int main(int argc, char* argv[])
{
  get_char_with_echo();
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

  auto file_stream = std::fstream(file_path, std::ios::in | std::ios::binary);
  if (!file_stream.is_open()) {
    std::cout << "Can't read file\n";
    return 1;
  }
  uint64_t total_size = 0;
  uint64_t deduped_size = 0;
  uint64_t delta_compressed_approx_size = 0;
  uint64_t delta_compressed_chunk_count = 0;
  
  std::vector<int> faiss_idx_to_chunk_id{};
  std::unordered_map<int, std::string> chunk_ssdeep_hashes{};

  std::unordered_map<std::string, std::vector<int>> known_hashes{};  // Find chunk pos by hash
  std::vector<std::string> chunk_hashes{};  // Find hash by chunk pos

  auto chunk_generator_start_time = std::chrono::high_resolution_clock::now();
  auto chunks = fastcdc::chunk_generator(file_stream, min_size, avg_size, max_size, true);
  auto chunk_generator_end_time = std::chrono::high_resolution_clock::now();
  auto hashing_start_time = std::chrono::high_resolution_clock::now();

  int chunk_i = 0;
  int pending_chunks = 0;
  std::vector<int32_t> pending_chunks_indexes(batch_size, 0);
  std::vector<uint8_t> pending_chunk_data(batch_size * max_size, 0);
  std::vector<std::bitset<64>> simhashes{};
  std::unordered_map<std::bitset<64>, int> simhashes_dict{};

  std::optional<std::string> similarity_locality_anchor{};
  for (auto& chunk : chunks) {
    if (chunk_i % 1000 == 0) std::cout << "\n%" + std::to_string((static_cast<float>(chunk_i) / chunks.size()) * 100) + "\n" << std::flush;
    total_size += chunk.length;
    // make hashes
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

    if (!known_hashes.contains(chunk.hash)) {
      chunk.lsh = simhash_data_xxhash(chunk.data.data(), chunk.data.size());
      auto simhash_base = hamming_base(chunk.lsh);
      if (!simhashes_dict.contains(simhash_base)) {
        faiss_idx_to_chunk_id.push_back(chunk_i);
        simhashes_dict[simhash_base] = chunk_i;
        simhashes.emplace_back(simhash_base);

        // Attempt to find similar block via DARE's DupAdj, checking if the next chunk from the similarity locality anchor is similar to this one
        std::optional<int> best_candidate_dist{};
        int best_candidate = 0;
        if (similarity_locality_anchor.has_value()) {
          for (const auto& anchor_instance_chunk_i : known_hashes[*similarity_locality_anchor]) {
            const auto similar_candidate_chunk = anchor_instance_chunk_i + 1;
            if (similar_candidate_chunk == chunk_i) continue;

            const auto& similar_chunk = chunks[similar_candidate_chunk];
            const auto new_dist = hamming_distance(chunk.lsh, similar_chunk.lsh);
            if (new_dist <= 99 && (!best_candidate_dist.has_value() || *best_candidate_dist > new_dist)) {
              best_candidate_dist = new_dist;
              best_candidate = similar_candidate_chunk;
            }
          }
        }
        if (best_candidate_dist.has_value()) {
          const auto& similar_chunk = chunks[best_candidate];
          auto saved_size = simulate_delta_encoding(chunk, similar_chunk);

          if (saved_size > 0) {
            std::cout << "WHOOHOO! DupAdj Success!\n" << std::flush;
            delta_compressed_approx_size += saved_size;
            delta_compressed_chunk_count++;
            similarity_locality_anchor = chunk.hash;
          }
        }
        else {
          similarity_locality_anchor = std::nullopt;
        }
      }
      else {
        const auto& similar_chunk = chunks[simhashes_dict[simhash_base]];

        auto saved_size = simulate_delta_encoding(chunk, similar_chunk);

        delta_compressed_approx_size += saved_size;
        delta_compressed_chunk_count++;
        similarity_locality_anchor = chunk.hash;
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

        auto saved_size = simulate_delta_encoding(chunk, similar_chunk);

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
  }
  auto hashing_end_time = std::chrono::high_resolution_clock::now();

  // Chunk stats
  std::cout << std::string("Chunk Sizes:    min ") + std::to_string(min_size) + " - avg " + std::to_string(avg_size) + " - max " + std::to_string(max_size) + "\n" << std::flush;
  std::cout << "Total chunk count: " + std::to_string(chunks.size()) + "\n" << std::flush;
  std::cout << "Total unique chunk count: " + std::to_string(known_hashes.size()) + "\n" << std::flush;
  std::cout << "Total delta compressed chunk count: " + std::to_string(delta_compressed_chunk_count) + "\n" << std::flush;

  // Throughput stats
  auto total_size_mbs = total_size / (1024.0 * 1024);
  std::printf("Chunk data total size:    %.1f MB\n", total_size_mbs);
  auto deduped_size_mbs = deduped_size / (1024.0 * 1024);
  std::printf("Chunk data deduped size:    %.1f MB\n", deduped_size_mbs);
  auto deltaed_size_mbs = delta_compressed_approx_size / (1024.0 * 1024);
  std::printf("Chunk data delta compressed size:    %.1f MB\n", deltaed_size_mbs);
  std::printf("Final size:    %.1f MB\n", total_size_mbs - deduped_size_mbs - deltaed_size_mbs);
  auto chunking_elapsed_nanoseconds = std::chrono::duration_cast<std::chrono::nanoseconds>(chunk_generator_end_time - chunk_generator_start_time).count();
  auto chunking_mb_per_nanosecond = total_size_mbs / chunking_elapsed_nanoseconds;
  std::printf("Chunking Throughput:    %.1f MB/s\n", chunking_mb_per_nanosecond * std::pow(10, 9));
  auto hashing_elapsed_nanoseconds = std::chrono::duration_cast<std::chrono::nanoseconds>(hashing_end_time - hashing_start_time).count();
  auto hashing_mb_per_nanosecond = total_size_mbs / hashing_elapsed_nanoseconds;
  std::printf("Hashing Throughput:    %.1f MB/s\n", hashing_mb_per_nanosecond * std::pow(10, 9));
  auto total_elapsed_nanoseconds = std::chrono::duration_cast<std::chrono::nanoseconds>(hashing_end_time - chunk_generator_start_time).count();
  auto total_mb_per_nanosecond = total_size_mbs / total_elapsed_nanoseconds;
  std::printf("Total Throughput:    %.1f MB/s\n", total_mb_per_nanosecond * std::pow(10, 9));
  std::printf("Total runtime:    %lld seconds\n", std::chrono::duration_cast<std::chrono::seconds>(hashing_end_time - chunk_generator_start_time).count());

  return 0;
}
