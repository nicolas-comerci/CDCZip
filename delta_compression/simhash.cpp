#include "simhash.hpp"

#include <span>

#include "contrib/xxHash/xxhash.h"

#include "cdc_algos/cdcz.hpp"

#if defined(__AVX2__)
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

void SimHahser64Bit::add_hash(std::bitset<64> chunk_hash) {
  // Update SimHash vector with the hash of the chunk
  // Using AVX/2 we don't have enough vector size to handle the whole 64bit hash, we should be able to do it with AVX512,
  // but that is less readily available in desktop chips, so we split the hash into 2 and process it that way
  const std::bitset<64> upper_chunk_hash = chunk_hash >> 32;
  upper_counter_vector = add_32bit_hash_to_simhash_counter(upper_chunk_hash.to_ulong(), upper_counter_vector);
  const std::bitset<64> lower_chunk_hash = (chunk_hash << 32) >> 32;
  lower_counter_vector = add_32bit_hash_to_simhash_counter(lower_chunk_hash.to_ulong(), lower_counter_vector);
}

std::bitset<64> SimHahser64Bit::digest() const {
  const uint32_t upper_chunk_hash = finalize_32bit_simhash_from_counter(upper_counter_vector);
  const uint32_t lower_chunk_hash = finalize_32bit_simhash_from_counter(lower_counter_vector);
  return (static_cast<uint64_t>(upper_chunk_hash) << 32) | lower_chunk_hash;
}
#else
void SimHahser64Bit::add_hash(std::bitset<64> chunk_hash) {
  for (uint64_t i = 0; i < 8; i++) {
    counters[63 - i] += chunk_hash.test(63) ? 1 : -1;
    counters[55 - i] += chunk_hash.test(55) ? 1 : -1;
    counters[47 - i] += chunk_hash.test(47) ? 1 : -1;
    counters[39 - i] += chunk_hash.test(39) ? 1 : -1;
    counters[31 - i] += chunk_hash.test(31) ? 1 : -1;
    counters[23 - i] += chunk_hash.test(23) ? 1 : -1;
    counters[15 - i] += chunk_hash.test(15) ? 1 : -1;
    counters[7 - i] += chunk_hash.test(7) ? 1 : -1;
    chunk_hash <<= 1;
  }
}

std::bitset<64> SimHahser64Bit::digest() const {
  std::bitset<64> simhash = 0;
  for (uint8_t bit_i = 0; bit_i < 64; bit_i++) {
    simhash[bit_i] = counters[bit_i] > 0 ? 1 : 0;
  }
  return simhash;
}
#endif

std::tuple<std::bitset<64>, std::vector<uint32_t>> simhash_data_xxhash_shingling(uint8_t* data, uint32_t data_len, uint32_t chunk_size) {
  std::tuple<std::bitset<64>, std::vector<uint32_t>> return_val{};
  auto* simhash = &std::get<0>(return_val);
  auto& minichunks_vec = std::get<1>(return_val);

  SimHahser64Bit simhasher;

  // Iterate over the data in chunks
  for (uint32_t i = 0; i < data_len; i += chunk_size) {
    // Calculate hash for current chunk
    const auto current_chunk_len = std::min(chunk_size, data_len - i);
    const uint64_t chunk_hash = XXH3_64bits(data + i, current_chunk_len);

    simhasher.add_hash(chunk_hash);
    minichunks_vec.emplace_back(current_chunk_len);
  }

  *simhash = simhasher.digest();
  minichunks_vec.shrink_to_fit();
  return return_val;
}

std::tuple<std::bitset<64>, std::vector<uint32_t>> simhash_data_xxhash_cdc(uint8_t* data, uint32_t data_len, uint32_t chunk_size) {
  std::tuple<std::bitset<64>, std::vector<uint32_t>> return_val{};
  auto* simhash = &std::get<0>(return_val);
  auto& minichunks_vec = std::get<1>(return_val);

  SimHahser64Bit simhasher;

  // Find the CDC minichunks and update the SimHash with their data
  const auto min_chunk_size = chunk_size / 2;
  auto [cut_offsets, cut_offsets_features] = find_cdc_cut_candidates<false>(
    std::span(data, data_len), min_chunk_size, chunk_size, chunk_size * 2
  );
  uint64_t previous_offset = 0;
  for (const auto& cut_point_candidate : cut_offsets) {
    if (cut_point_candidate.offset <= previous_offset) continue;
    if (cut_point_candidate.offset < previous_offset + min_chunk_size && cut_point_candidate.offset != data_len) continue;
    const auto minichunk_len = static_cast<const uint32_t>(cut_point_candidate.offset - previous_offset);
    // Calculate hash for current chunk
    const uint64_t chunk_hash = XXH3_64bits(data + previous_offset, minichunk_len);

    simhasher.add_hash(chunk_hash);
    minichunks_vec.emplace_back(minichunk_len);
    previous_offset = cut_point_candidate.offset;
  }

  *simhash = simhasher.digest();
  minichunks_vec.shrink_to_fit();
  return return_val;
}
