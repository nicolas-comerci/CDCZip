#ifndef SIMHASH_H
#define SIMHASH_H

#include <array>
#include <bitset>
#include <cstdint>
#include <vector>

#if defined(__AVX2__)
#include <immintrin.h>

__m256i movemask_inverse_epi8(const uint32_t mask);

__m256i add_32bit_hash_to_simhash_counter(const uint32_t new_hash, __m256i counter_vector);
uint32_t finalize_32bit_simhash_from_counter(const __m256i counter_vector);

class SimHahser64Bit {
public:
  void add_hash(std::bitset<64> chunk_hash);

  std::bitset<64> digest() const;
private:
  __m256i upper_counter_vector = _mm256_set1_epi8(0);
  __m256i lower_counter_vector = _mm256_set1_epi8(0);
};
#else
class SimHahser64Bit {
public:
  void add_hash(std::bitset<64> chunk_hash);
  std::bitset<64> digest() const;
private:
  std::array<int8_t, 64> counters{};
};
#endif

std::tuple<std::bitset<64>, std::vector<uint32_t>> simhash_data_xxhash_shingling(uint8_t* data, uint32_t data_len, uint32_t chunk_size);

std::tuple<std::bitset<64>, std::vector<uint32_t>> simhash_data_xxhash_cdc(uint8_t* data, uint32_t data_len, uint32_t chunk_size);

template<std::size_t bit_size>
uint64_t hamming_distance(const std::bitset<bit_size>& data1, const std::bitset<bit_size>& data2) {
  const auto val = data1 ^ data2;
  return val.count();
}

template<std::size_t bit_size>
std::size_t hamming_syndrome(const std::bitset<bit_size>& data) {
  int result = 0;
  std::bitset<bit_size> mask{ 0b1 };
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

#endif