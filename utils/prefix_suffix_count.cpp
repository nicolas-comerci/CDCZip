#include "prefix_suffix_count.hpp"

#include <cstring>

#if defined(__SSE2__) || defined(__AVX2__)
#include <immintrin.h>

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
#endif

uint64_t find_identical_prefix_byte_count(const std::span<const uint8_t> data1_span, const std::span<const uint8_t> data2_span) {
  const auto cmp_size = std::min(data1_span.size(), data2_span.size());
  uint64_t matching_data_count = 0;
  uint64_t i = 0;

#if defined(__AVX2__)
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
#endif

#if defined(__SSE2__)
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
#endif

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

#if defined(__AVX2__)
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
#endif

#if defined(__SSE2__)
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
#endif

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
