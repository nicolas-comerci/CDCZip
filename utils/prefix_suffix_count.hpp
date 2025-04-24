#ifndef PREFIX_SUFFIX_COUNT_H
#define PREFIX_SUFFIX_COUNT_H

#include <cstdint>
#include <span>

uint64_t find_identical_prefix_byte_count(const std::span<const uint8_t> data1_span, const std::span<const uint8_t> data2_span);
uint64_t find_identical_suffix_byte_count(const std::span<const uint8_t> data1_span, const std::span<const uint8_t> data2_span);

#endif