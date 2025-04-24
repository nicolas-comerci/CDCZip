#include "encoding.hpp"

#include <algorithm>

#include "contrib/xxHash/xxhash.h"

#include "utils/prefix_suffix_count.hpp"

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
