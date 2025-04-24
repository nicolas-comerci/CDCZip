#include "chunks.hpp"

#include <algorithm>
#include <utility>

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
    throw std::runtime_error("Verification error");
  }
#endif

  return { chunk_i, chunk_pos };
}
