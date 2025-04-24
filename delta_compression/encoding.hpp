#ifndef DELTA_ENCODING_H
#define DELTA_ENCODING_H

#include <vector>

#include "utils/chunks.hpp"
#include "utils/lz.hpp"

struct DeltaEncodingResult {
  uint64_t estimated_savings = 0;
  std::vector<LZInstruction> instructions;
};

DeltaEncodingResult simulate_delta_encoding_shingling(const utility::ChunkData& chunk, const utility::ChunkData& similar_chunk, uint32_t minichunk_size);
DeltaEncodingResult simulate_delta_encoding_using_minichunks(const utility::ChunkData& chunk, const utility::ChunkData& similar_chunk, uint32_t minichunk_size);

#endif
