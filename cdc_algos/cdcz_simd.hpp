#ifndef CDCZ_SIMD_H
#define CDCZ_SIMD_H

#include "cdcz.hpp"

// Precondition: Chunk invariance condition satisfied, that is, the data starts from the very beginning of the stream or after a chunk cutpoint we know for sure will be used
void find_cdc_cut_candidates_simd(
  std::vector<CutPointCandidate>& candidates,
  std::vector<std::vector<uint32_t>>& candidate_features,
  std::span<uint8_t> data,
  int32_t min_size,
  int32_t avg_size,
  int32_t max_size,
  uint32_t mask_hard,
  uint32_t mask_medium,
  uint32_t mask_easy,
  const CDCZ_CONFIG& cdcz_cfg,
  bool is_first_segment
);

void sscdc_first_stage(std::span<uint8_t> data, uint8_t* results_bitmap, uint32_t mask);
void sscdc_second_stage(
  const uint8_t* segment_results_bitmap,
  std::deque<utility::ChunkEntry>& process_pending_chunks,
  const uint64_t min_chunksize,
  const uint64_t max_chunksize,
  const uint64_t segment_length,
  const uint64_t segment_start_offset,
  uint64_t& prev_cut_offset
);

#endif
