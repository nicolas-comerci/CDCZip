#ifndef CDCZ_SERIAL_H
#define CDCZ_SERIAL_H

#include "cdcz.hpp"

// Precondition: Chunk invariance condition satisfied, that is, the data starts from the very beginning of the stream or after a chunk cutpoint we know for sure will be used
void find_cdc_cut_candidates_serial(
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

#endif
