#ifndef CDCZ_SIMD_H
#define CDCZ_SIMD_H

#include "cdcz.hpp"

class LaneStatus {
public:
  std::vector<std::vector<CutPointCandidate>> lane_results{};
  std::vector<bool> lane_achieved_chunk_invariance{};
  std::vector<std::vector<std::vector<uint32_t>>> lane_features_results{};
  std::vector<std::vector<uint32_t>> lane_current_features{};

  bool any_lane_marked_for_jump = false;
  std::vector<int32_t> jump_pos_vec{};

  std::vector<int32_t> minmax_adjustment_vec{};

  // SuperCDC's backup results, for each lane we need to save any valid ones until we reach a valid cut so we don't need them,
  // or reach max_size for a chunk and thus use the earliest backup.
  std::vector<std::queue<int32_t>> backup_cut_vec{};

  LaneStatus(size_t lane_count, bool is_first_segment) {
    lane_results.resize(lane_count);
    lane_achieved_chunk_invariance.resize(lane_count);
    lane_achieved_chunk_invariance[0] = is_first_segment;
    lane_features_results.resize(lane_count);
    lane_current_features.resize(lane_count);

    jump_pos_vec.resize(lane_count);
    minmax_adjustment_vec.resize(lane_count);
    backup_cut_vec.resize(lane_count);
  }
};

void process_lane_cut_candidate(
  const int32_t lane_i, int32_t pos, const CutPointCandidateType result_type, LaneStatus& lane_status,
  const int32_t min_size, const int32_t avg_size, const int32_t max_size, const int32_t bytes_per_lane, const int32_t lane_max_pos,
  const bool compute_features, const bool use_fastcdc_subminimum_skipping, const bool ignore_max_pos = false
);

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

#endif
