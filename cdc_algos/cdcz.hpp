#ifndef CDCZ_H
#define CDCZ_H

#include <span>
#include <queue>

#include "utils/chunks.hpp"

enum CutPointCandidateType : uint8_t {
  HARD_CUT_MASK,  // Satisfied harder mask before average size (FastCDC normalized chunking)
  EASY_CUT_MASK,  // Satisfied easier mask after average size (FastCDC normalized chunking)
  SUPERCDC_BACKUP_MASK,  // Satisfied SuperCDC backup mask because no other mask worked
  MAX_SIZE,  // Forcibly cut because the data size reached the chunk max allowed size
  EOF_CUT  // Forcibly cut because the data span reached its EOF
};

struct CutPointCandidate {
  CutPointCandidateType type;
  uint64_t offset;
};

struct CutPointCandidateWithContext {
  CutPointCandidate candidate{};
  uint32_t pattern = 0;
  std::vector<uint32_t> features{};
};

struct CDCZ_CONFIG {
  bool compute_features = false;
  bool use_fastcdc_subminimum_skipping = false;
  bool use_fastcdc_normalized_chunking = false;
  bool use_supercdc_minmax_adjustment = false;
  bool use_supercdc_backup_mask = false;
  bool simd_allowed = false;
};

inline uint64_t pad_size_for_alignment(uint64_t size, uint64_t alignment) { return ((size + alignment - 1) / alignment) * alignment; }

struct CdcCandidatesResult {
  std::vector<CutPointCandidate> candidates{};
  std::vector<std::vector<uint32_t>> candidatesFeatureResults{};
};

CdcCandidatesResult find_cdc_cut_candidates(std::span<uint8_t> data, const uint32_t min_size, const uint32_t avg_size, const uint32_t max_size, const CDCZ_CONFIG& cdcz_cfg, bool is_first_segment = true);

uint64_t select_cut_point_candidates(
  std::vector<CutPointCandidate>& new_cut_point_candidates,
  std::vector<std::vector<uint32_t>>& new_cut_point_candidates_features,
  std::deque<utility::ChunkEntry>& process_pending_chunks,
  std::queue<uint64_t>& supercdc_backup_pos,
  uint64_t last_used_cut_point,
  uint64_t segment_start_offset,
  std::span<uint8_t> segment_data,
  std::vector<uint8_t>& prev_segment_remaining_data,
  uint32_t min_size,
  uint32_t avg_size,
  uint32_t max_size,
  bool segments_eof,
  bool use_feature_extraction,
  bool is_first_segment,
  bool copy_chunk_data = true
);

inline auto promote_cut_candidate(const CDCZ_CONFIG& cdcz_cfg, uint32_t pattern, uint32_t mask_hard, uint32_t mask_medium, uint32_t mask_easy) -> CutPointCandidateType {
  // Set the candidate type according to the condition_mask we used
  CutPointCandidateType promoted_cut_type = cdcz_cfg.use_supercdc_backup_mask
    ? CutPointCandidateType::SUPERCDC_BACKUP_MASK
    : cdcz_cfg.use_fastcdc_normalized_chunking ? CutPointCandidateType::EASY_CUT_MASK : CutPointCandidateType::HARD_CUT_MASK;
  // Now we need to "promote" the cut condition if possible, that is, if we found a cut candidate that satisfied SuperCDC's backup mask,
  // but it also satisfies FastCDC's normalized chunking easy mask, or even the hard mask, we return it for the hardest condition available
  if (promoted_cut_type == CutPointCandidateType::SUPERCDC_BACKUP_MASK) {
    if (cdcz_cfg.use_fastcdc_normalized_chunking && !(pattern & mask_easy)) {
      promoted_cut_type = CutPointCandidateType::EASY_CUT_MASK;
    }
    else if (!cdcz_cfg.use_fastcdc_normalized_chunking && !(pattern & mask_medium)) {
      promoted_cut_type = CutPointCandidateType::HARD_CUT_MASK;
    }
  }
  // Note that the type can only be EASY here if normalized chunking is on, so no need for extra check
  if (promoted_cut_type == CutPointCandidateType::EASY_CUT_MASK && !(pattern & mask_hard)) {
    promoted_cut_type = CutPointCandidateType::HARD_CUT_MASK;
  }
  return promoted_cut_type;
}

inline auto is_chunk_invariance_condition_satisfied(
  bool is_prev_candidate_hard, uint64_t dist_with_prev, CutPointCandidateType new_candidate_type,
  uint64_t min_size, uint64_t avg_size, uint64_t max_size
) -> bool {
  return is_prev_candidate_hard &&
    // Given that the previous candidate is of HARD type, either it will be used, or it will be discarded which can only happen
    // if a cut was used with at most min_size distance before the previous candidate. Knowing the previous cut to be used is
    // at most at distance_w_prev_cut_candidate + min_size distance we can ensure we don't violate max_size if we use the current candidate.
    (dist_with_prev + min_size <= max_size) &&
    (
      // We also need to check that the current candidate is actually eligible, a HARD type cut needs to be at least min_size from the previous
      // cut to be valid, whereas an EASY cut needs to be at least avg_size from it.
      // Note that we check using distance_w_prev_cut_candidate, with the same logic as the check we did for the max_size, if it won't be used
      // then the actual distance to the previous cut will be even larger so these conditions will also validate the eligibility of the current
      // candidate in that case
      (new_candidate_type == CutPointCandidateType::HARD_CUT_MASK && dist_with_prev >= min_size) ||
      (new_candidate_type == CutPointCandidateType::EASY_CUT_MASK && dist_with_prev >= avg_size)
    );
}

#endif