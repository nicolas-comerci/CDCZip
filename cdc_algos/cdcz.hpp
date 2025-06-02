#ifndef CDCZ_H
#define CDCZ_H

#include <span>
#include <queue>

#include "cdc_algos/gear.hpp"
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
  CutPointCandidate candidate;
  uint32_t pattern = 0;
  std::vector<uint32_t> features;
};

struct CDCZ_CONFIG {
  bool compute_features = false;
  bool use_fastcdc_subminimum_skipping = true;
  bool use_fastcdc_normalized_chunking = true;
  bool use_supercdc_minmax_adjustment = true;
  bool use_supercdc_backup_mask = true;
  bool avx2_allowed = false;
};

// Precondition: Chunk invariance condition satisfied, that is, the data starts from the very beginning of the stream or after a chunk cutpoint we know for sure will be used
CutPointCandidateWithContext cdc_next_cutpoint(
  const std::span<uint8_t> data,
  uint32_t min_size,
  uint32_t avg_size,
  uint32_t max_size,
  uint32_t mask_hard,
  uint32_t mask_medium,
  uint32_t mask_easy,
  const CDCZ_CONFIG& cdcz_cfg,
  uint32_t initial_pattern = 0
);

struct CdcCandidatesResult {
  std::vector<CutPointCandidate> candidates;
  std::vector<std::vector<uint32_t>> candidatesFeatureResults;
};

CdcCandidatesResult find_cdc_cut_candidates(std::span<uint8_t> data, const uint32_t min_size, const uint32_t avg_size, const uint32_t max_size, const CDCZ_CONFIG& cdcz_cfg, bool is_first_segment = true);

#if defined(__AVX2__)
// Precondition: Chunk invariance condition satisfied, that is, the data starts from the very beginning of the stream or after a chunk cutpoint we know for sure will be used
void cdc_find_cut_points_with_invariance(
  std::vector<CutPointCandidate>& candidates,
  std::vector<std::vector<uint32_t>>& candidate_features,
  std::span<uint8_t> data,
  uint64_t base_offset,
  int32_t min_size,
  int32_t avg_size,
  int32_t max_size,
  uint32_t mask_hard,
  uint32_t mask_medium,
  uint32_t mask_easy,
  const CDCZ_CONFIG& cdcz_cfg
);
#endif

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

#endif