#include "cdcz.hpp"

#include <algorithm>
#include <deque>
#include <variant>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

#include "contrib/xxHash/xxhash.h"

#include "cdc_algos/gear.hpp"
#include "delta_compression/delta.hpp"

CutPointCandidateWithContext cdc_next_cutpoint(
  const std::span<uint8_t> data,
  uint32_t min_size,
  uint32_t avg_size,
  uint32_t max_size,
  uint32_t mask_hard,
  uint32_t mask_medium,
  uint32_t mask_easy,
  const CDCZ_CONFIG& cdcz_cfg,
  uint32_t initial_pattern
) {
  CutPointCandidateWithContext result;
  result.pattern = initial_pattern;
  const uint64_t size = data.size();
  uint32_t barrier;
  uint32_t i;

  if (cdcz_cfg.use_fastcdc_subminimum_skipping) {
    barrier = static_cast<uint32_t>(std::min<uint64_t>(avg_size, size));
    i = std::min(barrier, min_size);
  }
  else {
    barrier = static_cast<uint32_t>(std::min<uint64_t>(min_size, size));
    i = 0;
  }

  // if normalized chunking is disabled we set the harder and easier masks of normalized chunking to the "medium" of regular GEAR CDC,
  // which makes it behave the same way before and after the avg_size of chunk is reached, and thus works like regular GEAR CDC
  if (!cdcz_cfg.use_fastcdc_normalized_chunking) {
    mask_hard = mask_medium;
    mask_easy = mask_medium;
  }

  // SuperCDC's even easier "backup mask" and backup result, if mask_l fails to find a cutoff point before the max_size we use the backup result
  // gotten with the easier to meet mask_b cutoff point. This should make it much more unlikely that we have to forcefully end chunks at max_size,
  // which helps better preserve the content defined nature of chunks and thus increase dedup ratios.
  std::optional<uint32_t> backup_i{};
  uint32_t mask_backup = 0;
  if (cdcz_cfg.use_supercdc_backup_mask) {
    mask_backup = mask_easy << 1;
  }

  // SuperCDC's Min-Max adjustment of the Gear Hash on jump to minimum chunk size, should improve deduplication ratios by better preserving
  // the content defined nature of the Hashes.
  // We backtrack a little to ensure when we get to the i we actually wanted we have the exact same hash as if we hadn't skipped anything.
  uint32_t remaining_minmax_adjustment = 0;
  if (cdcz_cfg.use_fastcdc_subminimum_skipping) {
    if (cdcz_cfg.use_supercdc_minmax_adjustment) {
      remaining_minmax_adjustment = std::min<uint32_t>(i, 31);
      i -= remaining_minmax_adjustment;
    }
  }
  // If we don't have skipping to the minimum size enabled we advance GEAR up to the minimum
  else {
    while (i < barrier) {
      result.pattern = (result.pattern << 1) + GEAR_TABLE[data[i]];
      i++;
    }
    barrier = static_cast<uint32_t>(std::min<uint64_t>(avg_size, size));
  }

  // If we had minmax adjustment to do, we do it now before the main GEAR CDC loop starts
  while (remaining_minmax_adjustment > 0) {
    result.pattern = (result.pattern << 1) + GEAR_TABLE[data[i]];
    remaining_minmax_adjustment--;
    i++;
  }
  // If enabled, we use the GEAR hash values to extract features as described on ODESS paper, which allows us to do Delta compression on top of
  // dedup with very reduced computational overhead
  const auto process_feature_computation = [&result, compute_features = cdcz_cfg.compute_features]() {
    if (compute_features) {
      if (!(result.pattern & delta_comp_constants::CDS_SAMPLING_MASK)) {
        if (result.features.empty()) {
          result.features.resize(16);
          result.features.shrink_to_fit();
        }
        for (int feature_i = 0; feature_i < 16; feature_i++) {
          const auto& [mi, ai] = delta_comp_constants::N_Transform_Coefs[feature_i];
          result.features[feature_i] = std::max<uint32_t>(result.features[feature_i], (mi * result.pattern + ai) % (1LL << 32));
        }
      }
    }
    };

  // Okay, finally we do pretty standard GEAR CDC, not being for readjusting the barrier after avg_size for FastCDC's normalized chunking and
  // SuperCDC's backup cut condition if those are enabled
  while (i < barrier) {
    result.pattern = (result.pattern << 1) + GEAR_TABLE[data[i]];
    if (!(result.pattern & mask_hard)) {
      result.candidate.type = CutPointCandidateType::HARD_CUT_MASK;
      result.candidate.offset = i;
      return result;
    }
    process_feature_computation();
    i++;
  }
  barrier = static_cast<uint32_t>(std::min<uint64_t>(max_size, size));
  while (i < barrier) {
    result.pattern = (result.pattern << 1) + GEAR_TABLE[data[i]];
    if (!(result.pattern & mask_easy)) {
      result.candidate.type = CutPointCandidateType::EASY_CUT_MASK;
      result.candidate.offset = i;
      return result;
    }
    if (cdcz_cfg.use_supercdc_backup_mask) {
      if (!backup_i.has_value() && !(result.pattern & mask_backup)) backup_i = i;
    }
    process_feature_computation();
    i++;
  }

  if (cdcz_cfg.use_supercdc_backup_mask && backup_i.has_value()) {
    result.candidate.type = CutPointCandidateType::SUPERCDC_BACKUP_MASK;
    result.candidate.offset = *backup_i;
  }
  else {
    result.candidate.type = i == max_size ? CutPointCandidateType::MAX_SIZE : CutPointCandidateType::EOF_CUT;
    result.candidate.offset = i;
  }
  if (!cdcz_cfg.use_fastcdc_normalized_chunking && result.candidate.type == CutPointCandidateType::EASY_CUT_MASK) {
    // If we don't have normalized chunking enabled, the EASY/HARD cut conditions are the same so we ensure any cut candidate
    // that was found on the second part of the iteration is treated the same as the others by marking them as HARD as well
    result.candidate.type = CutPointCandidateType::HARD_CUT_MASK;
  }
  return result;
}

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

CutPointCandidateWithContext cdc_next_cutpoint_candidate(
  const std::span<uint8_t> data,
  uint32_t mask_hard,
  uint32_t mask_medium,
  uint32_t mask_easy,
  const CDCZ_CONFIG& cdcz_cfg,
  uint32_t initial_pattern
) {
  // We craft a CDCZ_CONFIG and settings to force cdc_next_cutpoint to give us the next possible candidate according to any of the available criteria,
  // given that the chunk invariance condition is not satisfied
  CDCZ_CONFIG all_candidates_cfg {
    .compute_features = cdcz_cfg.compute_features,
    .use_fastcdc_subminimum_skipping = false,  // we don't want anything skipped
    .use_fastcdc_normalized_chunking = true, // set this to true so cdc_next_cutpoint doesn't adjust the masks we will give it
    .use_supercdc_minmax_adjustment = false,  // as we don't use subminimum skipping, it makes no sense to attempt minmax_adjustment
    .use_supercdc_backup_mask = false,  // if we need to search for possible backup masks we will handle it by forcing mask_easy
  };
  uint32_t condition_mask = cdcz_cfg.use_fastcdc_normalized_chunking ? mask_easy : mask_medium;
  if (cdcz_cfg.use_supercdc_backup_mask) {
    // instead of using it as a backup, we just set the SuperCDC backup mask as the mask to be used, to force any position that satisfies it to be returned
    condition_mask = condition_mask << 1;
  }
  CutPointCandidateWithContext result = cdc_next_cutpoint(
    data, 0, data.size(), data.size(),
    condition_mask, condition_mask, condition_mask,
    all_candidates_cfg, initial_pattern
  );
  result.candidate.type = result.candidate.type == CutPointCandidateType::MAX_SIZE
		? CutPointCandidateType::EOF_CUT
		: promote_cut_candidate(cdcz_cfg, result.pattern, mask_hard, mask_medium, mask_easy);
  return result;
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

#if defined(__AVX2__)
int32_t _mm256_extract_epi32_var_indx(const __m256i vec, const unsigned int i) {
  const __m128i indx = _mm_cvtsi32_si128(i);
  const __m256i val = _mm256_permutevar8x32_epi32(vec, _mm256_castsi128_si256(indx));
  return _mm_cvtsi128_si32(_mm256_castsi256_si128(val));
}

__m256i _mm256_insert_epi32_var_indx(const __m256i vec, int32_t val, const uint8_t lane_i) {
  const auto val_vec = _mm256_set1_epi32(val);
  auto lane_flag = [&lane_i](const uint8_t lane_pos) -> int32_t {
    return lane_i == lane_pos ? static_cast<int32_t>(0xFFFFFFFF) : 0;
  };
  const auto lane_vec = _mm256_setr_epi32(lane_flag(0), lane_flag(1), lane_flag(2), lane_flag(3), lane_flag(4), lane_flag(5), lane_flag(6), lane_flag(7));
  return _mm256_castps_si256(_mm256_blendv_ps(_mm256_castsi256_ps(vec), _mm256_castsi256_ps(val_vec), _mm256_castsi256_ps(lane_vec)));
}

// Precondition: Chunk invariance condition satisfied, that is, the data starts from the very beginning of the stream or after a chunk cutpoint we know for sure will be used
// Precondition 2: data either has size multiple of 4, or can be safely extended to be (extra data will be ignored)
void cdc_find_cut_points_avx2(
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
  const CDCZ_CONFIG& cdcz_cfg,
  bool is_first_segment
) {
  std::array<std::vector<CutPointCandidate>, 8> lane_results{};
  std::array<std::vector<std::vector<uint32_t>>, 8> lane_features_results{};
  std::array<std::vector<uint32_t>, 8> lane_current_features{};

  std::array<bool, 8> lane_achieved_chunk_invariance{ is_first_segment, false, false, false, false, false, false, false };
  __m256i mask_hard_vec = _mm256_set1_epi32(static_cast<int>(mask_hard));
  __m256i mask_easy_vec = _mm256_set1_epi32(static_cast<int>(mask_easy));
  if (!cdcz_cfg.use_fastcdc_normalized_chunking) {
    mask_hard_vec = _mm256_set1_epi32(static_cast<int>(mask_medium));
    mask_easy_vec = _mm256_set1_epi32(static_cast<int>(mask_medium));
  }
  __m256i cmask = _mm256_set1_epi32(0xff);
  __m256i hash_vec = _mm256_set1_epi32(0);
  const __m256i zero_vec = _mm256_set1_epi32(0);
  const __m256i ones_vec = _mm256_set1_epi32(1);
  const __m256i window_size_minus_one_vec = _mm256_set1_epi32(31);

  constexpr int32_t lane_count = 8;

  // Highway's portable GatherIndex requires we read with 32bit/4byte alignment as we have int32_t vectors.
  // This way we ensure we never attempt to Gather from outside the data boundaries, the last < 4 bytes can be finished off manually.
  const auto data_adjusted_size = pad_size_for_alignment(data.size(), 4) - 4;
  {
	  const uint64_t bytes_per_lane_u64 = data_adjusted_size / lane_count;
	  if (bytes_per_lane_u64 > std::numeric_limits<int32_t>::max()) {
	    throw std::runtime_error("Unable to process data such that lanes positions would overflow");
	  }
  }
  int32_t bytes_per_lane = pad_size_for_alignment(static_cast<int32_t>(data_adjusted_size / lane_count), 4) - 4;  // Idem 32bit alignment for Gathers
  __m256i vindex = _mm256_setr_epi32(0, bytes_per_lane, 2 * bytes_per_lane, 3 * bytes_per_lane, 4 * bytes_per_lane, 5 * bytes_per_lane, 6 * bytes_per_lane, 7 * bytes_per_lane);

  if (cdcz_cfg.use_fastcdc_subminimum_skipping && is_first_segment) {
    vindex = _mm256_insert_epi32(vindex, static_cast<int32_t>(min_size), 0);
  }

  // For each lane, the last index they are allowed to access
  __m256i vindex_max = _mm256_add_epi32(vindex, _mm256_set1_epi32(bytes_per_lane));
  // Because we read 4bytes at a time we need to ensure we are not reading past the data end
  vindex_max = _mm256_insert_epi32(vindex_max, static_cast<int32_t>(data_adjusted_size), 7);

  // SuperCDC's even easier "backup mask" and backup result, if mask_easy fails to find a cutoff point before the max_size we use the backup result
  // gotten with the easier to meet mask_b cutoff point. This should make it much more unlikely that we have to forcefully end chunks at max_size,
  // which helps better preserve the content defined nature of chunks and thus increase dedup ratios.
  __m256i mask_b_vec = _mm256_set1_epi32(static_cast<int>(mask_easy << 1));
  std::array<std::queue<int32_t>, 8> backup_cut_vec{};

  // SuperCDC's Min-Max adjustment of the Gear Hash on jump to minimum chunk size, should improve deduplication ratios by better preserving
  // the content defined nature of the Hashes.
  // We backtrack a little to ensure when we get to the i we actually wanted we have the exact same hash as if we hadn't skipped anything.
  std::array<int32_t, 8> minmax_adjustment_vec{};
  minmax_adjustment_vec[0] = _mm256_extract_epi32(vindex, 0);
  minmax_adjustment_vec[1] = _mm256_extract_epi32(vindex, 1);
  minmax_adjustment_vec[2] = _mm256_extract_epi32(vindex, 2);
  minmax_adjustment_vec[3] = _mm256_extract_epi32(vindex, 3);
  minmax_adjustment_vec[4] = _mm256_extract_epi32(vindex, 4);
  minmax_adjustment_vec[5] = _mm256_extract_epi32(vindex, 5);
  minmax_adjustment_vec[6] = _mm256_extract_epi32(vindex, 6);
  minmax_adjustment_vec[7] = _mm256_extract_epi32(vindex, 7);
  if (cdcz_cfg.use_supercdc_minmax_adjustment) {
    vindex = _mm256_sub_epi32(vindex, window_size_minus_one_vec);
    // HACK FOR REALLY LOW min_size or use_fastcdc_subminimum_skipping = false
    if (_mm256_extract_epi32(vindex, 0) < 0) {
      vindex = _mm256_insert_epi32(vindex, 0, 0);
      minmax_adjustment_vec[0] = 0;
    }
  }

  __m256i mask_vec = mask_b_vec;
  if (!cdcz_cfg.use_supercdc_backup_mask) {
    mask_vec = cdcz_cfg.use_fastcdc_normalized_chunking ? mask_easy_vec : mask_hard_vec;
  }

  __m256i cds_mask_vec = _mm256_set1_epi32(delta_comp_constants::CDS_SAMPLING_MASK);

  bool any_lane_marked_for_jump = false;
  std::array<int32_t, 8> jump_pos_vec = { 0, 0, 0, 0, 0, 0, 0, 0 };
  auto mark_lane_for_jump = [&jump_pos_vec, &any_lane_marked_for_jump, &cdcz_cfg](const int32_t pos, const uint8_t lane_i) {
    if (!cdcz_cfg.use_fastcdc_subminimum_skipping) return;
    jump_pos_vec[lane_i] = pos;
    any_lane_marked_for_jump = true;
  };

  auto get_result_type_for_lane = [&cdcz_cfg, &mask_hard, &mask_medium, &mask_easy](int32_t pattern) {
    // Do the cut candidate promoting behavior
    return promote_cut_candidate(cdcz_cfg, pattern, mask_hard, mask_medium, mask_easy);
  };

  auto process_lane = [&](const uint8_t lane_i, int32_t pos, const CutPointCandidateType result_type, bool ignore_max_pos = false) {
    // Lane already finished, and we are on a pos for another lane (or after data end)!
    if (!ignore_max_pos && pos >= _mm256_extract_epi32_var_indx(vindex_max, lane_i)) return;
    // If we are still doing minmax adjustment ignore the cut
    if (pos < minmax_adjustment_vec[lane_i]) return;
    // if the lane is marked for jump already then ignore the cut as well
    if (jump_pos_vec[lane_i] != 0) return;

    if (lane_achieved_chunk_invariance[lane_i]) {
      int32_t prev_cut_pos = lane_results[lane_i].empty() ? lane_i * bytes_per_lane : static_cast<int32_t>(lane_results[lane_i].back().offset);
      int32_t dist_with_prev = pos - prev_cut_pos;

      // >= max_size and not > max_size here even if it would be theoretically unnecessary because that's what we do on single threaded FastCDC as well
      while (dist_with_prev >= max_size) {
        if (!backup_cut_vec[lane_i].empty() && pos > backup_cut_vec[lane_i].front()) {
          prev_cut_pos = backup_cut_vec[lane_i].front();
          backup_cut_vec[lane_i].pop();
          lane_results[lane_i].emplace_back(CutPointCandidateType::SUPERCDC_BACKUP_MASK, prev_cut_pos);
        }
        else {
          prev_cut_pos = pos - dist_with_prev + max_size;
          lane_results[lane_i].emplace_back(CutPointCandidateType::MAX_SIZE, prev_cut_pos);
        }
        // Clear any pending backup cut candidate that is now invalid, that is, before the new prev_cut_pos + avg_size
        while (!backup_cut_vec[lane_i].empty() && backup_cut_vec[lane_i].front() < prev_cut_pos + avg_size) {
          backup_cut_vec[lane_i].pop();
        }
        dist_with_prev = pos - prev_cut_pos;
      }

      if (result_type == CutPointCandidateType::SUPERCDC_BACKUP_MASK) {
        if (dist_with_prev < avg_size) return;
        backup_cut_vec[lane_i].emplace(pos);
        return;
      }
      else if (result_type == CutPointCandidateType::EASY_CUT_MASK && dist_with_prev < avg_size) {
        return;
      }
      else if (dist_with_prev < min_size) {
        return;
      }

      lane_results[lane_i].emplace_back(result_type, pos);
      if (cdcz_cfg.compute_features) {
        lane_features_results[lane_i].emplace_back(std::move(lane_current_features[lane_i]));
        lane_current_features[lane_i] = std::vector<uint32_t>();
      }
      while (!backup_cut_vec[lane_i].empty()) backup_cut_vec[lane_i].pop();
      mark_lane_for_jump(pos, lane_i);
    }
    else {
      if (!lane_results[lane_i].empty()) {
        auto& prevCutCandidate = lane_results[lane_i].back();
        const auto dist_with_prev = pos - prevCutCandidate.offset;
        const auto is_prev_candidate_hard = prevCutCandidate.type == CutPointCandidateType::HARD_CUT_MASK;
        // if this happens this lane is back in sync with non-segmented processing!
        if (is_chunk_invariance_condition_satisfied(is_prev_candidate_hard, dist_with_prev, result_type, min_size, avg_size, max_size)) {
          lane_achieved_chunk_invariance[lane_i] = true;
          mark_lane_for_jump(pos, lane_i);
        }
      }
      lane_results[lane_i].emplace_back(result_type, pos);
      if (cdcz_cfg.compute_features) {
        lane_features_results[lane_i].emplace_back(std::move(lane_current_features[lane_i]));
        lane_current_features[lane_i] = std::vector<uint32_t>();
      }
    }
  };

  auto sample_feature_value = [&lane_current_features, &hash_vec, &vindex, &minmax_adjustment_vec, &jump_pos_vec, &vindex_max, &cdcz_cfg] (uint32_t lane_i) {
    const auto lane_pos = _mm256_extract_epi32_var_indx(vindex, lane_i);
    // Lane already finished, and we are on a pos for another lane (or after data end)!
    if (lane_pos >= _mm256_extract_epi32_var_indx(vindex_max, lane_i)) return;
    // If we are still doing minmax adjustment ignore the feature sampling match
    if (lane_pos < minmax_adjustment_vec[lane_i]) return;
    // if the lane is marked for jump already then ignore feature sampling match
    if (jump_pos_vec[lane_i] != 0) return;

    uint32_t pattern = _mm256_extract_epi32_var_indx(hash_vec, lane_i);
    if (lane_current_features[lane_i].empty()) {
      lane_current_features[lane_i].resize(16);
      lane_current_features[lane_i].shrink_to_fit();
    }
    for (int feature_i = 0; feature_i < 16; feature_i++) {
      const auto& [mi, ai] = delta_comp_constants::N_Transform_Coefs[feature_i];
      lane_current_features[lane_i][feature_i] = std::max<uint32_t>(lane_current_features[lane_i][feature_i], (mi * pattern + ai) % (1LL << 32));
    }
  };

  auto adjust_lane_for_jump = [&minmax_adjustment_vec, &jump_pos_vec, &vindex, &hash_vec, &min_size, &cdcz_cfg, &data] (const uint8_t lane_i) {
    int32_t new_lane_pos = jump_pos_vec[lane_i];
    if (new_lane_pos == 0) return;
    new_lane_pos += min_size;

    if (cdcz_cfg.use_supercdc_minmax_adjustment) {
      minmax_adjustment_vec[lane_i] = new_lane_pos;
      auto adjustment = std::min(31, new_lane_pos);
      new_lane_pos -= adjustment;
    }
    else {
      hash_vec = _mm256_insert_epi32_var_indx(hash_vec, 0, lane_i);
    }

    // We might not be able to actually do a portable GatherIndex from this position, we adjust if necessary to align to 32bit boundary
    const int32_t aligned_new_lane_pos = pad_size_for_alignment(new_lane_pos, 4);
    const auto bytes_to_alignment = aligned_new_lane_pos - new_lane_pos;
    if (cdcz_cfg.use_supercdc_minmax_adjustment) {
      // This is fine, doing adjustment for more than 31 bytes won't change the GEAR hashing result
      new_lane_pos -= bytes_to_alignment > 0 ? 4 - bytes_to_alignment : 0;
    }
    else {
	    // If we are not doing minmax adjustment we need to serially calculate the GEAR hash until we are aligned so SIMD processing can continue
      int32_t pattern = 0;
      for (int32_t i = new_lane_pos; i < new_lane_pos + bytes_to_alignment; i++) {
        pattern = (pattern << 1) + GEAR_TABLE[data[i]];
      }
      hash_vec = _mm256_insert_epi32_var_indx(hash_vec, pattern, lane_i);
      new_lane_pos += bytes_to_alignment;
    }

    vindex = _mm256_insert_epi32_var_indx(vindex, new_lane_pos, lane_i);
  };

  while (true) {
    vindex = _mm256_min_epi32(vindex, vindex_max);  // prevent vindex from pointing to invalid memory
    const __m256i is_finished_vmask = _mm256_cmpgt_epi32(vindex_max, vindex);
    const int is_lane_not_finished = _mm256_movemask_ps(_mm256_castsi256_ps(is_finished_vmask));
    // If all lanes are finished, we break, else we continue and lanes that are already finished will ignore results
    if (is_lane_not_finished == 0) break;

    __m256i cbytes = _mm256_i32gather_epi32(reinterpret_cast<int const*>(data.data()), vindex, 1);

    uint32_t j = 0;
    while (j < 4) {
      hash_vec = _mm256_slli_epi32(hash_vec, 1);  // Shift all the hash values for each lane at the same time
      __m256i idx = _mm256_and_si256(cbytes, cmask);  // Get byte on the lower bits of the packed 32bit lanes
      cbytes = _mm256_srli_epi32(cbytes, 8);  // We already got the byte on the lower bits, we can shift right to later get the next byte
      // This gives us the GEAR hash values for each of the bytes we just got, scale by 4 because 32bits=4bytes
      __m256i tentry = _mm256_i32gather_epi32(reinterpret_cast<int const*>(GEAR_TABLE), idx, 4);
      hash_vec = _mm256_add_epi32(hash_vec, tentry);  // Add the values we got from the GEAR hash values to the values on the hash

      // Compare each packed int by bitwise AND with the mask and checking that its 0
      const __m256i hash_eq_mask = _mm256_cmpeq_epi32(_mm256_and_si256(hash_vec, mask_vec), zero_vec);
      int lane_has_result = _mm256_movemask_ps(_mm256_castsi256_ps(hash_eq_mask));

      if (cdcz_cfg.compute_features) {
        // Check if content defined sampling condition is satisfied and we should sample feature values for some lane
        __m256i hash_cds_masked = _mm256_and_si256(hash_vec, cds_mask_vec);
        __m256i hash_cds_eq_vmask = _mm256_cmpeq_epi32(hash_cds_masked, zero_vec);

        const int hash_cds_eq_mask = _mm256_movemask_ps(_mm256_castsi256_ps(hash_cds_eq_vmask));
        if (hash_cds_eq_mask != 0) {
          if (hash_cds_eq_mask & 0b00000001) sample_feature_value(0);
          if (hash_cds_eq_mask & 0b00000010) sample_feature_value(1);
          if (hash_cds_eq_mask & 0b00000100) sample_feature_value(2);
          if (hash_cds_eq_mask & 0b00001000) sample_feature_value(3);
          if (hash_cds_eq_mask & 0b00010000) sample_feature_value(4);
          if (hash_cds_eq_mask & 0b00100000) sample_feature_value(5);
          if (hash_cds_eq_mask & 0b01000000) sample_feature_value(6);
          if (hash_cds_eq_mask & 0b10000000) sample_feature_value(7);
        }
      }

      if (lane_has_result != 0) {
        lane_has_result &= is_lane_not_finished;

        if (lane_has_result != 0) {
          if (lane_has_result & 0b00000001) process_lane(0, _mm256_extract_epi32(vindex, 0), get_result_type_for_lane(_mm256_extract_epi32(hash_vec, 0)));
          if (lane_has_result & 0b00000010) process_lane(1, _mm256_extract_epi32(vindex, 1), get_result_type_for_lane(_mm256_extract_epi32(hash_vec, 1)));
          if (lane_has_result & 0b00000100) process_lane(2, _mm256_extract_epi32(vindex, 2), get_result_type_for_lane(_mm256_extract_epi32(hash_vec, 2)));
          if (lane_has_result & 0b00001000) process_lane(3, _mm256_extract_epi32(vindex, 3), get_result_type_for_lane(_mm256_extract_epi32(hash_vec, 3)));
          if (lane_has_result & 0b00010000) process_lane(4, _mm256_extract_epi32(vindex, 4), get_result_type_for_lane(_mm256_extract_epi32(hash_vec, 4)));
          if (lane_has_result & 0b00100000) process_lane(5, _mm256_extract_epi32(vindex, 5), get_result_type_for_lane(_mm256_extract_epi32(hash_vec, 5)));
          if (lane_has_result & 0b01000000) process_lane(6, _mm256_extract_epi32(vindex, 6), get_result_type_for_lane(_mm256_extract_epi32(hash_vec, 6)));
          if (lane_has_result & 0b10000000) process_lane(7, _mm256_extract_epi32(vindex, 7), get_result_type_for_lane(_mm256_extract_epi32(hash_vec, 7)));
        }
      }

      ++j;
      vindex = _mm256_add_epi32(vindex, ones_vec);  // advance 1 byte in the data for each lane
    }

    if (any_lane_marked_for_jump) {
      adjust_lane_for_jump(0);
      adjust_lane_for_jump(1);
      adjust_lane_for_jump(2);
      adjust_lane_for_jump(3);
      adjust_lane_for_jump(4);
      adjust_lane_for_jump(5);
      adjust_lane_for_jump(6);
      adjust_lane_for_jump(7);

      jump_pos_vec = { 0, 0, 0, 0, 0, 0, 0, 0 };
      any_lane_marked_for_jump = false;
    }
  }

  {
    // Deal with any trailing data sequentially
    uint64_t i = _mm256_extract_epi32(vindex_max, 7) - 31;
    // The hash might be nonsense if the last lane finished before the others, so we just recover it
    uint64_t remaining_minmax_adjustment = 31;
    uint32_t pattern = 0;

    // TODO: Check backup mask? anything else I am missing here?
    // TODO: Can trunc for max_size, though it should be handled by select cut point candidates anyway

    const auto mask = cdcz_cfg.use_supercdc_backup_mask
      ? mask_easy << 1
      : cdcz_cfg.use_fastcdc_normalized_chunking ? mask_easy : mask_medium;
    while (i < data.size()) {
      pattern = (pattern << 1) + GEAR_TABLE[data[i]];
      if (remaining_minmax_adjustment > 0) {
        --remaining_minmax_adjustment;
        ++i;
        continue;
      }
      if (!(pattern & mask)) process_lane(7, i, get_result_type_for_lane(pattern), true);
      ++i;
    }
  }

  for (uint64_t lane_i = 0; lane_i < 8; lane_i++) {
    auto& cut_points_list = lane_results[lane_i];
    for (uint64_t n = 0; n < cut_points_list.size(); n++) {
      const auto& cut_point_candidate = cut_points_list[n];
      candidates.emplace_back(cut_point_candidate.type, base_offset + cut_point_candidate.offset);
      if (cdcz_cfg.compute_features) {
        candidate_features.emplace_back(std::move(lane_features_results[lane_i][n]));
      }
    }

    // There might be some leftover SuperCDC backup cut candidates on the lane, just emit them and let the cut selection step fix it
    while (!backup_cut_vec[lane_i].empty()) {
      candidates.emplace_back(CutPointCandidateType::SUPERCDC_BACKUP_MASK, base_offset + backup_cut_vec[lane_i].front());
      backup_cut_vec[lane_i].pop();
      // TODO: This is nonsense!
      if (cdcz_cfg.compute_features) {
        candidate_features.emplace_back();
      }
    }
  }

  if (candidates.empty() || candidates.back().offset != base_offset + data.size()) {
    candidates.emplace_back(CutPointCandidateType::EOF_CUT, base_offset + data.size());
    if (cdcz_cfg.compute_features) {
      candidate_features.emplace_back();
    }
  }
}
#endif

CdcCandidatesResult find_cdc_cut_candidates(std::span<uint8_t> data, const uint32_t min_size, const uint32_t avg_size, const uint32_t max_size, const CDCZ_CONFIG& cdcz_cfg, bool is_first_segment) {
  CdcCandidatesResult result{};
  if (data.empty()) return result;

  const auto make_mask = [](uint32_t bits) { return 0xFFFFFFFF << (32 - bits); };
  const auto bits = std::lround(std::log2(avg_size));
  const auto mask_hard = make_mask(bits + 1);
  const auto mask_medium = make_mask(bits);
  const auto mask_easy = make_mask(bits - 1);

  if (
    data.size() < 1024 || !cdcz_cfg.avx2_allowed
#ifndef __AVX2__
    || true
#endif
    ) {
    CutPointCandidateWithContext cdc_return{};
    uint64_t base_offset = 0;

    // If this is not the first segment then we need to deal with the previous segment extended data and attempt to recover chunk invariance
    if (!is_first_segment) {
      cdc_return = cdc_next_cutpoint_candidate(data, mask_hard, mask_medium, mask_easy, cdcz_cfg, cdc_return.pattern);

      base_offset += cdc_return.candidate.offset;
      result.candidates.emplace_back(cdc_return.candidate.type, base_offset);
      if (cdcz_cfg.compute_features) {
        result.candidatesFeatureResults.emplace_back(std::move(cdc_return.features));
      }
      data = std::span(data.data() + cdc_return.candidate.offset, data.size() - cdc_return.candidate.offset);
      bool is_prev_candidate_hard = cdc_return.candidate.type == CutPointCandidateType::HARD_CUT_MASK;

      // And now we need to recover chunk invariance in accordance to the chunk invariance recovery condition.
      // We are guaranteed to be back in sync with what non-segmented CDC would have done as soon as we find a cut candidate with
      // distance_w_prev_cut_candidate >= min_size and distance_w_prev_cut_candidate + min_size <= max_size.
      // As soon as we find that, we can break from here and keep processing as if we were processing without segments,
      // which in particular means we can exploit jumps to min_size again.
      while (!data.empty()) {
        // We need to skip processing the first byte as we already have the pattern with GEAR having processed that byte.
        cdc_return = cdc_next_cutpoint_candidate(
          std::span(data.data() + 1, data.size() - 1), mask_hard, mask_medium, mask_easy, cdcz_cfg, cdc_return.pattern
        );
        cdc_return.candidate.offset = cdc_return.candidate.offset + 1; // +1 because of the first byte that we skipped

        base_offset += cdc_return.candidate.offset;
        result.candidates.emplace_back(cdc_return.candidate.type, base_offset);
        if (cdcz_cfg.compute_features) {
          result.candidatesFeatureResults.emplace_back(std::move(cdc_return.features));
        }
        data = std::span(data.data() + cdc_return.candidate.offset, data.size() - cdc_return.candidate.offset);

        // For a more in depth explanation of the chunk invariance recovery condition please refer to my CDCZ paper
        if (is_chunk_invariance_condition_satisfied(is_prev_candidate_hard, cdc_return.candidate.offset, cdc_return.candidate.type, min_size, avg_size, max_size)) {
          // We demonstrated the current cut candidate will be used!, no more uncertainty, back in sync with non-segmented processing!
          break;
        }

        is_prev_candidate_hard = cdc_return.candidate.type == CutPointCandidateType::HARD_CUT_MASK;
      }
    }

    while (!data.empty()) {
      if (base_offset == 0) {
        cdc_return = cdc_next_cutpoint(data, min_size, avg_size, max_size, mask_hard, mask_medium, mask_easy, cdcz_cfg, cdc_return.pattern);
      }
      else {
        // Once again we need to skip processing the first byte as we already have the pattern with GEAR having processed that byte.
        // But this time we also need to adjust the min/avg/max sizes so the chunks match the sizes as if we hadn't skipped that first byte.
        cdc_return = cdc_next_cutpoint(
          std::span(data.data() + 1, data.size() - 1), min_size - 1, avg_size - 1, max_size - 1, mask_hard, mask_medium, mask_easy, cdcz_cfg, cdc_return.pattern
        );
        cdc_return.candidate.offset = cdc_return.candidate.offset + 1; // +1 because of the first byte that we skipped
      }

      base_offset += cdc_return.candidate.offset;
      result.candidates.emplace_back(cdc_return.candidate.type, base_offset);
      if (cdcz_cfg.compute_features) {
        result.candidatesFeatureResults.emplace_back(std::move(cdc_return.features));
      }
      data = std::span(data.data() + cdc_return.candidate.offset, data.size() - cdc_return.candidate.offset);
    }
  }
#ifdef __AVX2__
  else {
    uint64_t base_offset = 0;
    cdc_find_cut_points_avx2(
      result.candidates, result.candidatesFeatureResults, data, base_offset,
      min_size, avg_size, max_size, mask_hard, mask_medium, mask_easy, cdcz_cfg, is_first_segment
    );
  }
#endif

  if (cdcz_cfg.compute_features) {
    result.candidatesFeatureResults.shrink_to_fit();
  }
  result.candidates.shrink_to_fit();
  return result;
}

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
  bool copy_chunk_data
) {
  // TODO: WE ARE DISCARDING FEATURES!!
  if (!segments_eof && !new_cut_point_candidates.empty() && new_cut_point_candidates.back().type == CutPointCandidateType::EOF_CUT) {
    new_cut_point_candidates.pop_back();
    if (use_feature_extraction) {
      new_cut_point_candidates_features.pop_back();
    }
  }

  utility::ChunkEntry* last_made_chunk = nullptr;
  uint64_t last_made_chunk_size = 0;
  auto get_last_cut_point = [last_used_cut_point, &last_made_chunk, &last_made_chunk_size]() {
    return last_made_chunk != nullptr ? last_made_chunk->offset + last_made_chunk_size : last_used_cut_point;
    };
  auto last_cut_point = get_last_cut_point();
  uint64_t segment_data_pos = 0;

  auto make_chunk = [&](uint64_t candidate_index, uint64_t cut_point_offset) {
    const auto chunk_size = cut_point_offset - last_cut_point;
    // if the last_cut_point is on the previous segment some data might have come from the prev_segment_remaining_data so it's
    // not counting towards our segment data position
    const auto chunk_size_in_segment = chunk_size > prev_segment_remaining_data.size() ? chunk_size - prev_segment_remaining_data.size() : 0;

    process_pending_chunks.emplace_back(last_cut_point);
    last_made_chunk = &process_pending_chunks.back();

    if (!prev_segment_remaining_data.empty()) {
      last_made_chunk_size = chunk_size;
      if (copy_chunk_data) {
        last_made_chunk->chunk_data->data.resize(chunk_size);
        std::copy_n(prev_segment_remaining_data.data(), chunk_size - chunk_size_in_segment, last_made_chunk->chunk_data->data.data());
        if (chunk_size_in_segment > 0) {
          std::copy_n(segment_data.data() + segment_data_pos, chunk_size_in_segment, last_made_chunk->chunk_data->data.data() + chunk_size - chunk_size_in_segment);
        }
      }
      if (chunk_size_in_segment > 0 || chunk_size == prev_segment_remaining_data.size()) {
        prev_segment_remaining_data.clear();
      }
      else {
        prev_segment_remaining_data.erase(prev_segment_remaining_data.begin(), prev_segment_remaining_data.begin() + static_cast<int64_t>(chunk_size));
      }
    }
    else {
      last_made_chunk_size = chunk_size;
      if (copy_chunk_data) {
        last_made_chunk->chunk_data->data.resize(chunk_size);
        std::copy_n(segment_data.data() + segment_data_pos, chunk_size, last_made_chunk->chunk_data->data.data());
      }
    }
    if (copy_chunk_data) {
      last_made_chunk->chunk_data->data.shrink_to_fit();
    }

    if (use_feature_extraction) {
      if (new_cut_point_candidates_features.size() >= candidate_index + 1) {
        auto& cut_point_candidate_features = new_cut_point_candidates_features[candidate_index];
        if (!cut_point_candidate_features.empty()) {
          // Takes 4 features (32bit(4byte) fingerprints, so 4 of them is 16bytes) and hash them into a single SuperFeature (seed used arbitrarily just because it needed one)
          last_made_chunk->chunk_data->super_features = {
            XXH32(cut_point_candidate_features.data(), 16, delta_comp_constants::CDS_SAMPLING_MASK),
            XXH32(cut_point_candidate_features.data() + 4, 16, delta_comp_constants::CDS_SAMPLING_MASK),
            XXH32(cut_point_candidate_features.data() + 8, 16, delta_comp_constants::CDS_SAMPLING_MASK),
            XXH32(cut_point_candidate_features.data() + 12, 16, delta_comp_constants::CDS_SAMPLING_MASK)
          };
          last_made_chunk->chunk_data->feature_sampling_failure = false;
        }
      }
    }

    segment_data_pos += chunk_size_in_segment;
    };

  // If the last used cut point is on data already on this segment (should be within the 31/window_size - 1 bytes extension of the previous segment)
  // we start from that pos on this segment
  if (last_cut_point > segment_start_offset) {
    segment_data_pos = last_cut_point - segment_start_offset;
  }

  auto trunc_with_max_size = [&last_cut_point, &supercdc_backup_pos, &make_chunk, avg_size, max_size]
    (uint64_t candidate_index, uint64_t up_to_pos, bool including = false) {
    while (up_to_pos >= last_cut_point + max_size) {
      while (!supercdc_backup_pos.empty()) {
        // Any saved SuperCDC backup candidate that is not valid (before avg_size is skipped)
        auto& backup_pos = supercdc_backup_pos.front();
        if (backup_pos < last_cut_point + avg_size) {
          supercdc_backup_pos.pop();
          continue;
        }
        break;
      }
      // If we didn't skip all SuperCDC backup candidates, and we have one that should be used, then use it.
      if (!supercdc_backup_pos.empty() && supercdc_backup_pos.front() < last_cut_point + max_size) {
        // TODO: these chunks will share features, which is not right, need to figure out a solution
        make_chunk(candidate_index, supercdc_backup_pos.front());
        last_cut_point = supercdc_backup_pos.front();
        supercdc_backup_pos.pop();
      }
      // Else we insert a max_size chunk if the max_size was reached before up_to_pos or up_to_pos was reached but the including flag is on
      else if (including || up_to_pos != last_cut_point + max_size) {
        // TODO: these chunks will share features, which is not right, need to figure out a solution
        make_chunk(candidate_index, last_cut_point + max_size);
        last_cut_point = last_cut_point + max_size;
      }
      else {
        // We can only be here id the including flag is off, and we reached up_to_pos with max size, so there is nothing left to do.
        break;
      }
    }
  };

    for (uint64_t i = 0; i < new_cut_point_candidates.size(); i++) {
      auto& cut_point_candidate = new_cut_point_candidates[i];
      if (!is_first_segment && cut_point_candidate.offset <= 31) {
        // Any supposed candidate on a segment which is not the first one, before the GEAR_WINDOW_SIZE - 1 is fully processed is either spurious
        // (an artifact of the GEAR hash not being fully loaded yet) or it already should have gotten returned at the end of the previous segment
        continue;
      }
      last_cut_point = get_last_cut_point();

      const uint64_t adjusted_cut_point_candidate = segment_start_offset + cut_point_candidate.offset;
      if (cut_point_candidate.type == CutPointCandidateType::SUPERCDC_BACKUP_MASK) {
        supercdc_backup_pos.emplace(adjusted_cut_point_candidate);
        continue;
      }
      trunc_with_max_size(i, adjusted_cut_point_candidate);
      last_cut_point = get_last_cut_point();

      // TODO: we are discarding the features (if we are computing them) along with the rejected cut point candidate, we need to roll those over
      if (
        (
          (adjusted_cut_point_candidate < last_cut_point + min_size) ||
          // Normalized chunking easy condition is not valid until we are at avg_size
          (adjusted_cut_point_candidate < last_cut_point + avg_size && cut_point_candidate.type != CutPointCandidateType::HARD_CUT_MASK)
          ) &&
        // If this is the segment at EOF and also the last candidate, we can't skip it, as there won't be any future cut point to complete the chunk
        !(segments_eof && i == new_cut_point_candidates.size() - 1)
        ) {
        continue;
      }

      make_chunk(i, segment_start_offset + cut_point_candidate.offset);
    }

    // We might have reached the end of the cut point candidates but there is enough data at the end of the segment that we need to enforce
    // the max_size for chunks, as we did between cut point candidates, just between the actual last candidate and the end of the segment this time
    last_cut_point = get_last_cut_point();
    const auto segment_end_pos = segment_start_offset + segment_data.size();
    trunc_with_max_size(0, segment_end_pos, true);

    // If there is more unused data at the end of the segment than the window_size - 1 bytes of the extension, we save that data
    // as it will need to be used for the next chunk
    const auto segment_data_tail_len = segment_data.size() - segment_data_pos;
    if (segment_data_tail_len > 31) {
      prev_segment_remaining_data.resize(segment_data_tail_len - 31);
      prev_segment_remaining_data.shrink_to_fit();
      std::copy_n(segment_data.data() + segment_data_pos, segment_data_tail_len - 31, prev_segment_remaining_data.data());
    }

    return last_made_chunk_size;
}
