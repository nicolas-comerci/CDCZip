#ifndef CDCZ_H
#define CDCZ_H

#include <span>
#include <variant>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

#include "cdc_algos/gear.hpp"
#include "delta_compression/delta.hpp"
#include "utils/chunks.hpp"

enum CutPointCandidateType : uint8_t {
  HARD_CUT_MASK,  // Satisfied harder mask before average size (FastCDC normalized chunking)
  EASY_CUT_MASK,  // Satisfied easier mask after average size (FastCDC normalized chunking)
  SUPERCDC_BACKUP_MASK,  // Satisfied SuperCDC backup mask because no other mask worked
  EOF_CUT  // Forcibly cut because the data span reached its EOF
};

struct CutPointCandidate {
  CutPointCandidateType type;
  uint64_t offset;
};

struct CutPointCandidateWithContext {
  CutPointCandidate candidate;
  uint32_t pattern;
  std::vector<uint32_t> features;
};

template<
  bool compute_features,
  bool use_fastcdc_subminimum_skipping = true,
  bool use_fastcdc_normalized_chunking = true,
  bool use_supercdc_minmax_adjustment = true,
  bool use_supercdc_backup_mask = true
>
CutPointCandidateWithContext cdc_next_cutpoint_candidate(
  const std::span<uint8_t> data,
  uint32_t min_size,
  uint32_t avg_size,
  uint32_t max_size,
  uint32_t mask_hard,
  uint32_t mask_medium,
  uint32_t mask_easy,
  uint32_t initial_pattern = 0
) {
  CutPointCandidateWithContext result;
  result.pattern = initial_pattern;
  uint32_t size = data.size();
  uint32_t barrier;
  uint32_t i;

  if constexpr (use_fastcdc_subminimum_skipping) {
    barrier = std::min(avg_size, size);
    i = std::min(barrier, min_size);
  }
  else {
    barrier = std::min(min_size, size);
    i = 0;
  }

  // if normalized chunking is disabled we set the harder and easier masks of normalized chunking to the "medium" of regular GEAR CDC,
  // which makes it behave the same way before and after the avg_size of chunk is reached, and thus works like regular GEAR CDC
  if constexpr (!use_fastcdc_normalized_chunking) {
    mask_hard = mask_medium;
    mask_easy = mask_medium;
  }

  // SuperCDC's even easier "backup mask" and backup result, if mask_l fails to find a cutoff point before the max_size we use the backup result
  // gotten with the easier to meet mask_b cutoff point. This should make it much more unlikely that we have to forcefully end chunks at max_size,
  // which helps better preserve the content defined nature of chunks and thus increase dedup ratios.
  std::optional<uint32_t> backup_i{};
  uint32_t mask_backup = 0;
  if constexpr (use_supercdc_backup_mask) {
    mask_backup = mask_easy >> 1;
  }

  // SuperCDC's Min-Max adjustment of the Gear Hash on jump to minimum chunk size, should improve deduplication ratios by better preserving
  // the content defined nature of the Hashes.
  // We backtrack a little to ensure when we get to the i we actually wanted we have the exact same hash as if we hadn't skipped anything.
  uint32_t remaining_minmax_adjustment = 0;
  if constexpr (use_supercdc_minmax_adjustment) {
    remaining_minmax_adjustment = std::min<uint32_t>(i, 31);
    i -= remaining_minmax_adjustment;
  }

  // If we don't have skipping to the minimum size enabled we advance GEAR up to the minimum
  if constexpr (!use_fastcdc_subminimum_skipping) {
    while (i < barrier) {
      result.pattern = (result.pattern >> 1) + GEAR_TABLE[data[i]];
      i++;
    }
    barrier = std::min(avg_size, size);
  }
  // And if we had minmax adjustment to do, we do it now as well
  if constexpr (use_supercdc_minmax_adjustment) {
    while (remaining_minmax_adjustment > 0) {
      result.pattern = (result.pattern >> 1) + GEAR_TABLE[data[i]];
      remaining_minmax_adjustment--;
      i++;
    }
  }
  // If enabled, we use the GEAR hash values to extract features as described on ODESS paper, which allows us to do Delta compression on top of
  // dedup with very reduced computational overhead
  const auto process_feature_computation = [&result]() {
    if constexpr (compute_features) {
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
    result.pattern = (result.pattern >> 1) + GEAR_TABLE[data[i]];
    if (!(result.pattern & mask_hard)) {
      result.candidate.type = CutPointCandidateType::HARD_CUT_MASK;
      result.candidate.offset = i;
      return result;
    }
    process_feature_computation();
    i++;
  }
  barrier = std::min(max_size, size);
  while (i < barrier) {
    result.pattern = (result.pattern >> 1) + GEAR_TABLE[data[i]];
    if (!(result.pattern & mask_easy)) {
      result.candidate.type = CutPointCandidateType::EASY_CUT_MASK;
      result.candidate.offset = i;
      return result;
    }
    if constexpr (use_supercdc_backup_mask) {
      if (!backup_i.has_value() && !(result.pattern & mask_backup)) backup_i = i;
    }
    process_feature_computation();
    i++;
  }

  if (use_supercdc_backup_mask && backup_i.has_value()) {
    result.candidate.type = CutPointCandidateType::SUPERCDC_BACKUP_MASK;
    result.candidate.offset = *backup_i;
  }
  else {
    result.candidate.type = CutPointCandidateType::EOF_CUT;
    result.candidate.offset = i;
  }
  return result;
}

struct CdcCandidatesResult {
  std::vector<CutPointCandidate> candidates;
  std::vector<std::vector<uint32_t>> candidatesFeatureResults;
};

template<
  bool compute_features,
  bool avx2_allowed = false,
  bool use_fastcdc_subminimum_skipping = true,
  bool use_fastcdc_normalized_chunking = true,
  bool use_supercdc_minmax_adjustment = true,
  bool use_supercdc_backup_mask = true
>
CdcCandidatesResult find_cdc_cut_candidates(std::span<uint8_t> data, uint32_t min_size, uint32_t avg_size, uint32_t max_size, bool is_first_segment = true) {
  CdcCandidatesResult result;
  if (data.empty()) return result;

  const auto make_mask = [](uint32_t bits) { return static_cast<uint32_t>(std::pow(2, bits)) - 1; };
  const auto bits = std::lround(std::log2(avg_size));
  const auto mask_hard = make_mask(bits + 1);
  const auto mask_medium = make_mask(bits);
  const auto mask_easy = make_mask(bits - 1);

  CutPointCandidateWithContext cdc_return{};
  uint64_t base_offset = 0;

  // If this is not the first segment then we need to deal with the previous segment extended data and attempt to recover chunk invariance
  if (!is_first_segment) {
    cdc_return = cdc_next_cutpoint_candidate<false, use_fastcdc_subminimum_skipping, use_fastcdc_normalized_chunking, use_supercdc_minmax_adjustment, use_supercdc_backup_mask>(
      data, 0, avg_size - 1, 4294967295, mask_hard, mask_medium, mask_easy, cdc_return.pattern
    );

    base_offset += cdc_return.candidate.offset;
    result.candidates.emplace_back(cdc_return.candidate.type, base_offset);
    if constexpr (compute_features) {
      result.candidatesFeatureResults.emplace_back();
    }
    data = std::span(data.data() + cdc_return.candidate.offset, data.size() - cdc_return.candidate.offset);

    // And now we need to recover chunk invariance in accordance to the chunk invariance recovery condition.
    // We are guaranteed to be back in sync with what non-segmented CDC would have done as soon as we find a cut candidate with
    // distance_w_prev_cut_candidate >= min_size and distance_w_prev_cut_candidate + min_size <= max_size.
    // As soon as we find that we can break from here and keep processing as if we were processing without segments,
    // which in particular means we can exploit jumps to min_size again.
    while (!data.empty()) {
      cdc_return = cdc_next_cutpoint_candidate<false, use_fastcdc_subminimum_skipping, use_fastcdc_normalized_chunking, use_supercdc_minmax_adjustment, use_supercdc_backup_mask>(
        std::span(data.data() + 1, data.size() - 1), 0, avg_size - 1, 4294967295,
        mask_hard, mask_medium, mask_easy, cdc_return.pattern
      );
      cdc_return.candidate.offset = cdc_return.candidate.offset + 1;

      base_offset += cdc_return.candidate.offset;
      result.candidates.emplace_back(cdc_return.candidate.type, base_offset);
      if constexpr (compute_features) {
        result.candidatesFeatureResults.emplace_back();
      }
      data = std::span(data.data() + cdc_return.candidate.offset, data.size() - cdc_return.candidate.offset);
      if (cdc_return.candidate.offset >= min_size && (cdc_return.candidate.offset + min_size <= max_size)) break;  // back in sync with non-segmented processing!
    }
  }

  if (
    data.size() < 1024 || !avx2_allowed
#ifndef __AVX2__
    || true
#endif
    ) {
    while (!data.empty()) {
      if (base_offset == 0) {
        cdc_return = cdc_next_cutpoint_candidate<compute_features, use_fastcdc_subminimum_skipping, use_fastcdc_normalized_chunking, use_supercdc_minmax_adjustment, use_supercdc_backup_mask>(
          data, min_size, avg_size, max_size, mask_hard, mask_medium, mask_easy, cdc_return.pattern
        );
      }
      else {
        cdc_return = cdc_next_cutpoint_candidate<compute_features, use_fastcdc_subminimum_skipping, use_fastcdc_normalized_chunking, use_supercdc_minmax_adjustment, use_supercdc_backup_mask>(
          std::span(data.data() + 1, data.size() - 1), min_size - 1, avg_size - 1, max_size - 1,
          mask_hard, mask_medium, mask_easy, cdc_return.pattern
        );
        cdc_return.candidate.offset = cdc_return.candidate.offset + 1;
      }

      base_offset += cdc_return.candidate.offset;
      result.candidates.emplace_back(cdc_return.candidate.type, base_offset);
      if constexpr (compute_features) {
        result.candidatesFeatureResults.emplace_back(std::move(cdc_return.features));
      }
      data = std::span(data.data() + cdc_return.candidate.offset, data.size() - cdc_return.candidate.offset);
    }
  }
#ifdef __AVX2__
  else {
    cdc_find_cut_points_with_invariance<compute_features, use_fastcdc_subminimum_skipping, use_fastcdc_normalized_chunking, use_supercdc_minmax_adjustment, use_supercdc_backup_mask>(
      result.candidates, result.candidatesFeatureResults, data, base_offset,
      min_size, avg_size, max_size, mask_hard, mask_medium, mask_easy
    );
  }
#endif

  if constexpr (compute_features) {
    result.candidatesFeatureResults.shrink_to_fit();
  }
  result.candidates.shrink_to_fit();
  return result;
}

#if defined(__AVX2__)
static int32_t _mm256_extract_epi32_var_indx(const __m256i vec, const unsigned int i) {
  const __m128i indx = _mm_cvtsi32_si128(i);
  const __m256i val = _mm256_permutevar8x32_epi32(vec, _mm256_castsi128_si256(indx));
  return _mm_cvtsi128_si32(_mm256_castsi256_si128(val));
}

static __m256i _mm256_insert_epi32_var_indx(const __m256i vec, int32_t val, const uint8_t lane_i) {
  const auto val_vec = _mm256_set1_epi32(val);
  auto lane_flag = [&lane_i](const uint8_t lane_pos) -> int32_t {
    return lane_i == lane_pos ? static_cast<int32_t>(0xFFFFFFFF) : 0;
    };
  const auto lane_vec = _mm256_setr_epi32(lane_flag(0), lane_flag(1), lane_flag(2), lane_flag(3), lane_flag(4), lane_flag(5), lane_flag(6), lane_flag(7));
  return _mm256_castps_si256(_mm256_blendv_ps(_mm256_castsi256_ps(vec), _mm256_castsi256_ps(val_vec), _mm256_castsi256_ps(lane_vec)));
}

// Precondition: Chunk invariance condition satisfied
template<
  bool compute_features,
  bool use_fastcdc_subminimum_skipping,
  bool use_fastcdc_normalized_chunking,
  bool use_supercdc_minmax_adjustment,
  bool use_supercdc_backup_mask,
  typename CandidateFeaturesResult = std::conditional_t<compute_features, std::vector<std::vector<uint32_t>>, std::monostate>
>
void cdc_find_cut_points_with_invariance(
  std::vector<CutPointCandidate>& candidates,
  CandidateFeaturesResult& candidate_features,
  std::span<uint8_t> data,
  uint64_t base_offset,
  int32_t min_size,
  int32_t avg_size,
  int32_t max_size,
  uint32_t mask_hard,
  uint32_t mask_medium,
  uint32_t mask_easy
) {
  std::array<std::vector<CutPointCandidate>, 8> lane_results{};
  std::array<CandidateFeaturesResult, 8> lane_features_results{};
  std::array<std::vector<uint32_t>, 8> lane_current_features{};

  std::array<bool, 8> lane_achieved_chunk_invariance{ true, false, false, false, false, false, false, false };
  __m256i mask_hard_vec = _mm256_set1_epi32(static_cast<int>(mask_hard));
  __m256i mask_easy_vec = _mm256_set1_epi32(static_cast<int>(mask_easy));
  if (!use_fastcdc_normalized_chunking) {
    mask_hard_vec = _mm256_set1_epi32(static_cast<int>(mask_medium));
    mask_easy_vec = _mm256_set1_epi32(static_cast<int>(mask_medium));
  }
  __m256i cmask = _mm256_set1_epi32(0xff);
  __m256i hash = _mm256_set1_epi32(0);
  const __m256i zero_vec = _mm256_set1_epi32(0);
  const __m256i ones_vec = _mm256_set1_epi32(1);
  const __m256i window_size_minus_one_vec = _mm256_set1_epi32(31);

  if (data.size() > std::numeric_limits<int32_t>::max()) {
    throw std::runtime_error("Unable to process data such that lanes positions would overflow");
  }
  int32_t bytes_per_lane = static_cast<int32_t>(data.size() / 8ULL);
  __m256i vindex = _mm256_setr_epi32(0, bytes_per_lane, 2 * bytes_per_lane, 3 * bytes_per_lane, 4 * bytes_per_lane, 5 * bytes_per_lane, 6 * bytes_per_lane, 7 * bytes_per_lane);

  // This vector has the max allowed size for each lane's current chunk, we start at the maximum int to essentially disable max size chunks, they are set as soon
  // as chunk invariant condition is satisfied, and from then on after starting a new chunk
  __m256i max_size_vec = _mm256_set1_epi32(std::numeric_limits<int32_t>::max());
  __m256i avg_size_vec = _mm256_set1_epi32(std::numeric_limits<int32_t>::max());

  if constexpr (use_fastcdc_subminimum_skipping) {
    vindex = _mm256_insert_epi32(vindex, static_cast<int32_t>(min_size), 0);
  }

  // For each lane, the last index they are allowed to access
  __m256i vindex_max = _mm256_add_epi32(vindex, _mm256_set1_epi32(bytes_per_lane));
  vindex_max = _mm256_insert_epi32(vindex_max, static_cast<int32_t>(data.size()), 7);
  // Because we read 4bytes at a time we need to ensure we are not reading past the data end
  __m256i vindex_max_avx2 = vindex_max;
  vindex_max_avx2 = _mm256_sub_epi32(vindex_max_avx2, _mm256_set_epi32(0, 0, 0, 0, 0, 0, 0, -4));

  // SuperCDC's even easier "backup mask" and backup result, if mask_easy fails to find a cutoff point before the max_size we use the backup result
  // gotten with the easier to meet mask_b cutoff point. This should make it much more unlikely that we have to forcefully end chunks at max_size,
  // which helps better preserve the content defined nature of chunks and thus increase dedup ratios.
  __m256i mask_b_vec = _mm256_set1_epi32(static_cast<int>(mask_easy >> 1));
  __m256i backup_cut_vec = max_size_vec;

  // SuperCDC's Min-Max adjustment of the Gear Hash on jump to minimum chunk size, should improve deduplication ratios by better preserving
  // the content defined nature of the Hashes.
  // We backtrack a little to ensure when we get to the i we actually wanted we have the exact same hash as if we hadn't skipped anything.
  __m256i minmax_adjustment_vec = window_size_minus_one_vec;
  if constexpr (use_supercdc_minmax_adjustment) {
    vindex = _mm256_sub_epi32(vindex, window_size_minus_one_vec);
    // HACK FOR REALLY LOW min_size or use_fastcdc_subminimum_skipping = false
    if (_mm256_extract_epi32(vindex, 0) < 0) {
      vindex = _mm256_insert_epi32(vindex, 0, 0);
      minmax_adjustment_vec = _mm256_insert_epi32(minmax_adjustment_vec, 0, 0);
    }
  }

  __m256i cds_mask_vec = _mm256_set1_epi32(delta_comp_constants::CDS_SAMPLING_MASK);

  __m256i lane_not_marked_for_jump_vec = _mm256_set1_epi32(0xFFFFFFFF);
  int lane_not_marked_for_jump = 0b11111111;
  __m256i jump_vec = zero_vec;

  __m256i minmax_adjustment_ready_vmask;

  auto get_result_type_for_lane = [](int lane_has_backup_result, int lane_has_easy_result, int lane_has_hard_result, uint32_t lane_i) {
    if (lane_has_hard_result & (0b1 << lane_i)) {
      return CutPointCandidateType::HARD_CUT_MASK;
    }
    if (lane_has_easy_result & (0b1 << lane_i)) {
      return CutPointCandidateType::EASY_CUT_MASK;
    }
    return CutPointCandidateType::SUPERCDC_BACKUP_MASK;
    };

  auto process_lane = [&lane_results, &lane_achieved_chunk_invariance, &min_size, &avg_size, &max_size, &bytes_per_lane, &vindex_max,
    &lane_not_marked_for_jump_vec, &jump_vec, &avg_size_vec, &max_size_vec, &backup_cut_vec, &lane_features_results, &lane_current_features]
    (const uint8_t lane_i, int32_t pos, const CutPointCandidateType result_type) {
    // Lane already finished, and we are on a pos for another lane (or after data end)!
    if (pos >= _mm256_extract_epi32_var_indx(vindex_max, lane_i)) return;

    if (lane_achieved_chunk_invariance[lane_i]) {
      if (lane_results[lane_i].empty()) {
        if (pos >= lane_i * bytes_per_lane + min_size) {
          if (use_supercdc_backup_mask && pos == _mm256_extract_epi32_var_indx(max_size_vec, lane_i) && pos > _mm256_extract_epi32_var_indx(backup_cut_vec, lane_i)) {
            pos = _mm256_extract_epi32_var_indx(backup_cut_vec, lane_i);
          }
          lane_results[lane_i].emplace_back(result_type, pos);
          if constexpr (compute_features) {
            lane_features_results[lane_i].emplace_back(std::move(lane_current_features[lane_i]));
            lane_current_features[lane_i] = std::vector<uint32_t>();
          }
          max_size_vec = _mm256_insert_epi32_var_indx(max_size_vec, pos + max_size, lane_i);
          avg_size_vec = _mm256_insert_epi32_var_indx(avg_size_vec, pos + avg_size, lane_i);
          lane_not_marked_for_jump_vec = _mm256_insert_epi32_var_indx(lane_not_marked_for_jump_vec, 0, lane_i);
          jump_vec = _mm256_insert_epi32_var_indx(jump_vec, pos, lane_i);
        }
      }
      else {
        const auto dist_with_prev = pos - lane_results[lane_i].back().offset;
        if (dist_with_prev >= static_cast<uint64_t>(min_size)) {
          if (use_supercdc_backup_mask && pos == _mm256_extract_epi32_var_indx(max_size_vec, lane_i) && pos > _mm256_extract_epi32_var_indx(backup_cut_vec, lane_i)) {
            pos = _mm256_extract_epi32_var_indx(backup_cut_vec, lane_i);
          }
          lane_results[lane_i].emplace_back(result_type, pos);
          if constexpr (compute_features) {
            lane_features_results[lane_i].emplace_back(std::move(lane_current_features[lane_i]));
            lane_current_features[lane_i] = std::vector<uint32_t>();
          }
          max_size_vec = _mm256_insert_epi32_var_indx(max_size_vec, pos + max_size, lane_i);
          avg_size_vec = _mm256_insert_epi32_var_indx(avg_size_vec, pos + avg_size, lane_i);
          lane_not_marked_for_jump_vec = _mm256_insert_epi32_var_indx(lane_not_marked_for_jump_vec, 0, lane_i);
          jump_vec = _mm256_insert_epi32_var_indx(jump_vec, pos, lane_i);
        }
      }
    }
    else {
      if (!lane_results[lane_i].empty()) {
        auto& prevCutCandidate = lane_results[lane_i].back();
        const auto dist_with_prev = pos - prevCutCandidate.offset;
        // if this happens this lane is back in sync with non-segmented processing!
        // HARD CUT type is required because easy cut or backup cut could get rejected if they happen before the avg_size, so we can't use them as reference
        if (dist_with_prev >= static_cast<uint64_t>(min_size) && (dist_with_prev + min_size <= static_cast<uint64_t>(max_size)) && prevCutCandidate.type == CutPointCandidateType::HARD_CUT_MASK) {
          lane_achieved_chunk_invariance[lane_i] = true;
          max_size_vec = _mm256_insert_epi32_var_indx(max_size_vec, pos + max_size, lane_i);
          avg_size_vec = _mm256_insert_epi32_var_indx(avg_size_vec, pos + avg_size, lane_i);
          lane_not_marked_for_jump_vec = _mm256_insert_epi32_var_indx(lane_not_marked_for_jump_vec, 0, lane_i);
          jump_vec = _mm256_insert_epi32_var_indx(jump_vec, pos, lane_i);
        }
      }
      lane_results[lane_i].emplace_back(result_type, pos);
      if constexpr (compute_features) {
        lane_features_results[lane_i].emplace_back(std::move(lane_current_features[lane_i]));
        lane_current_features[lane_i] = std::vector<uint32_t>();
      }
    }
    };

  auto sample_feature_value = [&lane_current_features](uint32_t lane_i, uint32_t pattern) {
    if (lane_current_features[lane_i].empty()) {
      lane_current_features[lane_i].resize(16);
      lane_current_features[lane_i].shrink_to_fit();
    }
    for (int feature_i = 0; feature_i < 16; feature_i++) {
      const auto& [mi, ai] = delta_comp_constants::N_Transform_Coefs[feature_i];
      lane_current_features[lane_i][feature_i] = std::max<uint32_t>(lane_current_features[lane_i][feature_i], (mi * pattern + ai) % (1LL << 32));
    }
    };

  auto adjust_lane_for_jump = [&minmax_adjustment_vec, &backup_cut_vec, &max_size_vec, &jump_vec, &vindex, &min_size](const uint8_t lane_i) {
    int32_t new_lane_pos = _mm256_extract_epi32_var_indx(jump_vec, lane_i);
    if constexpr (use_fastcdc_subminimum_skipping) {
      new_lane_pos += min_size;
    }

    if constexpr (use_supercdc_minmax_adjustment) {
      auto adjustment = std::min(31, new_lane_pos);
      new_lane_pos -= adjustment;
      minmax_adjustment_vec = _mm256_insert_epi32_var_indx(minmax_adjustment_vec, adjustment, lane_i);
    }
    if constexpr (use_supercdc_backup_mask) {
      backup_cut_vec = _mm256_insert_epi32_var_indx(backup_cut_vec, _mm256_extract_epi32_var_indx(max_size_vec, lane_i), lane_i);
    }

    vindex = _mm256_insert_epi32_var_indx(vindex, new_lane_pos, lane_i);
    };

  while (true) {
    vindex = _mm256_min_epi32(vindex, vindex_max_avx2);
    __m256i is_finish_vmask = _mm256_cmpgt_epi32(vindex_max_avx2, vindex);
    int is_lane_not_finished = _mm256_movemask_ps(_mm256_castsi256_ps(is_finish_vmask));
    // If all lanes are finished we break, else we continue and lanes that are already finished will ignore results
    if (is_lane_not_finished == 0) break;

    __m256i cbytes = _mm256_i32gather_epi32(reinterpret_cast<int const*>(data.data()), vindex, 1);

    uint32_t j = 0;
    while (j < 4) {
      hash = _mm256_srli_epi32(hash, 1);  // Shift all the hash values for each lane at the same time
      __m256i idx = _mm256_and_si256(cbytes, cmask);  // Get byte on the lower bits of the packed 32bit lanes
      cbytes = _mm256_srli_epi32(cbytes, 8);  // We already got the byte on the lower bits, we can shift right to later get the next byte
      // This gives us the GEAR hash values for each of the bytes we just got, scale by 4 because 32bits=4bytes
      __m256i tentry = _mm256_i32gather_epi32(reinterpret_cast<int const*>(GEAR_TABLE), idx, 4);
      hash = _mm256_add_epi32(hash, tentry);  // Add the values we got from the GEAR hash values to the values on the hash

      // Compare each packed int by bitwise AND with the mask and checking that its 0
      int lane_has_result = 0;
      int lane_has_backup_result = 0;
      int lane_has_easy_result = 0;
      int lane_has_hard_result = 0;
      if constexpr (use_supercdc_backup_mask) {
        __m256i hash_backup_masked = _mm256_and_si256(hash, mask_b_vec);
        __m256i hash_backup_eq_mask = _mm256_cmpeq_epi32(hash_backup_masked, zero_vec);
        lane_has_backup_result = _mm256_movemask_ps(_mm256_castsi256_ps(hash_backup_eq_mask));
        lane_has_result = lane_has_backup_result;
      }
      if (!use_supercdc_backup_mask || lane_has_result != 0) {
        if constexpr (use_fastcdc_normalized_chunking) {
          __m256i hash_easy_masked = _mm256_and_si256(hash, mask_easy_vec);
          __m256i hash_easy_eq_mask = _mm256_cmpeq_epi32(hash_easy_masked, zero_vec);
          lane_has_easy_result = _mm256_movemask_ps(_mm256_castsi256_ps(hash_easy_eq_mask));
          lane_has_result |= lane_has_easy_result;
        }
        if (!use_fastcdc_normalized_chunking || lane_has_result != 0) {
          __m256i hash_hard_masked = _mm256_and_si256(hash, mask_hard_vec);
          __m256i hash_hard_eq_mask = _mm256_cmpeq_epi32(hash_hard_masked, zero_vec);
          lane_has_hard_result = _mm256_movemask_ps(_mm256_castsi256_ps(hash_hard_eq_mask));
          lane_has_result |= lane_has_hard_result;
        }
      }

      if constexpr (use_supercdc_minmax_adjustment) {
        minmax_adjustment_ready_vmask = _mm256_cmpgt_epi32(ones_vec, minmax_adjustment_vec);
      }

      if constexpr (compute_features) {
        // Check if content defined sampling condition is satisfied and we should sample feature values for some lane
        __m256i hash_cds_masked = _mm256_and_si256(hash, cds_mask_vec);
        __m256i hash_cds_eq_vmask = _mm256_cmpeq_epi32(hash_cds_masked, zero_vec);
        // Ensure lane is not ready for jump or still doing min-max adjustment, in any of those cases we shouldn't sample
        hash_cds_eq_vmask = _mm256_and_si256(hash_cds_eq_vmask, lane_not_marked_for_jump_vec);
        if constexpr (use_supercdc_minmax_adjustment) {
          hash_cds_eq_vmask = _mm256_and_si256(hash_cds_eq_vmask, minmax_adjustment_ready_vmask);
        }
        int hash_cds_eq_mask = _mm256_movemask_ps(_mm256_castsi256_ps(hash_cds_eq_vmask));

        if (hash_cds_eq_mask != 0) {
          if (hash_cds_eq_mask & 0b00000001) sample_feature_value(0, _mm256_extract_epi32(hash, 0));
          if (hash_cds_eq_mask & 0b00000010) sample_feature_value(1, _mm256_extract_epi32(hash, 1));
          if (hash_cds_eq_mask & 0b00000100) sample_feature_value(2, _mm256_extract_epi32(hash, 2));
          if (hash_cds_eq_mask & 0b00001000) sample_feature_value(3, _mm256_extract_epi32(hash, 3));
          if (hash_cds_eq_mask & 0b00010000) sample_feature_value(4, _mm256_extract_epi32(hash, 4));
          if (hash_cds_eq_mask & 0b00100000) sample_feature_value(5, _mm256_extract_epi32(hash, 5));
          if (hash_cds_eq_mask & 0b01000000) sample_feature_value(6, _mm256_extract_epi32(hash, 6));
          if (hash_cds_eq_mask & 0b10000000) sample_feature_value(7, _mm256_extract_epi32(hash, 7));
        }
      }

      lane_not_marked_for_jump = _mm256_movemask_ps(_mm256_castsi256_ps(lane_not_marked_for_jump_vec));

      // We order the condition checks by probability, with reasonable min_size/masks that a lane hash a hash that satisfies the condition is already pretty rare,
      // so most of the time we are better checking that before bothering to check all the others conditions.
      // Other than the max_size which also needs to always be checked.
      if (lane_has_result != 0) {
        lane_has_result &= is_lane_not_finished & lane_not_marked_for_jump;
        if constexpr (use_supercdc_minmax_adjustment) {
          lane_has_result &= _mm256_movemask_ps(_mm256_castsi256_ps(minmax_adjustment_ready_vmask));
        }
      }

      if (lane_has_result != 0) {
        if (lane_has_result & 0b00000001) process_lane(0, _mm256_extract_epi32(vindex, 0), get_result_type_for_lane(lane_has_backup_result, lane_has_easy_result, lane_has_hard_result, 0));
        if (lane_has_result & 0b00000010) process_lane(1, _mm256_extract_epi32(vindex, 1), get_result_type_for_lane(lane_has_backup_result, lane_has_easy_result, lane_has_hard_result, 1));
        if (lane_has_result & 0b00000100) process_lane(2, _mm256_extract_epi32(vindex, 2), get_result_type_for_lane(lane_has_backup_result, lane_has_easy_result, lane_has_hard_result, 2));
        if (lane_has_result & 0b00001000) process_lane(3, _mm256_extract_epi32(vindex, 3), get_result_type_for_lane(lane_has_backup_result, lane_has_easy_result, lane_has_hard_result, 3));
        if (lane_has_result & 0b00010000) process_lane(4, _mm256_extract_epi32(vindex, 4), get_result_type_for_lane(lane_has_backup_result, lane_has_easy_result, lane_has_hard_result, 4));
        if (lane_has_result & 0b00100000) process_lane(5, _mm256_extract_epi32(vindex, 5), get_result_type_for_lane(lane_has_backup_result, lane_has_easy_result, lane_has_hard_result, 5));
        if (lane_has_result & 0b01000000) process_lane(6, _mm256_extract_epi32(vindex, 6), get_result_type_for_lane(lane_has_backup_result, lane_has_easy_result, lane_has_hard_result, 6));
        if (lane_has_result & 0b10000000) process_lane(7, _mm256_extract_epi32(vindex, 7), get_result_type_for_lane(lane_has_backup_result, lane_has_easy_result, lane_has_hard_result, 7));
      }

      if constexpr (use_supercdc_minmax_adjustment) {
        minmax_adjustment_vec = _mm256_sub_epi32(minmax_adjustment_vec, ones_vec);
      }

      ++j;
      vindex = _mm256_add_epi32(vindex, ones_vec);  // advance 1 byte in the data for each lane
    }

    if (lane_not_marked_for_jump != 0b11111111) {
      if (!(lane_not_marked_for_jump & 0b00000001)) { adjust_lane_for_jump(0); }
      if (!(lane_not_marked_for_jump & 0b00000010)) { adjust_lane_for_jump(1); }
      if (!(lane_not_marked_for_jump & 0b00000100)) { adjust_lane_for_jump(2); }
      if (!(lane_not_marked_for_jump & 0b00001000)) { adjust_lane_for_jump(3); }
      if (!(lane_not_marked_for_jump & 0b00010000)) { adjust_lane_for_jump(4); }
      if (!(lane_not_marked_for_jump & 0b00100000)) { adjust_lane_for_jump(5); }
      if (!(lane_not_marked_for_jump & 0b01000000)) { adjust_lane_for_jump(6); }
      if (!(lane_not_marked_for_jump & 0b10000000)) { adjust_lane_for_jump(7); }

      lane_not_marked_for_jump_vec = _mm256_set1_epi32(0xFFFFFFFF);
    }
  }

  uint64_t i = _mm256_extract_epi32(vindex, 7);
  auto pattern = _mm256_extract_epi32(hash, 7);  // Recover hash value from last lane
  // Deal with any trailing data sequentially
  // TODO: Check backup mask? anything else I am missing here?
  while (i < data.size()) {
    pattern = (pattern >> 1) + GEAR_TABLE[data[i]];
    if (!(pattern & mask_easy)) process_lane(7, i, CutPointCandidateType::EASY_CUT_MASK);
    ++i;
  }

  for (uint64_t lane = 0; lane < 8; lane++) {
    auto& cut_points_list = lane_results[lane];
    for (uint64_t n = 0; n < cut_points_list.size(); n++) {
      auto& cut_point_candidate = cut_points_list[n];
      candidates.emplace_back(cut_point_candidate.type, base_offset + cut_point_candidate.offset);
      if constexpr (compute_features) {
        candidate_features.emplace_back(std::move(lane_features_results[lane][n]));
      }
    }
  }

  if (candidates.empty() || candidates.back().offset != base_offset + data.size()) {
    // TODO: EOF_CUT and MAX_SIZE_CUT ARE THE SAME? SHOULDN'T THEY BE DIFFERENT?
    candidates.emplace_back(CutPointCandidateType::EOF_CUT, base_offset + data.size());
    if constexpr (compute_features) {
      candidate_features.emplace_back();
    }
  }
}
#endif

uint64_t select_cut_point_candidates(
  std::vector<CutPointCandidate>& new_cut_point_candidates,
  std::vector<std::vector<uint32_t>>& new_cut_point_candidates_features,
  std::deque<utility::ChunkEntry>& process_pending_chunks,
  uint64_t last_used_cut_point,
  uint64_t segment_start_offset,
  std::span<uint8_t> segment_data,
  std::vector<uint8_t>& prev_segment_remaining_data,
  uint32_t min_size,
  uint32_t avg_size,
  uint32_t max_size,
  bool segments_eof,
  bool use_feature_extraction,
  bool copy_chunk_data = true
);

#endif