#include "cdcz_serial.hpp"

// In case you are wondering why use Highway on the serial version, it's because most compilers are likely going to be able to do
// a good deal of optimization and autovectorization on their own, but if we are using a baseline target like SSE2, we lose on these
// "free" performance gains, so we do this to have them anyways
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "cdc_algos/cdcz_serial.cpp" 
#include <hwy/foreach_target.h>
#include <hwy/highway.h>

#include "cdc_algos/gear.hpp"
#include "delta_compression/delta.hpp"

HWY_BEFORE_NAMESPACE();

namespace CDCZ_SERIAL {
namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;

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
  CDCZ_CONFIG all_candidates_cfg{
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

static void find_cdc_cut_candidates_serial_impl(
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
) {
  CutPointCandidateWithContext cdc_return{};
  uint64_t base_offset = 0;

  // If this is not the first segment then we need to deal with the previous segment extended data and attempt to recover chunk invariance
  if (!is_first_segment) {
    cdc_return = cdc_next_cutpoint_candidate(data, mask_hard, mask_medium, mask_easy, cdcz_cfg, cdc_return.pattern);

    base_offset += cdc_return.candidate.offset;
    candidates.emplace_back(cdc_return.candidate.type, base_offset);
    if (cdcz_cfg.compute_features) {
      candidate_features.emplace_back(std::move(cdc_return.features));
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
      candidates.emplace_back(cdc_return.candidate.type, base_offset);
      if (cdcz_cfg.compute_features) {
        candidate_features.emplace_back(std::move(cdc_return.features));
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
    candidates.emplace_back(cdc_return.candidate.type, base_offset);
    if (cdcz_cfg.compute_features) {
      candidate_features.emplace_back(std::move(cdc_return.features));
    }
    data = std::span(data.data() + cdc_return.candidate.offset, data.size() - cdc_return.candidate.offset);
  }
}

}
}

HWY_AFTER_NAMESPACE();

#if HWY_ONCE

namespace CDCZ_SERIAL {
  HWY_EXPORT(find_cdc_cut_candidates_serial_impl);
}

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
) {
  HWY_DYNAMIC_DISPATCH(CDCZ_SERIAL::find_cdc_cut_candidates_serial_impl)(candidates, candidate_features, data, min_size, avg_size, max_size, mask_hard, mask_medium, mask_easy, cdcz_cfg, is_first_segment);
}

#endif
