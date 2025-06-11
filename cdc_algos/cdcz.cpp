#include "cdcz.hpp"

#include <algorithm>
#include <deque>
#include <variant>

#include "contrib/xxHash/xxhash.h"

#include "cdc_algos/cdcz_serial.hpp"
#include "cdc_algos/cdcz_simd.hpp"
#include "cdc_algos/gear.hpp"
#include "delta_compression/delta.hpp"

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "cdc_algos/cdcz.cpp" 
#include <hwy/foreach_target.h>
#include <hwy/highway.h>

HWY_BEFORE_NAMESPACE();

namespace CDCZ {
namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;

static CdcCandidatesResult find_cdc_cut_candidates_impl(
  std::span<uint8_t> data, const uint32_t min_size, const uint32_t avg_size, const uint32_t max_size, const CDCZ_CONFIG& cdcz_cfg, bool is_first_segment
) {
  CdcCandidatesResult result{};
  if (data.empty()) return result;

  const auto make_mask = [](uint32_t bits) { return 0xFFFFFFFF << (32 - bits); };
  const auto bits = std::lround(std::log2(avg_size));
  const auto mask_hard = make_mask(bits + 1);
  const auto mask_medium = make_mask(bits);
  const auto mask_easy = make_mask(bits - 1);

#ifndef NDEBUG
  constexpr auto lane_count = hn::MaxLanes(HWY_FULL(int32_t){});
#else
  constexpr auto lane_count = hn::Lanes(HWY_FULL(int32_t){});
#endif

  if (data.size() < 1024 || !cdcz_cfg.avx2_allowed || lane_count < 8) {
    find_cdc_cut_candidates_serial(
      result.candidates, result.candidatesFeatureResults, data,
      min_size, avg_size, max_size, mask_hard, mask_medium, mask_easy, cdcz_cfg, is_first_segment
    );
  }
  else {
    find_cdc_cut_candidates_simd(
      result.candidates, result.candidatesFeatureResults, data,
      min_size, avg_size, max_size, mask_hard, mask_medium, mask_easy, cdcz_cfg, is_first_segment
    );
  }

  if (cdcz_cfg.compute_features) {
    result.candidatesFeatureResults.shrink_to_fit();
  }
  result.candidates.shrink_to_fit();
  return result;
}

}
}

HWY_AFTER_NAMESPACE();

#if HWY_ONCE

namespace CDCZ {
  HWY_EXPORT(find_cdc_cut_candidates_impl);
}

CdcCandidatesResult find_cdc_cut_candidates(
  std::span<uint8_t> data, const uint32_t min_size, const uint32_t avg_size, const uint32_t max_size, const CDCZ_CONFIG& cdcz_cfg, bool is_first_segment
) {
  return HWY_DYNAMIC_DISPATCH(CDCZ::find_cdc_cut_candidates_impl)(data, min_size, avg_size, max_size, cdcz_cfg, is_first_segment);
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

#endif
