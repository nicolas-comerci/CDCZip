#include "cdcz.hpp"

#include <algorithm>
#include <deque>
#include <variant>
#include <queue>

#include "contrib/xxHash/xxhash.h"

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
  bool copy_chunk_data
) {
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
    const auto chunk_size_in_segment = chunk_size - prev_segment_remaining_data.size();

    process_pending_chunks.emplace_back(last_cut_point);
    last_made_chunk = &process_pending_chunks.back();

    if (!prev_segment_remaining_data.empty()) {
      last_made_chunk_size = chunk_size;
      if (copy_chunk_data) {
        last_made_chunk->chunk_data->data.resize(chunk_size);
        std::copy_n(prev_segment_remaining_data.data(), prev_segment_remaining_data.size(), last_made_chunk->chunk_data->data.data());
        std::copy_n(segment_data.data() + segment_data_pos, chunk_size_in_segment, last_made_chunk->chunk_data->data.data() + prev_segment_remaining_data.size());
      }
      prev_segment_remaining_data.clear();
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

  std::queue<uint64_t> supercdc_backup_pos;

  auto trunc_with_max_size = [&last_cut_point, &supercdc_backup_pos, &make_chunk, avg_size, max_size]
    (uint64_t candidate_index, uint64_t adjusted_cut_point_candidate) {
    while (adjusted_cut_point_candidate > last_cut_point + max_size) {
      while (!supercdc_backup_pos.empty()) {
        auto& backup_pos = supercdc_backup_pos.front();
        if (backup_pos <= last_cut_point + avg_size) {
          supercdc_backup_pos.pop();
          continue;
        }
        break;
      }
      if (!supercdc_backup_pos.empty() && supercdc_backup_pos.front() < last_cut_point + max_size) {
        // TODO: these chunks will share features, which is not right, need to figure out a solution
        make_chunk(candidate_index, supercdc_backup_pos.front());
        last_cut_point = supercdc_backup_pos.front();
        supercdc_backup_pos.pop();
      }
      else {
        // TODO: these chunks will share features, which is not right, need to figure out a solution
        make_chunk(candidate_index, last_cut_point + max_size);
        last_cut_point = last_cut_point + max_size;
      }
    }
    };

    for (uint64_t i = 0; i < new_cut_point_candidates.size(); i++) {
      auto& cut_point_candidate = new_cut_point_candidates[i];
      last_cut_point = get_last_cut_point();

      const uint64_t adjusted_cut_point_candidate = segment_start_offset + cut_point_candidate.offset;
      if (cut_point_candidate.type == CutPointCandidateType::SUPERCDC_BACKUP_MASK) {
        supercdc_backup_pos.emplace(adjusted_cut_point_candidate);
        continue;
      }
      trunc_with_max_size(i, adjusted_cut_point_candidate);

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
    trunc_with_max_size(0, segment_end_pos);

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