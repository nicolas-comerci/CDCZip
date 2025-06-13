#include "cdcz_simd.hpp"

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "cdc_algos/cdcz_simd.cpp" 
#include <hwy/foreach_target.h>
#include <hwy/highway.h>

#include "gear.hpp"
#include "delta_compression/delta.hpp"

HWY_BEFORE_NAMESPACE();

namespace CDCZ_SIMD {
namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;

constexpr HWY_FULL(int32_t) i32VecD {};

#ifndef NDEBUG
constexpr int32_t lane_count = hn::MaxLanes(i32VecD);
#else
constexpr int32_t lane_count = hn::Lanes(i32VecD);
#endif
using i32Vec = decltype(hn::Set(i32VecD, 0));

#define CDC_SIMD_ITER(cbytes)                                                                                                           \
do {                                                                                                                                    \
  hash_vec = hn::Shl(hash_vec, ones_vec);  /* Shift all the hash values for each lane at the same time */                               \
  i32Vec idx = hn::And(cbytes, cmask);  /* Get byte on the lower bits of the packed 32bit lanes */                                      \
  /* This gives us the GEAR hash values for each of the bytes we just got */                                                            \
  i32Vec tentry = hn::GatherIndex(i32VecD, reinterpret_cast<int const*>(GEAR_TABLE), idx);                                              \
  cbytes = hn::Shr(cbytes, eights_vec);  /* We already got the byte on the lower bits, we can shift right to later get the next byte */ \
  hash_vec = hn::Add(hash_vec, tentry);  /* Add the values we got from the GEAR hash values to the values on the hash */                \
																																																																				\
  /* Compare each packed int by bitwise AND with the masks and checking that its 0, and if so adding it to the results */               \
  if constexpr (use_supercdc_backup_mask) {                                                                                             \
		const auto hash_bck_eq_mask = hn::Eq(hn::And(hash_vec, mask_backup_vec), zero_vec);                                                 \
		candidates_backup_vmask = hn::Shl(candidates_backup_vmask, ones_vec);                                                               \
		candidates_backup_vmask = hn::MaskedAddOr(candidates_backup_vmask, hash_bck_eq_mask, candidates_backup_vmask, ones_vec);            \
  }                                                                                                                                     \
  if constexpr (use_fastcdc_normalized_chunking) {                                                                                      \
		const auto hash_easy_eq_mask = hn::Eq(hn::And(hash_vec, mask_easy_vec), zero_vec);                                                  \
		candidates_easy_vmask = hn::Shl(candidates_easy_vmask, ones_vec);                                                                   \
		candidates_easy_vmask = hn::MaskedAddOr(candidates_easy_vmask, hash_easy_eq_mask, candidates_easy_vmask, ones_vec);                 \
  }                                                                                                                                     \
  const auto hash_hard_eq_mask = hn::Eq(hn::And(hash_vec, mask_hard_vec), zero_vec);                                                    \
  candidates_hard_vmask = hn::Shl(candidates_hard_vmask, ones_vec);                                                                     \
  candidates_hard_vmask = hn::MaskedAddOr(candidates_hard_vmask, hash_hard_eq_mask, candidates_hard_vmask, ones_vec);                   \
  if constexpr (compute_features) {                                                                                                     \
		const auto hash_cds_eq_mask = hn::Eq(hn::And(hash_vec, mask_hard_vec), zero_vec);                                                   \
		candidates_cds_vmask = hn::Shl(candidates_cds_vmask, ones_vec);                                                                     \
		candidates_cds_vmask = hn::MaskedAddOr(candidates_cds_vmask, hash_cds_eq_mask, candidates_cds_vmask, ones_vec);                     \
  }                                                                                                                                     \
} while (0)

template <bool compute_features, bool use_fastcdc_normalized_chunking, bool use_supercdc_backup_mask>
void find_cdc_cut_candidates_simd_impl(
  std::vector<CutPointCandidate>& candidates,
  std::vector<std::vector<uint32_t>>& candidate_features,
  std::span<uint8_t> data,
  int32_t min_size,
  int32_t avg_size,
  int32_t max_size,
  uint32_t mask_hard,
  uint32_t mask_medium,
  uint32_t mask_easy,
	bool use_fastcdc_subminimum_skipping,
	bool use_supercdc_minmax_adjustment,
  bool is_first_segment
) {
  LaneStatus lane_status{ lane_count, is_first_segment };
  
  i32Vec mask_backup_vec = hn::Set(i32VecD, static_cast<int32_t>(mask_easy << 1));
  i32Vec mask_easy_vec = hn::Set(i32VecD, static_cast<int32_t>(mask_easy));
  // If we are not using normalized chunking then all of these cuts will be HARD candidates
  i32Vec mask_hard_vec = hn::Set(i32VecD, static_cast<int32_t>(use_fastcdc_normalized_chunking ? mask_hard : mask_medium));

  const i32Vec cmask = hn::Set(i32VecD, 0xff);
  i32Vec hash_vec = hn::Set(i32VecD, 0);
  const i32Vec zero_vec = hn::Set(i32VecD, 0);
  const i32Vec ones_vec = hn::Set(i32VecD, 1);
  const i32Vec twos_vec = hn::Set(i32VecD, 2);
  const i32Vec eights_vec = hn::Set(i32VecD, 8);
  const i32Vec window_size_minus_one_vec = hn::Set(i32VecD, 31);

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
  i32Vec vindex = hn::Mul(hn::Iota(i32VecD, 0), hn::Set(i32VecD, bytes_per_lane));
  if (use_fastcdc_subminimum_skipping && is_first_segment) {
    vindex = hn::InsertLane(vindex, 0, static_cast<int32_t>(min_size));
  }
  // For each lane, the last index they are allowed to access
  i32Vec vindex_max = hn::Add(vindex, hn::Set(i32VecD, bytes_per_lane));
  // Because we read 32bytes at a time we need to ensure we are not reading past the data end
  vindex_max = hn::InsertLane(vindex_max, lane_count - 1, static_cast<int32_t>(data_adjusted_size) - 32);

  // SuperCDC's Min-Max adjustment of the Gear Hash on jump to minimum chunk size, should improve deduplication ratios by better preserving
  // the content defined nature of the Hashes
  for (int32_t i = 0; i < lane_count; i++) {
    lane_status.minmax_adjustment_vec[i] = hn::ExtractLane(vindex, i);
  }
  // We backtrack a little to ensure when we get to the i we actually wanted we have the exact same hash as if we hadn't skipped anything.
  // Note that we do minmax adjustment at the start, even if it's off. Otherwise, we would get different results than serial version.
  vindex = hn::Sub(vindex, window_size_minus_one_vec);
  // If min_size is really low or use_fastcdc_subminimum_skipping = false then the first lane's pos might be negative now, fix.
  if (hn::ExtractLane(vindex, 0) < 0) {
    vindex = hn::InsertLane(vindex, 0, 0);
    lane_status.minmax_adjustment_vec[0] = 0;
  }
  // We need again to ensure the lane positions are 32bit aligned
  for (int32_t i = 0; i < lane_count; i++) {
    auto lane_pos = hn::ExtractLane(vindex, i);
    const int32_t aligned_lane_pos = pad_size_for_alignment(lane_pos, 4);
    const auto bytes_to_alignment = aligned_lane_pos - lane_pos;
    lane_pos -= bytes_to_alignment > 0 ? 4 - bytes_to_alignment : 0;
    vindex = hn::InsertLane(vindex, i, lane_pos);
  }

  i32Vec cds_mask_vec = hn::Set(i32VecD, delta_comp_constants::CDS_SAMPLING_MASK);

  auto sample_feature_value = [&lane_status, &hash_vec, &vindex, &vindex_max](uint32_t lane_i) {
    const auto lane_pos = hn::ExtractLane(vindex, lane_i);
    // Lane already finished, and we are on a pos for another lane (or after data end)!
    if (lane_pos >= hn::ExtractLane(vindex_max, lane_i)) return;
    // If we are still doing minmax adjustment ignore the feature sampling match
    if (lane_pos < lane_status.minmax_adjustment_vec[lane_i]) return;
    // if the lane is marked for jump already then ignore feature sampling match
    if (lane_status.jump_pos_vec[lane_i] != 0) return;

    uint32_t pattern = hn::ExtractLane(hash_vec, lane_i);
    if (lane_status.lane_current_features[lane_i].empty()) {
      lane_status.lane_current_features[lane_i].resize(16);
      lane_status.lane_current_features[lane_i].shrink_to_fit();
    }
    for (int32_t feature_i = 0; feature_i < 16; feature_i++) {
      const auto& [mi, ai] = delta_comp_constants::N_Transform_Coefs[feature_i];
      lane_status.lane_current_features[lane_i][feature_i] = std::max<uint32_t>(lane_status.lane_current_features[lane_i][feature_i], (mi * pattern + ai) % (1LL << 32));
    }
  };

  auto adjust_lane_for_jump = [&lane_status, &vindex, &hash_vec, &min_size, &data, use_supercdc_minmax_adjustment](const int32_t lane_i) {
    int32_t new_lane_pos = lane_status.jump_pos_vec[lane_i];
    if (new_lane_pos == 0) return;
    new_lane_pos += min_size;

    lane_status.minmax_adjustment_vec[lane_i] = new_lane_pos;
    if (use_supercdc_minmax_adjustment) {
      auto adjustment = std::min(31, new_lane_pos);
      new_lane_pos -= adjustment;
    }
    else {
      hash_vec = hn::InsertLane(hash_vec, lane_i, 0);
    }

    // We might not be able to actually do a portable GatherIndex from this position, we adjust if necessary to align to 32bit boundary
    const int32_t aligned_new_lane_pos = pad_size_for_alignment(new_lane_pos, 4);
    const auto bytes_to_alignment = aligned_new_lane_pos - new_lane_pos;
    if (use_supercdc_minmax_adjustment) {
      // This is fine, doing adjustment for more than 31 bytes won't change the GEAR hashing result
      new_lane_pos -= bytes_to_alignment > 0 ? 4 - bytes_to_alignment : 0;
    }
    else {
      // If we are not doing minmax adjustment we need to serially calculate the GEAR hash until we are aligned so SIMD processing can continue
      int32_t pattern = 0;
      for (int32_t i = new_lane_pos; i < new_lane_pos + bytes_to_alignment; i++) {
        pattern = (pattern << 1) + GEAR_TABLE[data[i]];
      }
      hash_vec = hn::InsertLane(hash_vec, lane_i, pattern);
      new_lane_pos += bytes_to_alignment;
    }

    vindex = hn::InsertLane(vindex, lane_i, new_lane_pos);
  };

  auto candidates_cds_vmask = hn::Set(i32VecD, 0);
  auto candidates_backup_vmask = hn::Set(i32VecD, 0);
  auto candidates_easy_vmask = hn::Set(i32VecD, 0);
  auto candidates_hard_vmask = hn::Set(i32VecD, 0);
  // This is a reference to the easiest candidates vector so that we can branchlessly do quick checks to see if any condition was satisfied
  while (true) {
    vindex = hn::Min(vindex, vindex_max);  // prevent vindex from pointing to invalid memory
    const auto is_lane_not_finished_mask = hn::Gt(vindex_max, vindex);
    // If all lanes are finished, we break, else we continue and lanes that are already finished will ignore results
    if (hn::AllFalse(i32VecD, is_lane_not_finished_mask)) break;

    auto vindex_int32_elem = hn::Shr(vindex, twos_vec);
    i32Vec bytes_1to4 = hn::GatherIndex(i32VecD, reinterpret_cast<int const*>(data.data()), vindex_int32_elem);
    i32Vec bytes_5to8 = hn::GatherIndex(i32VecD, reinterpret_cast<int const*>(data.data()), hn::Add(vindex_int32_elem, ones_vec));
    i32Vec bytes_9to12 = hn::GatherIndex(i32VecD, reinterpret_cast<int const*>(data.data()), hn::Add(vindex_int32_elem, twos_vec));
    i32Vec bytes_13to16 = hn::GatherIndex(i32VecD, reinterpret_cast<int const*>(data.data()), hn::Add(vindex_int32_elem, hn::Set(i32VecD, 3)));
    i32Vec bytes_17to20 = hn::GatherIndex(i32VecD, reinterpret_cast<int const*>(data.data()), hn::Add(vindex_int32_elem, hn::Set(i32VecD, 4)));
    i32Vec bytes_21to24 = hn::GatherIndex(i32VecD, reinterpret_cast<int const*>(data.data()), hn::Add(vindex_int32_elem, hn::Set(i32VecD, 5)));
    i32Vec bytes_25to28 = hn::GatherIndex(i32VecD, reinterpret_cast<int const*>(data.data()), hn::Add(vindex_int32_elem, hn::Set(i32VecD, 6)));
    i32Vec bytes_29to32 = hn::GatherIndex(i32VecD, reinterpret_cast<int const*>(data.data()), hn::Add(vindex_int32_elem, hn::Set(i32VecD, 7)));

    candidates_cds_vmask = hn::Set(i32VecD, 0);
    candidates_backup_vmask = hn::Set(i32VecD, 0);
    candidates_easy_vmask = hn::Set(i32VecD, 0);
    candidates_hard_vmask = hn::Set(i32VecD, 0);

    // Oh lord please forgive me, there has to be a better way of unrolling this thing
    CDC_SIMD_ITER(bytes_1to4);    CDC_SIMD_ITER(bytes_1to4);   CDC_SIMD_ITER(bytes_1to4);    CDC_SIMD_ITER(bytes_1to4);
    CDC_SIMD_ITER(bytes_5to8);    CDC_SIMD_ITER(bytes_5to8);   CDC_SIMD_ITER(bytes_5to8);    CDC_SIMD_ITER(bytes_5to8);
    CDC_SIMD_ITER(bytes_9to12);   CDC_SIMD_ITER(bytes_9to12);  CDC_SIMD_ITER(bytes_9to12);   CDC_SIMD_ITER(bytes_9to12);
    CDC_SIMD_ITER(bytes_13to16);  CDC_SIMD_ITER(bytes_13to16); CDC_SIMD_ITER(bytes_13to16);  CDC_SIMD_ITER(bytes_13to16);
    CDC_SIMD_ITER(bytes_17to20);  CDC_SIMD_ITER(bytes_17to20); CDC_SIMD_ITER(bytes_17to20);  CDC_SIMD_ITER(bytes_17to20);
    CDC_SIMD_ITER(bytes_21to24);  CDC_SIMD_ITER(bytes_21to24); CDC_SIMD_ITER(bytes_21to24);  CDC_SIMD_ITER(bytes_21to24);
    CDC_SIMD_ITER(bytes_25to28);  CDC_SIMD_ITER(bytes_25to28); CDC_SIMD_ITER(bytes_25to28);  CDC_SIMD_ITER(bytes_25to28);
    CDC_SIMD_ITER(bytes_29to32);  CDC_SIMD_ITER(bytes_29to32); CDC_SIMD_ITER(bytes_29to32);  CDC_SIMD_ITER(bytes_29to32);

    // TODO: Compute features stuff here
		if constexpr (compute_features) {
      //auto candidates_backup_mask = hn::IfThenElseZero(is_lane_not_finished_mask, candidates_backup_vmask);
		}

    if constexpr (use_supercdc_backup_mask) {
      candidates_backup_vmask = hn::IfThenElseZero(is_lane_not_finished_mask, candidates_backup_vmask);
    }
    else if constexpr (use_fastcdc_normalized_chunking) {
      candidates_easy_vmask = hn::IfThenElseZero(is_lane_not_finished_mask, candidates_easy_vmask);
    }
    else {
      candidates_hard_vmask = hn::IfThenElseZero(is_lane_not_finished_mask, candidates_hard_vmask);
    }
    
    const bool is_any_candidate = [&candidates_backup_vmask, &candidates_easy_vmask, &candidates_hard_vmask, &zero_vec]() {
      if constexpr (use_supercdc_backup_mask) return !hn::AllTrue(i32VecD, hn::Eq(candidates_backup_vmask, zero_vec));
      else if constexpr (use_fastcdc_normalized_chunking) return !hn::AllTrue(i32VecD, hn::Eq(candidates_easy_vmask, zero_vec));
      else return !hn::AllTrue(i32VecD, hn::Eq(candidates_hard_vmask, zero_vec));
    }();
    if (is_any_candidate) {
      int32_t candidates_backup_bits = 0;
      int32_t candidates_easy_bits = 0;
      int32_t candidates_hard_bits = 0;
      for (int32_t lane_i = 0; lane_i < lane_count; lane_i++) {
        if constexpr (use_supercdc_backup_mask) candidates_backup_bits = hn::ExtractLane(candidates_backup_vmask, lane_i);
        if constexpr (use_fastcdc_normalized_chunking) candidates_easy_bits = hn::ExtractLane(candidates_easy_vmask, lane_i);
        candidates_hard_bits = hn::ExtractLane(candidates_hard_vmask, lane_i);
        const int32_t lane_pos = hn::ExtractLane(vindex, lane_i);
        int32_t bit = 0;
        const auto& candidates_easiest_bits = [
          &candidates_backup_bits = candidates_backup_bits,
          &candidates_easy_bits = candidates_easy_bits,
          &candidates_hard_bits = candidates_hard_bits
        ]() -> int32_t& {
          if constexpr (use_supercdc_backup_mask) return candidates_backup_bits;
          else if constexpr (use_fastcdc_normalized_chunking) return candidates_easy_bits;
          else return candidates_hard_bits;
        }();
        while (candidates_easiest_bits != 0) {
          if (candidates_easiest_bits & (0b1 << 31)) {
            CutPointCandidateType cut_type = CutPointCandidateType::SUPERCDC_BACKUP_MASK;
            if constexpr (use_fastcdc_normalized_chunking) { if (candidates_easy_bits & (0b1 << 31)) cut_type = CutPointCandidateType::EASY_CUT_MASK; }
            if constexpr (!use_supercdc_backup_mask && !use_fastcdc_normalized_chunking) {
              cut_type = CutPointCandidateType::HARD_CUT_MASK;
            }
            else {
							if (candidates_hard_bits & (0b1 << 31)) cut_type = CutPointCandidateType::HARD_CUT_MASK;
            }
            process_lane_cut_candidate(
              lane_i, lane_pos + bit, cut_type, lane_status, min_size, avg_size, max_size, bytes_per_lane,
              hn::ExtractLane(vindex_max, lane_i), compute_features, use_fastcdc_subminimum_skipping);
          }
          candidates_backup_bits <<= 1;
          candidates_easy_bits <<= 1;
          candidates_hard_bits <<= 1;
          bit++;
        }
      }
    }

    vindex = hn::Add(vindex, hn::Set(i32VecD, 32));

    if (lane_status.any_lane_marked_for_jump) {
      for (int32_t lane_i = 0; lane_i < lane_count; lane_i++) { adjust_lane_for_jump(lane_i); }

      lane_status.jump_pos_vec = {};
      lane_status.jump_pos_vec.resize(lane_count);
      lane_status.any_lane_marked_for_jump = false;
    }
  }

  {
    // Deal with any trailing data sequentially
    const int32_t last_lane_i = lane_count - 1;
    int32_t i = hn::ExtractLane(vindex_max, last_lane_i) - 31;
    // The hash might be nonsense if the last lane finished before the others, so we just recover it
    uint64_t remaining_minmax_adjustment = 31;
    uint32_t pattern = 0;
    const auto mask = [mask_easy, mask_medium]() {
      if constexpr (use_supercdc_backup_mask) return mask_easy << 1;
      else if constexpr (use_fastcdc_normalized_chunking) return mask_easy; else return mask_medium;
    }();
    // TODO: Can trunc for max_size, though it should be handled by select cut point candidates anyway

    while (i < data.size()) {
      pattern = (pattern << 1) + GEAR_TABLE[data[i]];
      if (remaining_minmax_adjustment > 0) {
        --remaining_minmax_adjustment;
        ++i;
        continue;
      }
      if (!(pattern & mask)) {
        const auto cut_type = promote_cut_candidate(use_supercdc_backup_mask, use_fastcdc_normalized_chunking, pattern, mask_hard, mask_medium, mask_easy);
        process_lane_cut_candidate(last_lane_i, i, cut_type, lane_status, min_size, avg_size, max_size, bytes_per_lane,
          hn::ExtractLane(vindex_max, last_lane_i), compute_features, use_fastcdc_subminimum_skipping, true);
      }
      ++i;
    }
  }

  for (int32_t lane_i = 0; lane_i < lane_count; lane_i++) {
    auto& cut_points_list = lane_status.lane_results[lane_i];
    for (uint64_t n = 0; n < cut_points_list.size(); n++) {
      const auto& cut_point_candidate = cut_points_list[n];
      candidates.emplace_back(cut_point_candidate.type, cut_point_candidate.offset);
      if constexpr (compute_features) {
        candidate_features.emplace_back(std::move(lane_status.lane_features_results[lane_i][n]));
      }
    }

    // There might be some leftover SuperCDC backup cut candidates on the lane, just emit them and let the cut selection step fix it
    while (!lane_status.backup_cut_vec[lane_i].empty()) {
      candidates.emplace_back(CutPointCandidateType::SUPERCDC_BACKUP_MASK, lane_status.backup_cut_vec[lane_i].front());
      lane_status.backup_cut_vec[lane_i].pop();
      // TODO: This is nonsense!
      if constexpr (compute_features) {
        candidate_features.emplace_back();
      }
    }
  }

  if (candidates.empty() || candidates.back().offset != data.size()) {
    candidates.emplace_back(CutPointCandidateType::EOF_CUT, data.size());
    if constexpr (compute_features) {
      candidate_features.emplace_back();
    }
  }
}

static void find_cdc_cut_candidates_simd_wrapper(
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
  constexpr auto call = [](bool b) { return b ? 1 : 0; };
  constexpr auto key = [](bool b1, bool b2, bool b3) {
    return (b1 << 2) | (b2 << 1) | b3;
  };

  using FuncType = decltype(&find_cdc_cut_candidates_simd_impl<false, false, false>);

  static constexpr FuncType dispatchTable[8] = {
      &find_cdc_cut_candidates_simd_impl<false, false, false>,
      &find_cdc_cut_candidates_simd_impl<false, false, true>,
      &find_cdc_cut_candidates_simd_impl<false, true, false>,
      &find_cdc_cut_candidates_simd_impl<false, true, true>,
      &find_cdc_cut_candidates_simd_impl<true, false, false>,
      &find_cdc_cut_candidates_simd_impl<true, false, true>,
      &find_cdc_cut_candidates_simd_impl<true, true, false>,
      &find_cdc_cut_candidates_simd_impl<true, true, true>
  };

  const size_t idx = key(cdcz_cfg.compute_features, cdcz_cfg.use_fastcdc_normalized_chunking, cdcz_cfg.use_supercdc_backup_mask);
  dispatchTable[idx](
    candidates, candidate_features, data, min_size, avg_size, max_size, mask_hard, mask_medium, mask_easy,
    cdcz_cfg.use_fastcdc_subminimum_skipping, cdcz_cfg.use_supercdc_minmax_adjustment,
    is_first_segment
  );
}

}
}

HWY_AFTER_NAMESPACE();

#if HWY_ONCE

void process_lane_cut_candidate(
  const int32_t lane_i, int32_t pos, const CutPointCandidateType result_type, LaneStatus& lane_status,
  const int32_t min_size, const int32_t avg_size, const int32_t max_size, const int32_t bytes_per_lane, const int32_t lane_max_pos,
  const bool compute_features, const bool use_fastcdc_subminimum_skipping, const bool ignore_max_pos
  ) {
  auto mark_lane_for_jump = [&lane_status, use_fastcdc_subminimum_skipping](const int32_t pos, const int32_t lane_i) {
    if (!use_fastcdc_subminimum_skipping) return;
    lane_status.jump_pos_vec[lane_i] = pos;
    lane_status.any_lane_marked_for_jump = true;
  };

  // Lane already finished, and we are on a pos for another lane (or after data end)!
  if (!ignore_max_pos && pos >= lane_max_pos) return;
  // If we are still doing minmax adjustment ignore the cut
  if (pos < lane_status.minmax_adjustment_vec[lane_i]) return;
  // if the lane is marked for jump already then ignore the cut as well
  if (lane_status.jump_pos_vec[lane_i] != 0) return;

  if (lane_status.lane_achieved_chunk_invariance[lane_i]) {
    int32_t prev_cut_pos = lane_status.lane_results[lane_i].empty() ? lane_i * bytes_per_lane : static_cast<int32_t>(lane_status.lane_results[lane_i].back().offset);
    int32_t dist_with_prev = pos - prev_cut_pos;

    // >= max_size and not > max_size here even if it would be theoretically unnecessary because that's what we do on single threaded FastCDC as well
    while (dist_with_prev >= max_size) {
      if (!lane_status.backup_cut_vec[lane_i].empty() && pos > lane_status.backup_cut_vec[lane_i].front()) {
        prev_cut_pos = lane_status.backup_cut_vec[lane_i].front();
        lane_status.backup_cut_vec[lane_i].pop();
        lane_status.lane_results[lane_i].emplace_back(CutPointCandidateType::SUPERCDC_BACKUP_MASK, prev_cut_pos);
      }
      else {
        prev_cut_pos = pos - dist_with_prev + max_size;
        lane_status.lane_results[lane_i].emplace_back(CutPointCandidateType::MAX_SIZE, prev_cut_pos);
      }
      // Clear any pending backup cut candidate that is now invalid, that is, before the new prev_cut_pos + avg_size
      while (!lane_status.backup_cut_vec[lane_i].empty() && lane_status.backup_cut_vec[lane_i].front() < prev_cut_pos + avg_size) {
        lane_status.backup_cut_vec[lane_i].pop();
      }
      dist_with_prev = pos - prev_cut_pos;
    }

    if (result_type == CutPointCandidateType::SUPERCDC_BACKUP_MASK) {
      if (dist_with_prev < avg_size) return;
      lane_status.backup_cut_vec[lane_i].emplace(pos);
      return;
    }
    else if (result_type == CutPointCandidateType::EASY_CUT_MASK && dist_with_prev < avg_size) {
      return;
    }
    else if (dist_with_prev < min_size) {
      return;
    }

    lane_status.lane_results[lane_i].emplace_back(result_type, pos);
    if (compute_features) {
      lane_status.lane_features_results[lane_i].emplace_back(std::move(lane_status.lane_current_features[lane_i]));
      lane_status.lane_current_features[lane_i] = std::vector<uint32_t>();
    }
    while (!lane_status.backup_cut_vec[lane_i].empty()) lane_status.backup_cut_vec[lane_i].pop();
    mark_lane_for_jump(pos, lane_i);
  }
  else {
    if (!lane_status.lane_results[lane_i].empty()) {
      auto& prevCutCandidate = lane_status.lane_results[lane_i].back();
      const auto dist_with_prev = pos - prevCutCandidate.offset;
      const auto is_prev_candidate_hard = prevCutCandidate.type == CutPointCandidateType::HARD_CUT_MASK;
      // if this happens this lane is back in sync with non-segmented processing!
      if (is_chunk_invariance_condition_satisfied(is_prev_candidate_hard, dist_with_prev, result_type, min_size, avg_size, max_size)) {
        lane_status.lane_achieved_chunk_invariance[lane_i] = true;
        mark_lane_for_jump(pos, lane_i);
      }
    }
    lane_status.lane_results[lane_i].emplace_back(result_type, pos);
    if (compute_features) {
      lane_status.lane_features_results[lane_i].emplace_back(std::move(lane_status.lane_current_features[lane_i]));
      lane_status.lane_current_features[lane_i] = std::vector<uint32_t>();
    }
  }
};

namespace CDCZ_SIMD {
	HWY_EXPORT(find_cdc_cut_candidates_simd_wrapper);
}

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
) {
  HWY_DYNAMIC_DISPATCH(CDCZ_SIMD::find_cdc_cut_candidates_simd_wrapper)(candidates, candidate_features, data, min_size, avg_size, max_size, mask_hard, mask_medium, mask_easy, cdcz_cfg, is_first_segment);
}

#endif
