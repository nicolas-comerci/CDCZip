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
constexpr size_t lane_count = hn::MaxLanes(i32VecD);
#else
constexpr size_t lane_count = hn::Lanes(i32VecD);
#endif
using i32Vec = decltype(hn::Set(i32VecD, 0));

#define CDC_SIMD_ITER_IMPL(vbytes, supercdc_backup, normalized_chunking, features_cds)                                                  \
do {                                                                                                                                    \
  hash_vec = hn::Shl(hash_vec, ones_vec);  /* Shift all the hash values for each lane at the same time */                               \
  i32Vec lower_byte_vec = hn::And(vbytes, cmask);  /* Get byte on the lower bits of the packed 32bit lanes */                           \
  /* This gives us the GEAR hash values for each of the bytes we just got */                                                            \
  i32Vec gear_values = hn::GatherIndex(i32VecD, reinterpret_cast<int const*>(GEAR_TABLE), lower_byte_vec);                              \
  (vbytes) = hn::Shr(vbytes, eights_vec);/* We already got the byte on the lower bits, we can shift right to later get the next byte */ \
  hash_vec = hn::Add(hash_vec, gear_values);  /* Add the values we got from the GEAR hash values to the values on the hash */           \
																																																																				\
  if (features_cds) {                                                                                                                   \
		const auto hash_cds_eq_mask = hn::Eq(hn::And(hash_vec, mask_cds_vec), zero_vec);                                                    \
		candidates_cds_vmask = hn::Shl(candidates_cds_vmask, ones_vec);                                                                     \
		candidates_cds_vmask = hn::IfThenElse(hash_cds_eq_mask, hn::Or(candidates_cds_vmask, ones_vec), candidates_cds_vmask);              \
  }                                                                                                                                     \
  																																																																		  \
	candidates_backup_vmask = hn::Shl(candidates_backup_vmask, ones_vec);                                                                 \
  candidates_easy_vmask = hn::Shl(candidates_easy_vmask, ones_vec);                                                                     \
  candidates_hard_vmask = hn::Shl(candidates_hard_vmask, ones_vec);                                                                     \
  																																																																		  \
  /* Compare each packed int by bitwise AND with the masks and checking that its 0 */                                                   \
  if (supercdc_backup) {                                                                                                                \
		const auto hash_bck_eq_mask = hn::Eq(hn::And(hash_vec, mask_backup_vec), zero_vec);                                                 \
    /* Quitting early if possible to skip harder hash checks if those are bound to not find anything */                                 \
    if (hn::AllFalse(i32VecD, hash_bck_eq_mask)) continue;                                                                              \
		candidates_backup_vmask = hn::IfThenElse(hash_bck_eq_mask, hn::Or(candidates_backup_vmask, ones_vec), candidates_backup_vmask);     \
	}                                                                                                                                     \
  if (normalized_chunking) {                                                                                                            \
		const auto hash_easy_eq_mask = hn::Eq(hn::And(hash_vec, mask_easy_vec), zero_vec);                                                  \
    /* Quitting early as above */                                                                                                       \
    if (hn::AllFalse(i32VecD, hash_easy_eq_mask)) continue;                                                                             \
		candidates_easy_vmask = hn::IfThenElse(hash_easy_eq_mask, hn::Or(candidates_easy_vmask, ones_vec), candidates_easy_vmask);          \
  }                                                                                                                                     \
  const auto hash_hard_eq_mask = hn::Eq(hn::And(hash_vec, mask_hard_vec), zero_vec);                                                    \
  /* Quitting early as above */                                                                                                         \
  if (hn::AllFalse(i32VecD, hash_hard_eq_mask)) continue;                                                                               \
  candidates_hard_vmask = hn::IfThenElse(hash_hard_eq_mask, hn::Or(candidates_hard_vmask, ones_vec), candidates_hard_vmask);            \
} while (0)
#define CDC_SIMD_ITER(vbytes) CDC_SIMD_ITER_IMPL(vbytes, true, true, false)
#define CDC_SIMD_ITER_SSCDC(vbytes) CDC_SIMD_ITER_IMPL(vbytes, false, false, false)

static void find_cdc_cut_candidates_simd_impl(
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
  std::array<std::vector<CutPointCandidate>, lane_count> lane_results{};
  std::array<bool, lane_count> lane_achieved_chunk_invariance{};
  lane_achieved_chunk_invariance.fill(false);
  lane_achieved_chunk_invariance[0] = is_first_segment;
  std::array<std::vector<std::vector<uint32_t>>, lane_count> lane_features_results{};
  std::array<std::vector<uint32_t>, lane_count> lane_current_features{};

  const auto easiest_mask = cdcz_cfg.use_supercdc_backup_mask
    ? mask_easy << 1
    : cdcz_cfg.use_fastcdc_normalized_chunking ? mask_easy : mask_medium;
  i32Vec mask_backup_vec = hn::Set(i32VecD, static_cast<int32_t>(mask_easy << 1));
  i32Vec mask_easy_vec = hn::Set(i32VecD, static_cast<int32_t>(mask_easy));
  // If we are not using normalized chunking then all of these cuts will be HARD candidates
  i32Vec mask_hard_vec = hn::Set(i32VecD, static_cast<int32_t>(cdcz_cfg.use_fastcdc_normalized_chunking ? mask_hard : mask_medium));
  i32Vec mask_cds_vec = hn::Set(i32VecD, static_cast<int32_t>(delta_comp_constants::CDS_SAMPLING_MASK));

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
  if (data_adjusted_size > std::numeric_limits<int32_t>::max()) {
    throw std::runtime_error("Unable to process data such that lanes positions would overflow");
  }
  int32_t bytes_per_lane = pad_size_for_alignment(static_cast<int32_t>(data_adjusted_size / lane_count), 4) - 4;  // Idem 32bit alignment for Gathers
  i32Vec vindex = hn::Mul(hn::Iota(i32VecD, 0), hn::Set(i32VecD, bytes_per_lane));
  if (cdcz_cfg.use_fastcdc_subminimum_skipping && is_first_segment) {
    vindex = hn::InsertLane(vindex, 0, static_cast<int32_t>(min_size));
  }
  // For each lane, the last index they are allowed to access
  i32Vec vindex_max = hn::Add(vindex, hn::Set(i32VecD, bytes_per_lane));
  // Because we read 4bytes at a time we need to ensure we are not reading past the data end
  vindex_max = hn::InsertLane(vindex_max, lane_count - 1, static_cast<int32_t>(data_adjusted_size) - 32);

  // SuperCDC's backup results, for each lane we need to save any valid ones until we reach a valid cut so we don't need them,
  // or reach max_size for a chunk and thus use the earliest backup.
  std::array<std::queue<int32_t>, lane_count> backup_cut_vec{};

  // SuperCDC's Min-Max adjustment of the Gear Hash on jump to minimum chunk size, should improve deduplication ratios by better preserving
  // the content defined nature of the Hashes
  std::array<int32_t, lane_count> minmax_adjustment_vec{};
  for (int32_t i = 0; i < lane_count; i++) {
    minmax_adjustment_vec[i] = hn::ExtractLane(vindex, i);
  }
  // We backtrack a little to ensure when we get to the i we actually wanted we have the exact same hash as if we hadn't skipped anything.
  // Note that we do minmax adjustment at the start, even if it's off. Otherwise, we would get different results than serial version.
  vindex = hn::Sub(vindex, window_size_minus_one_vec);
  // If min_size is really low or use_fastcdc_subminimum_skipping = false then the first lane's pos might be negative now, fix.
  if (hn::ExtractLane(vindex, 0) < 0) {
    vindex = hn::InsertLane(vindex, 0, 0);
    minmax_adjustment_vec[0] = 0;
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

  bool any_lane_marked_for_jump = false;
  std::array<int32_t, lane_count> jump_pos_vec{};
  jump_pos_vec.fill(0);
  auto mark_lane_for_jump = [&jump_pos_vec, &any_lane_marked_for_jump, &cdcz_cfg](const int32_t pos, const int32_t lane_i) {
    if (!cdcz_cfg.use_fastcdc_subminimum_skipping) return;
    jump_pos_vec[lane_i] = pos;
    any_lane_marked_for_jump = true;
  };

  auto get_result_type_for_lane = [&cdcz_cfg, &mask_hard, &mask_medium, &mask_easy](int32_t pattern) {
    // Do the cut candidate promoting behavior
    return promote_cut_candidate(cdcz_cfg, pattern, mask_hard, mask_medium, mask_easy);
  };

  auto process_lane = [&](const int32_t lane_i, int32_t pos, const CutPointCandidateType result_type, bool ignore_max_pos = false) {
    // Lane already finished, and we are on a pos for another lane (or after data end)!
    if (!ignore_max_pos && pos >= hn::ExtractLane(vindex_max, lane_i)) return;
    // If we are still doing minmax adjustment ignore the cut
    if (pos < minmax_adjustment_vec[lane_i]) return;
    // if the lane is marked for jump already then ignore the cut as well
    if (jump_pos_vec[lane_i] != 0) return;

    if (lane_achieved_chunk_invariance[lane_i]) {
      int32_t prev_cut_pos = lane_results[lane_i].empty() ? lane_i * bytes_per_lane : static_cast<int32_t>(lane_results[lane_i].back().offset);
      // This should pretty much never happen but could happen if min_size < 32 (which you should never use as we use 32bit GEAR but that's besides the point)
      if (prev_cut_pos >= pos) return;
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

  auto sample_feature_value = [&lane_current_features, &hash_vec, &vindex, &minmax_adjustment_vec, &jump_pos_vec, &vindex_max, &cdcz_cfg](uint32_t lane_i) {
    const auto lane_pos = hn::ExtractLane(vindex, lane_i);
    // Lane already finished, and we are on a pos for another lane (or after data end)!
    if (lane_pos >= hn::ExtractLane(vindex_max, lane_i)) return;
    // If we are still doing minmax adjustment ignore the feature sampling match
    if (lane_pos < minmax_adjustment_vec[lane_i]) return;
    // if the lane is marked for jump already then ignore feature sampling match
    if (jump_pos_vec[lane_i] != 0) return;

    uint32_t pattern = hn::ExtractLane(hash_vec, lane_i);
    if (lane_current_features[lane_i].empty()) {
      lane_current_features[lane_i].resize(16);
      lane_current_features[lane_i].shrink_to_fit();
    }
    for (int32_t feature_i = 0; feature_i < 16; feature_i++) {
      const auto& [mi, ai] = delta_comp_constants::N_Transform_Coefs[feature_i];
      lane_current_features[lane_i][feature_i] = std::max<uint32_t>(lane_current_features[lane_i][feature_i], (mi * pattern + ai) % (1LL << 32));
    }
  };

  auto adjust_lane_for_jump = [&minmax_adjustment_vec, &jump_pos_vec, &vindex, &hash_vec, &min_size, &cdcz_cfg, &data](const int32_t lane_i) {
    int32_t new_lane_pos = jump_pos_vec[lane_i];
    if (new_lane_pos == 0) return;
    new_lane_pos += min_size;

    minmax_adjustment_vec[lane_i] = new_lane_pos;
    if (cdcz_cfg.use_supercdc_minmax_adjustment) {
      auto adjustment = std::min(31, new_lane_pos);
      new_lane_pos -= adjustment;
    }
    else {
      hash_vec = hn::InsertLane(hash_vec, lane_i, 0);
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
      hash_vec = hn::InsertLane(hash_vec, lane_i, pattern);
      new_lane_pos += bytes_to_alignment;
    }

    vindex = hn::InsertLane(vindex, lane_i, new_lane_pos);
  };

  auto candidates_backup_vmask = hn::Set(i32VecD, 0);
  auto candidates_easy_vmask = hn::Set(i32VecD, 0);
  auto candidates_hard_vmask = hn::Set(i32VecD, 0);
  auto candidates_cds_vmask = hn::Set(i32VecD, 0);

  auto& candidates_easiest_vmask = cdcz_cfg.use_supercdc_backup_mask
    ? candidates_backup_vmask
    : cdcz_cfg.use_fastcdc_normalized_chunking ? candidates_easy_vmask : candidates_hard_vmask;

  while (true) {
    vindex = hn::Min(vindex, vindex_max);  // prevent vindex from pointing to invalid memory
    const auto is_lane_not_finished_mask = hn::Gt(vindex_max, vindex);
    // If all lanes are finished, we break, else we continue and lanes that are already finished will ignore results
    if (hn::AllFalse(i32VecD, is_lane_not_finished_mask)) break;

    // Highway GatherIndex requires the index values to be element indexes as with an array, not by byte
    auto vindex_int32_elem = hn::Shr(vindex, twos_vec);
    i32Vec bytes_1to4 = hn::GatherIndex(i32VecD, reinterpret_cast<int const*>(data.data()), vindex_int32_elem);
    i32Vec bytes_5to8 = hn::GatherIndex(i32VecD, reinterpret_cast<int const*>(data.data()), hn::Add(vindex_int32_elem, ones_vec));
    i32Vec bytes_9to12 = hn::GatherIndex(i32VecD, reinterpret_cast<int const*>(data.data()), hn::Add(vindex_int32_elem, twos_vec));
    i32Vec bytes_13to16 = hn::GatherIndex(i32VecD, reinterpret_cast<int const*>(data.data()), hn::Add(vindex_int32_elem, hn::Set(i32VecD, 3)));
    i32Vec bytes_17to20 = hn::GatherIndex(i32VecD, reinterpret_cast<int const*>(data.data()), hn::Add(vindex_int32_elem, hn::Set(i32VecD, 4)));
    i32Vec bytes_21to24 = hn::GatherIndex(i32VecD, reinterpret_cast<int const*>(data.data()), hn::Add(vindex_int32_elem, hn::Set(i32VecD, 5)));
    i32Vec bytes_25to28 = hn::GatherIndex(i32VecD, reinterpret_cast<int const*>(data.data()), hn::Add(vindex_int32_elem, hn::Set(i32VecD, 6)));
    i32Vec bytes_29to32 = hn::GatherIndex(i32VecD, reinterpret_cast<int const*>(data.data()), hn::Add(vindex_int32_elem, hn::Set(i32VecD, 7)));

    // Oh lord please forgive me, there has to be a better way of unrolling this thing
    CDC_SIMD_ITER(bytes_1to4);    CDC_SIMD_ITER(bytes_1to4);   CDC_SIMD_ITER(bytes_1to4);    CDC_SIMD_ITER(bytes_1to4);
    CDC_SIMD_ITER(bytes_5to8);    CDC_SIMD_ITER(bytes_5to8);   CDC_SIMD_ITER(bytes_5to8);    CDC_SIMD_ITER(bytes_5to8);
    CDC_SIMD_ITER(bytes_9to12);   CDC_SIMD_ITER(bytes_9to12);  CDC_SIMD_ITER(bytes_9to12);   CDC_SIMD_ITER(bytes_9to12);
    CDC_SIMD_ITER(bytes_13to16);  CDC_SIMD_ITER(bytes_13to16); CDC_SIMD_ITER(bytes_13to16);  CDC_SIMD_ITER(bytes_13to16);
    CDC_SIMD_ITER(bytes_17to20);  CDC_SIMD_ITER(bytes_17to20); CDC_SIMD_ITER(bytes_17to20);  CDC_SIMD_ITER(bytes_17to20);
    CDC_SIMD_ITER(bytes_21to24);  CDC_SIMD_ITER(bytes_21to24); CDC_SIMD_ITER(bytes_21to24);  CDC_SIMD_ITER(bytes_21to24);
    CDC_SIMD_ITER(bytes_25to28);  CDC_SIMD_ITER(bytes_25to28); CDC_SIMD_ITER(bytes_25to28);  CDC_SIMD_ITER(bytes_25to28);
    CDC_SIMD_ITER(bytes_29to32);  CDC_SIMD_ITER(bytes_29to32); CDC_SIMD_ITER(bytes_29to32);  CDC_SIMD_ITER(bytes_29to32);

    // TODO: This is broken! sample_feature_value accesses the hash which might(most likely) not be appropriate for the pos anymore
    // Proper SIMD implementation probably using hn::IfThenElse or similar inside CDC_SIMD_ITER is required
    const auto candidates_cds_mask = hn::IfThenElseZero(is_lane_not_finished_mask, candidates_cds_vmask);
    if (!hn::AllTrue(i32VecD, hn::Eq(candidates_cds_mask, zero_vec))) {
      int32_t candidates_cds_bits = 0;
      for (int32_t lane_i = 0; lane_i < lane_count; lane_i++) {
        auto candidates_cds_bits = hn::ExtractLane(candidates_cds_vmask, lane_i);
        const int32_t lane_pos = hn::ExtractLane(vindex, lane_i);
        int32_t bit = 0;
        while (candidates_cds_bits != 0) {
          if (candidates_cds_bits & (0b1 << 31)) {
            sample_feature_value(lane_i);
          }
          candidates_cds_bits <<= 1;
          bit++;
        }
      }
    }

    candidates_easiest_vmask = hn::IfThenElseZero(is_lane_not_finished_mask, candidates_easiest_vmask);
    if (!hn::AllTrue(i32VecD, hn::Eq(candidates_easiest_vmask, zero_vec))) {
      int32_t candidates_backup_bits = 0;
      int32_t candidates_easy_bits = 0;
      int32_t candidates_hard_bits = 0;
      const auto& candidates_easiest_bits = cdcz_cfg.use_supercdc_backup_mask
        ? candidates_backup_bits
        : cdcz_cfg.use_fastcdc_normalized_chunking ? candidates_easy_bits : candidates_hard_bits;
      for (int32_t lane_i = 0; lane_i < lane_count; lane_i++) {
        candidates_backup_bits = hn::ExtractLane(candidates_backup_vmask, lane_i);
        candidates_easy_bits = hn::ExtractLane(candidates_easy_vmask, lane_i);
        candidates_hard_bits = hn::ExtractLane(candidates_hard_vmask, lane_i);
        const int32_t lane_pos = hn::ExtractLane(vindex, lane_i);
        int32_t bit = 0;
        while (candidates_easiest_bits != 0) {
          if (candidates_easiest_bits & (0b1 << 31)) {
            CutPointCandidateType cut_type = CutPointCandidateType::SUPERCDC_BACKUP_MASK;
            if (candidates_easy_bits & (0b1 << 31)) cut_type = CutPointCandidateType::EASY_CUT_MASK;
            if (candidates_hard_bits & (0b1 << 31)) cut_type = CutPointCandidateType::HARD_CUT_MASK;
            process_lane(lane_i, lane_pos + bit, cut_type);
          }
          candidates_backup_bits <<= 1;
          candidates_easy_bits <<= 1;
          candidates_hard_bits <<= 1;
          bit++;
        }
      }
    }

    vindex = hn::Add(vindex, hn::Set(i32VecD, 32));

    if (any_lane_marked_for_jump) {
      for (int32_t lane_i = 0; lane_i < lane_count; lane_i++) {
        adjust_lane_for_jump(lane_i);
      }

      jump_pos_vec = {};
      jump_pos_vec.fill(0);
      any_lane_marked_for_jump = false;
    }
  }

  {
    // Deal with any trailing data sequentially
    int32_t i = hn::ExtractLane(vindex_max, lane_count - 1) - 31;
    // The hash might be nonsense if the last lane finished before the others, so we just recover it
    uint64_t remaining_minmax_adjustment = 31;
    uint32_t pattern = 0;
    // TODO: Can trunc for max_size, though it should be handled by select cut point candidates anyway

    while (i < data.size()) {
      pattern = (pattern << 1) + GEAR_TABLE[data[i]];
      if (remaining_minmax_adjustment > 0) {
        --remaining_minmax_adjustment;
        ++i;
        continue;
      }
      if (!(pattern & easiest_mask)) process_lane(lane_count - 1, i, get_result_type_for_lane(pattern), true);
      ++i;
    }
  }

  for (int32_t lane_i = 0; lane_i < lane_count; lane_i++) {
    auto& cut_points_list = lane_results[lane_i];
    for (uint64_t n = 0; n < cut_points_list.size(); n++) {
      const auto& cut_point_candidate = cut_points_list[n];
      candidates.emplace_back(cut_point_candidate.type, cut_point_candidate.offset);
      if (cdcz_cfg.compute_features) {
        candidate_features.emplace_back(std::move(lane_features_results[lane_i][n]));
      }
    }

    // There might be some leftover SuperCDC backup cut candidates on the lane, just emit them and let the cut selection step fix it
    while (!backup_cut_vec[lane_i].empty()) {
      candidates.emplace_back(CutPointCandidateType::SUPERCDC_BACKUP_MASK, backup_cut_vec[lane_i].front());
      backup_cut_vec[lane_i].pop();
      // TODO: This is nonsense!
      if (cdcz_cfg.compute_features) {
        candidate_features.emplace_back();
      }
    }
  }

  if (candidates.empty() || candidates.back().offset != data.size()) {
    candidates.emplace_back(CutPointCandidateType::EOF_CUT, data.size());
    if (cdcz_cfg.compute_features) {
      candidate_features.emplace_back();
    }
  }
}

HWY_INLINE void i32VecScatter(std::uint8_t* base, const i32Vec& idx_vec, const i32Vec& val_vec, uint8_t scale_bytes) {
#if !defined(NDEBUG)
  if (!(scale_bytes == 1 || scale_bytes == 2 || scale_bytes == 4 || scale_bytes == 8)) abort();
#endif

  alignas(64) int32_t idx_tmp[hn::MaxLanes(i32VecD)];
  alignas(64) int32_t val_tmp[hn::MaxLanes(i32VecD)];
  hn::Store(idx_vec, i32VecD, idx_tmp);
  hn::Store(val_vec, i32VecD, val_tmp);

#if HWY_TARGET == HWY_AVX3 || HWY_TARGET == HWY_AVX3_DL || HWY_TARGET == HWY_AVX3_ZEN4 || HWY_TARGET == HWY_AVX3_SPR
  // AVX-512 native scatter for 16-lane vectors.
  if (lane_count == 16) {
    const __m512i v_idx = _mm512_loadu_si512(idx_tmp);
    const __m512i v_vals = _mm512_loadu_si512(val_tmp);
    _mm512_i32scatter_epi32(base, v_idx, v_vals, scale_bytes);
    return;
  }
#endif

  // Generic per-lane stores for any other arch.
  for (size_t i = 0; i < lane_count; ++i) {
    *reinterpret_cast<int*>(base + static_cast<size_t>(idx_tmp[i]) * scale_bytes) = val_tmp[i];
  }
}

static void sscdc_first_stage_impl(std::span<uint8_t> data, uint8_t* results_bitmap, uint32_t mask) {
  // For SS-CDC all cuts will be HARD candidates
  i32Vec mask_hard_vec = hn::Set(i32VecD, static_cast<int32_t>(mask));
  // All these are not used for SS-CDC so their values don't matter
  i32Vec mask_backup_vec = hn::Set(i32VecD, static_cast<int32_t>(mask));
  i32Vec mask_easy_vec = hn::Set(i32VecD, static_cast<int32_t>(mask));
  i32Vec mask_cds_vec = hn::Set(i32VecD, static_cast<int32_t>(mask));

  const i32Vec cmask = hn::Set(i32VecD, 0xff);
  i32Vec hash_vec = hn::Set(i32VecD, 0);
  const i32Vec zero_vec = hn::Set(i32VecD, 0);
  const i32Vec ones_vec = hn::Set(i32VecD, 1);
  const i32Vec twos_vec = hn::Set(i32VecD, 2);
  const i32Vec eights_vec = hn::Set(i32VecD, 8);

  // Highway's portable GatherIndex requires we read with 32bit/4byte alignment as we have int32_t vectors.
  // This way we ensure we never attempt to Gather from outside the data boundaries, the last < 4 bytes can be finished off manually.
  const auto data_adjusted_size = pad_size_for_alignment(data.size(), 4) - 4;
  if (data_adjusted_size > std::numeric_limits<int32_t>::max()) {
    throw std::runtime_error("Unable to process data such that lanes positions would overflow");
  }
  // For the bytes_per_lane we also need 32bit alignment for Gathers but also a whole number of 32byte results for results_bitmap
  const int32_t bytes_per_lane = pad_size_for_alignment(static_cast<int32_t>(data_adjusted_size / lane_count), 32) - 32;
  i32Vec vindex = hn::Mul(hn::Iota(i32VecD, 0), hn::Set(i32VecD, bytes_per_lane));
  const int32_t bitmap_bytes_per_lane = bytes_per_lane / 8;
  i32Vec bitmap_vindex = hn::Mul(hn::Iota(i32VecD, 0), hn::Set(i32VecD, bitmap_bytes_per_lane));

  // TODO: This should not be necessary for SS-CDC as there are no jumps, we should be easily able to precompute the needed amount of iterations
  // For each lane, the last index they are allowed to access
  i32Vec vindex_max = hn::Add(vindex, hn::Set(i32VecD, bytes_per_lane));

  // All lanes have to initialize their GEAR values with a full window of data before those are usable, like we do with SuperCDC min-max adjustment
  // Doing 32 bytes even if 31 should be enough, so we keep 32byte data alignment
  vindex = hn::Sub(vindex, hn::Set(i32VecD, 32));
  vindex = hn::InsertLane(vindex, 0, 0);
  bool gear_initialized = false;

  // For CDC_SIMD_ITER we will be using the hard vmask as there are no conditional cuts on SS-CDC, all cut candidates have to be honored,
  // as long as the minimum and maximum chunk size requirements are met
  auto candidates_hard_vmask = hn::Set(i32VecD, 0);

  // These are unnecessary for SS-CDC and unusued, but still needed for CDC_SIMD_ITER macro
  auto candidates_backup_vmask = hn::Set(i32VecD, 0);
  auto candidates_easy_vmask = hn::Set(i32VecD, 0);
  auto candidates_cds_vmask = hn::Set(i32VecD, 0);

  while (true) {
    vindex = hn::Min(vindex, vindex_max);  // prevent vindex from pointing to invalid memory
    const auto is_lane_not_finished_mask = hn::Gt(vindex_max, vindex);
    // If all lanes are finished, we break, else we continue and lanes that are already finished will ignore results
    if (hn::AllFalse(i32VecD, is_lane_not_finished_mask)) break;

    // Highway GatherIndex requires the index values to be element indexes as with an array, not by byte
    auto vindex_int32_elem = hn::Shr(vindex, twos_vec);
    i32Vec bytes_1to4 = hn::GatherIndex(i32VecD, reinterpret_cast<int const*>(data.data()), vindex_int32_elem);
    i32Vec bytes_5to8 = hn::GatherIndex(i32VecD, reinterpret_cast<int const*>(data.data()), hn::Add(vindex_int32_elem, ones_vec));
    i32Vec bytes_9to12 = hn::GatherIndex(i32VecD, reinterpret_cast<int const*>(data.data()), hn::Add(vindex_int32_elem, twos_vec));
    i32Vec bytes_13to16 = hn::GatherIndex(i32VecD, reinterpret_cast<int const*>(data.data()), hn::Add(vindex_int32_elem, hn::Set(i32VecD, 3)));
    i32Vec bytes_17to20 = hn::GatherIndex(i32VecD, reinterpret_cast<int const*>(data.data()), hn::Add(vindex_int32_elem, hn::Set(i32VecD, 4)));
    i32Vec bytes_21to24 = hn::GatherIndex(i32VecD, reinterpret_cast<int const*>(data.data()), hn::Add(vindex_int32_elem, hn::Set(i32VecD, 5)));
    i32Vec bytes_25to28 = hn::GatherIndex(i32VecD, reinterpret_cast<int const*>(data.data()), hn::Add(vindex_int32_elem, hn::Set(i32VecD, 6)));
    i32Vec bytes_29to32 = hn::GatherIndex(i32VecD, reinterpret_cast<int const*>(data.data()), hn::Add(vindex_int32_elem, hn::Set(i32VecD, 7)));

    // Oh lord please forgive me, there has to be a better way of unrolling this thing
    CDC_SIMD_ITER_SSCDC(bytes_1to4);    CDC_SIMD_ITER_SSCDC(bytes_1to4);   CDC_SIMD_ITER_SSCDC(bytes_1to4);    CDC_SIMD_ITER_SSCDC(bytes_1to4);
    CDC_SIMD_ITER_SSCDC(bytes_5to8);    CDC_SIMD_ITER_SSCDC(bytes_5to8);   CDC_SIMD_ITER_SSCDC(bytes_5to8);    CDC_SIMD_ITER_SSCDC(bytes_5to8);
    CDC_SIMD_ITER_SSCDC(bytes_9to12);   CDC_SIMD_ITER_SSCDC(bytes_9to12);  CDC_SIMD_ITER_SSCDC(bytes_9to12);   CDC_SIMD_ITER_SSCDC(bytes_9to12);
    CDC_SIMD_ITER_SSCDC(bytes_13to16);  CDC_SIMD_ITER_SSCDC(bytes_13to16); CDC_SIMD_ITER_SSCDC(bytes_13to16);  CDC_SIMD_ITER_SSCDC(bytes_13to16);
    CDC_SIMD_ITER_SSCDC(bytes_17to20);  CDC_SIMD_ITER_SSCDC(bytes_17to20); CDC_SIMD_ITER_SSCDC(bytes_17to20);  CDC_SIMD_ITER_SSCDC(bytes_17to20);
    CDC_SIMD_ITER_SSCDC(bytes_21to24);  CDC_SIMD_ITER_SSCDC(bytes_21to24); CDC_SIMD_ITER_SSCDC(bytes_21to24);  CDC_SIMD_ITER_SSCDC(bytes_21to24);
    CDC_SIMD_ITER_SSCDC(bytes_25to28);  CDC_SIMD_ITER_SSCDC(bytes_25to28); CDC_SIMD_ITER_SSCDC(bytes_25to28);  CDC_SIMD_ITER_SSCDC(bytes_25to28);
    CDC_SIMD_ITER_SSCDC(bytes_29to32);  CDC_SIMD_ITER_SSCDC(bytes_29to32); CDC_SIMD_ITER_SSCDC(bytes_29to32);  CDC_SIMD_ITER_SSCDC(bytes_29to32);

    candidates_hard_vmask = hn::IfThenElseZero(is_lane_not_finished_mask, candidates_hard_vmask);
    if (gear_initialized && !hn::AllTrue(i32VecD, hn::Eq(candidates_hard_vmask, zero_vec))) {
      i32VecScatter(results_bitmap, bitmap_vindex, candidates_hard_vmask, 1);
    }

    vindex = hn::Add(vindex, hn::Set(i32VecD, 32));
    if (gear_initialized) {
      bitmap_vindex = hn::Add(bitmap_vindex, hn::Set(i32VecD, 4));  // 32bits=4bytes per lane
    }
    else {
      // Set lane 0 to the start, with this now all the lanes are at their actual starting positions with GEAR values from a full prior data window
      vindex = hn::InsertLane(vindex, 0, 0);
      gear_initialized = true;
    }
  }

  {
    // Deal with any trailing data sequentially
    int32_t i = hn::ExtractLane(vindex_max, lane_count - 1) - 31;
    // The hash might be nonsense if the last lane finished before the others, so we just recover it
    uint64_t remaining_minmax_adjustment = 31;
    uint32_t pattern = 0;

    while (i < data.size()) {
      pattern = (pattern << 1) + GEAR_TABLE[data[i]];
      if (remaining_minmax_adjustment > 0) {
        --remaining_minmax_adjustment;
        ++i;
        continue;
      }
      if (!(pattern & mask)) {
        const uint32_t byte_pos = i / 8u;
        const uint8_t bit_pos = i % 8u;
        results_bitmap[byte_pos] = results_bitmap[byte_pos] | static_cast<uint8_t>(1u << bit_pos);
      }
      ++i;
    }
  }
}

static void sscdc_second_stage_impl(
  const uint8_t* segment_results_bitmap,
  std::deque<utility::ChunkEntry>& process_pending_chunks,
  const uint64_t min_chunksize,
  const uint64_t max_chunksize,
  const uint64_t segment_length,
  const uint64_t segment_start_offset,
  const bool is_last_segment,
  uint64_t& prev_cut_offset
) {
  const auto vec_bit_width = lane_count * 32; // lanes of 32bit ints
  const i32Vec zero_v = hn::Set(i32VecD, 0);

  uint64_t in_segment_current_offset = 0;
  while (in_segment_current_offset < segment_length && segment_length - in_segment_current_offset >= vec_bit_width) {
    i32Vec bits = hn::LoadU(i32VecD, reinterpret_cast<const int32_t*>(&segment_results_bitmap[(in_segment_current_offset) / 8]));
    auto lane_has_result_bitmask = hn::Ne(bits, zero_v);
    if (hn::AllFalse(i32VecD, lane_has_result_bitmask)) {
      in_segment_current_offset += vec_bit_width;
      continue;
    }

    auto candidate_lane = hn::FindKnownFirstTrue(i32VecD, lane_has_result_bitmask);
    bool valid_cutoff_found = false;
    uint64_t in_segment_candidate_offset = in_segment_current_offset;
    for (; candidate_lane < lane_count; candidate_lane++) {
      if (valid_cutoff_found) break;
      uint32_t lane_bits = hn::ExtractLane(bits, candidate_lane);
      if (lane_bits == 0) continue;

      std::bitset<32> lane_bitset{ lane_bits };
      for (int8_t bit = 31; bit >= 0; bit--) {
        auto is_supposed_cut = lane_bitset.test(bit);
        in_segment_candidate_offset = in_segment_current_offset + (candidate_lane * 32u) + (31 - bit);
        if (is_supposed_cut) {
          const auto candidate_offset = in_segment_candidate_offset + segment_start_offset;
          // Before attempting to use this new cut candidate, ensure we haven't exceeded max chunk size since the previous cut
          while (prev_cut_offset + max_chunksize < candidate_offset) {
            prev_cut_offset = prev_cut_offset + max_chunksize;
            process_pending_chunks.emplace_back(prev_cut_offset);
          }
          const bool is_too_small = (candidate_offset < (prev_cut_offset + min_chunksize));
          if (!is_too_small) {
            valid_cutoff_found = true;
            break;
          }
        }
      }
    }
    if (
      !valid_cutoff_found ||
      // If it's not the first segment, the first 31 bytes have invalid results as the GEAR hash hasn't had a full data window processed yet
      (segment_start_offset != 0 && in_segment_candidate_offset <= 31)
    ) {
      in_segment_current_offset += vec_bit_width;
      continue;
    }

    prev_cut_offset = in_segment_candidate_offset + segment_start_offset;
    process_pending_chunks.emplace_back(prev_cut_offset);
    const uint64_t next_valid_offset = in_segment_candidate_offset + min_chunksize;
    // Jump to the previous byte from where we actually want to jump to (if not already byte aligned).
    // This way we can continue doing SIMD comparisons without missing anything, prev_cut_offset checks will prevent
    // any cutoff before min_chunksize on any of those bits case they have a cutoff candidate
    in_segment_current_offset = next_valid_offset - (next_valid_offset % vec_bit_width);
  }

  const auto is_cut = [&segment_results_bitmap, &prev_cut_offset, segment_start_offset, min_chunksize](uint64_t offset) {
    if (offset + segment_start_offset < prev_cut_offset + min_chunksize) return false;
    uint8_t bitmap_byte = segment_results_bitmap[offset / 8];
    return ((bitmap_byte >> (offset % 8)) & 0x1) == 1;
  };
  while (in_segment_current_offset < segment_length) {
    if (is_cut(in_segment_current_offset)) {
      prev_cut_offset = in_segment_current_offset + segment_start_offset;
      process_pending_chunks.emplace_back(prev_cut_offset);
      in_segment_current_offset = std::min(in_segment_current_offset + min_chunksize, segment_length);
      continue;
    }
    in_segment_current_offset++;
  }

  if (is_last_segment) {
    const auto last_offset = segment_start_offset + segment_length;
    while (prev_cut_offset + max_chunksize < last_offset) {
      prev_cut_offset = prev_cut_offset + max_chunksize;
      process_pending_chunks.emplace_back(prev_cut_offset);
    }
  }
}

}
}

HWY_AFTER_NAMESPACE();

#if HWY_ONCE

namespace CDCZ_SIMD {
	HWY_EXPORT(find_cdc_cut_candidates_simd_impl);
	HWY_EXPORT(sscdc_first_stage_impl);
  HWY_EXPORT(sscdc_second_stage_impl);
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
  HWY_DYNAMIC_DISPATCH(CDCZ_SIMD::find_cdc_cut_candidates_simd_impl)(candidates, candidate_features, data, min_size, avg_size, max_size, mask_hard, mask_medium, mask_easy, cdcz_cfg, is_first_segment);
}

void sscdc_first_stage(std::span<uint8_t> data, uint8_t* results_bitmap, uint32_t mask) {
  HWY_DYNAMIC_DISPATCH(CDCZ_SIMD::sscdc_first_stage_impl)(data, results_bitmap, mask);
}

void sscdc_second_stage(
  const uint8_t* segment_results_bitmap,
  std::deque<utility::ChunkEntry>& process_pending_chunks,
  const uint64_t min_chunksize,
  const uint64_t max_chunksize,
  const uint64_t segment_length,
  const uint64_t segment_start_offset,
  const bool is_last_segment,
  uint64_t& prev_cut_offset
) {
  HWY_DYNAMIC_DISPATCH(CDCZ_SIMD::sscdc_second_stage_impl)(
    segment_results_bitmap, process_pending_chunks, min_chunksize, max_chunksize, segment_length, segment_start_offset, is_last_segment, prev_cut_offset
  );
}

#endif
