#include "cdcz_test.hpp"
#include "cdc_algos/cdcz.hpp"

#include <chrono>
#include <fstream>
#include <span>
#include <unordered_set>

#include "cdc_algos/gear.hpp"
#include "contrib/task_pool.h"
#include "contrib/xxHash/xxhash.h"
#include "utils/chunks.hpp"
#include "utils/console_utils.hpp"

struct ChunkTrace {
  uint64_t offset;
  uint64_t size;
  uint64_t hash;
};

inline void* portable_aligned_alloc(std::size_t alignment, std::size_t size) {
#ifdef _WIN32
  return _aligned_malloc(size, alignment);
#elif defined(__APPLE__) || defined(__MACH__)
  // macOS: posix_memalign is safest
  void* ptr = nullptr;
  if (posix_memalign(&ptr, alignment, size) != 0) {
    return nullptr;
  }
  return ptr;
#else
  // Linux and others with C11 support
  return std::aligned_alloc(alignment, size);
#endif
}

inline void portable_aligned_free(void* ptr) {
#ifdef _WIN32
  _aligned_free(ptr);
#else
  std::free(ptr);
#endif
}

void cdcz_test_mode(const std::string& file_path, uint64_t file_size, std::unordered_map<std::string, std::string>& cli_params) {
  print_to_console("TEST MODE\n");

  const uint64_t thread_count = std::stoi(cli_params["threads"]);
  const bool is_use_mt = thread_count > 1;
  const bool is_simulate_mt = !cli_params["simulate_mt"].empty();
  const bool is_use_simd = !cli_params["simd"].empty();

  const std::string trace_out_path = cli_params["trace_out_file_path"];
  auto trace_out = std::fstream();
  if (!trace_out_path.empty()) {
    trace_out.open(trace_out_path, std::ios::out | std::ios::trunc);
    if (!trace_out.is_open()) {
      print_to_console("Can't read trace out file\n");
      exit(1);
    }
  }
  const std::string trace_in_path = cli_params["trace_in_file_path"];
  std::vector<ChunkTrace> trace_inputs{};
  if (!trace_in_path.empty()) {
    auto trace_in = std::fstream(trace_in_path, std::ios::in);
    if (!trace_in.is_open()) {
      print_to_console("Can't read trace in file\n");
      exit(1);
    }
    print_to_console("WARNING: using trace input to ensure chunks produced are expected ones, this will penalize Throughput slightly\n");
    std::string line;
    while (std::getline(trace_in, line)) {
      auto first_space_pos = line.find(' ');
      auto second_space_pos = line.find(' ', first_space_pos + 1);
      trace_inputs.emplace_back(
        std::stoull(line.substr(0, first_space_pos)),
        std::stoull(line.substr(first_space_pos + 1, second_space_pos)),
        std::stoull(line.substr(second_space_pos + 1, line.size() - second_space_pos - 1))
      );
    }
  }

  const uint64_t min_size = 2048;
  const uint64_t avg_size = 8ull * 1024;
  const uint64_t max_size = 64ull * 1024;

  const auto file_size_mb = file_size / (1024.0 * 1024);
  const int alignment = 64;  // TODO: use proper alignment for arch, hardcoding 64 as it works well for most cases I am using

  auto file_data = static_cast<uint8_t*>(portable_aligned_alloc(alignment, file_size));

  auto file_stream = std::fstream(file_path, std::ios::in | std::ios::binary);
  if (!file_stream.is_open()) {
    print_to_console("Can't read test input file\n");
    exit(1);
  }
  {
    uint64_t remaining = file_size;
    char* data_ptr = reinterpret_cast<char*>(file_data);
    while (remaining > 0) {
      const auto read_amt = static_cast<std::streamsize>(std::min<uint64_t>(100ull * 1024 * 1024, remaining));
      file_stream.read(data_ptr, read_amt);
      const auto actually_read = file_stream.gcount();
      if (actually_read != read_amt) {
        print_to_console("Couldn't read whole file\n");
        exit(1);
      }
      remaining -= actually_read;
      data_ptr += actually_read;
    }
  }

  std::unordered_set<uint64_t> chunk_hashes{};
  uint64_t deduped_size = 0;
  TaskPool threadPool = TaskPool(thread_count);

  // For SIMD we are using lanes with signed 32bit integers, if we use segments that are too large we risk overflows, this way we have pretty big segments
  // with no risk of overflows.
  // On my tests (x86 intel and amd only) 128mb segments seem to perform the best.
  const uint64_t segment_size = std::min<uint64_t>(std::numeric_limits<int32_t>::max() / 16, file_size / thread_count);

  std::vector<std::span<uint8_t>> segments;
  if (is_use_mt || is_use_simd) {
    auto data_span = std::span<uint8_t>(file_data, file_size);
    while (!data_span.empty()) {
      const uint64_t segment_expected_size = segment_size + 31;
      const uint64_t this_segment_size = std::min<uint64_t>(data_span.size(), segment_expected_size);

      segments.emplace_back(data_span.data(), this_segment_size);
      auto& data_segment_span = segments.back();

      if (this_segment_size == segment_expected_size) {
        data_span = std::span<uint8_t>(data_span.data() + data_segment_span.size() - 31, data_span.size() - data_segment_span.size() + 31);
      }
      else {
        // last segment, set empty span so we quit
        data_span = std::span<uint8_t>(data_span.data(), 0);
      }
    }
  }
  else {
    segments.emplace_back(file_data, file_size);
  }

  std::deque<utility::ChunkEntry> process_pending_chunks{};
  std::vector<uint8_t> prev_segment_remaining_data{};
  uint64_t current_segment_i = 0;
  uint64_t current_segment_pos = 0;

  auto make_chunk_hash = [&segments, &current_segment_i, &current_segment_pos]
  (uint64_t chunk_size, XXH3_state_t* hash_state) {
    auto in_segment_data = std::min(chunk_size, segments[current_segment_i].size() - current_segment_pos);
    if (in_segment_data == 0) {
      current_segment_i++;
      current_segment_pos = 31;
      in_segment_data = std::min(chunk_size, segments[current_segment_i].size() - current_segment_pos);
    }
    XXH3_64bits_reset(hash_state);
    XXH3_64bits_update(hash_state, segments[current_segment_i].data() + current_segment_pos, in_segment_data);
    current_segment_pos += in_segment_data;
    if (in_segment_data < chunk_size) {
      current_segment_i++;
      current_segment_pos = 31;
      in_segment_data = chunk_size - in_segment_data;
      XXH3_64bits_update(hash_state, segments[current_segment_i].data() + current_segment_pos, in_segment_data);
      current_segment_pos += in_segment_data;
    }

    return XXH3_64bits_digest(hash_state);
    };

  auto chunking_start_time = std::chrono::high_resolution_clock::now();
  auto chunking_end_time = std::chrono::high_resolution_clock::now();

  const bool do_trace_input_check = !trace_inputs.empty();
  // If no form of parallelism was used then chunks resulting from select_cut_point_candidates should just be the candidates
  const bool do_all_candidates_selected_check = !is_use_mt && !is_use_simd;
  auto do_chunk_correctness_checks = [&process_pending_chunks, &trace_inputs, do_trace_input_check, do_all_candidates_selected_check]
  (uint64_t& chunk_idx, uint64_t last_chunk_size, CdcCandidatesResult& result, uint64_t segment_i, uint64_t segment_size, uint64_t last_cut_offset, uint64_t curr_segment_start_offset) {
    if (do_trace_input_check || do_all_candidates_selected_check) {
      while (chunk_idx != process_pending_chunks.size()) {
        const uint64_t next_chunk_idx = chunk_idx + 1;
        const uint64_t pending_chunk_size = next_chunk_idx == process_pending_chunks.size() ? last_chunk_size : process_pending_chunks[next_chunk_idx].offset - process_pending_chunks[chunk_idx].offset;
        if (do_all_candidates_selected_check) {
          const uint64_t prev_candidate_offset = chunk_idx >= 1 ? result.candidates[chunk_idx - 1].offset : 0;
          const uint64_t candidate_chunk_size = result.candidates[chunk_idx].offset - prev_candidate_offset;
          if (candidate_chunk_size != pending_chunk_size) {
            print_to_console(
              "select_cut_point_candidates messed up candidate selection on segment {} with size {}, chunk {}, we needed chunk_size of {} but got {}",
              segment_i, segment_size, chunk_idx, candidate_chunk_size, pending_chunk_size
            );
            exit(1);
          }
        }
        if (do_trace_input_check && (pending_chunk_size != trace_inputs[chunk_idx].size || process_pending_chunks[chunk_idx].offset != trace_inputs[chunk_idx].offset)) {
          print_to_console(
            "Chunk size mismatch for chunk idx {}. Actual: {}, Expected: {}.\n",
            chunk_idx, pending_chunk_size, trace_inputs[chunk_idx].size
          );
          print_to_console(
            "On segment {} with size {}, last_cut_offset: {}, segment offset: {}, chunk offset: {}\n",
            segment_i, segment_size, last_cut_offset, curr_segment_start_offset, trace_inputs[chunk_idx].offset
          );
          exit(1);
        }
        ++chunk_idx;
      }
    }
  };

  {
    std::deque<std::future<CdcCandidatesResult>> cdc_candidates_futures{};
    std::queue<uint64_t> supercdc_backup_pos{};
    CdcCandidatesResult result;
    bool is_first_segment = true;
    uint64_t curr_segment_start_offset = 0;
    uint64_t last_cut_offset = 0;

    uint64_t chunk_idx = 0;

    chunking_start_time = std::chrono::high_resolution_clock::now();
    if (!is_simulate_mt) {
      for (uint64_t i = 0; i < segments.size(); i++) {
        cdc_candidates_futures.emplace_back(
          threadPool.addTask(
            [&segments, is_first_segment, i, &min_size, &avg_size, &max_size, &is_use_simd]() {
              const CDCZ_CONFIG cfg{ .avx2_allowed = is_use_simd };
              return find_cdc_cut_candidates(segments[i], min_size, avg_size, max_size, cfg, is_first_segment);
            }
          )
        );
        is_first_segment = false;
      }

      for (uint64_t i = 0; i < segments.size(); i++) {
        auto& future = cdc_candidates_futures.front();
        result = future.get();

        auto last_chunk_size = select_cut_point_candidates(
          result.candidates,
          result.candidatesFeatureResults,
          process_pending_chunks,
          supercdc_backup_pos,
          last_cut_offset,
          curr_segment_start_offset,
          segments[i],
          prev_segment_remaining_data,
          min_size,
          avg_size,
          max_size,
          i == segments.size() - 1,
          false,
          is_first_segment,
#ifdef NDEBUG
          false
#else
          true
#endif
        );

        do_chunk_correctness_checks(chunk_idx, last_chunk_size, result, i, segments[i].size(), last_cut_offset, curr_segment_start_offset);

        last_cut_offset = process_pending_chunks.back().offset + last_chunk_size;
        curr_segment_start_offset += std::min<uint64_t>(segments[i].size(), segment_size);

        cdc_candidates_futures.pop_front();
      }
    }
    else {
      const CDCZ_CONFIG cfg{ .avx2_allowed = is_use_simd };
      for (uint64_t i = 0; i < segments.size(); i++) {
        result = find_cdc_cut_candidates(segments[i], min_size, avg_size, max_size, cfg, is_first_segment);

        auto last_chunk_size = select_cut_point_candidates(
          result.candidates,
          result.candidatesFeatureResults,
          process_pending_chunks,
          supercdc_backup_pos,
          last_cut_offset,
          curr_segment_start_offset,
          segments[i],
          prev_segment_remaining_data,
          min_size,
          avg_size,
          max_size,
          i == segments.size() - 1,
          false,
          is_first_segment,
#ifdef NDEBUG
          false
#else
          true
#endif
        );

        do_chunk_correctness_checks(chunk_idx, last_chunk_size, result, i, segments[i].size(), last_cut_offset, curr_segment_start_offset);

        last_cut_offset = process_pending_chunks.back().offset + last_chunk_size;
        curr_segment_start_offset += std::min<uint64_t>(segments[i].size(), segment_size);

        is_first_segment = false;
      }
    }
    chunking_end_time = std::chrono::high_resolution_clock::now();
  }

  XXH3_state_t* hash_state = XXH3_createState();
  auto chunks_iter = process_pending_chunks.cbegin();
  uint64_t chunk_idx = 0;
  auto dedup_start_time = std::chrono::high_resolution_clock::now();
  while (chunks_iter != process_pending_chunks.cend()) {
    const utility::ChunkEntry& chunk = *chunks_iter;
    ++chunks_iter;
    const uint64_t next_chunk_offset = chunks_iter == process_pending_chunks.cend() ? file_size : chunks_iter->offset;

    const uint64_t chunk_size = next_chunk_offset - chunk.offset;
    const uint64_t hash = make_chunk_hash(chunk_size, hash_state);
#ifndef NDEBUG
    auto hash_from_chunk_data = XXH3_64bits(chunk.chunk_data->data.data(), chunk.chunk_data->data.size());
    if (hash != hash_from_chunk_data) {
      print_to_console("ERROR: BAD CHUNK HASH, WILL MESS UP DEDUPLICATION\n");
    }
#endif
    if (!trace_inputs.empty() && trace_inputs.size() <= chunk_idx) {
      print_to_console("Chunk idx {} is more than whats available on the trace input.\n", chunk_idx);
      exit(1);
    }
    if (!trace_inputs.empty() && trace_inputs.size() > chunk_idx && trace_inputs[chunk_idx].hash != hash) {
      print_to_console(
        "Chunk idx {} has hash {} and it should have {}.\n", chunk_idx, trace_inputs[chunk_idx].hash, hash
      );
      exit(1);
    }
    ++chunk_idx;
    if (!trace_out_path.empty()) {
      const std::string trace_line = std::to_string(chunk.offset) + " " + std::to_string(chunk_size) + " " + std::to_string(hash) + std::string("\n");
      trace_out.write(trace_line.c_str(), trace_line.size());
    }

    if (chunk_hashes.contains(hash)) {
      deduped_size += chunk_size;
    }
    else {
      chunk_hashes.emplace(hash);
    }
  }
  auto dedup_end_time = std::chrono::high_resolution_clock::now();
  XXH3_freeState(hash_state);
  if (!trace_out_path.empty()) trace_out.close();

  auto chunking_elapsed_time = std::chrono::duration_cast<std::chrono::nanoseconds>(chunking_end_time - chunking_start_time);
  auto chunking_elapsed_time_ns = chunking_elapsed_time.count();
  const auto chunking_mb_per_nanosecond = file_size_mb / chunking_elapsed_time_ns;

  auto dedup_elapsed_time = std::chrono::duration_cast<std::chrono::nanoseconds>(dedup_end_time - dedup_start_time);
  auto dedup_elapsed_time_ns = dedup_elapsed_time.count();
  const auto dedup_mb_per_nanosecond = file_size_mb / dedup_elapsed_time_ns;

  auto test_elapsed_time = std::chrono::duration_cast<std::chrono::nanoseconds>(dedup_end_time - chunking_start_time);
  auto test_elapsed_time_ns = test_elapsed_time.count();
  const auto test_mb_per_nanosecond = file_size_mb / test_elapsed_time_ns;

  const auto file_size_after_dedup = file_size - deduped_size;

  print_to_console("Tested file size (bytes):            {}\n", file_size);
  print_to_console("Tested Chunking Throughput:          {:.1f} MB/s\n", chunking_mb_per_nanosecond * std::pow(10, 9));
  print_to_console("Tested Dedup Throughput:             {:.1f} MB/s\n", dedup_mb_per_nanosecond * std::pow(10, 9));
  print_to_console("Tested Chunking+Dedup Throughput:    {:.1f} MB/s\n", test_mb_per_nanosecond * std::pow(10, 9));
  print_to_console("Deduped size (bytes):                {}\n", deduped_size);
  print_to_console("File size after dedup (bytes):       {}\n", file_size_after_dedup);
  print_to_console("DER:                                 {:f}\n", static_cast<float>(file_size) / static_cast<float>(file_size_after_dedup));
  print_to_console("Total chunk count:                   {}\n", process_pending_chunks.size());
  print_to_console("Average chunk size:                  {}\n", file_size_after_dedup/ process_pending_chunks.size());
  print_to_console("Total chunking runtime:              {} seconds\n", std::chrono::duration_cast<std::chrono::seconds>(chunking_elapsed_time).count());
  print_to_console("Total dedup runtime:                 {} seconds\n", std::chrono::duration_cast<std::chrono::seconds>(dedup_elapsed_time).count());
  print_to_console("Total test runtime:                  {} seconds\n", std::chrono::duration_cast<std::chrono::seconds>(test_elapsed_time).count());

  // Bleh, dirty hack for quicker exit
  exit(0);
  portable_aligned_free(file_data);
}

void calc_gear_at_pos(const std::string& file_path, uint64_t file_size, uint64_t pos) {
  auto file_stream = std::fstream(file_path, std::ios::in | std::ios::binary);
  if (!file_stream.is_open()) {
    print_to_console("Can't read test input file\n");
    exit(1);
  }

  auto minmax_adjustment = std::min<uint64_t>(pos, 31);
  file_stream.seekg(pos - minmax_adjustment);
  std::vector<uint8_t> data{};
  data.resize(minmax_adjustment + 1);

  file_stream.read(reinterpret_cast<char*>(data.data()), minmax_adjustment + 1);
  if (file_stream.gcount() != minmax_adjustment + 1) {
    print_to_console("Can't read test input file\n");
    exit(1);
  }

  uint32_t pattern = 0;
  for (uint64_t i = 0; i < minmax_adjustment + 1; i++) {
    pattern = (pattern << 1) + GEAR_TABLE[data[i]];
  }

  std::bitset<32> pattern_bitset = pattern;
  print_to_console("The GEAR hash at requested position {} is {}\n", pos, pattern_bitset.to_string());
  exit(0);
}
