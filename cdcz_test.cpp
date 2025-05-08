#include "cdcz_test.hpp"
#include "cdc_algos/cdcz.hpp"

#include <chrono>
#include <fstream>
#include <span>
#include <unordered_set>

#include "contrib/task_pool.h"
#include "contrib/xxHash/xxhash.h"
#include "utils/chunks.hpp"
#include "utils/console_utils.hpp"

struct ChunkTrace {
  uint64_t hash;
  uint64_t size;
};

void cdcz_test_mode(const std::string& file_path, uint64_t file_size, const std::string& trace_out_path, const std::string& trace_in_path) {
  print_to_console("TEST MODE\n");

  auto trace_out = std::fstream();
  if (!trace_out_path.empty()) {
    trace_out.open(trace_out_path, std::ios::out | std::ios::trunc);
    if (!trace_out.is_open()) {
      print_to_console("Can't read trace out file\n");
      exit(1);
    }
  }
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
      auto space_pos = line.find(' ');
      trace_inputs.emplace_back(std::stoull(line.substr(0, space_pos)), std::stoull(line.substr(space_pos + 1, line.size() - space_pos - 1)));
    }
  }

  const uint64_t min_size = 2048;
  const uint64_t avg_size = 8ull * 1024;
  const uint64_t max_size = 64ull * 1024;

  const auto file_size_mb = file_size / (1024.0 * 1024);
  const int alignment = 32;

#ifndef __clang__
  auto file_data = static_cast<uint8_t*>(_aligned_malloc(file_size, alignment));
#else
  const uint64_t adjusted_file_size = ((file_size + alignment - 1) / alignment) * alignment;
  auto file_data = static_cast<uint8_t*>(std::aligned_alloc(32, adjusted_file_size));
#endif

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

  const bool is_use_mt = true;
  const bool is_use_simd = true;
  TaskPool threadPool = TaskPool(is_use_mt ? std::thread::hardware_concurrency() : 1);

  // For SIMD we are using lanes with signed 32bit integers, if we use segments that are too large we risk overflows, this way we have pretty big segments
  // with no risk of overflows.
  const uint64_t segment_size = std::min<uint64_t>(std::numeric_limits<int32_t>::max() / 2, file_size / std::thread::hardware_concurrency());

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

  {
    std::deque<std::future<CdcCandidatesResult>> cdc_candidates_futures;
    CdcCandidatesResult result;
    bool is_first_segment = true;
    uint64_t curr_segment_start_offset = 0;
    uint64_t current_offset = 0;

    uint64_t chunk_idx = 0;
    auto pending_chunks_iter = process_pending_chunks.cbegin();

    chunking_start_time = std::chrono::high_resolution_clock::now();
    for (uint64_t i = 0; i < segments.size(); i++) {
      cdc_candidates_futures.emplace_back(
        threadPool.addTask(
          [&segments, is_first_segment, i, &min_size, &avg_size, &max_size, &is_use_simd]() {
            const CDCZ_CONFIG cfg{ .avx2_allowed = is_use_simd};
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
        current_offset,
        curr_segment_start_offset,
        segments[i],
        prev_segment_remaining_data,
        min_size,
        avg_size,
        max_size,
        i == segments.size() - 1,
        false,
#ifdef NDEBUG
        false
#else
        true
#endif
      );

      if (!trace_inputs.empty()) {
        if (i == 0) {
          pending_chunks_iter = process_pending_chunks.cbegin();
        }
        else {
          ++pending_chunks_iter;
        }
        while (pending_chunks_iter != process_pending_chunks.cend()) {
          auto next_pending_chunk_iter = pending_chunks_iter + 1;
          uint64_t chunk_size = next_pending_chunk_iter == process_pending_chunks.cend() ? last_chunk_size : next_pending_chunk_iter->offset - pending_chunks_iter->offset;
          if (chunk_size != trace_inputs[chunk_idx].size) {
            print_to_console(
              "Chunk size mismatch for chunk idx " + std::to_string(chunk_idx) + ". Actual: " + std::to_string(chunk_size) +
              ", Expected: " + std::to_string(trace_inputs[chunk_idx].size) + ".\n" +
              "On segment " + std::to_string(i) + ", current_offset: " + std::to_string(current_offset) + ", segment offset: " + std::to_string(curr_segment_start_offset) + "\n"
            );
            exit(1);
          }
          pending_chunks_iter = next_pending_chunk_iter;
          ++chunk_idx;
        }
        --pending_chunks_iter;  // leave the iter pointing to the last actual chunk, so for next segment when there are new chunks we can get to them from here
      }

      current_offset = process_pending_chunks.back().offset + last_chunk_size;
      curr_segment_start_offset += std::min<uint64_t>(segments[i].size(), segment_size);

      cdc_candidates_futures.pop_front();
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
    auto hash_from_chunk_data = XXH3_64bits(prevChunk->chunk_data->data.data(), prevChunk->chunk_data->data.size());
    if (hash != hash_from_chunk_data) {
      print_to_console("ERROR: BAD CHUNK HASH, WILL MESS UP DEDUPLICATION\n");
    }
#endif
    if (!trace_inputs.empty() && trace_inputs.size() <= chunk_idx) {
      print_to_console("Chunk idx " + std::to_string(chunk_idx) + " is more than whats available on the trace input.\n");
      exit(1);
    }
    if (!trace_inputs.empty() && trace_inputs.size() > chunk_idx && trace_inputs[chunk_idx].hash != hash) {
      print_to_console(
        "Chunk idx " + std::to_string(chunk_idx) + " has hash " + std::to_string(trace_inputs[chunk_idx].hash) + " and it should have " + std::to_string(hash) + ".\n"
      );
      exit(1);
    }
    ++chunk_idx;
    if (!trace_out_path.empty()) {
      const std::string trace_line = std::to_string(hash) + " " + std::to_string(chunk_size) + std::string("\n");
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

  print_to_console("Tested file size (bytes):    " + std::to_string(file_size) + "\n");
  print_to_console("Tested Chunking Throughput:    %.1f MB/s\n", chunking_mb_per_nanosecond * std::pow(10, 9));
  print_to_console("Tested Dedup Throughput:    %.1f MB/s\n", dedup_mb_per_nanosecond * std::pow(10, 9));
  print_to_console("Tested Chunking+Dedup Throughput:    %.1f MB/s\n", test_mb_per_nanosecond * std::pow(10, 9));
  print_to_console("Deduped size (bytes):    " + std::to_string(deduped_size) + "\n");
  print_to_console("File size after dedup (bytes):    " + std::to_string(file_size - deduped_size) + "\n");
  print_to_console("DER:    %f\n", static_cast<float>(file_size) / static_cast<float>(file_size - deduped_size));
  print_to_console("Total chunk count:    " + std::to_string(process_pending_chunks.size()) + "\n");
  print_to_console("Total chunking runtime:    " + std::to_string(std::chrono::duration_cast<std::chrono::seconds>(chunking_elapsed_time).count()) + " seconds\n");
  print_to_console("Total dedup runtime:    " + std::to_string(std::chrono::duration_cast<std::chrono::seconds>(dedup_elapsed_time).count()) + " seconds\n");
  print_to_console("Total test runtime:    " + std::to_string(std::chrono::duration_cast<std::chrono::seconds>(test_elapsed_time).count()) + " seconds\n");

#ifndef __clang__
  _aligned_free(file_data);
#else
  std::free(file_data);
#endif
}
