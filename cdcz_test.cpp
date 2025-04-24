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

void cdcz_test_mode(const std::string& file_path, uint64_t file_size) {
  print_to_console("TEST MODE\n");

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
    print_to_console("Can't read file\n");
    exit(1);
  }
  {
    uint64_t remaining = file_size;
    char* data_ptr = reinterpret_cast<char*>(file_data);
    while (remaining > 0) {
      const auto read_amt = static_cast<std::streamsize>(std::min<uint64_t>(100 * 1024 * 1024, remaining));
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

  std::vector<CdcCandidatesResult> results;
  results.resize(segments.size());

  std::deque<utility::ChunkEntry> process_pending_chunks{};
  std::vector<uint8_t> prev_segment_remaining_data{};
  uint64_t current_offset = 0;
  uint64_t curr_segment_start_offset = 0;
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

  //print_to_console("dale gil\n");
  //get_char_with_echo();

  auto cdc_func = [&min_size, &avg_size, &max_size, &is_use_simd](std::span<uint8_t> segment_data, bool is_first_segment) {
    if (is_use_simd) {
      return find_cdc_cut_candidates<false, true>(segment_data, min_size, avg_size, max_size, is_first_segment);
    }
    else {
      return find_cdc_cut_candidates<false, false>(segment_data, min_size, avg_size, max_size, is_first_segment);
    }
    };

  bool is_first_segment = true;

  auto chunking_start_time = std::chrono::high_resolution_clock::now();
  std::deque<std::future<CdcCandidatesResult>> cdc_candidates_futures;
  for (uint64_t i = 0; i < segments.size(); i++) {
    cdc_candidates_futures.emplace_back(
      threadPool.addTask(
        [&cdc_func, &segments, is_first_segment, i]() mutable {
          return cdc_func(segments[i], is_first_segment);
        }
      )
    );
    is_first_segment = false;
  }

  for (uint64_t i = 0; i < segments.size(); i++) {
    auto& future = cdc_candidates_futures.front();
    results[i] = future.get();
    cdc_candidates_futures.pop_front();
  }

  //exit(0);

  for (uint64_t i = 0; i < results.size(); i++) {
    auto last_chunk_size = select_cut_point_candidates(
      results[i].candidates,
      results[i].candidatesFeatureResults,
      process_pending_chunks,
      current_offset,
      curr_segment_start_offset,
      segments[i],
      prev_segment_remaining_data,
      min_size,
      avg_size,
      max_size,
      i == results.size() - 1,
      false,
#ifdef NDEBUG
      false
#else
      true
#endif
    );

    current_offset = process_pending_chunks.back().offset + last_chunk_size;
    curr_segment_start_offset += std::min<uint64_t>(segments[i].size(), segment_size);
  }
  auto chunking_end_time = std::chrono::high_resolution_clock::now();

  XXH3_state_t* hash_state = XXH3_createState();
  utility::ChunkEntry* prevChunk = nullptr;
  auto dedup_start_time = std::chrono::high_resolution_clock::now();
  for (auto& chunk : process_pending_chunks) {
    if (prevChunk == nullptr) {
      prevChunk = &chunk;
      continue;
    }
    const auto chunk_size = chunk.offset - prevChunk->offset;
    auto hash = make_chunk_hash(chunk_size, hash_state);
#ifndef NDEBUG
    auto hash_from_chunk_data = XXH3_64bits(prevChunk->chunk_data->data.data(), prevChunk->chunk_data->data.size());
    if (hash != hash_from_chunk_data) {
      print_to_console("ERROR: BAD CHUNK HASH, WILL MESS UP DEDUPLICATION\n");
    }
#endif
    if (chunk_hashes.contains(hash)) {
      deduped_size += chunk_size;
    }
    else {
      chunk_hashes.emplace(hash);
    }

    prevChunk = &chunk;
  }
  {
    const auto chunk_size = file_size - prevChunk->offset;
    auto hash = make_chunk_hash(chunk_size, hash_state);
#ifndef NDEBUG
    auto hash_from_chunk_data = XXH3_64bits(prevChunk->chunk_data->data.data(), prevChunk->chunk_data->data.size());
    if (hash != hash_from_chunk_data) {
      print_to_console("ERROR: BAD CHUNK HASH, WILL MESS UP DEDUPLICATION\n");
    }
#endif
    if (chunk_hashes.contains(hash)) {
      deduped_size += chunk_size;
    }
    else {
      chunk_hashes.emplace(hash);
    }
  }
  XXH3_freeState(hash_state);
  auto dedup_end_time = std::chrono::high_resolution_clock::now();

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
