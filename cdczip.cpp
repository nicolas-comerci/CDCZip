#include <algorithm>
#include <cmath>
#include <istream>
#include <iostream>
#include <fstream>
#include <utility>
#include <string>
#include <random>
#include <cstdio>
#include <chrono>
#include <span>
#include <array>
#include <list>
#include <unordered_map>
#include <bitset>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <filesystem>
#include <functional>
#include <ranges>
#include <stack>
#include <unordered_set>
#include <thread>

#if defined(__AVX2__) || defined(__SSE2__)
#include <immintrin.h>
#endif

#include "contrib/stream.h"
#include "contrib/task_pool.h"
#include "contrib/xxHash/xxhash.h"

#include "cdczip_decompress.hpp"
#include "cdcz_test.hpp"

#include "utils/chunks.hpp"
#include "utils/circular_buffer.hpp"
#include "utils/circular_vector.hpp"
#include "utils/console_utils.hpp"
#include "utils/io_utils.hpp"
#include "utils/lz.hpp"
#include "cdc_algos/cdcz.hpp"
#include "delta_compression/delta.hpp"
#include "delta_compression/encoding.hpp"
#include "delta_compression/simhash.hpp"

namespace constants {
  // Smallest acceptable value for the minimum chunk size.
  static constexpr uint32_t MINIMUM_MIN = 64;
  // Largest acceptable value for the minimum chunk size.
  static constexpr uint32_t MINIMUM_MAX = 67'108'864;
  // Smallest acceptable value for the average chunk size.
  static constexpr uint32_t AVERAGE_MIN = 256;
  // Largest acceptable value for the average chunk size.
  static constexpr uint32_t AVERAGE_MAX = 268'435'456;
  // Smallest acceptable value for the maximum chunk size.
  static constexpr uint32_t MAXIMUM_MIN = 1024;
  // Largest acceptable value for the maximum chunk size.
  static constexpr uint32_t MAXIMUM_MAX = 1'073'741'824;
}

int main(int argc, char* argv[]) {
#ifndef NDEBUG
  print_to_console("STARTED IN DEBUG MODE, PRESS ENTER TO CONTINUE, NOW WOULD BE A GOOD TIME TO ATTACH A DEBUGGER IF YOU NEED TO.\n");
  get_char_with_echo();
#endif
  std::string input_path{ argv[1] };
  bool do_decompression = false;

  std::unordered_map<std::string, std::string> cli_params;
  cli_params["threads"] = std::to_string(std::thread::hardware_concurrency());
  for (uint64_t param_idx = 2; param_idx < argc; param_idx++) {
    auto param = std::string(argv[param_idx]);
    if (param.starts_with("--threads=")) {
      constexpr auto opt_size = std::string("--threads=").size();
      const auto thread_count_str = param.substr(opt_size, param.size() - opt_size);
	  if (std::stoi(thread_count_str) != 0) {  // if 0 we just keep the auto-detected concurrency count
          cli_params["threads"] = param.substr(opt_size, param.size() - opt_size);
	  }
    }
    else if (param.starts_with("--simd")) {
        cli_params["simd"] = "true";
    }
    else if (param.starts_with("--dict=")) {
        constexpr auto opt_size = std::string("--dict=").size();
        cli_params["max_dict"] = param.substr(opt_size, param.size() - opt_size);
    }
    else if (param.starts_with("-d=")) {
      do_decompression = true;
      cli_params["decomp_file_path"] = param.substr(3, param.size() - 3);
    }
    else if (param.starts_with("--test")) {
      cli_params["test_mode"] = "true";
    }
    else if (param.starts_with("--gear-at-pos=")) {
      constexpr auto opt_size = std::string("--gear-at-pos=").size();
      cli_params["calc_gear_at_pos"] = param.substr(opt_size, param.size() - opt_size);
    }
    else if (param.starts_with("--trace-out=")) {
      constexpr auto opt_size = std::string("--trace-out=").size();
      cli_params["trace_out_file_path"] = param.substr(opt_size, param.size() - opt_size);
    }
    else if (param.starts_with("--trace-in=")) {
      constexpr auto opt_size = std::string("--trace-in=").size();
      cli_params["trace_in_file_path"] = param.substr(opt_size, param.size() - opt_size);
    }
    else if (param.starts_with("--simulate-mt")) {
        cli_params["simulate_mt"] = "true";
    }
    else {
      print_to_console("Bad arg: {}\n", param);
      exit(1);
    }
  }

  auto input_stream = std::ifstream();
  if (input_path == "-") {
      set_std_handle_binary_mode(StdHandles::STDIN_HANDLE);
      reinterpret_cast<std::istream*>(&input_stream)->rdbuf(std::cin.rdbuf());
  }
  else {
      input_stream.open(input_path, std::ios::in | std::ios::binary);
      if (!input_stream.is_open()) {
          print_to_console("Can't read file\n");
          return 1;
      }
  }

  if (do_decompression) {  // decompress
    std::array<char, 12> header;
    input_stream.read(header.data(), 12);
    if (header[0] != 'C' || header[1] != 'D' || header[2] != 'C' || header[3] != 'Z') {
      print_to_console("Input is not a CDCZip deduplicated stream!\n");
      exit(1);
    }
    print_to_console("Decompressing...\n");
    uint64_t dict_size = *reinterpret_cast<uint64_t*>(&header[4]);

    uint8_t flags;
    input_stream.read(reinterpret_cast<char*>(&flags), 1);

    auto decompress_start_time = std::chrono::high_resolution_clock::now();

    std::string& decomp_file_path = cli_params["decomp_file_path"];
    const bool just_hash = decomp_file_path == "hash";
    auto decomp_file_stream = std::fstream();
    std::unique_ptr<FakeIOStream> decomp_hash_stream;
    if (just_hash) {
      decomp_hash_stream = std::make_unique<FakeIOStream>();
    }
    else if (decomp_file_path == "-") {
      if (dict_size == 0) {
        print_to_console("Can't decompress file to stdout as it is flagged as seekable stream only decompressable");
        exit(1);
      }
      set_std_handle_binary_mode(StdHandles::STDOUT_HANDLE);
      static_cast<std::ostream*>(&decomp_file_stream)->rdbuf(std::cout.rdbuf());
    }
    else {
      decomp_file_stream.open(decomp_file_path, std::ios::in | std::ios::out | std::ios::binary | std::ios::trunc);
    }

    auto wrapped_input_stream = WrappedIStreamInputStream(&input_stream);
    auto bit_input_stream = BitInputStream(wrapped_input_stream);

    if (just_hash) {
      decompress(*decomp_hash_stream, bit_input_stream, dict_size);
      decomp_hash_stream->print_hash();
    }
    else {
      decompress(decomp_file_stream, bit_input_stream, dict_size);
    }

    auto decompress_end_time = std::chrono::high_resolution_clock::now();
    print_to_console("Decompression finished in {} seconds!\n", std::chrono::duration_cast<std::chrono::seconds>(decompress_end_time - decompress_start_time).count());
    return 0;
  }

  uint64_t file_size = input_path == "-" ? 0 : std::filesystem::file_size(input_path);

  if (cli_params["test_mode"] == "true") {
    cdcz_test_mode(input_path, file_size, cli_params);
    exit(0);
  }
  if (!cli_params["calc_gear_at_pos"].empty()) {
    calc_gear_at_pos(input_path, file_size, std::stoull(cli_params["calc_gear_at_pos"]));
    exit(0);
  }

  uint32_t avg_size = 256;
  uint32_t min_size = avg_size / 4;
  uint32_t max_size = avg_size * 8;

  if (constants::MINIMUM_MIN > min_size || min_size > constants::MINIMUM_MAX) throw std::runtime_error("Bad minimum size");
  if (constants::AVERAGE_MIN > avg_size || avg_size > constants::AVERAGE_MAX) throw std::runtime_error("Bad average size");
  if (constants::MAXIMUM_MIN > max_size || max_size > constants::MAXIMUM_MAX) throw std::runtime_error("Bad maximum size");

  //bert_params params;
  //params.model = R"(C:\Users\Administrator\Desktop\fastcdc_test\paraphrase-MiniLM-L3-v2-GGML-q4_0.bin)";
  //params.n_threads = 20;
  //bert_ctx* bctx = bert_load_from_file(params.model);
  //const int bert_max_tokens_num = bert_n_max_tokens(bctx);
  //std::vector<bert_vocab_id> bert_feature_tokens(bert_max_tokens_num);
  int bert_tokens_num = 0;
  //const auto embeddings_dim = bert_n_embd(bctx);
  //std::vector<float> bert_embeddings(embeddings_dim);

  uint64_t batch_size = 100;

  //faiss::IndexLSH index(max_size / 4, 64, false);
  std::vector<float> search_results(5 * batch_size, 0);
  //std::vector<faiss::idx_t> search_result_labels(5 * batch_size, -1);
  //printf("is_trained = %s\n", index.is_trained ? "true" : "false");

  uint64_t total_size = 0;
  uint64_t deduped_size = 0;
  uint64_t delta_compressed_approx_size = 0;
  uint64_t delta_compressed_chunk_count = 0;

  ChunkIndex known_hashes{};  // Find chunk pos by hash

  uint64_t chunk_i = 0;
  uint64_t chunk_id = 0;
  uint64_t last_reduced_chunk_i = 0;
  std::vector<int32_t> pending_chunks_indexes(batch_size, 0);
  std::vector<uint8_t> pending_chunk_data(batch_size * max_size, 0);
  std::unordered_map<std::bitset<64>, uint64_t> simhashes_dict{};

  // Find chunk_i that has a given SuperFeature
  std::unordered_map<uint32_t, std::list<uint64_t>> superfeatures_dict{};

  const auto cdc_thread_count = std::stoi(cli_params["threads"]);

  const uint32_t simhash_chunk_size = std::max<uint32_t>(16, min_size / 8);
  const uint32_t max_allowed_dist = 32;
  const uint32_t delta_mode = 2;
  // Don't even bother saving chunk as delta chunk if savings are too little and overhead will probably negate them
  const uint64_t min_delta_saving = std::min(min_size * 2, avg_size);

  std::tuple<std::bitset<64>, std::vector<uint32_t>> (*simhash_func)(uint8_t * data, uint32_t data_len, uint32_t minichunk_size) = nullptr;
  DeltaEncodingResult(*simulate_delta_encoding_func)(const utility::ChunkData& chunk, const utility::ChunkData& similar_chunk, uint32_t minichunk_size) = nullptr;
  if (delta_mode == 0) {
    // Fastest delta mode, SimHash and delta encoding using Shingling
    simhash_func = &simhash_data_xxhash_shingling;
    simulate_delta_encoding_func = &simulate_delta_encoding_shingling;
  }
  else if (delta_mode == 1) {
    // Balanced delta mode, SimHashes with features using CDC but delta encoding using Shingling
    simhash_func = &simhash_data_xxhash_cdc;
    simulate_delta_encoding_func = &simulate_delta_encoding_shingling;
  }
  else {
    // Best? delta mode, both SimHashes and delta encoding using CDC
    simhash_func = &simhash_data_xxhash_cdc;
    simulate_delta_encoding_func = &simulate_delta_encoding_using_minichunks;
  }

  const bool output_disabled = false;

  const bool use_dupadj = true;
  const bool use_dupadj_backwards = true;
  // If enabled then DupAdj will attempt other chunks with the same hashes as the adjacent ones in an attempt to find other possible similar chunks
  const bool exhaustive_dupadj = false;
  const bool use_generalized_resemblance_detection = false;
  const bool use_feature_extraction = false;
  // Because of data locality, chunks close to the current one might be similar to it, we attempt to find the most similar out of the previous ones in this window,
  // and use it if it's good enough
  const int resemblance_attempt_window = 0;
  const bool use_match_extension = true;
  const bool use_match_extension_backwards = true;

  // if false, the first similar block found by any method will be used and other methods won't run, if true, all methods will run and the most similar block found will be used
  const bool attempt_multiple_delta_methods = true;
  // Only try to delta encode against the candidate chunk with the lesser hamming distance and closest to the current chunk
  // (because of data locality it's likely to be the best)
  const bool only_try_best_delta_match = true;
  const bool only_try_min_dist_delta_matches = false;
  const bool keep_first_delta_match = true;
  const bool is_any_delta_on = use_dupadj || use_dupadj_backwards || use_feature_extraction || use_generalized_resemblance_detection || resemblance_attempt_window > 0;

  const bool verify_delta_coding = false;
  const bool verify_dumps = false;
  const bool verify_addInstruction = false;
  const bool verify_chunk_offsets = false;

  const std::optional<uint64_t> dictionary_size_limit = cli_params.contains("max_dict")
    ? std::optional(static_cast<uint64_t>(std::stoi(cli_params["max_dict"])) * 1024 * 1024)
    : std::nullopt;
  uint64_t dictionary_size_used = 0;
  // If true, a regular LZ77 Data Window will be used, which results in easiest/fastest decompression, otherwise matches will be outputted in terms of the chunks
  // which a more complex and slows decompression process, but it allows for better deduplication ratios in heavily duplicated data
  const bool lz77_mode = true;
  uint64_t first_non_out_of_range_chunk_i = 0;

  circular_vector<utility::ChunkEntry> chunks{};
  
  auto wrapped_input_stream = IStreamWrapper(&input_stream);

  std::optional<uint64_t> similarity_locality_anchor_i{};

  auto verify_file_stream = std::fstream();
  if (input_path != "-") {
    verify_file_stream.open(input_path, std::ios::in | std::ios::binary);
  }
  std::vector<uint8_t> verify_buffer{};
  std::vector<uint8_t> verify_buffer_delta{};

  auto dump_file = std::ofstream(/*file_path + ".cdcz", std::ios::out | std::ios::binary | std::ios::trunc*/);
  set_std_handle_binary_mode(StdHandles::STDOUT_HANDLE);
  reinterpret_cast<std::ostream*>(&dump_file)->rdbuf(std::cout.rdbuf());
  dump_file.put('C');
  dump_file.put('D');
  dump_file.put('C');
  dump_file.put('Z');
  if (dictionary_size_limit.has_value()) {
    dump_file.write(reinterpret_cast<const char*>(&*dictionary_size_limit), 8);
  }
  else {
    dump_file.put(0); dump_file.put(0); dump_file.put(0); dump_file.put(0); dump_file.put(0); dump_file.put(0); dump_file.put(0); dump_file.put(0);
  }
  // Dump header config flags
  dump_file.put(lz77_mode ? 0b1 : 0);

  LZInstructionManager lz_manager{ &chunks, use_match_extension_backwards, use_match_extension, &dump_file };

  static constexpr auto similar_chunk_tuple_cmp = [](const std::tuple<uint64_t, uint64_t>& a, const std::tuple<uint64_t, uint64_t>& b) {
    // 1st member is hamming dist, less hamming dist, better result, we want those tuple first, so we say they compare as lesser
    // 2nd member is chunk idx, more means more recent chunks, larger idx, to be prioritized because they are closer (locality principle)
    if (std::get<0>(a) < std::get<0>(b) || (std::get<0>(a) == std::get<0>(b) && std::get<1>(a) > std::get<1>(b))) return true;
    return false;
  };

  uint64_t chunk_generator_execution_time_ns = 0;
  uint64_t hashing_execution_time_ns = 0;
  uint64_t simhashing_execution_time_ns = 0;

  const auto calc_simhash_if_needed = [&simhash_func, &simhashing_execution_time_ns, &simhash_chunk_size](utility::ChunkEntry& chunk) {
    if (!chunk.chunk_data->minichunks.empty()) return;
    const auto simhashing_start_time = std::chrono::high_resolution_clock::now();
    std::tie(chunk.chunk_data->lsh, chunk.chunk_data->minichunks) = simhash_func(chunk.chunk_data->data.data(), chunk.chunk_data->data.size(), simhash_chunk_size);
    const auto simhashing_end_time = std::chrono::high_resolution_clock::now();
    simhashing_execution_time_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(simhashing_end_time - simhashing_start_time).count();
  };

  // SS-CDC style data segmenting to enable chunk invariant multithreading
  const uint64_t segment_size = 10 * 1024 * 1024;
  const auto segment_batch_size = segment_size * cdc_thread_count;
  std::vector<uint8_t> prev_segment_remaining_data{};
  uint64_t segment_start_offset = 0;
  std::deque<utility::ChunkEntry> process_pending_chunks{};
  std::queue<uint64_t> supercdc_backup_pos{};

  uint64_t current_offset = 0;

  auto find_cdc_cut_candidates_in_thread = [min_size, avg_size, max_size]
  (std::vector<uint8_t>&& segment_data, uint64_t segment_start_offset, bool is_eof_segment, bool is_first_segment) {
    CdcCandidatesResult cdc_candidates_result;
    const CDCZ_CONFIG cfg{ .compute_features = use_feature_extraction};
    cdc_candidates_result = find_cdc_cut_candidates(segment_data, min_size, avg_size, max_size, cfg, is_first_segment);
    return std::tuple(std::move(cdc_candidates_result), std::move(segment_data), segment_start_offset, is_eof_segment, is_first_segment);
  };

  std::deque<std::future<decltype(std::function{find_cdc_cut_candidates_in_thread})::result_type>> cdc_candidates_futures;
  bool is_first_segment = true;

  std::future<std::vector<uint8_t>> load_next_segment_batch_future;
  auto load_next_segment_batch = [&wrapped_input_stream, segment_batch_size]
  (std::array<uint8_t, 31>&& _prev_segment_extend_data) -> std::vector<uint8_t> {
    std::array<uint8_t, 31> prev_segment_extend_data = std::move(_prev_segment_extend_data);
    std::vector<uint8_t> new_segment_batch_data;
    new_segment_batch_data.resize(segment_batch_size + 31);  // +31 bytes (our GEAR window size - 1) at the end so it overlaps with next segment as described in SS-CDC

    // We get the 31byte extension to be at the start of the new segment data
    std::copy_n(prev_segment_extend_data.data(), 31, new_segment_batch_data.data());

    // And attempt to load remaining data for the new segments including next segment extension
    wrapped_input_stream.read(new_segment_batch_data.data() + 31, segment_batch_size);
    new_segment_batch_data.resize(31 + wrapped_input_stream.gcount());

    new_segment_batch_data.shrink_to_fit();
    return new_segment_batch_data;
  };

  auto launch_cdc_threads =
  [&cdc_candidates_futures, &find_cdc_cut_candidates_in_thread, &load_next_segment_batch_future, &load_next_segment_batch,
  cdc_thread_count, segment_size, segment_batch_size, &segment_start_offset, &is_first_segment, &chunk_generator_execution_time_ns]
  (const std::vector<uint8_t>& segment_batch_data) {
    auto chunk_generator_start_time = std::chrono::high_resolution_clock::now();
    auto segment_batch_data_span = std::span(segment_batch_data);
    for (uint64_t i = 0; i < cdc_thread_count; i++) {
      if (segment_batch_data_span.empty()) break;
      auto current_segment_data = std::vector<uint8_t>();
      current_segment_data.resize(std::min<uint64_t>(segment_batch_data_span.size(), segment_size + 31));
      current_segment_data.shrink_to_fit();
      std::copy_n(segment_batch_data_span.data(), current_segment_data.size(), current_segment_data.data());
      bool segments_eof = current_segment_data.size() != segment_size + 31;

      const auto span_advance_size = std::min<uint64_t>(current_segment_data.size(), segment_size);

      cdc_candidates_futures.emplace_back(
        globalTaskPool.addTask(
          [&find_cdc_cut_candidates_in_thread, current_segment_data = std::move(current_segment_data), segments_eof, segment_start_offset, &is_first_segment]() mutable {
            return find_cdc_cut_candidates_in_thread(std::move(current_segment_data), segment_start_offset, segments_eof, is_first_segment);
          }
        )
      );
      segment_start_offset += span_advance_size;
      segment_batch_data_span = std::span(segment_batch_data_span.data() + span_advance_size, segment_batch_data_span.size() - span_advance_size);
      is_first_segment = false;
    }
    auto chunk_generator_end_time = std::chrono::high_resolution_clock::now();
    chunk_generator_execution_time_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(chunk_generator_end_time - chunk_generator_start_time).count();

    // If not at EOF we already start loading the next segment batch data
    if (segment_batch_data.size() == segment_batch_size + 31) {
      std::array<uint8_t, 31> segment_extend_data;
      std::copy_n(segment_batch_data.data() + segment_batch_size, 31, segment_extend_data.data());
      load_next_segment_batch_future = globalTaskPool.addTask(
        [segment_extend_data = std::move(segment_extend_data), &load_next_segment_batch]() mutable {
          return load_next_segment_batch(std::move(segment_extend_data));
        }
      );
    }
  };

  auto get_more_pending_chunks_from_cdc_futures =
  [&cdc_candidates_futures, &process_pending_chunks, &prev_segment_remaining_data, &current_offset, &supercdc_backup_pos, min_size, avg_size, max_size]
  (bool force_wait) {
    if (cdc_candidates_futures.size() > 0) {
      auto candidate_future_status = cdc_candidates_futures.front().wait_for(std::chrono::nanoseconds::zero());
      if (force_wait || candidate_future_status == std::future_status::ready || process_pending_chunks.empty()) {
        auto future = std::move(cdc_candidates_futures.front());
        auto [
          new_cut_points_results,
          segment_data,
          curr_segment_start_offset,
          segments_eof,
          is_first_segment
        ] = future.get();
        cdc_candidates_futures.pop_front();

        select_cut_point_candidates(
          new_cut_points_results.candidates,
          new_cut_points_results.candidatesFeatureResults,
          process_pending_chunks,
          supercdc_backup_pos,
          !process_pending_chunks.empty() ? process_pending_chunks.back().offset + process_pending_chunks.back().chunk_data->data.size() : current_offset,
          curr_segment_start_offset,
          segment_data,
          prev_segment_remaining_data,
          min_size,
          avg_size,
          max_size,
          segments_eof && cdc_candidates_futures.size() == 0,
          use_feature_extraction,
          is_first_segment
        );
      }
    }
  };

  auto total_runtime_start_time = std::chrono::high_resolution_clock::now();

  {
    std::vector<uint8_t> segment_batch_data;
    segment_batch_data.resize(segment_size * cdc_thread_count + 31);  // +31 bytes (our GEAR window size - 1) at the end so it overlaps with next segment as described in SS-CDC
    wrapped_input_stream.read(segment_batch_data.data(), segment_batch_size + 31);
    segment_batch_data.resize(wrapped_input_stream.gcount());
    segment_batch_data.shrink_to_fit();

    launch_cdc_threads(segment_batch_data);
    get_more_pending_chunks_from_cdc_futures(true);
  }

  while (!process_pending_chunks.empty()) {
    if (chunk_i > 0 && dictionary_size_limit.has_value() && dictionary_size_used > *dictionary_size_limit) {
      const auto prev_first_non_out_of_range_chunk_i = first_non_out_of_range_chunk_i;
    
      while (dictionary_size_used > *dictionary_size_limit) {
        auto& previous_chunk = chunks[first_non_out_of_range_chunk_i];

        // Remove the out of range chunks from the indexes so they can't be used for dedup or delta
        auto& hash_list = known_hashes[previous_chunk.chunk_data->hash].instances_idx;
        if (hash_list.size() == 1) {
          known_hashes.erase(previous_chunk.chunk_data->hash);

          // The chunk has no duplicate left so by removing it we are effectively taking it out of the dictionary.
          dictionary_size_used -= previous_chunk.chunk_data->data.size();
        }
        else {
          hash_list.pop_front();
          hash_list.shrink_to_fit();
          // In LZ77 mode we always adjust the used window, because the space is spent for every instance of the chunk within the data window
          if (lz77_mode) dictionary_size_used -= previous_chunk.chunk_data->data.size();
        }

        if (use_generalized_resemblance_detection) {
          const auto base = hamming_base(previous_chunk.chunk_data->lsh);
          if (simhashes_dict[base] <= first_non_out_of_range_chunk_i) {
            simhashes_dict.erase(base);
          }
        }

        if (use_feature_extraction) {
          for (auto& sf : previous_chunk.chunk_data->super_features) {
            // SuperFeature index might have already been deleted, so we need to do proper check here
            auto sf_list_iter = superfeatures_dict.find(sf);
            if (sf_list_iter != superfeatures_dict.end()) {
              auto& sf_list = sf_list_iter->second;
              if (sf_list.front() == first_non_out_of_range_chunk_i) {
                if (sf_list.size() == 1) {
                  superfeatures_dict.erase(sf);
                }
                else {
                  sf_list.pop_front();
                }
              }
            }
          }
        }

        first_non_out_of_range_chunk_i++;
      }

      uint64_t earliest_allowed_offset = chunks[first_non_out_of_range_chunk_i].offset;
      if (!output_disabled) lz_manager.dump(verify_file_stream, verify_dumps, earliest_allowed_offset);

      // Actually remove the chunk's data, we had to wait until after dump to do this because dumping required the data to still be there
      for (uint64_t i = prev_first_non_out_of_range_chunk_i; i < first_non_out_of_range_chunk_i; i++) {
        // Delete its data
        chunks[i].chunk_data = nullptr;
        chunks.pop_front();
      }

      chunks.shrink_to_fit();
    }

    chunks.emplace_back(std::move(process_pending_chunks.front()));
    process_pending_chunks.pop_front();
    auto& chunk = chunks.back();

    std::optional<uint64_t> prev_similarity_locality_anchor_i = similarity_locality_anchor_i;
    similarity_locality_anchor_i = std::nullopt;
    if (chunk_i % 50000 == 0) print_to_console("\n%{}\n", (static_cast<float>(current_offset) / file_size) * 100);

    std::span<uint8_t> data_span = chunk.chunk_data->data;
    total_size += data_span.size();

    if (verify_chunk_offsets) {
      std::vector<uint8_t> verify_buffer_orig_data{};
      std::fstream verify_file{};
      verify_file.open(R"(C:\Users\Administrator\Documents\dedup_proj\Datasets\LNX-IMG\LNX-IMG.tar)", std::ios_base::in | std::ios_base::binary);

      verify_buffer_orig_data.resize(data_span.size());
      // Read original data
      verify_file.seekg(current_offset);
      verify_file.read(reinterpret_cast<char*>(verify_buffer_orig_data.data()), data_span.size());
      // Ensure data matches
      if (std::memcmp(verify_buffer_orig_data.data(), data_span.data(), data_span.size()) != 0) {
        print_to_console("Error while verifying current_offset at offset {}\n", current_offset);
        throw std::runtime_error("Verification error");
      }
    }

    // make hashes
    const auto hashing_start_time = std::chrono::high_resolution_clock::now();
    /*
    // SHA1 boost hash
    boost::uuids::detail::sha1 hasher;
    hasher.process_bytes(chunk.data.data(), chunk.data.size());
    chunk.hash = get_sha1_hash(hasher);
    */
    
    // xxHash
    chunk.chunk_data->hash = XXH3_64bits(data_span.data(), data_span.size());
    const auto hashing_end_time = std::chrono::high_resolution_clock::now();
    hashing_execution_time_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(hashing_end_time - hashing_start_time).count();

    std::optional<uint64_t> duplicate_chunk_i = std::nullopt;
    // If last chunk was a duplicate or similar chunk, it's possible the next chunk from there is a duplicate of the current chunk, and if it is,
    // it's likely it will allow for better reduction (or at least less LZInstructions and match extension attempts)
    if (prev_similarity_locality_anchor_i.has_value()) {
      const auto prev_similarity_anchor_next_chunk_i = *prev_similarity_locality_anchor_i + 1;
      if (prev_similarity_anchor_next_chunk_i >= first_non_out_of_range_chunk_i) {
        auto& prev_similarity_anchor_next_chunk = chunks[prev_similarity_anchor_next_chunk_i];
        if (prev_similarity_anchor_next_chunk.chunk_data->hash == chunk.chunk_data->hash) {
          duplicate_chunk_i = prev_similarity_anchor_next_chunk_i;
        }
      }
    }
    if (!duplicate_chunk_i.has_value() && known_hashes.contains(chunk.chunk_data->hash)) {
      const auto duplicate_chunk_i_candidate = known_hashes[chunk.chunk_data->hash].instances_idx.back();
      duplicate_chunk_i = duplicate_chunk_i_candidate >= first_non_out_of_range_chunk_i ? std::optional{duplicate_chunk_i_candidate} : std::nullopt;
    }
    const bool is_duplicate_chunk = duplicate_chunk_i.has_value();

    std::vector<LZInstruction> new_instructions;
    if (!is_duplicate_chunk) {
      // Attempt resemblance detection and delta compression, first by using generalized resemblance detection
      std::vector<std::tuple<uint64_t, uint64_t>> similar_chunks{};
      if (use_generalized_resemblance_detection) {
        calc_simhash_if_needed(chunk);
        const auto simhash_base = hamming_base(chunk.chunk_data->lsh);
        if (simhashes_dict.contains(simhash_base)) {
          similar_chunks.emplace_back(0, simhashes_dict[simhash_base]);
        }
        // If there is already a match via generalized resemblance detection, we still overwrite the index with this newer chunk.
        // Because of data locality, a more recent chunk on the data stream is more likely to yield good results
        simhashes_dict[simhash_base] = chunk_i;
      }

      // Resemblance detection on the resemblance attempt window, exploiting the principle of data locality, there is a high likelihood data chunks close
      // to the current chunk are fairly similar, just check distance for all chunks within the window
      if (resemblance_attempt_window > 0 && (similar_chunks.empty() || attempt_multiple_delta_methods)) {
        calc_simhash_if_needed(chunk);
        std::vector<std::tuple<uint64_t, uint64_t>> resemblance_attempt_window_similar_chunks{};

        std::optional<uint64_t> best_candidate_dist{};
        uint64_t best_candidate_i = 0;
        for (uint64_t i = 1; i <= resemblance_attempt_window && i <= chunk_i; i++) {
          const auto candidate_i = chunk_i - i;
          if (candidate_i < first_non_out_of_range_chunk_i) break;
          auto& similar_chunk = chunks[candidate_i];

          calc_simhash_if_needed(similar_chunk);
          const auto new_dist = hamming_distance(chunk.chunk_data->lsh, similar_chunk.chunk_data->lsh);
          if (
            new_dist <= max_allowed_dist &&
            // If new_dist is the same as the best so far, privilege chunks with higher index as they are closer to the current chunk
            // and data locality suggests it should be a better match
            (!only_try_best_delta_match || (!best_candidate_dist.has_value() || *best_candidate_dist > new_dist || (*best_candidate_dist == new_dist && candidate_i > best_candidate_i)))
          ) {
            if (only_try_best_delta_match || best_candidate_dist.has_value() && new_dist < *best_candidate_dist)  {
              resemblance_attempt_window_similar_chunks.clear();
            }
            best_candidate_i = candidate_i;
            best_candidate_dist = new_dist;
            resemblance_attempt_window_similar_chunks.emplace_back(new_dist, candidate_i);
          }
        }
        if (!resemblance_attempt_window_similar_chunks.empty()) {
          similar_chunks.insert(similar_chunks.end(), resemblance_attempt_window_similar_chunks.begin(), resemblance_attempt_window_similar_chunks.end());
        }
      }

      // Resemblance detection via feature extraction and superfeature matching
      // If we couldn't sample the chunk's features we completely skip any attempt to match or register superfeatures, as we could not extract them for this chunk
      if (use_feature_extraction && !chunk.chunk_data->feature_sampling_failure) {
        // If we don't have a similar chunk yet, attempt to find one by SuperFeature matching
        if (similar_chunks.empty() || attempt_multiple_delta_methods) {
          calc_simhash_if_needed(chunk);
          std::vector<std::tuple<uint64_t, uint64_t>> feature_extraction_similar_chunks{};

          std::optional<uint64_t> best_candidate_dist{};
          uint64_t best_candidate_i = 0;
          // We will rank potential similar chunks by their amount of matching SuperFeatures (so we select the most similar chunk possible)
          std::unordered_map<uint64_t, uint64_t> chunk_rank{};

          for (const auto& sf : chunk.chunk_data->super_features) {
            if (superfeatures_dict.contains(sf)) {
              for (const auto& candidate_i : superfeatures_dict[sf]) {
                if (candidate_i < first_non_out_of_range_chunk_i) continue;
                chunk_rank[candidate_i] += 1;
                auto& similar_chunk = chunks[candidate_i];
                calc_simhash_if_needed(similar_chunk);
                const auto new_dist = hamming_distance(chunk.chunk_data->lsh, similar_chunk.chunk_data->lsh);
                if (
                  new_dist <= max_allowed_dist &&
                  // If new_dist is the same as the best so far, privilege chunks with higher index as they are closer to the current chunk
                  // and data locality suggests it should be a better match
                  (!only_try_best_delta_match || (!best_candidate_dist.has_value() || *best_candidate_dist > new_dist || (*best_candidate_dist == new_dist && candidate_i > best_candidate_i)))
                ) {
                  if (only_try_best_delta_match || best_candidate_dist.has_value() && new_dist < *best_candidate_dist) {
                    feature_extraction_similar_chunks.clear();
                  }
                  best_candidate_i = candidate_i;
                  best_candidate_dist = new_dist;
                  feature_extraction_similar_chunks.emplace_back(new_dist, candidate_i);
                }
              }
            }
          }
          if (!feature_extraction_similar_chunks.empty()) {
            similar_chunks.insert(similar_chunks.end(), feature_extraction_similar_chunks.begin(), feature_extraction_similar_chunks.end());
          }

          // If all SuperFeatures match, remove the previous chunk from the Index, prevents the index from getting pointlessly large, specially for some pathological cases
          for (const auto& [candidate_i, sf_count] : chunk_rank) {
            if (sf_count == 4) {
              for (const auto& sf : chunk.chunk_data->super_features) {
                auto& sf_list = superfeatures_dict[sf];
                if (sf_list.size() == 1) {
                  superfeatures_dict.erase(sf);
                }
                else {
                  auto sf_iter = std::find(sf_list.begin(), sf_list.end(), candidate_i);
                  if (sf_iter != sf_list.end()) {
                    sf_list.erase(sf_iter);
                  }
                }
              }
            }
          }

          // Register this chunk's SuperFeatures so that matches may be found with subsequent chunks
          for (const auto& sf : chunk.chunk_data->super_features) {
            superfeatures_dict[sf].push_back(chunk_i);
          }
        }
      }

      // If we don't have a similar chunk yet, attempt to find similar block via DARE's DupAdj,
      // checking if the next chunk from the similarity locality anchor is similar to this one
      if (use_dupadj && (similar_chunks.empty() || attempt_multiple_delta_methods)) {
        calc_simhash_if_needed(chunk);
        std::vector<std::tuple<uint64_t, uint64_t>> dupadj_similar_chunks{};

        std::optional<uint64_t> best_candidate_dist{};
        uint64_t best_candidate_i = 0;
        if (prev_similarity_locality_anchor_i.has_value()) {
          circular_vector<uint64_t> similarity_anchor_i_list;
          circular_vector<uint64_t>* chunks_with_similarity_anchor_hash;
          if (exhaustive_dupadj) {
            chunks_with_similarity_anchor_hash = &known_hashes[chunks[*prev_similarity_locality_anchor_i].chunk_data->hash].instances_idx;
          }
          else {
            similarity_anchor_i_list.emplace_back(*prev_similarity_locality_anchor_i);
            chunks_with_similarity_anchor_hash = &similarity_anchor_i_list;
          }
          for (const auto& anchor_instance_chunk_i : *chunks_with_similarity_anchor_hash) {
            const auto similar_candidate_chunk_i = anchor_instance_chunk_i + 1;
            if (similar_candidate_chunk_i == chunk_i) continue;

            auto& similar_candidate = chunks[similar_candidate_chunk_i];
            calc_simhash_if_needed(similar_candidate);
            const auto new_dist = hamming_distance(chunk.chunk_data->lsh, similar_candidate.chunk_data->lsh);
            if (
              new_dist <= max_allowed_dist &&
              // If new_dist is the same as the best so far, privilege chunks with higher index as they are closer to the current chunk
              // and data locality suggests it should be a better match
              (!only_try_best_delta_match || (!best_candidate_dist.has_value() || *best_candidate_dist > new_dist || (*best_candidate_dist == new_dist && similar_candidate_chunk_i > best_candidate_i)))
            ) {
              if (only_try_best_delta_match || best_candidate_dist.has_value() && new_dist < *best_candidate_dist) {
                dupadj_similar_chunks.clear();
              }
              best_candidate_i = similar_candidate_chunk_i;
              best_candidate_dist = new_dist;
              dupadj_similar_chunks.emplace_back(new_dist, similar_candidate_chunk_i);
            }
          }
        }
        if (!dupadj_similar_chunks.empty()) {
          similar_chunks.insert(similar_chunks.end(), dupadj_similar_chunks.begin(), dupadj_similar_chunks.end());
        }
      }

      // If any of the methods detected chunks that appear to be similar, attempt delta encoding
      DeltaEncodingResult best_encoding_result;
      if (!similar_chunks.empty()) {
        std::ranges::sort(similar_chunks, similar_chunk_tuple_cmp);

        uint64_t best_saved_size = 0;
        uint64_t best_similar_chunk_i;
        uint64_t best_similar_chunk_dist = 999;
        for (const auto& [similar_chunk_dist, similar_chunk_i] : similar_chunks) {
          if (only_try_min_dist_delta_matches && best_similar_chunk_dist < similar_chunk_dist) continue;

          const DeltaEncodingResult encoding_result = simulate_delta_encoding_func(*chunk.chunk_data, *chunks[similar_chunk_i].chunk_data, simhash_chunk_size);
          const auto& saved_size = encoding_result.estimated_savings;
          if (saved_size > best_saved_size) {
            best_saved_size = saved_size;
            best_encoding_result = encoding_result;
            best_similar_chunk_i = similar_chunk_i;
            best_similar_chunk_dist = similar_chunk_dist;
            if (keep_first_delta_match) break;
          }

          if (only_try_best_delta_match) break;
        }
        if (best_saved_size > 0 && best_saved_size > min_delta_saving) {
          delta_compressed_chunk_count++;
          similarity_locality_anchor_i = best_similar_chunk_i;
        }
      }

      if (similarity_locality_anchor_i.has_value()) {
        auto& similar_chunk = chunks[*similarity_locality_anchor_i];
        uint64_t chunk_offset_pos = 0;

        if (verify_delta_coding) {
          verify_buffer.resize(chunk.chunk_data->data.size());
          verify_buffer_delta.resize(chunk.chunk_data->data.size());

          verify_file_stream.seekg(chunk.offset);
          verify_file_stream.read(reinterpret_cast<char*>(verify_buffer.data()), chunk.chunk_data->data.size());
        }

        for (auto& instruction : best_encoding_result.instructions) {
          const auto offset = chunk.offset + chunk_offset_pos;
          // On delta instructions the offsets are relative to the reference chunk base offset
          const auto reference_offset = similar_chunk.offset + instruction.offset;
          if (instruction.type == LZInstructionType::INSERT) {
            new_instructions.emplace_back(LZInstructionType::INSERT, offset, instruction.size);
          }
          else {
            new_instructions.emplace_back(LZInstructionType::COPY, reference_offset, instruction.size);
          }
          if (verify_delta_coding) {
            verify_file_stream.seekg(instruction.type == LZInstructionType::INSERT ? offset : reference_offset);
            verify_file_stream.read(reinterpret_cast<char*>(verify_buffer_delta.data()) + chunk_offset_pos, instruction.size);

            if (std::memcmp(verify_buffer_delta.data() + chunk_offset_pos, verify_buffer.data() + chunk_offset_pos, instruction.size) != 0) {
              print_to_console("Delta coding data verification mismatch!\n");
              throw std::runtime_error("Verification error");
            }
          }
          chunk_offset_pos += instruction.size;
        }
        if (chunk_offset_pos != chunk.chunk_data->data.size()) {
          print_to_console("Delta coding size mismatch: chunk_size/delta size {}/{}\n", chunk.chunk_data->data.size(), chunk_offset_pos);
          throw std::runtime_error("Verification error");
        }
      }
    }
    else {
      similarity_locality_anchor_i = duplicate_chunk_i;
      const auto& previous_chunk_instance = chunks[*duplicate_chunk_i];

      // Important to copy LSH in case this duplicate chunk ends up being candidate for Delta encoding via DupAdj or something
      chunk.chunk_data = previous_chunk_instance.chunk_data;

      if (use_generalized_resemblance_detection) {
        // Overwrite index for generalized resemblance detection with this newer chunk.
        // Because of data locality, a more recent chunk on the data stream is more likely to yield good results
        const auto simhash_base = hamming_base(chunk.chunk_data->lsh);
        simhashes_dict[simhash_base] = chunk_i;
      }
      // TODO: DO THE SAME FOR SUPERFEATURES
      // TODO: OR DON'T? ATTEMPTED IT AND IT HURT COMPRESSION A LOT

      new_instructions.emplace_back(LZInstructionType::COPY, previous_chunk_instance.offset, chunk.chunk_data->data.size());
    }
    if (!is_duplicate_chunk) {
      known_hashes[chunk.chunk_data->hash].chunk_id = chunk_id++;
    }
    known_hashes[chunk.chunk_data->hash].instances_idx.emplace_back(chunk_i);

    // if similarity_locality_anchor has a value it means we either deduplicated the chunk or delta compressed it
    if (similarity_locality_anchor_i.has_value()) {
      // set new last_reduced_chunk_i, if there are previous chunks that haven't been deduped/delta'd attempt to do so via backwards DupAdj
      uint64_t prev_last_reduced_chunk_i = last_reduced_chunk_i;
      last_reduced_chunk_i = chunk_i;

      // For us to be able to backtrack the previously last reduced chunk needs to not be the previous one but at least 2 before,
      // else we would be backtracking to that already reduced chunk
      if (chunk_i - prev_last_reduced_chunk_i >= 2 && use_dupadj_backwards) {
        uint64_t backtrack_similarity_anchor_i = *similarity_locality_anchor_i;
        auto backtrack_chunk_i = chunk_i - 1;

        std::vector<DeltaEncodingResult> dupadj_results{};

        for (; backtrack_chunk_i > prev_last_reduced_chunk_i; backtrack_chunk_i--) {
          auto& backtrack_chunk = chunks[backtrack_chunk_i];
          std::vector<std::tuple<uint64_t, uint64_t>> backwards_dupadj_similar_chunks{};
          std::optional<uint64_t> best_candidate_dist{};
          uint64_t best_candidate_i = 0;

          circular_vector<uint64_t> backtrack_anchor_i_list;
          circular_vector<uint64_t>* chunks_with_backtrack_anchor_hash;
          if (exhaustive_dupadj) {
            chunks_with_backtrack_anchor_hash = &known_hashes[chunks[backtrack_similarity_anchor_i].chunk_data->hash].instances_idx;
          }
          else {
            backtrack_anchor_i_list.emplace_back(backtrack_similarity_anchor_i);
            chunks_with_backtrack_anchor_hash = &backtrack_anchor_i_list;
          }
          for (const auto& anchor_instance_chunk_i : *chunks_with_backtrack_anchor_hash) {
            // if anchor_instance_chunk_i is <= first_non_out_of_range_chunk_i then when we look at the previous one we are out of range
            if (anchor_instance_chunk_i <= first_non_out_of_range_chunk_i /*|| anchor_instance_chunk_i == 0*/) continue;
            // if anchor_instance_chunk_i is larger than backtrack_chunk_i then that chunk is in the future, so not really backtracking
            if (anchor_instance_chunk_i > backtrack_chunk_i) break;
            const auto similar_candidate_chunk_i = anchor_instance_chunk_i - 1;
            auto& similar_chunk = chunks[similar_candidate_chunk_i];

            calc_simhash_if_needed(backtrack_chunk);
            calc_simhash_if_needed(similar_chunk);
            const auto new_dist = hamming_distance(backtrack_chunk.chunk_data->lsh, similar_chunk.chunk_data->lsh);
            if (
              new_dist <= max_allowed_dist &&
              // If new_dist is the same as the best so far, privilege chunks with higher index as they are closer to the current chunk
              // and data locality suggests it should be a better match
              (!only_try_best_delta_match || (!best_candidate_dist.has_value() || *best_candidate_dist > new_dist || (*best_candidate_dist == new_dist && similar_candidate_chunk_i > best_candidate_i)))
            ) {
              if (only_try_best_delta_match || best_candidate_dist.has_value() && new_dist < *best_candidate_dist) {
                backwards_dupadj_similar_chunks.clear();
              }
              best_candidate_i = similar_candidate_chunk_i;
              best_candidate_dist = new_dist;
              backwards_dupadj_similar_chunks.emplace_back(new_dist, similar_candidate_chunk_i);
            }
          }

          if (backwards_dupadj_similar_chunks.empty()) break;
          std::ranges::sort(backwards_dupadj_similar_chunks, similar_chunk_tuple_cmp);

          uint64_t best_saved_size = 0;
          std::optional<uint64_t> best_similar_chunk_i;
          uint64_t best_similar_chunk_dist = 999;
          DeltaEncodingResult best_encoding_result;
          for (const auto& [similar_chunk_dist, similar_chunk_i] : backwards_dupadj_similar_chunks) {
            if (only_try_min_dist_delta_matches && best_similar_chunk_dist < similar_chunk_dist) continue;
            if (similar_chunk_i < first_non_out_of_range_chunk_i) continue;
            auto& similar_chunk = chunks[similar_chunk_i];

            const DeltaEncodingResult encoding_result = simulate_delta_encoding_func(*backtrack_chunk.chunk_data, *similar_chunk.chunk_data, simhash_chunk_size);
            const auto& saved_size = encoding_result.estimated_savings;
            if (saved_size > best_saved_size) {
              best_saved_size = saved_size;
              best_encoding_result = encoding_result;
              best_similar_chunk_i = similar_chunk_i;
              best_similar_chunk_dist = similar_chunk_dist;
              if (keep_first_delta_match) break;
            }

            if (only_try_best_delta_match) break;
          }

          if (!best_similar_chunk_i.has_value()) break;
          const auto& similar_chunk = chunks[*best_similar_chunk_i];

          if (best_saved_size > 0 && best_saved_size > min_delta_saving) {
            for (auto& instruction : best_encoding_result.instructions) {
              instruction.offset += instruction.type == LZInstructionType::INSERT ? backtrack_chunk.offset : similar_chunk.offset;
            }

            delta_compressed_chunk_count++;
            backtrack_similarity_anchor_i = best_candidate_i;

            dupadj_results.emplace_back(std::move(best_encoding_result));
          }
          else {
            break;
          }
        }

        if (!dupadj_results.empty()) {
          uint64_t revertSize = 0;
          for (auto& delta_result : dupadj_results) {
            for (auto& instruction : delta_result.instructions) {
              revertSize += instruction.size;
            }
          }

          // Iterating in reverse as the results are ordered from latest to earliest chunk, and they need to be added from earliest to latest
          lz_manager.revertInstructionSize(revertSize);
          auto lz_offset = chunk.offset - revertSize;
          for (auto& delta_result : std::ranges::reverse_view(dupadj_results)) {
            for (auto& instruction : delta_result.instructions) {
              const auto instruction_size = instruction.size;
              const auto prev_estimated_savings = lz_manager.accumulatedSavings();
              const auto instruction_type = instruction.type;
              lz_manager.addInstruction(std::move(instruction), lz_offset, verify_addInstruction, chunks[first_non_out_of_range_chunk_i].offset);
              lz_offset += instruction_size;

              if (instruction_type == LZInstructionType::COPY) {
                // If the instruction was a COPY we need to adapt the data reduction counts
                // The LZManager might have extended or rejected the instruction, so we need to handle that
                // TODO: this is actually not accurate, some match extension is counted as regular dedup/delta reduction
                // but at least the count won't overflow anymore
                const auto instruction_savings = lz_manager.accumulatedSavings() - prev_estimated_savings;
                delta_compressed_approx_size += std::min(instruction_size, instruction_savings);
              }
            }
          }
        }
      }

      auto lz_offset = chunk.offset;
      for (auto& instruction : new_instructions) {
        const auto instruction_size = instruction.size;
        const auto prev_estimated_savings = lz_manager.accumulatedSavings();
        const auto instruction_type = instruction.type;
        lz_manager.addInstruction(std::move(instruction), lz_offset, verify_addInstruction, chunks[first_non_out_of_range_chunk_i].offset);
        lz_offset += instruction_size;

        if (instruction_type == LZInstructionType::COPY) {
          // If the instruction was a COPY we need to adapt the data reduction counts
          // The LZManager might have extended or rejected the instruction, so we need to handle that
          // TODO: this is actually not accurate, some match extension is counted as regular dedup/delta reduction
          // but at least the count won't overflow anymore
          const auto instruction_savings = lz_manager.accumulatedSavings() - prev_estimated_savings;

          if (is_duplicate_chunk) {
            deduped_size += std::min(instruction_size, instruction_savings);
          }
          else {
            delta_compressed_approx_size += std::min(instruction_size, instruction_savings);
          }
        }
      }
    }
    else {
      lz_manager.addInstruction({ .type = LZInstructionType::INSERT, .offset = chunk.offset, .size = chunk.chunk_data->data.size() }, chunk.offset, verify_addInstruction, chunks[first_non_out_of_range_chunk_i].offset);
    }

    dictionary_size_used += !lz77_mode && is_duplicate_chunk ? 0 : chunk.chunk_data->data.size();
    chunk_i++;
    current_offset += chunk.chunk_data->data.size();

    get_more_pending_chunks_from_cdc_futures(process_pending_chunks.empty());
    // If we have a single batch or less worth of cdc futures and the future for loading the next batch is ready we already launch the loading of the new next batch
    if (
      cdc_candidates_futures.size() <= cdc_thread_count &&
      process_pending_chunks.size() <= 1000 &&
      load_next_segment_batch_future.valid() &&
      load_next_segment_batch_future.wait_for(std::chrono::nanoseconds::zero()) == std::future_status::ready
    ) {
      auto segment_extend_data = load_next_segment_batch_future.get();
      //process_pending_chunks.shrink_to_fit();
      //cdc_candidates_futures.shrink_to_fit();
      launch_cdc_threads(segment_extend_data);
    }
    // If we were to run out of pending chunks but the next segment batch is not ready yet, we wait for it and get some pending chunks so we can continue
    if (process_pending_chunks.empty() && load_next_segment_batch_future.valid()) {
      auto segment_extend_data = load_next_segment_batch_future.get();
      //process_pending_chunks.shrink_to_fit();
      //cdc_candidates_futures.shrink_to_fit();
      launch_cdc_threads(segment_extend_data);
      get_more_pending_chunks_from_cdc_futures(true);
    }
  }

  print_to_console("Final offset: {}\n", current_offset);

  auto total_dedup_end_time = std::chrono::high_resolution_clock::now();

  print_to_console("Total dedup time:    {} seconds\n", std::chrono::duration_cast<std::chrono::seconds>(total_dedup_end_time - total_runtime_start_time).count());

  // Dump any remaining data
  if (!output_disabled) lz_manager.dump(verify_file_stream, verify_dumps, std::nullopt, true);

  auto dump_end_time = std::chrono::high_resolution_clock::now();

  // Chunk stats
  print_to_console("Chunk Sizes:    min {} - avg {} - max {}\n", min_size, avg_size, max_size);
  print_to_console("Total chunk count: {}\n", chunks.fullSize());
  print_to_console("In dictionary chunk count: {}\n", chunks.size());
  print_to_console("In memory chunk count: {}\n", chunks.innerVecSize());
  print_to_console("Real AVG chunk size: {}\n", total_size / chunks.fullSize());
  print_to_console("Total unique chunk count: {}\n", chunk_id);
  print_to_console("Total delta compressed chunk count: {}\n", delta_compressed_chunk_count);

  const auto total_accumulated_savings = lz_manager.accumulatedSavings();
  const auto match_omitted_size = lz_manager.omittedSmallMatchSize();
  const auto match_extension_saved_size = lz_manager.accumulatedExtendedBackwardsSavings() + lz_manager.accumulatedExtendedForwardsSavings();
  // Results stats
  const auto total_size_mbs = total_size / (1024.0 * 1024);
  print_to_console("Chunk data total size:    {:.1f} MB\n", total_size_mbs);
  const auto deduped_size_mbs = deduped_size / (1024.0 * 1024);
  print_to_console("Chunk data deduped size:    {:.1f} MB\n", deduped_size_mbs);
  const auto deltaed_size_mbs = delta_compressed_approx_size / (1024.0 * 1024);
  print_to_console("Chunk data delta compressed size:    {:.1f} MB\n", deltaed_size_mbs);
  const auto extension_size_mbs = match_extension_saved_size / (1024.0 * 1024);
  print_to_console("Match extended size:    {:.1f} MB\n", extension_size_mbs);
  const auto omitted_size_mbs = match_omitted_size / (1024.0 * 1024);
  print_to_console("Match omitted size (matches too small):    {:.1f} MB\n", omitted_size_mbs);
  const auto total_accummulated_savings_mbs = total_accumulated_savings / (1024.0 * 1024);
  print_to_console("Total estimated reduced size:    {:.1f} MB\n", total_accummulated_savings_mbs);
  print_to_console("Final size:    {:.1f} MB\n", total_size_mbs - total_accummulated_savings_mbs);

  print_to_console("\n");

  // Throughput stats
  const auto chunking_mb_per_nanosecond = total_size_mbs / chunk_generator_execution_time_ns;
  print_to_console("Chunking Throughput:    {:.1f} MB/s\n", chunking_mb_per_nanosecond * std::pow(10, 9));
  const auto hashing_mb_per_nanosecond = total_size_mbs / hashing_execution_time_ns;
  print_to_console("Hashing Throughput:    {:.1f} MB/s\n", hashing_mb_per_nanosecond * std::pow(10, 9));
  const auto simhashing_mb_per_nanosecond = total_size_mbs / simhashing_execution_time_ns;
  print_to_console("SimHashing Throughput:    {:.1f} MB/s\n", simhashing_mb_per_nanosecond * std::pow(10, 9));
  const auto total_elapsed_nanoseconds = std::chrono::duration_cast<std::chrono::nanoseconds>(total_dedup_end_time - total_runtime_start_time).count();
  const auto total_mb_per_nanosecond = total_size_mbs / total_elapsed_nanoseconds;
  print_to_console("Total Throughput:    {:.1f} MB/s\n", total_mb_per_nanosecond * std::pow(10, 9));
  print_to_console("Total LZ instructions:    {}\n", lz_manager.instructionCount());
  print_to_console("Total dedup time:    {} seconds\n", std::chrono::duration_cast<std::chrono::seconds>(total_dedup_end_time - total_runtime_start_time).count());
  print_to_console("Dump time:    {} seconds\n", std::chrono::duration_cast<std::chrono::seconds>(dump_end_time - total_dedup_end_time).count());
  print_to_console("Total runtime:    {} seconds\n", std::chrono::duration_cast<std::chrono::seconds>(dump_end_time - total_runtime_start_time).count());

  print_to_console("Processing finished, press enter to quit.\n");
  //get_char_with_echo();
  exit(0);  // Dirty, dirty, dirty, but should be fine as long all threads have finished, for exiting quickly until I refactor the codebase a little
  return 0;
}
