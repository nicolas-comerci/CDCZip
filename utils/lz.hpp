#ifndef LZ_UITLS_H
#define LZ_UITLS_H

#include <cstring>
#include <fstream>

#include "contrib/bitstream.h"

#include "chunks.hpp"
#include "console_utils.hpp"
#include "io_utils.hpp"
#include "prefix_suffix_count.hpp"

enum LZInstructionType : uint8_t {
  COPY,
  INSERT,
  DELTA
};

struct LZInstruction {
  LZInstructionType type;
  uint64_t offset;  // For COPY: previous offset; For INSERT: offset on the original stream; For Delta: previous offset of original data
  uint64_t size;  // How much data to be copied or inserted, or size of the delta original data

  auto operator<=>(const LZInstruction&) const = default;
};

class LZInstructionManager {
  circular_vector<utility::ChunkEntry>* chunks;
  circular_vector<LZInstruction> instructions;

  std::ostream* ostream;
  WrappedOStreamOutputStream output_stream;
  BitOutputStream bit_output_stream;

  const bool use_match_extension_backwards;
  const bool use_match_extension;

  uint64_t accumulated_savings = 0;
  uint64_t accumulated_extended_backwards_savings = 0;
  uint64_t accumulated_extended_forwards_savings = 0;
  uint64_t omitted_small_match_size = 0;
  uint64_t outputted_up_to_offset = 0;
  uint64_t outputted_lz_instructions = 0;

  uint64_t check_backwards_extend_size(const LZInstruction& instruction, uint64_t current_offset, uint64_t earliest_allowed_offset) const {
    uint64_t extended_backwards_size = 0;
    const bool can_backwards_extend_instruction = instruction.offset > earliest_allowed_offset;
    if (!use_match_extension_backwards || instruction.type != LZInstructionType::COPY || instruction.size == 0 || !can_backwards_extend_instruction)
      return extended_backwards_size;

    auto prevInstruction_iter = instructions.cend();
    prevInstruction_iter = prevInstruction_iter - 1;
    const LZInstruction* prevInstruction = &*prevInstruction_iter;

    auto [instruction_chunk_i, instruction_chunk_pos] = get_chunk_i_and_pos_for_offset(*chunks, instruction.offset - 1);
    utility::ChunkEntry* instruction_chunk = &(*chunks)[instruction_chunk_i];
#ifndef NDEBUG
    if (instruction_chunk->offset + instruction_chunk_pos != instruction.offset - 1) {
      print_to_console("BACKWARD MATCH EXTENSION NEW INSTRUCTION OFFSET MISMATCH\n");
      throw std::runtime_error("Verification error");
    }
#endif

    uint64_t extended_instruction_offset = current_offset;
    uint64_t prevInstruction_eaten_size = 0;
    const bool can_backwards_extend_prevInstruction = extended_instruction_offset > earliest_allowed_offset;
    while (can_backwards_extend_prevInstruction) {
      // TODO: figure out why this happens, most likely some match extension is not properly cleaning up INSERTs that are shrunk into nothingness
      if (prevInstruction->size == 0) {
        if (prevInstruction_iter == instructions.cbegin()) {
          break;
        }
        --prevInstruction_iter;
        prevInstruction_eaten_size = 0;
        prevInstruction = &*prevInstruction_iter;
        continue;
      }

      auto [prevInstruction_chunk_i, prevInstruction_chunk_pos] = get_chunk_i_and_pos_for_offset(*chunks, extended_instruction_offset - 1);
      utility::ChunkEntry* prevInstruction_chunk = &(*chunks)[prevInstruction_chunk_i];
#ifndef NDEBUG
      if (prevInstruction_chunk->offset + prevInstruction_chunk_pos != extended_instruction_offset - 1) {
        print_to_console("BACKWARD MATCH EXTENSION PREVIOUS INSTRUCTION OFFSET MISMATCH\n");
        throw std::runtime_error("Verification error");
      }
#endif

      bool stop_matching = false;
      while (true) {
        const auto bytes_remaining_for_prevInstruction_to_earliest_allowed_offset = extended_instruction_offset - earliest_allowed_offset;
        // bytes_remaining_for_prevInstruction_backtrack is including the current byte, which is why we +1 to bytes_remaining_for_prevInstruction_to_earliest_allowed_offset,
        // otherwise this would be 0 when we are on the actual earliest_allowed_offset
        const auto bytes_remaining_for_prevInstruction_backtrack = std::min(
          bytes_remaining_for_prevInstruction_to_earliest_allowed_offset + 1,
          std::min(prevInstruction->size - prevInstruction_eaten_size, prevInstruction_chunk_pos + 1)
        );

        const auto prevInstruction_backtrack_data = std::span(
          prevInstruction_chunk->chunk_data->data.data() + prevInstruction_chunk_pos - (bytes_remaining_for_prevInstruction_backtrack - 1),
          bytes_remaining_for_prevInstruction_backtrack
        );
        const auto instruction_backtrack_data = std::span(instruction_chunk->chunk_data->data.data(), instruction_chunk_pos + 1);

        uint64_t matched_amt = find_identical_suffix_byte_count(prevInstruction_backtrack_data, instruction_backtrack_data);
        if (matched_amt == 0) {
          stop_matching = true;
          break;
        }

        extended_instruction_offset -= matched_amt;
        prevInstruction_eaten_size += matched_amt;
        extended_backwards_size += matched_amt;

        // Can't keep extending backwards, any previous data is disallowed (presumably to accomodate max matching distance)
        if (extended_instruction_offset < earliest_allowed_offset) {
          stop_matching = true;
          break;
        }
        if (extended_instruction_offset < earliest_allowed_offset) {
          stop_matching = true;
          break;
        }

        if (instruction_chunk_pos >= matched_amt) {
          instruction_chunk_pos -= matched_amt;
        }
        else if (instruction_chunk_i == 0 || instruction_chunk->offset == earliest_allowed_offset) {
          stop_matching = true;
          break;
        }
        else {
          instruction_chunk_i--;
          instruction_chunk = &(*chunks)[instruction_chunk_i];
          instruction_chunk_pos = instruction_chunk->chunk_data->data.size() - 1;
        }

        if (prevInstruction->size == prevInstruction_eaten_size) {
          if (prevInstruction_iter == instructions.cbegin()) {
            stop_matching = true;
            break;
          }
          --prevInstruction_iter;
          prevInstruction_eaten_size = 0;
          prevInstruction = &*prevInstruction_iter;
          break;
        }

        if (prevInstruction_chunk_pos >= matched_amt) {
          prevInstruction_chunk_pos -= matched_amt;
        }
        else if (prevInstruction_chunk_i == 0 || prevInstruction_chunk->offset == earliest_allowed_offset) {
          stop_matching = true;
          break;
        }
        else {
          prevInstruction_chunk_i--;
          prevInstruction_chunk = &(*chunks)[prevInstruction_chunk_i];
          prevInstruction_chunk_pos = prevInstruction_chunk->chunk_data->data.size() - 1;
        }
      }
      if (stop_matching) break;
    }

    return extended_backwards_size;
  }

  uint64_t check_forwards_extend_size(const LZInstruction& instruction, uint64_t earliest_allowed_offset) const {
    uint64_t extended_forwards_savings = 0;
    uint64_t extended_forwards_size = 0;
    const LZInstruction& prevInstruction = instructions.back();
    if (!use_match_extension || prevInstruction.type != LZInstructionType::COPY)
      return extended_forwards_size;

    uint64_t prevInstruction_offset = prevInstruction.offset + prevInstruction.size;

    const bool can_forward_extend_prevInstruction = prevInstruction_offset >= earliest_allowed_offset;
    if (can_forward_extend_prevInstruction) {
      auto [prevInstruction_chunk_i, prevInstruction_chunk_pos] = get_chunk_i_and_pos_for_offset(*chunks, prevInstruction_offset);
      utility::ChunkEntry* prevInstruction_chunk = &(*chunks)[prevInstruction_chunk_i];

      auto [instruction_chunk_i, instruction_chunk_pos] = get_chunk_i_and_pos_for_offset(*chunks, instruction.offset);
      utility::ChunkEntry* instruction_chunk = &(*chunks)[instruction_chunk_i];
#ifndef NDEBUG
      if (instruction_chunk->offset + instruction_chunk_pos != instruction.offset || instruction_chunk_pos >= instruction_chunk->chunk_data->data.size()) {
        throw std::runtime_error("Verification error");
      }
#endif

      while (extended_forwards_size < instruction.size) {
        const auto prevInstruction_extend_data = std::span(
          prevInstruction_chunk->chunk_data->data.data() + prevInstruction_chunk_pos,
          prevInstruction_chunk->chunk_data->data.size() - prevInstruction_chunk_pos
        );
        const auto instruction_extend_data = std::span(
          instruction_chunk->chunk_data->data.data() + instruction_chunk_pos,
          std::min(instruction_chunk->chunk_data->data.size() - instruction_chunk_pos, instruction.size - extended_forwards_size)
        );

        uint64_t matched_amt = find_identical_prefix_byte_count(prevInstruction_extend_data, instruction_extend_data);
        if (matched_amt == 0) break;

        extended_forwards_size += matched_amt;
        //prevInstruction.size += matched_amt;
        //instruction.offset += matched_amt;
        //instruction.size -= matched_amt;
        if (instruction.type == LZInstructionType::INSERT) extended_forwards_savings += matched_amt;

        prevInstruction_chunk_pos += matched_amt;
        if (prevInstruction_chunk_pos == prevInstruction_chunk->chunk_data->data.size()) {
          prevInstruction_chunk_i++;
          prevInstruction_chunk = &(*chunks)[prevInstruction_chunk_i];
          prevInstruction_chunk_pos = 0;
        }

        if (instruction.size == extended_forwards_size) {
          break;
        }

        instruction_chunk_pos += matched_amt;
        if (instruction_chunk_pos == instruction_chunk->chunk_data->data.size()) {
          instruction_chunk_i++;
          instruction_chunk = &(*chunks)[instruction_chunk_i];
          instruction_chunk_pos = 0;
        }
      }
    }
    return extended_forwards_size;
  }

  void verify_copy_instruction_data(char* buffer_ptr, char* verify_buffer_ptr, const LZInstruction& instruction, uint64_t prev_outputted_up_to_offset) {
    auto [copy_instruction_chunk_i, copy_instruction_chunk_pos] = get_chunk_i_and_pos_for_offset(*chunks, instruction.offset);
    auto [curr_instruction_chunk_i, curr_instruction_chunk_pos] = get_chunk_i_and_pos_for_offset(*chunks, outputted_up_to_offset);
    auto* copy_instruction_chunk = &(*chunks)[copy_instruction_chunk_i];
    auto* curr_instruction_chunk = &(*chunks)[curr_instruction_chunk_i];

    uint64_t remaining_size = instruction.size;
    while (remaining_size > 0) {
      if (copy_instruction_chunk->chunk_data->data.size() == copy_instruction_chunk_pos) {
        copy_instruction_chunk_i++;
        copy_instruction_chunk_pos = 0;
        copy_instruction_chunk = &(*chunks)[copy_instruction_chunk_i];
      }
      if (curr_instruction_chunk->chunk_data->data.size() == curr_instruction_chunk_pos) {
        curr_instruction_chunk_i++;
        curr_instruction_chunk_pos = 0;
        curr_instruction_chunk = &(*chunks)[curr_instruction_chunk_i];
      }
      auto cmp_size = std::min<uint64_t>(
        copy_instruction_chunk->chunk_data->data.size() - copy_instruction_chunk_pos,
        curr_instruction_chunk->chunk_data->data.size() - curr_instruction_chunk_pos
      );
      cmp_size = std::min(cmp_size, remaining_size);
      const auto copy_chunk_data = copy_instruction_chunk->chunk_data->data.data() + copy_instruction_chunk_pos;
      const auto curr_chunk_data = curr_instruction_chunk->chunk_data->data.data() + curr_instruction_chunk_pos;

      if (std::memcmp(curr_chunk_data, verify_buffer_ptr, cmp_size) != 0) {
        print_to_console("ERROR ON CURR CHUNK DATA!\n");
        print_to_console("With prev offset " + std::to_string(prev_outputted_up_to_offset) + "\n");
        throw std::runtime_error("Verification error");
      }
      verify_buffer_ptr += cmp_size;
      if (std::memcmp(copy_chunk_data, buffer_ptr, cmp_size) != 0) {
        print_to_console("ERROR ON COPY CHUNK DATA!\n");
        print_to_console("With prev offset " + std::to_string(prev_outputted_up_to_offset) + "\n");
        throw std::runtime_error("Verification error");
      }
      buffer_ptr += cmp_size;

      if (std::memcmp(copy_chunk_data, curr_chunk_data, cmp_size) != 0) {
        print_to_console("Error while verifying outputted match with chunk data at offset " + std::to_string(outputted_up_to_offset) + "\n");
        print_to_console("With prev offset " + std::to_string(prev_outputted_up_to_offset) + "\n");
        throw std::runtime_error("Verification error");
      }
      remaining_size -= cmp_size;
      copy_instruction_chunk_pos += cmp_size;
      curr_instruction_chunk_pos += cmp_size;
    }
  }

public:
  explicit LZInstructionManager(circular_vector<utility::ChunkEntry>* _chunks, bool _use_match_extension_backwards, bool _use_match_extension, std::ostream* _ostream)
    : chunks(_chunks), ostream(_ostream), output_stream(_ostream), bit_output_stream(output_stream),
    use_match_extension_backwards(_use_match_extension_backwards), use_match_extension(_use_match_extension) {
  }

  ~LZInstructionManager() {
    bit_output_stream.flush();
    ostream->flush();
  }

  uint64_t instructionCount() const {
    return outputted_lz_instructions;
  }

  uint64_t accumulatedSavings() const {
    return accumulated_savings;
  }
  uint64_t accumulatedExtendedBackwardsSavings() const {
    return accumulated_extended_backwards_savings;
  }
  uint64_t accumulatedExtendedForwardsSavings() const {
    return accumulated_extended_forwards_savings;
  }

  uint64_t omittedSmallMatchSize() const {
    return omitted_small_match_size;
  }

  void addInstruction(LZInstruction&& instruction, uint64_t current_offset, bool verify, std::optional<uint64_t> _earliest_allowed_offset = std::nullopt) {
    uint64_t earliest_allowed_offset = _earliest_allowed_offset.has_value() ? *_earliest_allowed_offset : 0;
#ifndef NDEBUG
    if (instruction.type == LZInstructionType::INSERT && instruction.offset != current_offset) {
      print_to_console("INSERT LZInstruction added is not at current offset!");
      throw std::runtime_error("Verification error");
    }
#endif
    if (instructions.size() == 0) {
      instructions.emplace_back(std::move(instruction));
      return;
    }

    // If same type of instruction, and it starts from the offset at the end of the previous instruction we just extend that one
    LZInstruction* prevInstruction = &instructions.back();
    if (
      prevInstruction->type == instruction.type &&
      prevInstruction->offset + prevInstruction->size == instruction.offset
      ) {
      prevInstruction->size += instruction.size;
      if (prevInstruction->type == LZInstructionType::COPY) {
        accumulated_savings += instruction.size;
      }
    }
    else {
      std::vector<uint8_t> verify_buffer_orig_data{};
      std::vector<uint8_t> verify_buffer_instruction_data{};
      uint64_t verify_end_offset = current_offset + instruction.size;
      if (verify) {
        std::fstream verify_file{};
        verify_file.open(R"(C:\Users\Administrator\Documents\dedup_proj\Datasets\LNX-IMG\LNX-IMG.tar)", std::ios_base::in | std::ios_base::binary);

        verify_buffer_orig_data.resize(prevInstruction->size);
        verify_buffer_instruction_data.resize(prevInstruction->size);
        // Read prevInstruction original data
        verify_file.seekg(current_offset - prevInstruction->size);
        verify_file.read(reinterpret_cast<char*>(verify_buffer_orig_data.data()), prevInstruction->size);
        // Read data according to prevInstruction
        verify_file.seekg(prevInstruction->offset);
        verify_file.read(reinterpret_cast<char*>(verify_buffer_instruction_data.data()), prevInstruction->size);
        // Ensure data matches
        if (std::memcmp(verify_buffer_orig_data.data(), verify_buffer_instruction_data.data(), prevInstruction->size) != 0) {
          print_to_console("Error while verifying addInstruction prevInstruction at offset " + std::to_string(current_offset) + "\n");
          throw std::runtime_error("Verification error");
        }

        verify_buffer_orig_data.resize(instruction.size);
        verify_buffer_instruction_data.resize(instruction.size);
        // Read instruction original data
        verify_file.seekg(current_offset);
        verify_file.read(reinterpret_cast<char*>(verify_buffer_orig_data.data()), instruction.size);
        // Read data according to instruction
        verify_file.seekg(instruction.offset);
        verify_file.read(reinterpret_cast<char*>(verify_buffer_instruction_data.data()), instruction.size);
        // Ensure data matches
        if (std::memcmp(verify_buffer_orig_data.data(), verify_buffer_instruction_data.data(), instruction.size) != 0) {
          print_to_console("Error while verifying addInstruction instruction at offset " + std::to_string(current_offset) + "\n");
          throw std::runtime_error("Verification error");
        }
      }

      const uint64_t original_instruction_size = instruction.size;
      uint64_t extended_forwards_savings = 0;
      uint64_t extended_backwards_savings = 0;

      uint64_t forwards_extend_possible_size = check_forwards_extend_size(instruction, earliest_allowed_offset);
      const bool is_forwards_extend_eats_prevInstruction = forwards_extend_possible_size == instruction.size;
      if (is_forwards_extend_eats_prevInstruction) {
        prevInstruction->size += forwards_extend_possible_size;
        instruction.offset += forwards_extend_possible_size;
        instruction.size -= forwards_extend_possible_size;
        if (instruction.type == LZInstructionType::INSERT) {
          extended_forwards_savings += forwards_extend_possible_size;
        }
      }
      else {
        uint64_t backwards_extend_possible_size = check_backwards_extend_size(instruction, current_offset, earliest_allowed_offset);
        const bool is_backwards_extend_eats_prevInstruction = backwards_extend_possible_size >= prevInstruction->size;

        if (is_backwards_extend_eats_prevInstruction || prevInstruction->type == LZInstructionType::INSERT) {
          while (backwards_extend_possible_size > 0) {
            prevInstruction = &instructions.back();

            uint64_t prevInstruction_reduced_size = 0;
            const bool is_prevInstruction_INSERT = prevInstruction->type == LZInstructionType::INSERT;
            if (backwards_extend_possible_size >= prevInstruction->size) {
              prevInstruction_reduced_size = prevInstruction->size;
              instructions.pop_back();
            }
            else if (!is_prevInstruction_INSERT) {
              // If prevInstruction is a COPY, but we can't eat it completely, we skip it.
              // It's mostly pointless to reduce it, and that prior instruction is likely for data that is
              // more distant, which might make it more compressible.
              break;
            }
            else {
              prevInstruction_reduced_size = backwards_extend_possible_size;
              prevInstruction->size -= backwards_extend_possible_size;
            }

            backwards_extend_possible_size -= prevInstruction_reduced_size;
            if (is_prevInstruction_INSERT) {
              extended_backwards_savings += prevInstruction_reduced_size;
            }

            instruction.offset -= prevInstruction_reduced_size;
            instruction.size += prevInstruction_reduced_size;
          }
          prevInstruction = &instructions.back();
        }
        else if (forwards_extend_possible_size > 0) {
          prevInstruction->size += forwards_extend_possible_size;
          instruction.offset += forwards_extend_possible_size;
          instruction.size -= forwards_extend_possible_size;
          if (instruction.type == LZInstructionType::INSERT) {
            extended_forwards_savings += forwards_extend_possible_size;
          }
        }
      }

      if (verify) {
        std::fstream verify_file{};
        verify_file.open(R"(C:\Users\Administrator\Documents\dedup_proj\Datasets\LNX-IMG\LNX-IMG.tar)", std::ios_base::in | std::ios_base::binary);
        const auto data_count = prevInstruction->size + instruction.size;

        verify_buffer_orig_data.resize(data_count);
        verify_buffer_instruction_data.resize(data_count);

        uint64_t orig_data_start = verify_end_offset - data_count;

        // Read original data
        verify_file.seekg(orig_data_start);
        verify_file.read(reinterpret_cast<char*>(verify_buffer_orig_data.data()), data_count);
        // Read data according to the instructions
        verify_file.seekg(prevInstruction->offset);
        verify_file.read(reinterpret_cast<char*>(verify_buffer_instruction_data.data()), prevInstruction->size);
        verify_file.seekg(instruction.offset);
        verify_file.read(reinterpret_cast<char*>(verify_buffer_instruction_data.data()) + prevInstruction->size, instruction.size);
        // Ensure data matches
        if (std::memcmp(verify_buffer_orig_data.data(), verify_buffer_instruction_data.data(), data_count) != 0) {
          print_to_console("Error while verifying addInstruction at offset " + std::to_string(current_offset) + "\n");
          throw std::runtime_error("Verification error");
        }
      }

      // If instruction is COPY then any forward extending is actually just retreading ground from what was extended backwards and
      // at most from the instruction itself, so it's all already counted there
      if (instruction.type == LZInstructionType::COPY) accumulated_savings += extended_backwards_savings + original_instruction_size;
      else accumulated_savings += extended_forwards_savings;
      accumulated_extended_backwards_savings += extended_backwards_savings;
      accumulated_extended_forwards_savings += extended_forwards_savings;

      // If the whole instruction is consumed by extending the previous COPY, then just quit, there is no instruction to add anymore
      if (instruction.size == 0) {
        return;
      }
      // If we are adding an INSERT, then the previous COPY was already extended backwards and/or forwards as much as it could.
      // If it's still so small that the overhead of outputting the extra instruction is larger than the deduplication we would get
      // we just extend this insert so that we save that overhead
      if (instruction.type == LZInstructionType::INSERT && prevInstruction->type == LZInstructionType::COPY && prevInstruction->size < 128) {
        accumulated_savings -= prevInstruction->size;
        omitted_small_match_size += prevInstruction->size;
        prevInstruction->type = LZInstructionType::INSERT;
        prevInstruction->offset = instruction.offset - prevInstruction->size;
        prevInstruction->size = instruction.size + prevInstruction->size;
        return;
      }
      instructions.emplace_back(std::move(instruction));
    }
  }

  void revertInstructionSize(uint64_t size) {
    LZInstruction* prevInstruction = &instructions.back();
    while (prevInstruction->size <= size) {
      if (prevInstruction->type == LZInstructionType::COPY) accumulated_savings -= size;
      size -= prevInstruction->size;
      instructions.pop_back();
      prevInstruction = &instructions.back();
    }
    prevInstruction->size -= size;
    if (prevInstruction->type == LZInstructionType::COPY) accumulated_savings -= size;
  }

  void dump(std::istream& istream, bool verify_copies, std::optional<uint64_t> up_to_offset = std::nullopt, bool flush = false) {
    std::vector<char> buffer;

    auto prev_outputted_up_to_offset = outputted_up_to_offset;
    while (instructions.size() != 0) {
      if (up_to_offset.has_value() && outputted_up_to_offset > *up_to_offset) break;
      auto instruction = std::move(instructions.front());
      instructions.pop_front();

      bit_output_stream.put(instruction.type, 8);
      bit_output_stream.putVLI(instruction.size);
      if (instruction.type == LZInstructionType::INSERT) {
        auto [instruction_chunk_i, instruction_chunk_pos] = get_chunk_i_and_pos_for_offset(*chunks, instruction.offset);
        buffer.resize(instruction.size);
        uint64_t written = 0;
        while (written < instruction.size) {
          auto& instruction_chunk = (*chunks)[instruction_chunk_i];
          uint64_t to_read_from_chunk = std::min(instruction.size - written, instruction_chunk.chunk_data->data.size() - instruction_chunk_pos);
          std::copy_n(instruction_chunk.chunk_data->data.data() + instruction_chunk_pos, to_read_from_chunk, buffer.data() + written);
          written += to_read_from_chunk;

          instruction_chunk_i++;
          instruction_chunk_pos = 0;
        }
        bit_output_stream.putBytes(reinterpret_cast<const uint8_t*>(buffer.data()), instruction.size);
      }
      else {
        bit_output_stream.putVLI(outputted_up_to_offset - instruction.offset);
        if (verify_copies) {
          istream.seekg(instruction.offset);
          buffer.resize(instruction.size);
          istream.read(buffer.data(), instruction.size);

          thread_local std::vector<char> verify_buffer;
          istream.seekg(outputted_up_to_offset);
          verify_buffer.resize(instruction.size);
          istream.read(verify_buffer.data(), instruction.size);

          if (std::memcmp(verify_buffer.data(), buffer.data(), instruction.size) != 0) {
            print_to_console("Error while verifying outputted match at offset " + std::to_string(outputted_up_to_offset) + "\n");
            print_to_console("With prev offset " + std::to_string(prev_outputted_up_to_offset) + "\n");
            throw std::runtime_error("Verification error");
          }

          //verify_copy_instruction_data(buffer.data(), verify_buffer.data(), instruction, prev_outputted_up_to_offset);
        }
      }
      prev_outputted_up_to_offset = outputted_up_to_offset;
      outputted_up_to_offset += instruction.size;
      outputted_lz_instructions++;
    }

    instructions.shrink_to_fit();
    if (flush) {
      bit_output_stream.flush();
      output_stream.flush();
    }
  }
};

#endif
