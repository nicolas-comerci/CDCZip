#ifndef CDCZIP_DECOMPRESS_H
#define CDCZIP_DECOMPRESS_H

#include "contrib/bitstream.h"

#include "utils/circular_buffer.hpp"
#include "utils/console_utils.hpp"
#include "utils/lz.hpp"

template<typename T>
void decompress(T& output_iostream, BitInputStream& bit_input_stream, uint64_t dict_size) {
  std::vector<char> read_buffer;
  uint64_t current_offset = 0;
  auto lzDict = CircularBuffer(dict_size);
  bool isSeekMode = dict_size == 0;  // If dict_size is zero then we are in SEEK mode, and we just seek back in the output file itself to get the data

  auto instruction = bit_input_stream.get(8);
  while (!bit_input_stream.eof()) {
    const auto eof = output_iostream.eof();
    const auto fail = output_iostream.fail();
    const auto bad = output_iostream.bad();
    if (eof || fail || bad) {
      print_to_console("Something wrong bad during decompression\n");
      exit(1);
    }

    uint64_t size = bit_input_stream.getVLI();

    if (instruction == LZInstructionType::INSERT) {
      read_buffer.resize(size);
      const auto read_amt = bit_input_stream.readBytes(reinterpret_cast<unsigned char*>(read_buffer.data()), size);
      if (read_amt != size) throw std::runtime_error("Unable to read expected INSERT data");
      output_iostream.write(read_buffer.data(), size);

      if (!isSeekMode) {  // Add INSERTed data to the lzDict
        auto [pre_wrap_data, post_wrap_data] = lzDict.getSpansForWrite(size);
        std::copy_n(read_buffer.data(), pre_wrap_data.size(), pre_wrap_data.data());
        if (!post_wrap_data.empty()) {
          std::copy_n(read_buffer.data() + pre_wrap_data.size(), post_wrap_data.size(), post_wrap_data.data());
        }
      }
    }
    else {  // LZInstructionType::COPY
      output_iostream.flush();
      uint64_t relative_offset = bit_input_stream.getVLI();
      uint64_t offset = current_offset - relative_offset;

      // A COPY instruction might be overlapping with itself, which means we need to keep copying data already copied within the
      // same COPY instruction (usually because of repeating patterns in data)
      // In this case we have to only read again the non overlapping data
      const uint64_t prev_write_pos = output_iostream.tellp();
      const auto actual_read_size = std::min(size, prev_write_pos - offset);
      read_buffer.resize(actual_read_size);

      if (isSeekMode) {
        output_iostream.seekg(offset);
        output_iostream.read(read_buffer.data(), actual_read_size);
        output_iostream.seekp(prev_write_pos);
      }
      else {
        auto [pre_wrap_data, post_wrap_data] = lzDict.getSpansForRead(relative_offset, true);
#ifndef NDEBUG
        output_iostream.seekg(offset);
        output_iostream.read(read_buffer.data(), actual_read_size);
        output_iostream.seekp(prev_write_pos);

        const auto pre_wrap_data_size = std::min<uint64_t>(pre_wrap_data.size(), actual_read_size);
        auto cmp_result = std::memcmp(read_buffer.data(), pre_wrap_data.data(), pre_wrap_data_size);
        if (cmp_result != 0) throw std::runtime_error("Data gotten from the dict pre_wrap_data is different than the one from the file");
        if (pre_wrap_data_size < actual_read_size) {
          cmp_result = std::memcmp(read_buffer.data() + pre_wrap_data_size, post_wrap_data.data(), actual_read_size - pre_wrap_data_size);
          if (cmp_result != 0) throw std::runtime_error("Data gotten from the dict post_wrap_data is different than the one from the file");
        }
#endif
        uint64_t data_size_from_pre_wrap_data = std::min(actual_read_size, pre_wrap_data.size());
        if (data_size_from_pre_wrap_data > 0) std::copy_n(pre_wrap_data.data(), data_size_from_pre_wrap_data, read_buffer.data());
        if (uint64_t data_size_from_post_wrap_data = actual_read_size - data_size_from_pre_wrap_data) {
          std::copy_n(post_wrap_data.data(), data_size_from_post_wrap_data, read_buffer.data() + data_size_from_pre_wrap_data);
        }
      }

      {
        uint64_t remaining_size = size;
        while (remaining_size > 0) {
          const auto write_size = std::min(remaining_size, actual_read_size);
          output_iostream.write(read_buffer.data(), write_size);
          remaining_size -= write_size;
        }
      }

      if (!isSeekMode) {  // COPY the data at the end of the lzDict
        auto [pre_wrap_write_data, post_wrap_write_data] = lzDict.getSpansForWrite(size);
        uint64_t remaining_size = size;
        while (remaining_size > 0) {
          const auto write_size = std::min(remaining_size, actual_read_size);
          auto read_buffer_span = std::span(read_buffer.data(), write_size);
          if (!pre_wrap_write_data.empty()) {
            const auto pre_wrap_data_write_size = std::min(pre_wrap_write_data.size(), write_size);
            std::copy_n(read_buffer_span.data(), pre_wrap_data_write_size, pre_wrap_write_data.data());
            pre_wrap_write_data = std::span(pre_wrap_write_data.data() + pre_wrap_data_write_size, pre_wrap_write_data.size() - pre_wrap_data_write_size);
            read_buffer_span = std::span(read_buffer_span.data() + pre_wrap_data_write_size, read_buffer_span.size() - pre_wrap_data_write_size);
          }
          if (!read_buffer_span.empty()) {  // if we still have data in the read buffer span then we must dump it into the post_wrap_write_data span
            std::copy_n(read_buffer_span.data(), read_buffer_span.size(), post_wrap_write_data.data());
            post_wrap_write_data = std::span(post_wrap_write_data.data() + read_buffer_span.size(), post_wrap_write_data.size() - read_buffer_span.size());
          }
          remaining_size -= write_size;
        }
      }
    }

    current_offset += size;
    instruction = bit_input_stream.get(8);
  }
}

#endif