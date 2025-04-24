#include "io_utils.hpp"

#include <algorithm>
#include <format>

#include "contrib/xxHash/xxhash.h"

#include "console_utils.hpp"

IStreamWrapper::IStreamWrapper(std::istream* _istream) : istream(_istream) {}
void IStreamWrapper::read(void* buf, std::streamsize count) {
  istream->read(static_cast<char*>(buf), count);
}
std::streamsize IStreamWrapper::gcount() {
  return istream->gcount();
}

IStreamMem::IStreamMem(const uint8_t* _buf, uint32_t _len) : membuf(_buf), len(_len) {}
void IStreamMem::read(void* buf, std::streamsize count) {
  if (pos >= len) {
    _gcount = 0;
    return;
  }

  auto to_read = std::min<std::streamsize>(len - pos, count);
  std::copy_n(membuf + pos, to_read, static_cast<uint8_t*>(buf));
  _gcount = to_read;
  pos += to_read;
}
std::streamsize IStreamMem::gcount() {
  return _gcount;
}

void dump_vector_to_ostream_with_guard(std::vector<uint8_t>&& _output_buffer, uint64_t buffer_used_len, std::ostream* ostream) {
  std::vector<uint8_t> output_buffer = std::move(_output_buffer);
  ostream->write(reinterpret_cast<const char*>(output_buffer.data()), buffer_used_len);
}

void FakeIOStream::write(char* buffer, uint64_t size) {
  if (pos + size > data.size()) {
    data.resize(pos + size);
  }
  std::copy_n(buffer, size, data.data() + pos);
  pos = pos + size;
}

void FakeIOStream::read(char* buffer, uint64_t size) {
  const auto actual_read_size = std::min(data.size() - pos, size);
  std::copy_n(data.data() + pos, actual_read_size, buffer);
  pos = pos + actual_read_size;
}

void FakeIOStream::print_hash() {
  const auto hash = XXH64(data.data(), data.size(), 0);
  auto hash_str = std::format("{:x}", hash);
  std::ranges::transform(hash_str, hash_str.begin(), ::toupper);
  print_to_console("XXH64 hash: " + hash_str + "\n");
}
