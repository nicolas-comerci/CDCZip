#ifndef IO_UTILS_H
#define IO_UTILS_H

#include <algorithm>
#include <istream>
#include <thread>
#include <vector>

#include "contrib/stream.h"

class IStreamLike {
public:
  virtual ~IStreamLike() = default;
  virtual void read(void* buf, std::streamsize count) = 0;
  virtual std::streamsize gcount() = 0;
};

class IStreamWrapper : public IStreamLike {
  std::istream* istream;
public:
  IStreamWrapper(std::istream* _istream);
  ~IStreamWrapper() override = default;

  void read(void* buf, std::streamsize count) override;
  std::streamsize gcount() override;
};

class IStreamMem : public IStreamLike {
  const uint8_t* membuf;
  uint32_t len;
  uint32_t pos = 0;
  std::streamsize _gcount = 0;
public:
  IStreamMem(const uint8_t* _buf, uint32_t _len);
  ~IStreamMem() override = default;

  void read(void* buf, std::streamsize count) override;
  std::streamsize gcount() override;
};

class WrappedIStreamInputStream : public InputStream {
private:
  std::istream* istream;

public:
  WrappedIStreamInputStream(std::istream* _istream) : istream(_istream) {}

  bool eof() const override { return istream->eof(); }
  size_t read(unsigned char* buffer, const size_t size) override {
    istream->read(reinterpret_cast<char*>(buffer), size);
    return istream->gcount();
  }
};

void dump_vector_to_ostream_with_guard(std::vector<uint8_t>&& _output_buffer, uint64_t buffer_used_len, std::ostream* ostream);

class WrappedOStreamOutputStream : public OutputStream {
private:
  std::ostream* ostream;
  uint64_t buffer_size;
  std::vector<uint8_t> output_buffer;
  uint64_t buffer_used_len = 0;
  std::thread dump_thread;

  void write_with_thread() {
    if (dump_thread.joinable()) dump_thread.join();
    dump_thread = std::thread(dump_vector_to_ostream_with_guard, std::move(output_buffer), buffer_used_len, ostream);
    buffer_used_len = 0;
    output_buffer.resize(buffer_size);
    output_buffer.shrink_to_fit();
  }

public:
  explicit WrappedOStreamOutputStream(std::ostream* _ostream, uint64_t _buffer_size = 200ull * 1024 * 1024) : ostream(_ostream), buffer_size(_buffer_size) {
    output_buffer.resize(buffer_size);
    output_buffer.shrink_to_fit();
  }

  ~WrappedOStreamOutputStream() override {
    flush();
  }

  void flush() {
    if (buffer_used_len > 0) {
      write_with_thread();
      dump_thread.join();
    }
    ostream->flush();
  }

  size_t write(const unsigned char* buffer, const size_t size) override {
    if (size > output_buffer.size()) {
      if (buffer_used_len > 0) {
        ostream->write(reinterpret_cast<const char*>(output_buffer.data()), buffer_used_len);
        buffer_used_len = 0;
      }
      ostream->write(reinterpret_cast<const char*>(buffer), size);
      return size;
    }
    if (size + buffer_used_len > output_buffer.size()) {
      write_with_thread();
    }
    std::copy_n(buffer, size, output_buffer.data() + buffer_used_len);
    buffer_used_len += size;
    return size;
  }
};

class FakeIOStream {
public:
  FakeIOStream() { data.reserve(50ULL * 1024 * 1024 * 1024); }

  void write(char* buffer, uint64_t size);
  void flush() {}

  void read(char* buffer, uint64_t size);

  uint64_t tellp() { return pos; }
  uint64_t tellg() { return pos; }
  void seekg(uint64_t offset) { pos = offset; }
  void seekp(uint64_t offset) { pos = offset; }

  bool eof() { return false; }
  bool bad() { return false; }
  bool fail() { return false; }

  void print_hash();

private:
  uint64_t pos = 0;
  std::vector<char> data;
};

#endif
