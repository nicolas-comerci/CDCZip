#ifndef CIRCULAR_BUFFER_H
#define CIRCULAR_BUFFER_H

class CircularBuffer {
public:
  explicit CircularBuffer(uint64_t size) {
    data.resize(size);
  }

  std::tuple<std::span<const uint8_t>, std::span<const uint8_t>> getSpansForRead(uint64_t size = 0, bool fromEnd = false) const {
    if (size == 0) size = m_usedSize;
    if (size > m_usedSize) throw std::runtime_error("Tried to read more data than available on CircularBuffer");
    std::tuple<std::span<const uint8_t>, std::span<const uint8_t>> ret_val;
    std::span<const uint8_t>& pre_wrapping_data = std::get<0>(ret_val);
    std::span<const uint8_t>& post_wrapping_data = std::get<1>(ret_val);

    // If the circular buffer is not full and wrapped around yet then m_dataStartOffset=0 so we get all the data in the pre_wrapping_data.
    // Otherwise m_usedSize=data.size() and we get all the data after m_dataStartOffset in the pre_wrapping_data, and the rest in the post_wrapping_data.
    pre_wrapping_data = std::span(data.data() + m_dataStartOffset, m_usedSize - m_dataStartOffset);
    post_wrapping_data = std::span(data.data(), m_dataStartOffset);

    // If the whole circular buffer data wasn't requested we adjust the spans so that only the requested amount (from the start or end)
    // is returned
    if (size != m_usedSize) {
      if (fromEnd) {
        if (post_wrapping_data.size() >= size) {
          const auto excess_data_size = post_wrapping_data.size() - size;
          post_wrapping_data = std::span(post_wrapping_data.data() + excess_data_size, post_wrapping_data.size() - excess_data_size);
          // We are getting data from the end and the post warpping data is already more than enough, so pre_wrapping_data get cleared
          pre_wrapping_data = std::span(data.data(), 0);
        }
        size -= post_wrapping_data.size();

        if (size > 0) {
#ifndef NDEBUG
          if (pre_wrapping_data.size() < size) throw std::runtime_error("getSpansForRead fromEnd somehow we don't have enough data!");
#endif
          const auto excess_data_size = pre_wrapping_data.size() - size;
          pre_wrapping_data = std::span(pre_wrapping_data.data() + excess_data_size, pre_wrapping_data.size() - excess_data_size);
        }
      }
      else {
        if (pre_wrapping_data.size() >= size) {
          const auto excess_data_size = pre_wrapping_data.size() - size;
          pre_wrapping_data = std::span(pre_wrapping_data.data(), pre_wrapping_data.size() - excess_data_size);
          post_wrapping_data = std::span(data.data(), 0);
        }
        size -= pre_wrapping_data.size();

        if (size > 0) {
#ifndef NDEBUG
          if (post_wrapping_data.size() < size) throw std::runtime_error("getSpansForRead somehow we don't have enough data!");
#endif
          const auto excess_data_size = post_wrapping_data.size() - size;
          post_wrapping_data = std::span(post_wrapping_data.data(), post_wrapping_data.size() - excess_data_size);
        }
      }
    }
    return ret_val;
  }

  std::tuple<std::span<uint8_t>, std::span<uint8_t>> getSpansForWrite(const uint64_t size) {
    if (size > data.size()) throw std::runtime_error("Tried to write more data than available size on CircularBuffer");
    uint64_t remaining_size = size;
    std::tuple<std::span<uint8_t>, std::span<uint8_t>> ret_val;
    std::span<uint8_t>& pre_wrapping_data = std::get<0>(ret_val);
    std::span<uint8_t>& post_wrapping_data = std::get<1>(ret_val);

    // First we set the span for the data before we need to wrap around the circular buffer
    const auto unused_size = data.size() - m_usedSize;
    if (unused_size > 0) {  // CircularBuffer not even full, can just add data at the end
      const uint64_t to_add_size = std::min(size, unused_size);
      pre_wrapping_data = std::span(data.data() + m_usedSize, to_add_size);
      m_usedSize += to_add_size;
      remaining_size -= to_add_size;
    }
    else {  // CircularBuffer is already full, get data after the m_dataStartOffset
      const uint64_t non_wrapped_data_end = std::min(m_dataStartOffset + size, data.size());
      const uint64_t non_wrapped_size = non_wrapped_data_end - m_dataStartOffset;
      pre_wrapping_data = std::span(data.data() + m_dataStartOffset, non_wrapped_size);
      remaining_size -= non_wrapped_size;

      m_dataStartOffset += non_wrapped_size;
      if (m_dataStartOffset == data.size()) m_dataStartOffset = 0;
    }
    if (remaining_size == 0) return ret_val;  // If we got the size requested with just the pre_wrapping_data we quit here

#ifndef NDEBUG
    if (m_dataStartOffset != 0) throw std::runtime_error("Wrapping around but m_dataStartOffset is not 0!");
#endif
    // If we are here, there is some data size that we need to take, after wrapping to the beginning of the circular buffer
    post_wrapping_data = std::span(data.data(), remaining_size);
    m_dataStartOffset = remaining_size;
#ifndef NDEBUG
    if (pre_wrapping_data.size() + post_wrapping_data.size() != size) throw std::runtime_error("getSpansForWrite got incorrect data size!");
#endif
    return ret_val;
  }

private:
  std::vector<uint8_t> data;
  uint64_t m_dataStartOffset = 0;
  uint64_t m_usedSize = 0;
};

#endif
