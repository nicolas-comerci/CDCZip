#ifndef CHUNK_UTILS_H
#define CHUNK_UTILS_H

#include <array>
#include <bitset>
#include <cstdint>
#include <memory>
#include <unordered_map>

#include "circular_vector.hpp"

namespace utility {
  class ChunkData {
  public:
    std::bitset<64> hash;
    std::bitset<64> lsh;

    std::vector<uint8_t> data = std::vector<uint8_t>(0);
    // Smaller chunks inside this chunk, used for delta compression
    std::vector<uint32_t> minichunks = std::vector<uint32_t>(0);

    std::array<uint32_t, 4> super_features{};
    bool feature_sampling_failure = true;

    explicit ChunkData() = default;
  };
  class ChunkEntry {
  public:
    uint64_t offset = 0;
    std::shared_ptr<ChunkData> chunk_data;
    explicit ChunkEntry() = default;
    explicit ChunkEntry(uint64_t _offset) : offset(_offset), chunk_data(std::make_shared<ChunkData>()) {}
  };
}

struct ChunkIndexEntry {
  uint64_t chunk_id;
  circular_vector<uint64_t> instances_idx;
};

using ChunkIndex = std::unordered_map<std::bitset<64>, ChunkIndexEntry>;

std::tuple<uint64_t, uint64_t> get_chunk_i_and_pos_for_offset(circular_vector<utility::ChunkEntry>& chunks, uint64_t offset);

#endif
