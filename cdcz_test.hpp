#ifndef CDCZ_TEST_H
#define CDCZ_TEST_H

#include <cstdint>
#include <string>
#include <unordered_map>

void cdcz_test_mode(const std::string& file_path, uint64_t file_size, std::unordered_map<std::string, std::string>& cli_params);

#endif