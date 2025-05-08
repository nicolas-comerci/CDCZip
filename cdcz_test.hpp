#ifndef CDCZ_TEST_H
#define CDCZ_TEST_H

#include <cstdint>
#include <string>

void cdcz_test_mode(const std::string& file_path, uint64_t file_size, const std::string& trace_out_path, const std::string& trace_in_path);

#endif