#ifndef CONSOLE_UTILS_H
#define CONSOLE_UTILS_H
#include <string>
#include <format>

typedef enum {
  STDIN_HANDLE = 0,
  STDOUT_HANDLE = 1,
  STDERR_HANDLE = 2,
} StdHandles;
void set_std_handle_binary_mode(StdHandles handle);

void print_to_console(const std::string& fmt);

template<class... Args>
void print_to_console(const std::string& fmt, Args&&... args) {
  return print_to_console(std::vformat(fmt, std::make_format_args(args...)));
}

void printf_to_console(const char* fmt, ...);

int get_char_with_echo();

#endif
