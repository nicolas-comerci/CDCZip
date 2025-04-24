#ifndef CONSOLE_UTILS_H
#define CONSOLE_UTILS_H
#include <string>

typedef enum {
  STDIN_HANDLE = 0,
  STDOUT_HANDLE = 1,
  STDERR_HANDLE = 2,
} StdHandles;
void set_std_handle_binary_mode(StdHandles handle);

void print_to_console(const std::string& format);

void print_to_console(const char* fmt, ...);

int get_char_with_echo();

#endif
