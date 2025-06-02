#include "console_utils.hpp"

#include <cassert>
#include <cstdarg>
#include <fcntl.h>

#ifndef __unix
#include <io.h>
#include <conio.h>
#endif

#ifndef __unix
void set_std_handle_binary_mode(StdHandles handle) { std::ignore = _setmode(handle, O_BINARY); }
#else
void set_std_handle_binary_mode(StdHandles handle) {}
#endif

#ifndef _WIN32
#include <unistd.h>
int ttyfd = -1;
#endif

void print_to_console(const std::string& fmt) {
#ifdef _WIN32
  for (char chr : fmt) {
    _putch(chr);
  }
#else
  if (ttyfd < 0)
    ttyfd = open("/dev/tty", O_RDWR);
  write(ttyfd, fmt.c_str(), fmt.length());
#endif
}

void printf_to_console(const char* fmt, ...) {
  va_list args;
  va_start(args, fmt);
  va_list args_copy;
  va_copy(args_copy, args);
  int length = std::vsnprintf(nullptr, 0, fmt, args);
  va_end(args);
  assert(length >= 0);

  char* buf = new char[length + 1];
  (void)std::vsnprintf(buf, length + 1, fmt, args_copy);
  va_end(args_copy);

  std::string str(buf);
  delete[] buf;
  print_to_console(str);
}

int get_char_with_echo() {
#ifndef __unix
  return _getche();
#else
  return fgetc(stdin);
#endif
}
