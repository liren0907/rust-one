#include "math.h"
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>

// A hardcoded string with Pi to ~160 decimal places.
const char *PI_STR_LITERAL =
    "3."
    "14159265358979323846264338327950288419716939937510582097494459230781640628"
    "62089986280348253421170679821480865132823066470938446095505822317253594081"
    "28";

extern "C" {
char *calculate_pi(int digits) {
  if (digits < 0)
    digits = 0;

  const int max_available_digits = strlen(PI_STR_LITERAL) - 2;
  digits = std::min(digits, max_available_digits);

  const int total_len = 2 + digits + 1; // "3." + digits + null

  char *c_str = (char *)malloc(total_len);
  if (c_str == NULL) {
    std::cerr << "Memory allocation failed!" << std::endl;
    return NULL;
  }

  strncpy(c_str, PI_STR_LITERAL, total_len - 1);
  c_str[total_len - 1] = '\0';

  return c_str;
}

void free_pi_string(char *pi_str) {
  if (pi_str != NULL) {
    free(pi_str);
  }
}
}
