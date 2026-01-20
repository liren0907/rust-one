#ifndef MATH_H
#define MATH_H

#ifdef __cplusplus
extern "C" {
#endif

// Calculates Pi to 'digits' decimal places and returns it as a C string.
// The caller is responsible for freeing the returned string using free_pi_string.
char* calculate_pi(int digits);

// Frees the memory allocated by calculate_pi.
void free_pi_string(char* pi_str);

#ifdef __cplusplus
}
#endif

#endif // MATH_H
