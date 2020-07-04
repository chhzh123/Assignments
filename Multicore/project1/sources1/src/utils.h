/*
 * In general, you don't need to modify this file to finish hw1.
 */
#ifndef _INCL_UTILS
#define _INCL_UTILS

#include "time.h"

// divup calculates n / m and would round it up if the remainder is non-zero.
int divup(int n, int m) {
    return n % m == 0 ? n/m : n/m + 1;
}

// getTime gets the local time in nanoseconds.
long getTime() {
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return (long)ts.tv_sec * 1000000000L + ts.tv_nsec;
}

// W_CHK macro is used to check if a file write is successfully or not.
#define W_CHK(call)                                         \
{                                                           \
    const int written = call;                               \
    if (written == 0) {                                     \
        printf("error written\n");                          \
        exit(1);                                            \
    }                                                       \
}                                                           \

#endif