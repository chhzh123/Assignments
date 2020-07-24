/*
 * You should guarantee that your final submitted source code files are
 * compatible with an unmodified version of this file.
 */
#ifndef __INCL_GENERATOR
#define __INCL_GENERATOR

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

// Use double to ensure that RAND_MAX can be precisely represented as a floating
// point number.
const double DOUBLE_RAND_MAX = double(RAND_MAX);

// getRandNum generates a number in [0, 1] randomly.
float getRandNum() {
    return rand() / DOUBLE_RAND_MAX;
}

// isGeneratorReady indicates whether the seed is set.
bool isGeneratorReady = false;

// setRandSeed should be called once and only once.
void setRandSeed(int seed) {
    srand(seed);
    isGeneratorReady = true;
}

// getSample generates a sample with m search points and n reference points in
// k-dimensional space.
void getSample(int k, int m, int n, float **searchPoints,
               float **referencePoints) {
    assert(isGeneratorReady);
    float *tmp;

    tmp = (float*)malloc(sizeof(float) * k * m);
    assert(tmp != NULL);
    for (int i = 0; i < k * m; i++) {
        tmp[i] = getRandNum();
    }
    *searchPoints = tmp;

    tmp = (float*)malloc(sizeof(float) * k * n);
    assert(tmp != NULL);
    for (int i = 0; i < k * n; i++) {
        tmp[i] = getRandNum();
    }
    *referencePoints = tmp;
}

#endif