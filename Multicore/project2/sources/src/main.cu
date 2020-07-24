/*
 * You should guarantee that your final submitted source code files are
 * compatible with an unmodified version of this file.
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "utils.h"
#include "generator.h"
#include "core.h"

// func is a function pointer which is compatible with "cudaCallback"s.
void (*func)(int, int, int, float*, float*, int**);

// calcDistance is for calculating the precise Euclidean distance.
float calcDistance(int k, int mInd, int nInd, float *searchPoints,
                   float *referencePoints) {
    float squareSum = 0;
    float diff;
    for (int i = 0; i < k; i++) {
        diff = searchPoints[k*mInd+i] - referencePoints[k*nInd+i];
        squareSum += (diff * diff);
    }
    return sqrt(squareSum);
}

// samplesConfig decides the scale of test samples.
int samplesConfig[] = {
    3,  1,      2,
    3,  2,      8,

    3,  1,      1024,
    3,  1,      65536,
    16, 1,      65536,

    3,  1024,   1024,
    3,  1024,   65536,
    16, 1024,   65536,
};
int numSamples = 0;

// seed is used to control the sample generation.
int seed = 1000;

// Timestamp variables for measuring running time.
long st, et;

// baselineResults store the results of baseline for checking your algo
// variants.
int **baselineResults = NULL;

// testCnt tracks the number of tests.
int testCnt = 0;

void test() {
    testCnt++;

    // Generate samples and invoke the provided callback in one-by-one.
    setRandSeed(seed);
    for (int i = 0; i < numSamples; i++) {
        int k = samplesConfig[3*i];
        int m = samplesConfig[3*i+1];
        int n = samplesConfig[3*i+2];
        float *searchPoints, *referencePoints;
        getSample(k, m, n, &searchPoints, &referencePoints);
        
        // Invoke the callback.
        int *results;
        st = getTime();
        (*func)(k, m, n, searchPoints, referencePoints, &results);
        et = getTime();
        printf("Callback%d, %2d, %4d, %5d, %10.3fms\n", testCnt, k, m, n,
            (et - st) / 1e6);

        // De-allocate the memory spaces.
        free(searchPoints);
        free(referencePoints);

        if (baselineResults[i] == NULL) {
            baselineResults[i] = results;
        } else {
            // Check correctness.
            int errors = 0;
            for (int j = 0; j < m; j++) {
                if (baselineResults[i][j] == results[j]) {
                    continue;
                } else {
                    float d1 = calcDistance(k, j, baselineResults[i][j],
                        searchPoints, referencePoints);
                    float d2 = calcDistance(k, j, results[j],
                        searchPoints, referencePoints);
                    if (d1 - d2 < -1e-3 || d1 - d2 > 1e-3) {
                        errors++;
                    }
                }
            }
            printf("errors/total w.r.t. baseline: %d/%d\n\n", errors, m);
            free(results);
        }
    }
}

int main() {
    // Get the number of samples.
    numSamples = sizeof(samplesConfig) / (3 * sizeof(*samplesConfig));

    // Initialize the baseline results list.
    baselineResults = (int **)malloc(sizeof(int *) * numSamples);
    for (int i = 0; i < numSamples; i++) {
        baselineResults[i] = NULL;
    }

#ifdef CALLBACK1
    printf("\non running CALLBACK1...\n");
    func = &CALLBACK1;
    test();
#endif

#ifdef CALLBACK2
    printf("\non running CALLBACK2...\n");
    func = &CALLBACK2;
    test();
#endif

#ifdef CALLBACK3
    printf("\non running CALLBACK3...\n");
    func = &CALLBACK3;
    test();
#endif

#ifdef CALLBACK4
    printf("\non running CALLBACK4...\n");
    func = &CALLBACK4;
    test();
#endif

#ifdef CALLBACK5
    printf("\non running CALLBACK5...\n");
    func = &CALLBACK5;
    test();
#endif

#ifdef CALLBACK6
    printf("\non running CALLBACK6...\n");
    func = &CALLBACK6;
    test();
#endif

#ifdef CALLBACK7
    printf("\non running CALLBACK7...\n");
    func = &CALLBACK7;
    test();
#endif

#ifdef CALLBACK8
    printf("\non running CALLBACK8...\n");
    func = &CALLBACK8;
    test();
#endif

#ifdef CALLBACK9
    printf("\non running CALLBACK9...\n");
    func = &CALLBACK9;
    test();
#endif

#ifdef CALLBACK10
    printf("\non running CALLBACK10...\n");
    func = &CALLBACK10;
    test();
#endif

    if (baselineResults != NULL) {
        for (int i = 0; i < numSamples; i++) {
            free(baselineResults[i]);
        }
        free(baselineResults);
    }
}