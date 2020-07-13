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

int main() {
    // Timestamp variables for measuring running time
    long st, et;

    // Open a stream for writing out results in text
    FILE *outStream = fopen("results.csv", "w");
    if (outStream == NULL) {
        printf("failed to open the output file\n");
        return -1;
    }

    // Generate samples and invoke the provided callback in a one-by-one fashion
    int seed = 1000;
    setRandSeed(seed);

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
    int numSamples = sizeof(samplesConfig) / (3 * sizeof(*samplesConfig));

    for (int i = 0; i < numSamples; i++) {
        int k = samplesConfig[3*i];
        int m = samplesConfig[3*i+1];
        int n = samplesConfig[3*i+2];
        float *searchPoints, *referencePoints;
        getSample(k, m, n, &searchPoints, &referencePoints);
        
        // Modify and print out some small samples for easy checking by hand
        if (i < 2) {
            printf("Small sample %d:\n---\n", i);

            printf("Search points:\n");
            for (int mInd = 0; mInd < m; mInd++) {
                printf("- [ ");
                for (int kInd = 0; kInd < k; kInd++) {
                    if (i < 2) {
                        // For hand checking only (protect in case you are
                        // intending to inspect more samples)
                        searchPoints[3*mInd+kInd] = 
                            roundf(searchPoints[3*mInd+kInd] * 10) / 10;
                    }
                    printf("%.2f ", searchPoints[3*mInd+kInd]);
                }
                printf("]\n");
            }

            printf("Reference points:\n");
            for (int nInd = 0; nInd < n; nInd++) {
                printf("- %d: [ ", nInd);
                for (int kInd = 0; kInd < k; kInd++) {
                    if (i < 2) {
                        // For hand checking only (protect in case you are
                        // intending to inspect more samples)
                        referencePoints[3*nInd+kInd] = 
                            roundf(referencePoints[3*nInd+kInd] * 10) / 10;
                    }
                    printf("%.2f ", referencePoints[3*nInd+kInd]);
                }
                printf("]\n");
            }
        } else {
            printf("Sample %d:\n---\n", i);
        }

        // Invoke the callback
        int *results;
        st = getTime();
        cudaCallback(k, m, n, searchPoints, referencePoints, &results);
        et = getTime();

        if (i < 2) {
            printf("Results:\n");
            for (int mInd = 0; mInd < m; mInd++) {
                printf("- %d (%.3f)\n", results[mInd], calcDistance(k, mInd,
                    results[mInd], searchPoints, referencePoints));
            }
        }

        printf("cudaCallback: %.3f ms\n\n", (et - st) / 1e6);

        // Write the results out to the output stream
        char buffer[128];
        for (int mInd = 0; mInd < m; mInd++) {
            sprintf(buffer, "%d,", results[mInd]);
            W_CHK(fputs(buffer, outStream));
        }
        W_CHK(fputs("\n", outStream));
        for (int mInd = 0; mInd < m; mInd++) {
            sprintf(buffer, "%.3f,", calcDistance(k, mInd,
                    results[mInd], searchPoints, referencePoints));
            W_CHK(fputs(buffer, outStream));
        }
        W_CHK(fputs("\n", outStream));

        printf("\n");
        // De-allocate the memory spaces
        free(searchPoints);
        free(referencePoints);
        free(results);
    }

    // Close the output stream
    fclose(outStream);
    return 0;
}
