/*
 * In general, you don't need to modify this file to finish hw1.
 */
#include <stdio.h>
#include <stdlib.h>
#include "config.h"
#include "utils.h"
#include "samples.h"
#include "core.h"

int main() {
    // Timestamp variables
    long st, et;

    // Open the data file
    FILE *stream = fopen(dataPath, "rb");
    if (stream == NULL) {
        printf("failed to open the data file\n");
        return -1;
    }

    // Open a stream to write out results in text
    FILE *outStream = fopen(outputPath, "w");
    if (outStream == NULL) {
        printf("failed to open the output file\n");
        return -1;
    }

    // Read in and process the samples one-by-one
    int width, height, size;
    float *sample, *result;
    while (getNextSample(stream, &width, &height, &sample) != 0) {
        size = width * height;
        printf("%d * %d\n", width, height);

        if (printSample) {
            // Print out a small portion of the sample
            printf("sample:\n");
            for (int j = height - 5; j < height; j++) {
                for (int i = width - 5; i < width; i++) {
                    printf("%8.5f ", sample[j*width+i]);
                }
                printf("\n");
            }
        }

        // Invoke the callback
        st = getTime();
        cudaCallback(width, height, sample, &result);
        et = getTime();

        if (printResult) {
            // Print out a small portion of the result
            printf("result:\n");
            for (int j = height - 5; j < height; j++) {
                for (int i = width - 5; i < width; i++) {
                    printf("%8.5f ", result[j*width+i]);
                }
                printf("\n");
            }
        }

        printf("cudaCallback: %.3f ms\n\n", (et - st) / 1e6);

        // Write the result to the output stream
        char buffer[128];
        sprintf(buffer, "%d,", width);
        W_CHK(fputs(buffer, outStream));
        sprintf(buffer, "%d,", height);
        W_CHK(fputs(buffer, outStream));
        for (int i = 0; i < size; i++) {
            sprintf(buffer, "%.5f,", result[i]);
            W_CHK(fputs(buffer, outStream));
        }
        W_CHK(fputs("\n", outStream));

        // De-allocate the sample and the result
        free(sample);
        sample = NULL;
        free(result);
        result = NULL;
    }

    // Close the output stream
    fclose(outStream);
    return 0;
}