#include "core.h"

extern void cudaCallback(int k, int m, int n, float *searchPoints,
                         float *referencePoints, int **results) {

    int *tmp = (int*)malloc(sizeof(int)*m);
    int minIndex;
    float minSquareSum, diff, squareSum;

    // Iterate over all search points
    for (int mInd = 0; mInd < m; mInd++) {
        minSquareSum = -1;
        // Iterate over all reference points
        for (int nInd = 0; nInd < n; nInd++) {
            squareSum = 0;
            for (int kInd = 0; kInd < k; kInd++) {
                diff = searchPoints[k*mInd+kInd] - referencePoints[k*nInd+kInd];
                squareSum += (diff * diff);
            }
            if (minSquareSum < 0 || squareSum < minSquareSum) {
                minSquareSum = squareSum;
                minIndex = nInd;
            }
        }
        tmp[mInd] = minIndex;
    }

    *results = tmp;
    // Note that you don't have to free searchPoints, referencePoints, and
    // *results by yourself
}