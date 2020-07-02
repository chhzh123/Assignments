#ifndef _INCL_CONFIG
#define _INCL_CONFIG

// Whether to print a portion of each sample (disable: 0)
int printSample = 1;
// Whether to print a portion of each result (disable: 0)
int printResult = 1;

/*
 * Don't change the settings below if you are going to use the rexec utility
 */

// The relative path to the data file
char *dataPath = "./data.bin";
// The relative path to the output results
char *outputPath = "./results.csv";

#endif