/*
 * This code is part of the hw1 of multicore programming in SYSU
 * Copyright (c) 2020 Hongzheng Chen
 * Email: chenhzh37@mail2.sysu.edu.cn
 * 
 * This file is the header of OpenMP implementation
 *   that calculates the central entropy of each point in a matrix
 */

#ifndef _INCL_CORE
#define _INCL_CORE

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

/*!
 * Core execution part of OpenMP
 *   that calculates the central entropy of each point.
 * \param width The width of the input matrix.
 * \param height The height of the input matrix.
 * \param input The input matrix.
 * \param output The output matrix.
 * \return void. Results will be put in output.
 */
void kernel(int size, int width, int height, float *input, float *output);

/*!
 * Wrapper of the OpenMP kernel
 *   used to be called in the main function
 * \param width The width of the input matrix.
 * \param height The height of the input matrix.
 * \param sample The input matrix.
 * \param result The output matrix.
 * \return void. Results will be put in result.
 */
void openmpCallback(int width, int height, float *sample, float **result);

static float log_table[26] = {0, // do not use index 0!
                              0.0,
                              0.6931471805599453,
                              1.0986122886681098,
                              1.3862943611198906,
                              1.6094379124341003,
                              1.791759469228055,
                              1.9459101490553132,
                              2.0794415416798357,
                              2.1972245773362196,
                              2.302585092994046,
                              2.3978952727983707,
                              2.4849066497880004,
                              2.5649493574615367,
                              2.6390573296152584,
                              2.70805020110221,
                              2.772588722239781,
                              2.833213344056216,
                              2.8903717578961645,
                              2.9444389791664403,
                              2.995732273553991,
                              3.044522437723423,
                              3.091042453358316,
                              3.1354942159291497,
                              3.1780538303479458,
                              3.2188758248682006};

#endif