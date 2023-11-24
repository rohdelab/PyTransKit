/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 *
 * GPReduce.h
 *
 * Code generation for function 'GPReduce'
 *
 */

#pragma once

/* Include files */
#include "GPReduce_types.h"
#include "rtwtypes.h"
#include "emlrt.h"
#include "mex.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Function Declarations */
void GPReduce(const emlrtStack *sp, const emxArray_real_T *b_I,
              emxArray_real_T *IResult);

/* End of code generation (GPReduce.h) */
