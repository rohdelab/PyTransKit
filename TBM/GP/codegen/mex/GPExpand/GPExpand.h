/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 *
 * GPExpand.h
 *
 * Code generation for function 'GPExpand'
 *
 */

#pragma once

/* Include files */
#include "GPExpand_types.h"
#include "rtwtypes.h"
#include "emlrt.h"
#include "mex.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Function Declarations */
void GPExpand(const emlrtStack *sp, const emxArray_real_T *b_I,
              const real_T newdim[3], emxArray_real_T *IResult);

/* End of code generation (GPExpand.h) */
