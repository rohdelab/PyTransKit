/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 *
 * sumMatrixIncludeNaN.c
 *
 * Code generation for function 'sumMatrixIncludeNaN'
 *
 */

/* Include files */
#include "sumMatrixIncludeNaN.h"
#include "rt_nonfinite.h"

/* Function Definitions */
real_T sumColumnB(const real_T x[125])
{
  real_T y;
  int32_T k;
  y = x[0];
  for (k = 0; k < 124; k++) {
    y += x[k + 1];
  }
  return y;
}

/* End of code generation (sumMatrixIncludeNaN.c) */
