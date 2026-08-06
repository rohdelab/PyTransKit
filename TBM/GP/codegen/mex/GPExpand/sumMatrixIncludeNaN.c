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
real_T sumColumnB(const real_T x_data[], int32_T vlen)
{
  real_T y;
  int32_T k;
  y = x_data[0];
  for (k = 0; k <= vlen - 2; k++) {
    y += x_data[k + 1];
  }
  return y;
}

/* End of code generation (sumMatrixIncludeNaN.c) */
