/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 *
 * GPExpand_initialize.c
 *
 * Code generation for function 'GPExpand_initialize'
 *
 */

/* Include files */
#include "GPExpand_initialize.h"
#include "GPExpand_data.h"
#include "_coder_GPExpand_mex.h"
#include "rt_nonfinite.h"

/* Function Definitions */
void GPExpand_initialize(void)
{
  emlrtStack st = {
      NULL, /* site */
      NULL, /* tls */
      NULL  /* prev */
  };
  mex_InitInfAndNan();
  mexFunctionCreateRootTLS();
  emlrtBreakCheckR2012bFlagVar = emlrtGetBreakCheckFlagAddressR2012b();
  st.tls = emlrtRootTLSGlobal;
  emlrtClearAllocCountR2012b(&st, false, 0U, NULL);
  emlrtEnterRtStackR2012b(&st);
  emlrtFirstTimeR2012b(emlrtRootTLSGlobal);
}

/* End of code generation (GPExpand_initialize.c) */
