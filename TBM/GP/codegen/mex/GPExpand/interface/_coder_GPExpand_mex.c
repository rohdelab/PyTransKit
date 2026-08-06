/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 *
 * _coder_GPExpand_mex.c
 *
 * Code generation for function '_coder_GPExpand_mex'
 *
 */

/* Include files */
#include "_coder_GPExpand_mex.h"
#include "GPExpand_data.h"
#include "GPExpand_initialize.h"
#include "GPExpand_terminate.h"
#include "_coder_GPExpand_api.h"
#include "rt_nonfinite.h"

/* Function Definitions */
void GPExpand_mexFunction(int32_T nlhs, mxArray *plhs[1], int32_T nrhs,
                          const mxArray *prhs[2])
{
  emlrtStack st = {
      NULL, /* site */
      NULL, /* tls */
      NULL  /* prev */
  };
  const mxArray *outputs;
  st.tls = emlrtRootTLSGlobal;
  /* Check for proper number of arguments. */
  if (nrhs != 2) {
    emlrtErrMsgIdAndTxt(&st, "EMLRT:runTime:WrongNumberOfInputs", 5, 12, 2, 4,
                        8, "GPExpand");
  }
  if (nlhs > 1) {
    emlrtErrMsgIdAndTxt(&st, "EMLRT:runTime:TooManyOutputArguments", 3, 4, 8,
                        "GPExpand");
  }
  /* Call the function. */
  GPExpand_api(prhs, &outputs);
  /* Copy over outputs to the caller. */
  emlrtReturnArrays(1, &plhs[0], &outputs);
}

void mexFunction(int32_T nlhs, mxArray *plhs[], int32_T nrhs,
                 const mxArray *prhs[])
{
  mexAtExit(&GPExpand_atexit);
  /* Module initialization. */
  GPExpand_initialize();
  /* Dispatch the entry-point. */
  GPExpand_mexFunction(nlhs, plhs, nrhs, prhs);
  /* Module termination. */
  GPExpand_terminate();
}

emlrtCTX mexFunctionCreateRootTLS(void)
{
  emlrtCreateRootTLSR2022a(&emlrtRootTLSGlobal, &emlrtContextGlobal, NULL, 1,
                           NULL, (const char_T *)"UTF-8", true);
  return emlrtRootTLSGlobal;
}

/* End of code generation (_coder_GPExpand_mex.c) */
