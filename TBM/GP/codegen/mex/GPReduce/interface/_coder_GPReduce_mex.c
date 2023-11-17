/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 *
 * _coder_GPReduce_mex.c
 *
 * Code generation for function '_coder_GPReduce_mex'
 *
 */

/* Include files */
#include "_coder_GPReduce_mex.h"
#include "GPReduce_data.h"
#include "GPReduce_initialize.h"
#include "GPReduce_terminate.h"
#include "_coder_GPReduce_api.h"
#include "rt_nonfinite.h"

/* Function Definitions */
void GPReduce_mexFunction(int32_T nlhs, mxArray *plhs[1], int32_T nrhs,
                          const mxArray *prhs[1])
{
  emlrtStack st = {
      NULL, /* site */
      NULL, /* tls */
      NULL  /* prev */
  };
  const mxArray *outputs;
  st.tls = emlrtRootTLSGlobal;
  /* Check for proper number of arguments. */
  if (nrhs != 1) {
    emlrtErrMsgIdAndTxt(&st, "EMLRT:runTime:WrongNumberOfInputs", 5, 12, 1, 4,
                        8, "GPReduce");
  }
  if (nlhs > 1) {
    emlrtErrMsgIdAndTxt(&st, "EMLRT:runTime:TooManyOutputArguments", 3, 4, 8,
                        "GPReduce");
  }
  /* Call the function. */
  GPReduce_api(prhs[0], &outputs);
  /* Copy over outputs to the caller. */
  emlrtReturnArrays(1, &plhs[0], &outputs);
}

void mexFunction(int32_T nlhs, mxArray *plhs[], int32_T nrhs,
                 const mxArray *prhs[])
{
  mexAtExit(&GPReduce_atexit);
  /* Module initialization. */
  GPReduce_initialize();
  /* Dispatch the entry-point. */
  GPReduce_mexFunction(nlhs, plhs, nrhs, prhs);
  /* Module termination. */
  GPReduce_terminate();
}

emlrtCTX mexFunctionCreateRootTLS(void)
{
  emlrtCreateRootTLSR2022a(&emlrtRootTLSGlobal, &emlrtContextGlobal, NULL, 1,
                           NULL, (const char_T *)"UTF-8", true);
  return emlrtRootTLSGlobal;
}

/* End of code generation (_coder_GPReduce_mex.c) */
