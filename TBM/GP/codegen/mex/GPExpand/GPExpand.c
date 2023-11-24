/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 *
 * GPExpand.c
 *
 * Code generation for function 'GPExpand'
 *
 */

/* Include files */
#include "GPExpand.h"
#include "GPExpand_data.h"
#include "GPExpand_emxutil.h"
#include "GPExpand_types.h"
#include "find.h"
#include "rt_nonfinite.h"
#include "sumMatrixIncludeNaN.h"
#include "mwmathutil.h"
#include <string.h>

/* Variable Definitions */
static emlrtBCInfo emlrtBCI = {
    -1,                                                        /* iFirst */
    -1,                                                        /* iLast */
    56,                                                        /* lineNo */
    8,                                                         /* colNo */
    "I2",                                                      /* aName */
    "GPExpand",                                                /* fName */
    "/Users/natasha/Downloads/TBM_software/TBM/GP/GPExpand.m", /* pName */
    0                                                          /* checkKind */
};

static emlrtBCInfo b_emlrtBCI = {
    -1,                                                        /* iFirst */
    -1,                                                        /* iLast */
    56,                                                        /* lineNo */
    19,                                                        /* colNo */
    "I2",                                                      /* aName */
    "GPExpand",                                                /* fName */
    "/Users/natasha/Downloads/TBM_software/TBM/GP/GPExpand.m", /* pName */
    0                                                          /* checkKind */
};

static emlrtBCInfo c_emlrtBCI = {
    -1,                                                        /* iFirst */
    -1,                                                        /* iLast */
    56,                                                        /* lineNo */
    30,                                                        /* colNo */
    "I2",                                                      /* aName */
    "GPExpand",                                                /* fName */
    "/Users/natasha/Downloads/TBM_software/TBM/GP/GPExpand.m", /* pName */
    0                                                          /* checkKind */
};

static emlrtECInfo emlrtECI = {
    -1,                                                       /* nDims */
    56,                                                       /* lineNo */
    3,                                                        /* colNo */
    "GPExpand",                                               /* fName */
    "/Users/natasha/Downloads/TBM_software/TBM/GP/GPExpand.m" /* pName */
};

static emlrtECInfo b_emlrtECI = {
    -1,                                                       /* nDims */
    57,                                                       /* lineNo */
    3,                                                        /* colNo */
    "GPExpand",                                               /* fName */
    "/Users/natasha/Downloads/TBM_software/TBM/GP/GPExpand.m" /* pName */
};

static emlrtBCInfo d_emlrtBCI = {
    -1,                                                        /* iFirst */
    -1,                                                        /* iLast */
    57,                                                        /* lineNo */
    38,                                                        /* colNo */
    "I2",                                                      /* aName */
    "GPExpand",                                                /* fName */
    "/Users/natasha/Downloads/TBM_software/TBM/GP/GPExpand.m", /* pName */
    0                                                          /* checkKind */
};

static emlrtBCInfo e_emlrtBCI = {
    -1,                                                        /* iFirst */
    -1,                                                        /* iLast */
    57,                                                        /* lineNo */
    26,                                                        /* colNo */
    "I2",                                                      /* aName */
    "GPExpand",                                                /* fName */
    "/Users/natasha/Downloads/TBM_software/TBM/GP/GPExpand.m", /* pName */
    0                                                          /* checkKind */
};

static emlrtECInfo c_emlrtECI = {
    -1,                                                       /* nDims */
    57,                                                       /* lineNo */
    23,                                                       /* colNo */
    "GPExpand",                                               /* fName */
    "/Users/natasha/Downloads/TBM_software/TBM/GP/GPExpand.m" /* pName */
};

static emlrtECInfo d_emlrtECI = {
    -1,                                                       /* nDims */
    58,                                                       /* lineNo */
    3,                                                        /* colNo */
    "GPExpand",                                               /* fName */
    "/Users/natasha/Downloads/TBM_software/TBM/GP/GPExpand.m" /* pName */
};

static emlrtBCInfo f_emlrtBCI = {
    -1,                                                        /* iFirst */
    -1,                                                        /* iLast */
    58,                                                        /* lineNo */
    40,                                                        /* colNo */
    "I2",                                                      /* aName */
    "GPExpand",                                                /* fName */
    "/Users/natasha/Downloads/TBM_software/TBM/GP/GPExpand.m", /* pName */
    0                                                          /* checkKind */
};

static emlrtBCInfo g_emlrtBCI = {
    -1,                                                        /* iFirst */
    -1,                                                        /* iLast */
    58,                                                        /* lineNo */
    28,                                                        /* colNo */
    "I2",                                                      /* aName */
    "GPExpand",                                                /* fName */
    "/Users/natasha/Downloads/TBM_software/TBM/GP/GPExpand.m", /* pName */
    0                                                          /* checkKind */
};

static emlrtECInfo e_emlrtECI = {
    -1,                                                       /* nDims */
    58,                                                       /* lineNo */
    23,                                                       /* colNo */
    "GPExpand",                                               /* fName */
    "/Users/natasha/Downloads/TBM_software/TBM/GP/GPExpand.m" /* pName */
};

static emlrtECInfo f_emlrtECI = {
    -1,                                                       /* nDims */
    59,                                                       /* lineNo */
    3,                                                        /* colNo */
    "GPExpand",                                               /* fName */
    "/Users/natasha/Downloads/TBM_software/TBM/GP/GPExpand.m" /* pName */
};

static emlrtBCInfo h_emlrtBCI = {
    -1,                                                        /* iFirst */
    -1,                                                        /* iLast */
    59,                                                        /* lineNo */
    42,                                                        /* colNo */
    "I2",                                                      /* aName */
    "GPExpand",                                                /* fName */
    "/Users/natasha/Downloads/TBM_software/TBM/GP/GPExpand.m", /* pName */
    0                                                          /* checkKind */
};

static emlrtBCInfo i_emlrtBCI = {
    -1,                                                        /* iFirst */
    -1,                                                        /* iLast */
    59,                                                        /* lineNo */
    30,                                                        /* colNo */
    "I2",                                                      /* aName */
    "GPExpand",                                                /* fName */
    "/Users/natasha/Downloads/TBM_software/TBM/GP/GPExpand.m", /* pName */
    0                                                          /* checkKind */
};

static emlrtECInfo g_emlrtECI = {
    -1,                                                       /* nDims */
    59,                                                       /* lineNo */
    23,                                                       /* colNo */
    "GPExpand",                                               /* fName */
    "/Users/natasha/Downloads/TBM_software/TBM/GP/GPExpand.m" /* pName */
};

static emlrtRTEInfo emlrtRTEI = {
    62,                                                       /* lineNo */
    11,                                                       /* colNo */
    "GPExpand",                                               /* fName */
    "/Users/natasha/Downloads/TBM_software/TBM/GP/GPExpand.m" /* pName */
};

static emlrtRTEInfo b_emlrtRTEI = {
    63,                                                       /* lineNo */
    12,                                                       /* colNo */
    "GPExpand",                                               /* fName */
    "/Users/natasha/Downloads/TBM_software/TBM/GP/GPExpand.m" /* pName */
};

static emlrtRTEInfo c_emlrtRTEI = {
    64,                                                       /* lineNo */
    13,                                                       /* colNo */
    "GPExpand",                                               /* fName */
    "/Users/natasha/Downloads/TBM_software/TBM/GP/GPExpand.m" /* pName */
};

static emlrtBCInfo j_emlrtBCI = {
    -1,                                                        /* iFirst */
    -1,                                                        /* iLast */
    68,                                                        /* lineNo */
    12,                                                        /* colNo */
    "I",                                                       /* aName */
    "GPExpand",                                                /* fName */
    "/Users/natasha/Downloads/TBM_software/TBM/GP/GPExpand.m", /* pName */
    0                                                          /* checkKind */
};

static emlrtDCInfo emlrtDCI = {
    68,                                                        /* lineNo */
    12,                                                        /* colNo */
    "GPExpand",                                                /* fName */
    "/Users/natasha/Downloads/TBM_software/TBM/GP/GPExpand.m", /* pName */
    1                                                          /* checkKind */
};

static emlrtBCInfo k_emlrtBCI = {
    -1,                                                        /* iFirst */
    -1,                                                        /* iLast */
    68,                                                        /* lineNo */
    25,                                                        /* colNo */
    "I",                                                       /* aName */
    "GPExpand",                                                /* fName */
    "/Users/natasha/Downloads/TBM_software/TBM/GP/GPExpand.m", /* pName */
    0                                                          /* checkKind */
};

static emlrtDCInfo b_emlrtDCI = {
    68,                                                        /* lineNo */
    25,                                                        /* colNo */
    "GPExpand",                                                /* fName */
    "/Users/natasha/Downloads/TBM_software/TBM/GP/GPExpand.m", /* pName */
    1                                                          /* checkKind */
};

static emlrtBCInfo l_emlrtBCI = {
    -1,                                                        /* iFirst */
    -1,                                                        /* iLast */
    68,                                                        /* lineNo */
    38,                                                        /* colNo */
    "I",                                                       /* aName */
    "GPExpand",                                                /* fName */
    "/Users/natasha/Downloads/TBM_software/TBM/GP/GPExpand.m", /* pName */
    0                                                          /* checkKind */
};

static emlrtDCInfo c_emlrtDCI = {
    68,                                                        /* lineNo */
    38,                                                        /* colNo */
    "GPExpand",                                                /* fName */
    "/Users/natasha/Downloads/TBM_software/TBM/GP/GPExpand.m", /* pName */
    1                                                          /* checkKind */
};

static emlrtBCInfo m_emlrtBCI = {
    -1,                                                        /* iFirst */
    -1,                                                        /* iLast */
    69,                                                        /* lineNo */
    14,                                                        /* colNo */
    "IResult",                                                 /* aName */
    "GPExpand",                                                /* fName */
    "/Users/natasha/Downloads/TBM_software/TBM/GP/GPExpand.m", /* pName */
    0                                                          /* checkKind */
};

static emlrtBCInfo n_emlrtBCI = {
    -1,                                                        /* iFirst */
    -1,                                                        /* iLast */
    69,                                                        /* lineNo */
    21,                                                        /* colNo */
    "IResult",                                                 /* aName */
    "GPExpand",                                                /* fName */
    "/Users/natasha/Downloads/TBM_software/TBM/GP/GPExpand.m", /* pName */
    0                                                          /* checkKind */
};

static emlrtBCInfo o_emlrtBCI = {
    -1,                                                        /* iFirst */
    -1,                                                        /* iLast */
    69,                                                        /* lineNo */
    27,                                                        /* colNo */
    "IResult",                                                 /* aName */
    "GPExpand",                                                /* fName */
    "/Users/natasha/Downloads/TBM_software/TBM/GP/GPExpand.m", /* pName */
    0                                                          /* checkKind */
};

static emlrtDCInfo d_emlrtDCI = {
    55,                                                        /* lineNo */
    14,                                                        /* colNo */
    "GPExpand",                                                /* fName */
    "/Users/natasha/Downloads/TBM_software/TBM/GP/GPExpand.m", /* pName */
    1                                                          /* checkKind */
};

static emlrtDCInfo e_emlrtDCI = {
    20,                                                        /* lineNo */
    17,                                                        /* colNo */
    "GPExpand",                                                /* fName */
    "/Users/natasha/Downloads/TBM_software/TBM/GP/GPExpand.m", /* pName */
    1                                                          /* checkKind */
};

static emlrtDCInfo f_emlrtDCI = {
    20,                                                        /* lineNo */
    17,                                                        /* colNo */
    "GPExpand",                                                /* fName */
    "/Users/natasha/Downloads/TBM_software/TBM/GP/GPExpand.m", /* pName */
    4                                                          /* checkKind */
};

static emlrtRTEInfo e_emlrtRTEI = {
    20,                                                       /* lineNo */
    1,                                                        /* colNo */
    "GPExpand",                                               /* fName */
    "/Users/natasha/Downloads/TBM_software/TBM/GP/GPExpand.m" /* pName */
};

static emlrtRTEInfo f_emlrtRTEI = {
    55,                                                       /* lineNo */
    3,                                                        /* colNo */
    "GPExpand",                                               /* fName */
    "/Users/natasha/Downloads/TBM_software/TBM/GP/GPExpand.m" /* pName */
};

static emlrtRTEInfo g_emlrtRTEI = {
    57,                                                       /* lineNo */
    13,                                                       /* colNo */
    "GPExpand",                                               /* fName */
    "/Users/natasha/Downloads/TBM_software/TBM/GP/GPExpand.m" /* pName */
};

static emlrtRTEInfo h_emlrtRTEI = {
    57,                                                       /* lineNo */
    35,                                                       /* colNo */
    "GPExpand",                                               /* fName */
    "/Users/natasha/Downloads/TBM_software/TBM/GP/GPExpand.m" /* pName */
};

static emlrtRTEInfo i_emlrtRTEI = {
    58,                                                       /* lineNo */
    13,                                                       /* colNo */
    "GPExpand",                                               /* fName */
    "/Users/natasha/Downloads/TBM_software/TBM/GP/GPExpand.m" /* pName */
};

static emlrtRTEInfo j_emlrtRTEI = {
    58,                                                       /* lineNo */
    35,                                                       /* colNo */
    "GPExpand",                                               /* fName */
    "/Users/natasha/Downloads/TBM_software/TBM/GP/GPExpand.m" /* pName */
};

static emlrtRTEInfo k_emlrtRTEI = {
    59,                                                       /* lineNo */
    13,                                                       /* colNo */
    "GPExpand",                                               /* fName */
    "/Users/natasha/Downloads/TBM_software/TBM/GP/GPExpand.m" /* pName */
};

static emlrtRTEInfo l_emlrtRTEI = {
    59,                                                       /* lineNo */
    35,                                                       /* colNo */
    "GPExpand",                                               /* fName */
    "/Users/natasha/Downloads/TBM_software/TBM/GP/GPExpand.m" /* pName */
};

/* Function Definitions */
void GPExpand(const emlrtStack *sp, const emxArray_real_T *b_I,
              const real_T newdim[3], emxArray_real_T *IResult)
{
  static const real_T dv[5] = {0.05, 0.25, 0.4, 0.25, 0.05};
  emxArray_real_T *I2;
  emxArray_real_T *d_I2;
  emxArray_real_T *e_I2;
  emxArray_real_T *f_I2;
  real_T Wt3[125];
  const real_T *I_data;
  real_T b;
  real_T *I2_data;
  real_T *IResult_data;
  real_T *b_I2_data;
  int32_T idxi_data[5];
  int32_T idxj_data[5];
  int32_T idxk_data[5];
  int32_T A_size[3];
  int32_T c_I2[3];
  int32_T g_I2[2];
  int32_T idxi_size[2];
  int32_T idxj_size[2];
  int32_T iv[2];
  int32_T Wt3_tmp;
  int32_T b_I2;
  int32_T b_i;
  int32_T b_loop_ub;
  int32_T i;
  int32_T i1;
  int32_T i2;
  int32_T i3;
  int32_T i4;
  int32_T i5;
  int32_T i6;
  int32_T j;
  int32_T k;
  int32_T loop_ub;
  I_data = b_I->data;
  emlrtHeapReferenceStackEnterFcnR2012b((emlrtCTX)sp);
  /* %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% */
  /*  Expand an image as per the Gaussian Pyramid. */
  /* %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% */
  /*  newdim = zeros(1,3);  */
  /*  newdim(1) = dim(1); newdim(2) = dim(2); newdim(3) = dim(3);   */
  /*  for i = 1:numel(dim) */
  /*      if mod(dim(i),2)==0 */
  /*          newdim(i) = dim(i)*2; */
  /*      else  */
  /*          newdim(i) = dim(i)*2-1;  */
  /*      end */
  /*  end */
  if (!(newdim[0] >= 0.0)) {
    emlrtNonNegativeCheckR2012b(newdim[0], &f_emlrtDCI, (emlrtCTX)sp);
  }
  if (newdim[0] != (int32_T)muDoubleScalarFloor(newdim[0])) {
    emlrtIntegerCheckR2012b(newdim[0], &e_emlrtDCI, (emlrtCTX)sp);
  }
  if (!(newdim[1] >= 0.0)) {
    emlrtNonNegativeCheckR2012b(newdim[1], &f_emlrtDCI, (emlrtCTX)sp);
  }
  if (newdim[1] != (int32_T)muDoubleScalarFloor(newdim[1])) {
    emlrtIntegerCheckR2012b(newdim[1], &e_emlrtDCI, (emlrtCTX)sp);
  }
  if (!(newdim[2] >= 0.0)) {
    emlrtNonNegativeCheckR2012b(newdim[2], &f_emlrtDCI, (emlrtCTX)sp);
  }
  if (newdim[2] != (int32_T)muDoubleScalarFloor(newdim[2])) {
    emlrtIntegerCheckR2012b(newdim[2], &e_emlrtDCI, (emlrtCTX)sp);
  }
  i = IResult->size[0] * IResult->size[1] * IResult->size[2];
  IResult->size[0] = (int32_T)newdim[0];
  IResult->size[1] = (int32_T)newdim[1];
  IResult->size[2] = (int32_T)newdim[2];
  emxEnsureCapacity_real_T(sp, IResult, i, &e_emlrtRTEI);
  IResult_data = IResult->data;
  loop_ub = (int32_T)newdim[0] * (int32_T)newdim[1] * (int32_T)newdim[2];
  for (i = 0; i < loop_ub; i++) {
    IResult_data[i] = 0.0;
  }
  /*  Initialize the array in the beginning .. */
  for (i = 0; i < 125; i++) {
    Wt3[i] = 1.0;
  }
  for (b_i = 0; b_i < 5; b_i++) {
    b = dv[b_i];
    for (i = 0; i < 5; i++) {
      for (i1 = 0; i1 < 5; i1++) {
        Wt3_tmp = (b_i + 5 * i1) + 25 * i;
        Wt3[Wt3_tmp] *= b;
      }
      for (i1 = 0; i1 < 5; i1++) {
        Wt3_tmp = (i1 + 5 * b_i) + 25 * i;
        Wt3[Wt3_tmp] *= b;
      }
    }
    for (i = 0; i < 5; i++) {
      for (i1 = 0; i1 < 5; i1++) {
        Wt3_tmp = (i1 + 5 * i) + 25 * b_i;
        Wt3[Wt3_tmp] *= b;
      }
    }
    if (*emlrtBreakCheckR2012bFlagVar != 0) {
      emlrtBreakCheckR2012b((emlrtCTX)sp);
    }
  }
  /* 		%% Pad the boundaries */
  if ((real_T)b_I->size[0] + 2.0 != b_I->size[0] + 2) {
    emlrtIntegerCheckR2012b((real_T)b_I->size[0] + 2.0, &d_emlrtDCI,
                            (emlrtCTX)sp);
  }
  if ((real_T)b_I->size[1] + 2.0 != b_I->size[1] + 2) {
    emlrtIntegerCheckR2012b((real_T)b_I->size[1] + 2.0, &d_emlrtDCI,
                            (emlrtCTX)sp);
  }
  if ((real_T)b_I->size[2] + 2.0 != b_I->size[2] + 2) {
    emlrtIntegerCheckR2012b((real_T)b_I->size[2] + 2.0, &d_emlrtDCI,
                            (emlrtCTX)sp);
  }
  emxInit_real_T(sp, &I2, 3, &f_emlrtRTEI);
  i = I2->size[0] * I2->size[1] * I2->size[2];
  I2->size[0] = (int32_T)(uint32_T)((real_T)b_I->size[0] + 2.0);
  I2->size[1] = (int32_T)(uint32_T)((real_T)b_I->size[1] + 2.0);
  I2->size[2] = (int32_T)(uint32_T)((real_T)b_I->size[2] + 2.0);
  emxEnsureCapacity_real_T(sp, I2, i, &f_emlrtRTEI);
  I2_data = I2->data;
  loop_ub = (int32_T)(uint32_T)((real_T)b_I->size[0] + 2.0) *
            (int32_T)(uint32_T)((real_T)b_I->size[1] + 2.0) *
            (int32_T)(uint32_T)((real_T)b_I->size[2] + 2.0);
  for (i = 0; i < loop_ub; i++) {
    I2_data[i] = 0.0;
  }
  if (b_I->size[0] + 1U < 2U) {
    i = 0;
    i1 = 0;
  } else {
    i = 1;
    if (((int32_T)(b_I->size[0] + 1U) < 1) ||
        ((int32_T)(b_I->size[0] + 1U) >
         (int32_T)(uint32_T)((real_T)b_I->size[0] + 2.0))) {
      emlrtDynamicBoundsCheckR2012b(
          (int32_T)(b_I->size[0] + 1U), 1,
          (int32_T)(uint32_T)((real_T)b_I->size[0] + 2.0), &emlrtBCI,
          (emlrtCTX)sp);
    }
    i1 = (int32_T)(b_I->size[0] + 1U);
  }
  if (b_I->size[1] + 1U < 2U) {
    b_I2 = 0;
    i2 = 0;
  } else {
    b_I2 = 1;
    if (((int32_T)(b_I->size[1] + 1U) < 1) ||
        ((int32_T)(b_I->size[1] + 1U) >
         (int32_T)(uint32_T)((real_T)b_I->size[1] + 2.0))) {
      emlrtDynamicBoundsCheckR2012b(
          (int32_T)(b_I->size[1] + 1U), 1,
          (int32_T)(uint32_T)((real_T)b_I->size[1] + 2.0), &b_emlrtBCI,
          (emlrtCTX)sp);
    }
    i2 = (int32_T)(b_I->size[1] + 1U);
  }
  if (b_I->size[2] + 1U < 2U) {
    i3 = 0;
    i4 = 0;
  } else {
    i3 = 1;
    if (((int32_T)(b_I->size[2] + 1U) < 1) ||
        ((int32_T)(b_I->size[2] + 1U) >
         (int32_T)(uint32_T)((real_T)b_I->size[2] + 2.0))) {
      emlrtDynamicBoundsCheckR2012b(
          (int32_T)(b_I->size[2] + 1U), 1,
          (int32_T)(uint32_T)((real_T)b_I->size[2] + 2.0), &c_emlrtBCI,
          (emlrtCTX)sp);
    }
    i4 = (int32_T)(b_I->size[2] + 1U);
  }
  c_I2[0] = i1 - i;
  c_I2[1] = i2 - b_I2;
  c_I2[2] = i4 - i3;
  emlrtSubAssignSizeCheckR2012b(&c_I2[0], 3, &b_I->size[0], 3, &emlrtECI,
                                (emlrtCTX)sp);
  loop_ub = b_I->size[2];
  for (i1 = 0; i1 < loop_ub; i1++) {
    b_loop_ub = b_I->size[1];
    for (i2 = 0; i2 < b_loop_ub; i2++) {
      Wt3_tmp = b_I->size[0];
      for (i4 = 0; i4 < Wt3_tmp; i4++) {
        I2_data[((i + i4) + I2->size[0] * (b_I2 + i2)) +
                I2->size[0] * I2->size[1] * (i3 + i1)] =
            I_data[(i4 + b_I->size[0] * i2) + b_I->size[0] * b_I->size[1] * i1];
      }
    }
  }
  emxInit_real_T(sp, &d_I2, 3, &g_emlrtRTEI);
  loop_ub = I2->size[1];
  b_loop_ub = I2->size[2];
  i = d_I2->size[0] * d_I2->size[1] * d_I2->size[2];
  d_I2->size[0] = 1;
  d_I2->size[1] = I2->size[1];
  d_I2->size[2] = I2->size[2];
  emxEnsureCapacity_real_T(sp, d_I2, i, &g_emlrtRTEI);
  b_I2_data = d_I2->data;
  for (i = 0; i < b_loop_ub; i++) {
    for (i1 = 0; i1 < loop_ub; i1++) {
      b_I2_data[i1 + d_I2->size[1] * i] =
          I2_data[(I2->size[0] * i1 + I2->size[0] * I2->size[1] * i) + 1];
    }
  }
  c_I2[0] = 1;
  c_I2[1] = I2->size[1];
  c_I2[2] = I2->size[2];
  emlrtSubAssignSizeCheckR2012b(&c_I2[0], 3, &d_I2->size[0], 3, &b_emlrtECI,
                                (emlrtCTX)sp);
  Wt3_tmp = I2->size[1] - 1;
  loop_ub = I2->size[2] - 1;
  i = d_I2->size[0] * d_I2->size[1] * d_I2->size[2];
  d_I2->size[0] = 1;
  d_I2->size[1] = I2->size[1];
  d_I2->size[2] = I2->size[2];
  emxEnsureCapacity_real_T(sp, d_I2, i, &g_emlrtRTEI);
  b_I2_data = d_I2->data;
  for (i = 0; i <= loop_ub; i++) {
    for (i1 = 0; i1 <= Wt3_tmp; i1++) {
      b_I2_data[i1 + d_I2->size[1] * i] =
          I2_data[(I2->size[0] * i1 + I2->size[0] * I2->size[1] * i) + 1];
    }
  }
  loop_ub = d_I2->size[2];
  for (i = 0; i < loop_ub; i++) {
    b_loop_ub = d_I2->size[1];
    for (i1 = 0; i1 < b_loop_ub; i1++) {
      I2_data[I2->size[0] * i1 + I2->size[0] * I2->size[1] * i] =
          b_I2_data[i1 + d_I2->size[1] * i];
    }
  }
  if (I2->size[0] < 1) {
    emlrtDynamicBoundsCheckR2012b(I2->size[0], 1, I2->size[0], &e_emlrtBCI,
                                  (emlrtCTX)sp);
  }
  if ((I2->size[0] - 1 < 1) || (I2->size[0] - 1 > I2->size[0])) {
    emlrtDynamicBoundsCheckR2012b(I2->size[0] - 1, 1, I2->size[0], &d_emlrtBCI,
                                  (emlrtCTX)sp);
  }
  loop_ub = I2->size[1];
  b_loop_ub = I2->size[2];
  i = d_I2->size[0] * d_I2->size[1] * d_I2->size[2];
  d_I2->size[0] = 1;
  d_I2->size[1] = I2->size[1];
  d_I2->size[2] = I2->size[2];
  emxEnsureCapacity_real_T(sp, d_I2, i, &h_emlrtRTEI);
  b_I2_data = d_I2->data;
  for (i = 0; i < b_loop_ub; i++) {
    for (i1 = 0; i1 < loop_ub; i1++) {
      b_I2_data[i1 + d_I2->size[1] * i] =
          I2_data[((I2->size[0] + I2->size[0] * i1) +
                   I2->size[0] * I2->size[1] * i) -
                  2];
    }
  }
  c_I2[0] = 1;
  c_I2[1] = I2->size[1];
  c_I2[2] = I2->size[2];
  emlrtSubAssignSizeCheckR2012b(&c_I2[0], 3, &d_I2->size[0], 3, &c_emlrtECI,
                                (emlrtCTX)sp);
  Wt3_tmp = I2->size[1] - 1;
  loop_ub = I2->size[2] - 1;
  b_I2 = I2->size[0] - 1;
  i = d_I2->size[0] * d_I2->size[1] * d_I2->size[2];
  d_I2->size[0] = 1;
  d_I2->size[1] = I2->size[1];
  d_I2->size[2] = I2->size[2];
  emxEnsureCapacity_real_T(sp, d_I2, i, &h_emlrtRTEI);
  b_I2_data = d_I2->data;
  for (i = 0; i <= loop_ub; i++) {
    for (i1 = 0; i1 <= Wt3_tmp; i1++) {
      b_I2_data[i1 + d_I2->size[1] * i] =
          I2_data[((I2->size[0] + I2->size[0] * i1) +
                   I2->size[0] * I2->size[1] * i) -
                  2];
    }
  }
  loop_ub = d_I2->size[2];
  for (i = 0; i < loop_ub; i++) {
    b_loop_ub = d_I2->size[1];
    for (i1 = 0; i1 < b_loop_ub; i1++) {
      I2_data[(b_I2 + I2->size[0] * i1) + I2->size[0] * I2->size[1] * i] =
          b_I2_data[i1 + d_I2->size[1] * i];
    }
  }
  emxFree_real_T(sp, &d_I2);
  emxInit_real_T(sp, &e_I2, 3, &i_emlrtRTEI);
  loop_ub = I2->size[0];
  b_loop_ub = I2->size[2];
  i = e_I2->size[0] * e_I2->size[1] * e_I2->size[2];
  e_I2->size[0] = I2->size[0];
  e_I2->size[1] = 1;
  e_I2->size[2] = I2->size[2];
  emxEnsureCapacity_real_T(sp, e_I2, i, &i_emlrtRTEI);
  b_I2_data = e_I2->data;
  for (i = 0; i < b_loop_ub; i++) {
    for (i1 = 0; i1 < loop_ub; i1++) {
      b_I2_data[i1 + e_I2->size[0] * i] =
          I2_data[(i1 + I2->size[0]) + I2->size[0] * I2->size[1] * i];
    }
  }
  c_I2[0] = I2->size[0];
  c_I2[1] = 1;
  c_I2[2] = I2->size[2];
  emlrtSubAssignSizeCheckR2012b(&c_I2[0], 3, &e_I2->size[0], 3, &d_emlrtECI,
                                (emlrtCTX)sp);
  Wt3_tmp = I2->size[0] - 1;
  loop_ub = I2->size[2] - 1;
  i = e_I2->size[0] * e_I2->size[1] * e_I2->size[2];
  e_I2->size[0] = I2->size[0];
  e_I2->size[1] = 1;
  e_I2->size[2] = I2->size[2];
  emxEnsureCapacity_real_T(sp, e_I2, i, &i_emlrtRTEI);
  b_I2_data = e_I2->data;
  for (i = 0; i <= loop_ub; i++) {
    for (i1 = 0; i1 <= Wt3_tmp; i1++) {
      b_I2_data[i1 + e_I2->size[0] * i] =
          I2_data[(i1 + I2->size[0]) + I2->size[0] * I2->size[1] * i];
    }
  }
  loop_ub = e_I2->size[2];
  for (i = 0; i < loop_ub; i++) {
    b_loop_ub = e_I2->size[0];
    for (i1 = 0; i1 < b_loop_ub; i1++) {
      I2_data[i1 + I2->size[0] * I2->size[1] * i] =
          b_I2_data[i1 + e_I2->size[0] * i];
    }
  }
  if (I2->size[1] < 1) {
    emlrtDynamicBoundsCheckR2012b(I2->size[1], 1, I2->size[1], &g_emlrtBCI,
                                  (emlrtCTX)sp);
  }
  if ((I2->size[1] - 1 < 1) || (I2->size[1] - 1 > I2->size[1])) {
    emlrtDynamicBoundsCheckR2012b(I2->size[1] - 1, 1, I2->size[1], &f_emlrtBCI,
                                  (emlrtCTX)sp);
  }
  loop_ub = I2->size[0];
  b_loop_ub = I2->size[2];
  i = e_I2->size[0] * e_I2->size[1] * e_I2->size[2];
  e_I2->size[0] = I2->size[0];
  e_I2->size[1] = 1;
  e_I2->size[2] = I2->size[2];
  emxEnsureCapacity_real_T(sp, e_I2, i, &j_emlrtRTEI);
  b_I2_data = e_I2->data;
  for (i = 0; i < b_loop_ub; i++) {
    for (i1 = 0; i1 < loop_ub; i1++) {
      b_I2_data[i1 + e_I2->size[0] * i] =
          I2_data[(i1 + I2->size[0] * (I2->size[1] - 2)) +
                  I2->size[0] * I2->size[1] * i];
    }
  }
  c_I2[0] = I2->size[0];
  c_I2[1] = 1;
  c_I2[2] = I2->size[2];
  emlrtSubAssignSizeCheckR2012b(&c_I2[0], 3, &e_I2->size[0], 3, &e_emlrtECI,
                                (emlrtCTX)sp);
  Wt3_tmp = I2->size[0] - 1;
  loop_ub = I2->size[2] - 1;
  b_I2 = I2->size[1] - 1;
  i = e_I2->size[0] * e_I2->size[1] * e_I2->size[2];
  e_I2->size[0] = I2->size[0];
  e_I2->size[1] = 1;
  e_I2->size[2] = I2->size[2];
  emxEnsureCapacity_real_T(sp, e_I2, i, &j_emlrtRTEI);
  b_I2_data = e_I2->data;
  for (i = 0; i <= loop_ub; i++) {
    for (i1 = 0; i1 <= Wt3_tmp; i1++) {
      b_I2_data[i1 + e_I2->size[0] * i] =
          I2_data[(i1 + I2->size[0] * (I2->size[1] - 2)) +
                  I2->size[0] * I2->size[1] * i];
    }
  }
  loop_ub = e_I2->size[2];
  for (i = 0; i < loop_ub; i++) {
    b_loop_ub = e_I2->size[0];
    for (i1 = 0; i1 < b_loop_ub; i1++) {
      I2_data[(i1 + I2->size[0] * b_I2) + I2->size[0] * I2->size[1] * i] =
          b_I2_data[i1 + e_I2->size[0] * i];
    }
  }
  emxFree_real_T(sp, &e_I2);
  emxInit_real_T(sp, &f_I2, 2, &k_emlrtRTEI);
  g_I2[0] = I2->size[0];
  g_I2[1] = I2->size[1];
  iv[0] = I2->size[0];
  iv[1] = I2->size[1];
  emlrtSubAssignSizeCheckR2012b(&g_I2[0], 2, &iv[0], 2, &f_emlrtECI,
                                (emlrtCTX)sp);
  Wt3_tmp = I2->size[0] - 1;
  loop_ub = I2->size[1] - 1;
  i = f_I2->size[0] * f_I2->size[1];
  f_I2->size[0] = I2->size[0];
  f_I2->size[1] = I2->size[1];
  emxEnsureCapacity_real_T(sp, f_I2, i, &k_emlrtRTEI);
  b_I2_data = f_I2->data;
  for (i = 0; i <= loop_ub; i++) {
    for (i1 = 0; i1 <= Wt3_tmp; i1++) {
      b_I2_data[i1 + f_I2->size[0] * i] =
          I2_data[(i1 + I2->size[0] * i) + I2->size[0] * I2->size[1]];
    }
  }
  loop_ub = f_I2->size[1];
  for (i = 0; i < loop_ub; i++) {
    b_loop_ub = f_I2->size[0];
    for (i1 = 0; i1 < b_loop_ub; i1++) {
      I2_data[i1 + I2->size[0] * i] = b_I2_data[i1 + f_I2->size[0] * i];
    }
  }
  if (I2->size[2] < 1) {
    emlrtDynamicBoundsCheckR2012b(I2->size[2], 1, I2->size[2], &i_emlrtBCI,
                                  (emlrtCTX)sp);
  }
  if ((I2->size[2] - 1 < 1) || (I2->size[2] - 1 > I2->size[2])) {
    emlrtDynamicBoundsCheckR2012b(I2->size[2] - 1, 1, I2->size[2], &h_emlrtBCI,
                                  (emlrtCTX)sp);
  }
  g_I2[0] = I2->size[0];
  g_I2[1] = I2->size[1];
  iv[0] = I2->size[0];
  iv[1] = I2->size[1];
  emlrtSubAssignSizeCheckR2012b(&g_I2[0], 2, &iv[0], 2, &g_emlrtECI,
                                (emlrtCTX)sp);
  Wt3_tmp = I2->size[0] - 1;
  loop_ub = I2->size[1] - 1;
  b_I2 = I2->size[2] - 1;
  i = f_I2->size[0] * f_I2->size[1];
  f_I2->size[0] = I2->size[0];
  f_I2->size[1] = I2->size[1];
  emxEnsureCapacity_real_T(sp, f_I2, i, &l_emlrtRTEI);
  b_I2_data = f_I2->data;
  for (i = 0; i <= loop_ub; i++) {
    for (i1 = 0; i1 <= Wt3_tmp; i1++) {
      b_I2_data[i1 + f_I2->size[0] * i] =
          I2_data[(i1 + I2->size[0] * i) +
                  I2->size[0] * I2->size[1] * (I2->size[2] - 2)];
    }
  }
  loop_ub = f_I2->size[1];
  for (i = 0; i < loop_ub; i++) {
    b_loop_ub = f_I2->size[0];
    for (i1 = 0; i1 < b_loop_ub; i1++) {
      I2_data[(i1 + I2->size[0] * i) + I2->size[0] * I2->size[1] * b_I2] =
          b_I2_data[i1 + f_I2->size[0] * i];
    }
  }
  emxFree_real_T(sp, &f_I2);
  /* clear I2; */
  i = (int32_T)((newdim[0] - 1.0) + 1.0);
  emlrtForLoopVectorCheckR2021a(0.0, 1.0, newdim[0] - 1.0, mxDOUBLE_CLASS,
                                (int32_T)((newdim[0] - 1.0) + 1.0), &emlrtRTEI,
                                (emlrtCTX)sp);
  if (i - 1 >= 0) {
    i5 = (int32_T)((newdim[1] - 1.0) + 1.0);
    if (i5 - 1 >= 0) {
      i6 = (int32_T)((newdim[2] - 1.0) + 1.0);
    }
  }
  for (b_i = 0; b_i < i; b_i++) {
    emlrtForLoopVectorCheckR2021a(0.0, 1.0, newdim[1] - 1.0, mxDOUBLE_CLASS,
                                  (int32_T)((newdim[1] - 1.0) + 1.0),
                                  &b_emlrtRTEI, (emlrtCTX)sp);
    for (j = 0; j < i5; j++) {
      emlrtForLoopVectorCheckR2021a(0.0, 1.0, newdim[2] - 1.0, mxDOUBLE_CLASS,
                                    (int32_T)((newdim[2] - 1.0) + 1.0),
                                    &c_emlrtRTEI, (emlrtCTX)sp);
      for (k = 0; k < i6; k++) {
        real_T A_data[125];
        real_T tmp_data[125];
        real_T pixeli[5];
        real_T pixelj[5];
        real_T pixelk[5];
        boolean_T x[5];
        for (Wt3_tmp = 0; Wt3_tmp < 5; Wt3_tmp++) {
          b = ((real_T)b_i - ((real_T)Wt3_tmp + -2.0)) / 2.0 + 2.0;
          pixeli[Wt3_tmp] = b;
          x[Wt3_tmp] = (muDoubleScalarFloor(b) == b);
        }
        eml_find(x, idxk_data, g_I2);
        loop_ub = g_I2[1];
        idxi_size[1] = g_I2[1];
        if (loop_ub - 1 >= 0) {
          memcpy(&idxi_data[0], &idxk_data[0], loop_ub * sizeof(int32_T));
        }
        for (Wt3_tmp = 0; Wt3_tmp < 5; Wt3_tmp++) {
          b = ((real_T)j - ((real_T)Wt3_tmp + -2.0)) / 2.0 + 2.0;
          pixelj[Wt3_tmp] = b;
          x[Wt3_tmp] = (muDoubleScalarFloor(b) == b);
        }
        eml_find(x, idxk_data, g_I2);
        b_loop_ub = g_I2[1];
        idxj_size[1] = g_I2[1];
        if (b_loop_ub - 1 >= 0) {
          memcpy(&idxj_data[0], &idxk_data[0], b_loop_ub * sizeof(int32_T));
        }
        for (Wt3_tmp = 0; Wt3_tmp < 5; Wt3_tmp++) {
          b = ((real_T)k - ((real_T)Wt3_tmp + -2.0)) / 2.0 + 2.0;
          pixelk[Wt3_tmp] = b;
          x[Wt3_tmp] = (muDoubleScalarFloor(b) == b);
        }
        eml_find(x, idxk_data, g_I2);
        Wt3_tmp = g_I2[1];
        for (i1 = 0; i1 < Wt3_tmp; i1++) {
          for (b_I2 = 0; b_I2 < b_loop_ub; b_I2++) {
            for (i2 = 0; i2 < loop_ub; i2++) {
              real_T d;
              real_T d1;
              b = pixeli[idxi_data[i2] - 1];
              if (b != muDoubleScalarFloor(b)) {
                emlrtIntegerCheckR2012b(b, &emlrtDCI, (emlrtCTX)sp);
              }
              if ((int32_T)b > I2->size[0]) {
                emlrtDynamicBoundsCheckR2012b((int32_T)b, 1, I2->size[0],
                                              &j_emlrtBCI, (emlrtCTX)sp);
              }
              d = pixelj[idxj_data[b_I2] - 1];
              if (d != muDoubleScalarFloor(d)) {
                emlrtIntegerCheckR2012b(d, &b_emlrtDCI, (emlrtCTX)sp);
              }
              if ((int32_T)d > I2->size[1]) {
                emlrtDynamicBoundsCheckR2012b((int32_T)d, 1, I2->size[1],
                                              &k_emlrtBCI, (emlrtCTX)sp);
              }
              d1 = pixelk[idxk_data[i1] - 1];
              if (d1 != muDoubleScalarFloor(d1)) {
                emlrtIntegerCheckR2012b(d1, &c_emlrtDCI, (emlrtCTX)sp);
              }
              if ((int32_T)d1 > I2->size[2]) {
                emlrtDynamicBoundsCheckR2012b((int32_T)d1, 1, I2->size[2],
                                              &l_emlrtBCI, (emlrtCTX)sp);
              }
              tmp_data[(i2 + loop_ub * b_I2) + loop_ub * b_loop_ub * i1] =
                  I2_data[(((int32_T)b + I2->size[0] * ((int32_T)d - 1)) +
                           I2->size[0] * I2->size[1] * ((int32_T)d1 - 1)) -
                          1];
            }
          }
        }
        i1 = idxi_size[1];
        b_I2 = idxj_size[1];
        Wt3_tmp = g_I2[1];
        A_size[0] = loop_ub;
        A_size[1] = b_loop_ub;
        A_size[2] = g_I2[1];
        for (i2 = 0; i2 < Wt3_tmp; i2++) {
          for (i3 = 0; i3 < b_loop_ub; i3++) {
            for (i4 = 0; i4 < loop_ub; i4++) {
              A_data[(i4 + i1 * i3) + i1 * b_I2 * i2] =
                  tmp_data[(i4 + loop_ub * i3) + loop_ub * b_loop_ub * i2] *
                  Wt3[((idxi_data[i4] + 5 * (idxj_data[i3] - 1)) +
                       25 * (idxk_data[i2] - 1)) -
                      1];
            }
          }
        }
        i1 = A_size[0] * A_size[1] * A_size[2];
        if (i1 == 0) {
          b = 0.0;
        } else {
          b = sumColumnB(A_data, i1);
        }
        if (((int32_T)(b_i + 1U) < 1) ||
            ((int32_T)(b_i + 1U) > IResult->size[0])) {
          emlrtDynamicBoundsCheckR2012b((int32_T)(b_i + 1U), 1,
                                        IResult->size[0], &m_emlrtBCI,
                                        (emlrtCTX)sp);
        }
        if (((int32_T)(j + 1U) < 1) || ((int32_T)(j + 1U) > IResult->size[1])) {
          emlrtDynamicBoundsCheckR2012b((int32_T)(j + 1U), 1, IResult->size[1],
                                        &n_emlrtBCI, (emlrtCTX)sp);
        }
        if (((int32_T)(k + 1U) < 1) || ((int32_T)(k + 1U) > IResult->size[2])) {
          emlrtDynamicBoundsCheckR2012b((int32_T)(k + 1U), 1, IResult->size[2],
                                        &o_emlrtBCI, (emlrtCTX)sp);
        }
        IResult_data[(b_i + IResult->size[0] * j) +
                     IResult->size[0] * IResult->size[1] * k] = 8.0 * b;
        if (*emlrtBreakCheckR2012bFlagVar != 0) {
          emlrtBreakCheckR2012b((emlrtCTX)sp);
        }
      }
      if (*emlrtBreakCheckR2012bFlagVar != 0) {
        emlrtBreakCheckR2012b((emlrtCTX)sp);
      }
    }
    if (*emlrtBreakCheckR2012bFlagVar != 0) {
      emlrtBreakCheckR2012b((emlrtCTX)sp);
    }
  }
  emxFree_real_T(sp, &I2);
  emlrtHeapReferenceStackLeaveFcnR2012b((emlrtCTX)sp);
}

/* End of code generation (GPExpand.c) */
