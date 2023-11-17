/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 *
 * GPReduce.c
 *
 * Code generation for function 'GPReduce'
 *
 */

/* Include files */
#include "GPReduce.h"
#include "GPReduce_data.h"
#include "GPReduce_emxutil.h"
#include "GPReduce_types.h"
#include "rt_nonfinite.h"
#include "sumMatrixIncludeNaN.h"
#include "mwmathutil.h"

/* Variable Definitions */
static emlrtBCInfo emlrtBCI = {
    -1,                                                        /* iFirst */
    -1,                                                        /* iLast */
    45,                                                        /* lineNo */
    8,                                                         /* colNo */
    "I2",                                                      /* aName */
    "GPReduce",                                                /* fName */
    "/Users/natasha/Downloads/TBM_software/TBM/GP/GPReduce.m", /* pName */
    0                                                          /* checkKind */
};

static emlrtBCInfo b_emlrtBCI = {
    -1,                                                        /* iFirst */
    -1,                                                        /* iLast */
    45,                                                        /* lineNo */
    19,                                                        /* colNo */
    "I2",                                                      /* aName */
    "GPReduce",                                                /* fName */
    "/Users/natasha/Downloads/TBM_software/TBM/GP/GPReduce.m", /* pName */
    0                                                          /* checkKind */
};

static emlrtBCInfo c_emlrtBCI = {
    -1,                                                        /* iFirst */
    -1,                                                        /* iLast */
    45,                                                        /* lineNo */
    30,                                                        /* colNo */
    "I2",                                                      /* aName */
    "GPReduce",                                                /* fName */
    "/Users/natasha/Downloads/TBM_software/TBM/GP/GPReduce.m", /* pName */
    0                                                          /* checkKind */
};

static emlrtECInfo emlrtECI = {
    -1,                                                       /* nDims */
    45,                                                       /* lineNo */
    3,                                                        /* colNo */
    "GPReduce",                                               /* fName */
    "/Users/natasha/Downloads/TBM_software/TBM/GP/GPReduce.m" /* pName */
};

static emlrtECInfo b_emlrtECI = {
    -1,                                                       /* nDims */
    46,                                                       /* lineNo */
    3,                                                        /* colNo */
    "GPReduce",                                               /* fName */
    "/Users/natasha/Downloads/TBM_software/TBM/GP/GPReduce.m" /* pName */
};

static emlrtECInfo c_emlrtECI = {
    -1,                                                       /* nDims */
    46,                                                       /* lineNo */
    23,                                                       /* colNo */
    "GPReduce",                                               /* fName */
    "/Users/natasha/Downloads/TBM_software/TBM/GP/GPReduce.m" /* pName */
};

static emlrtBCInfo d_emlrtBCI = {
    -1,                                                        /* iFirst */
    -1,                                                        /* iLast */
    46,                                                        /* lineNo */
    58,                                                        /* colNo */
    "I2",                                                      /* aName */
    "GPReduce",                                                /* fName */
    "/Users/natasha/Downloads/TBM_software/TBM/GP/GPReduce.m", /* pName */
    0                                                          /* checkKind */
};

static emlrtBCInfo e_emlrtBCI = {
    -1,                                                        /* iFirst */
    -1,                                                        /* iLast */
    46,                                                        /* lineNo */
    46,                                                        /* colNo */
    "I2",                                                      /* aName */
    "GPReduce",                                                /* fName */
    "/Users/natasha/Downloads/TBM_software/TBM/GP/GPReduce.m", /* pName */
    0                                                          /* checkKind */
};

static emlrtECInfo d_emlrtECI = {
    -1,                                                       /* nDims */
    46,                                                       /* lineNo */
    43,                                                       /* colNo */
    "GPReduce",                                               /* fName */
    "/Users/natasha/Downloads/TBM_software/TBM/GP/GPReduce.m" /* pName */
};

static emlrtBCInfo f_emlrtBCI = {
    -1,                                                        /* iFirst */
    -1,                                                        /* iLast */
    46,                                                        /* lineNo */
    86,                                                        /* colNo */
    "I2",                                                      /* aName */
    "GPReduce",                                                /* fName */
    "/Users/natasha/Downloads/TBM_software/TBM/GP/GPReduce.m", /* pName */
    0                                                          /* checkKind */
};

static emlrtBCInfo g_emlrtBCI = {
    -1,                                                        /* iFirst */
    -1,                                                        /* iLast */
    46,                                                        /* lineNo */
    72,                                                        /* colNo */
    "I2",                                                      /* aName */
    "GPReduce",                                                /* fName */
    "/Users/natasha/Downloads/TBM_software/TBM/GP/GPReduce.m", /* pName */
    0                                                          /* checkKind */
};

static emlrtECInfo e_emlrtECI = {
    -1,                                                       /* nDims */
    46,                                                       /* lineNo */
    69,                                                       /* colNo */
    "GPReduce",                                               /* fName */
    "/Users/natasha/Downloads/TBM_software/TBM/GP/GPReduce.m" /* pName */
};

static emlrtECInfo f_emlrtECI = {
    -1,                                                       /* nDims */
    47,                                                       /* lineNo */
    3,                                                        /* colNo */
    "GPReduce",                                               /* fName */
    "/Users/natasha/Downloads/TBM_software/TBM/GP/GPReduce.m" /* pName */
};

static emlrtECInfo g_emlrtECI = {
    -1,                                                       /* nDims */
    47,                                                       /* lineNo */
    23,                                                       /* colNo */
    "GPReduce",                                               /* fName */
    "/Users/natasha/Downloads/TBM_software/TBM/GP/GPReduce.m" /* pName */
};

static emlrtBCInfo h_emlrtBCI = {
    -1,                                                        /* iFirst */
    -1,                                                        /* iLast */
    47,                                                        /* lineNo */
    60,                                                        /* colNo */
    "I2",                                                      /* aName */
    "GPReduce",                                                /* fName */
    "/Users/natasha/Downloads/TBM_software/TBM/GP/GPReduce.m", /* pName */
    0                                                          /* checkKind */
};

static emlrtBCInfo i_emlrtBCI = {
    -1,                                                        /* iFirst */
    -1,                                                        /* iLast */
    47,                                                        /* lineNo */
    48,                                                        /* colNo */
    "I2",                                                      /* aName */
    "GPReduce",                                                /* fName */
    "/Users/natasha/Downloads/TBM_software/TBM/GP/GPReduce.m", /* pName */
    0                                                          /* checkKind */
};

static emlrtECInfo h_emlrtECI = {
    -1,                                                       /* nDims */
    47,                                                       /* lineNo */
    43,                                                       /* colNo */
    "GPReduce",                                               /* fName */
    "/Users/natasha/Downloads/TBM_software/TBM/GP/GPReduce.m" /* pName */
};

static emlrtBCInfo j_emlrtBCI = {
    -1,                                                        /* iFirst */
    -1,                                                        /* iLast */
    47,                                                        /* lineNo */
    88,                                                        /* colNo */
    "I2",                                                      /* aName */
    "GPReduce",                                                /* fName */
    "/Users/natasha/Downloads/TBM_software/TBM/GP/GPReduce.m", /* pName */
    0                                                          /* checkKind */
};

static emlrtBCInfo k_emlrtBCI = {
    -1,                                                        /* iFirst */
    -1,                                                        /* iLast */
    47,                                                        /* lineNo */
    74,                                                        /* colNo */
    "I2",                                                      /* aName */
    "GPReduce",                                                /* fName */
    "/Users/natasha/Downloads/TBM_software/TBM/GP/GPReduce.m", /* pName */
    0                                                          /* checkKind */
};

static emlrtECInfo i_emlrtECI = {
    -1,                                                       /* nDims */
    47,                                                       /* lineNo */
    69,                                                       /* colNo */
    "GPReduce",                                               /* fName */
    "/Users/natasha/Downloads/TBM_software/TBM/GP/GPReduce.m" /* pName */
};

static emlrtECInfo j_emlrtECI = {
    -1,                                                       /* nDims */
    48,                                                       /* lineNo */
    3,                                                        /* colNo */
    "GPReduce",                                               /* fName */
    "/Users/natasha/Downloads/TBM_software/TBM/GP/GPReduce.m" /* pName */
};

static emlrtECInfo k_emlrtECI = {
    -1,                                                       /* nDims */
    48,                                                       /* lineNo */
    23,                                                       /* colNo */
    "GPReduce",                                               /* fName */
    "/Users/natasha/Downloads/TBM_software/TBM/GP/GPReduce.m" /* pName */
};

static emlrtBCInfo l_emlrtBCI = {
    -1,                                                        /* iFirst */
    -1,                                                        /* iLast */
    48,                                                        /* lineNo */
    62,                                                        /* colNo */
    "I2",                                                      /* aName */
    "GPReduce",                                                /* fName */
    "/Users/natasha/Downloads/TBM_software/TBM/GP/GPReduce.m", /* pName */
    0                                                          /* checkKind */
};

static emlrtBCInfo m_emlrtBCI = {
    -1,                                                        /* iFirst */
    -1,                                                        /* iLast */
    48,                                                        /* lineNo */
    50,                                                        /* colNo */
    "I2",                                                      /* aName */
    "GPReduce",                                                /* fName */
    "/Users/natasha/Downloads/TBM_software/TBM/GP/GPReduce.m", /* pName */
    0                                                          /* checkKind */
};

static emlrtECInfo l_emlrtECI = {
    -1,                                                       /* nDims */
    48,                                                       /* lineNo */
    43,                                                       /* colNo */
    "GPReduce",                                               /* fName */
    "/Users/natasha/Downloads/TBM_software/TBM/GP/GPReduce.m" /* pName */
};

static emlrtBCInfo n_emlrtBCI = {
    -1,                                                        /* iFirst */
    -1,                                                        /* iLast */
    48,                                                        /* lineNo */
    90,                                                        /* colNo */
    "I2",                                                      /* aName */
    "GPReduce",                                                /* fName */
    "/Users/natasha/Downloads/TBM_software/TBM/GP/GPReduce.m", /* pName */
    0                                                          /* checkKind */
};

static emlrtBCInfo o_emlrtBCI = {
    -1,                                                        /* iFirst */
    -1,                                                        /* iLast */
    48,                                                        /* lineNo */
    76,                                                        /* colNo */
    "I2",                                                      /* aName */
    "GPReduce",                                                /* fName */
    "/Users/natasha/Downloads/TBM_software/TBM/GP/GPReduce.m", /* pName */
    0                                                          /* checkKind */
};

static emlrtECInfo m_emlrtECI = {
    -1,                                                       /* nDims */
    48,                                                       /* lineNo */
    69,                                                       /* colNo */
    "GPReduce",                                               /* fName */
    "/Users/natasha/Downloads/TBM_software/TBM/GP/GPReduce.m" /* pName */
};

static emlrtBCInfo p_emlrtBCI = {
    -1,                                                        /* iFirst */
    -1,                                                        /* iLast */
    54,                                                        /* lineNo */
    12,                                                        /* colNo */
    "I",                                                       /* aName */
    "GPReduce",                                                /* fName */
    "/Users/natasha/Downloads/TBM_software/TBM/GP/GPReduce.m", /* pName */
    0                                                          /* checkKind */
};

static emlrtBCInfo q_emlrtBCI = {
    -1,                                                        /* iFirst */
    -1,                                                        /* iLast */
    54,                                                        /* lineNo */
    20,                                                        /* colNo */
    "I",                                                       /* aName */
    "GPReduce",                                                /* fName */
    "/Users/natasha/Downloads/TBM_software/TBM/GP/GPReduce.m", /* pName */
    0                                                          /* checkKind */
};

static emlrtBCInfo r_emlrtBCI = {
    -1,                                                        /* iFirst */
    -1,                                                        /* iLast */
    54,                                                        /* lineNo */
    28,                                                        /* colNo */
    "I",                                                       /* aName */
    "GPReduce",                                                /* fName */
    "/Users/natasha/Downloads/TBM_software/TBM/GP/GPReduce.m", /* pName */
    0                                                          /* checkKind */
};

static emlrtBCInfo s_emlrtBCI = {
    -1,                                                        /* iFirst */
    -1,                                                        /* iLast */
    55,                                                        /* lineNo */
    14,                                                        /* colNo */
    "IResult",                                                 /* aName */
    "GPReduce",                                                /* fName */
    "/Users/natasha/Downloads/TBM_software/TBM/GP/GPReduce.m", /* pName */
    0                                                          /* checkKind */
};

static emlrtBCInfo t_emlrtBCI = {
    -1,                                                        /* iFirst */
    -1,                                                        /* iLast */
    55,                                                        /* lineNo */
    18,                                                        /* colNo */
    "IResult",                                                 /* aName */
    "GPReduce",                                                /* fName */
    "/Users/natasha/Downloads/TBM_software/TBM/GP/GPReduce.m", /* pName */
    0                                                          /* checkKind */
};

static emlrtBCInfo u_emlrtBCI = {
    -1,                                                        /* iFirst */
    -1,                                                        /* iLast */
    55,                                                        /* lineNo */
    22,                                                        /* colNo */
    "IResult",                                                 /* aName */
    "GPReduce",                                                /* fName */
    "/Users/natasha/Downloads/TBM_software/TBM/GP/GPReduce.m", /* pName */
    0                                                          /* checkKind */
};

static emlrtDCInfo emlrtDCI = {
    44,                                                        /* lineNo */
    14,                                                        /* colNo */
    "GPReduce",                                                /* fName */
    "/Users/natasha/Downloads/TBM_software/TBM/GP/GPReduce.m", /* pName */
    1                                                          /* checkKind */
};

static emlrtRTEInfo emlrtRTEI = {
    10,                                                       /* lineNo */
    1,                                                        /* colNo */
    "GPReduce",                                               /* fName */
    "/Users/natasha/Downloads/TBM_software/TBM/GP/GPReduce.m" /* pName */
};

static emlrtRTEInfo b_emlrtRTEI = {
    44,                                                       /* lineNo */
    3,                                                        /* colNo */
    "GPReduce",                                               /* fName */
    "/Users/natasha/Downloads/TBM_software/TBM/GP/GPReduce.m" /* pName */
};

static emlrtRTEInfo c_emlrtRTEI = {
    46,                                                       /* lineNo */
    13,                                                       /* colNo */
    "GPReduce",                                               /* fName */
    "/Users/natasha/Downloads/TBM_software/TBM/GP/GPReduce.m" /* pName */
};

static emlrtRTEInfo d_emlrtRTEI = {
    46,                                                       /* lineNo */
    33,                                                       /* colNo */
    "GPReduce",                                               /* fName */
    "/Users/natasha/Downloads/TBM_software/TBM/GP/GPReduce.m" /* pName */
};

static emlrtRTEInfo e_emlrtRTEI = {
    46,                                                       /* lineNo */
    55,                                                       /* colNo */
    "GPReduce",                                               /* fName */
    "/Users/natasha/Downloads/TBM_software/TBM/GP/GPReduce.m" /* pName */
};

static emlrtRTEInfo f_emlrtRTEI = {
    46,                                                       /* lineNo */
    83,                                                       /* colNo */
    "GPReduce",                                               /* fName */
    "/Users/natasha/Downloads/TBM_software/TBM/GP/GPReduce.m" /* pName */
};

static emlrtRTEInfo g_emlrtRTEI = {
    47,                                                       /* lineNo */
    13,                                                       /* colNo */
    "GPReduce",                                               /* fName */
    "/Users/natasha/Downloads/TBM_software/TBM/GP/GPReduce.m" /* pName */
};

static emlrtRTEInfo h_emlrtRTEI = {
    47,                                                       /* lineNo */
    33,                                                       /* colNo */
    "GPReduce",                                               /* fName */
    "/Users/natasha/Downloads/TBM_software/TBM/GP/GPReduce.m" /* pName */
};

static emlrtRTEInfo i_emlrtRTEI = {
    47,                                                       /* lineNo */
    55,                                                       /* colNo */
    "GPReduce",                                               /* fName */
    "/Users/natasha/Downloads/TBM_software/TBM/GP/GPReduce.m" /* pName */
};

static emlrtRTEInfo j_emlrtRTEI = {
    47,                                                       /* lineNo */
    83,                                                       /* colNo */
    "GPReduce",                                               /* fName */
    "/Users/natasha/Downloads/TBM_software/TBM/GP/GPReduce.m" /* pName */
};

static emlrtRTEInfo k_emlrtRTEI = {
    48,                                                       /* lineNo */
    13,                                                       /* colNo */
    "GPReduce",                                               /* fName */
    "/Users/natasha/Downloads/TBM_software/TBM/GP/GPReduce.m" /* pName */
};

static emlrtRTEInfo l_emlrtRTEI = {
    48,                                                       /* lineNo */
    33,                                                       /* colNo */
    "GPReduce",                                               /* fName */
    "/Users/natasha/Downloads/TBM_software/TBM/GP/GPReduce.m" /* pName */
};

static emlrtRTEInfo m_emlrtRTEI = {
    48,                                                       /* lineNo */
    55,                                                       /* colNo */
    "GPReduce",                                               /* fName */
    "/Users/natasha/Downloads/TBM_software/TBM/GP/GPReduce.m" /* pName */
};

static emlrtRTEInfo n_emlrtRTEI = {
    48,                                                       /* lineNo */
    83,                                                       /* colNo */
    "GPReduce",                                               /* fName */
    "/Users/natasha/Downloads/TBM_software/TBM/GP/GPReduce.m" /* pName */
};

/* Function Definitions */
void GPReduce(const emlrtStack *sp, const emxArray_real_T *b_I,
              emxArray_real_T *IResult)
{
  static const real_T dv[5] = {0.05, 0.25, 0.4, 0.25, 0.05};
  emxArray_real_T *I2;
  emxArray_real_T *c_I2;
  emxArray_real_T *e_I2;
  emxArray_real_T *f_I2;
  real_T Wt3[125];
  const real_T *I_data;
  real_T *I2_data;
  real_T *IResult_data;
  real_T *b_I2_data;
  int32_T b_I2[3];
  int32_T g_I2[2];
  int32_T iv[2];
  int32_T Wt3_tmp;
  int32_T b_i;
  int32_T b_loop_ub;
  int32_T d_I2;
  int32_T i;
  int32_T i1;
  int32_T i2;
  int32_T i3;
  int32_T i4;
  int32_T i5;
  int32_T i6;
  int32_T k;
  int32_T loop_ub;
  int32_T newdim_idx_0;
  int32_T newdim_idx_1;
  int32_T newdim_idx_2;
  I_data = b_I->data;
  emlrtHeapReferenceStackEnterFcnR2012b((emlrtCTX)sp);
  /* %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% */
  /*  Reduce an image applying Gaussian Pyramid. */
  /* %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% */
  newdim_idx_0 = (int32_T)muDoubleScalarCeil((real_T)b_I->size[0] * 0.5);
  newdim_idx_1 = (int32_T)muDoubleScalarCeil((real_T)b_I->size[1] * 0.5);
  newdim_idx_2 = (int32_T)muDoubleScalarCeil((real_T)b_I->size[2] * 0.5);
  i = IResult->size[0] * IResult->size[1] * IResult->size[2];
  IResult->size[0] = newdim_idx_0;
  IResult->size[1] = newdim_idx_1;
  IResult->size[2] = newdim_idx_2;
  emxEnsureCapacity_real_T(sp, IResult, i, &emlrtRTEI);
  IResult_data = IResult->data;
  loop_ub = newdim_idx_0 * newdim_idx_1 * newdim_idx_2;
  for (i = 0; i < loop_ub; i++) {
    IResult_data[i] = 0.0;
  }
  /*  Initialize the array in the beginning .. */
  for (i = 0; i < 125; i++) {
    Wt3[i] = 1.0;
  }
  for (b_i = 0; b_i < 5; b_i++) {
    real_T b;
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
  /* 		%% Pad the boundaries. */
  if ((real_T)b_I->size[0] + 4.0 != b_I->size[0] + 4) {
    emlrtIntegerCheckR2012b((real_T)b_I->size[0] + 4.0, &emlrtDCI,
                            (emlrtCTX)sp);
  }
  if ((real_T)b_I->size[1] + 4.0 != b_I->size[1] + 4) {
    emlrtIntegerCheckR2012b((real_T)b_I->size[1] + 4.0, &emlrtDCI,
                            (emlrtCTX)sp);
  }
  if ((real_T)b_I->size[2] + 4.0 != b_I->size[2] + 4) {
    emlrtIntegerCheckR2012b((real_T)b_I->size[2] + 4.0, &emlrtDCI,
                            (emlrtCTX)sp);
  }
  emxInit_real_T(sp, &I2, 3, &b_emlrtRTEI);
  i = I2->size[0] * I2->size[1] * I2->size[2];
  I2->size[0] = (int32_T)(uint32_T)((real_T)b_I->size[0] + 4.0);
  I2->size[1] = (int32_T)(uint32_T)((real_T)b_I->size[1] + 4.0);
  I2->size[2] = (int32_T)(uint32_T)((real_T)b_I->size[2] + 4.0);
  emxEnsureCapacity_real_T(sp, I2, i, &b_emlrtRTEI);
  I2_data = I2->data;
  loop_ub = (int32_T)(uint32_T)((real_T)b_I->size[0] + 4.0) *
            (int32_T)(uint32_T)((real_T)b_I->size[1] + 4.0) *
            (int32_T)(uint32_T)((real_T)b_I->size[2] + 4.0);
  for (i = 0; i < loop_ub; i++) {
    I2_data[i] = 0.0;
  }
  if (b_I->size[0] + 2U < 3U) {
    i = 0;
    i1 = 0;
  } else {
    i = 2;
    if (((int32_T)(b_I->size[0] + 2U) < 1) ||
        ((int32_T)(b_I->size[0] + 2U) >
         (int32_T)(uint32_T)((real_T)b_I->size[0] + 4.0))) {
      emlrtDynamicBoundsCheckR2012b(
          (int32_T)(b_I->size[0] + 2U), 1,
          (int32_T)(uint32_T)((real_T)b_I->size[0] + 4.0), &emlrtBCI,
          (emlrtCTX)sp);
    }
    i1 = (int32_T)(b_I->size[0] + 2U);
  }
  if (b_I->size[1] + 2U < 3U) {
    i2 = 0;
    i3 = 0;
  } else {
    i2 = 2;
    if (((int32_T)(b_I->size[1] + 2U) < 1) ||
        ((int32_T)(b_I->size[1] + 2U) >
         (int32_T)(uint32_T)((real_T)b_I->size[1] + 4.0))) {
      emlrtDynamicBoundsCheckR2012b(
          (int32_T)(b_I->size[1] + 2U), 1,
          (int32_T)(uint32_T)((real_T)b_I->size[1] + 4.0), &b_emlrtBCI,
          (emlrtCTX)sp);
    }
    i3 = (int32_T)(b_I->size[1] + 2U);
  }
  if (b_I->size[2] + 2U < 3U) {
    i4 = 0;
    i5 = 0;
  } else {
    i4 = 2;
    if (((int32_T)(b_I->size[2] + 2U) < 1) ||
        ((int32_T)(b_I->size[2] + 2U) >
         (int32_T)(uint32_T)((real_T)b_I->size[2] + 4.0))) {
      emlrtDynamicBoundsCheckR2012b(
          (int32_T)(b_I->size[2] + 2U), 1,
          (int32_T)(uint32_T)((real_T)b_I->size[2] + 4.0), &c_emlrtBCI,
          (emlrtCTX)sp);
    }
    i5 = (int32_T)(b_I->size[2] + 2U);
  }
  b_I2[0] = i1 - i;
  b_I2[1] = i3 - i2;
  b_I2[2] = i5 - i4;
  emlrtSubAssignSizeCheckR2012b(&b_I2[0], 3, &b_I->size[0], 3, &emlrtECI,
                                (emlrtCTX)sp);
  loop_ub = b_I->size[2];
  for (i1 = 0; i1 < loop_ub; i1++) {
    b_loop_ub = b_I->size[1];
    for (i3 = 0; i3 < b_loop_ub; i3++) {
      Wt3_tmp = b_I->size[0];
      for (i5 = 0; i5 < Wt3_tmp; i5++) {
        I2_data[((i + i5) + I2->size[0] * (i2 + i3)) +
                I2->size[0] * I2->size[1] * (i4 + i1)] =
            I_data[(i5 + b_I->size[0] * i3) + b_I->size[0] * b_I->size[1] * i1];
      }
    }
  }
  emxInit_real_T(sp, &c_I2, 3, &c_emlrtRTEI);
  loop_ub = I2->size[1];
  b_loop_ub = I2->size[2];
  i = c_I2->size[0] * c_I2->size[1] * c_I2->size[2];
  c_I2->size[0] = 1;
  c_I2->size[1] = I2->size[1];
  c_I2->size[2] = I2->size[2];
  emxEnsureCapacity_real_T(sp, c_I2, i, &c_emlrtRTEI);
  b_I2_data = c_I2->data;
  for (i = 0; i < b_loop_ub; i++) {
    for (i1 = 0; i1 < loop_ub; i1++) {
      b_I2_data[i1 + c_I2->size[1] * i] =
          I2_data[(I2->size[0] * i1 + I2->size[0] * I2->size[1] * i) + 2];
    }
  }
  b_I2[0] = 1;
  b_I2[1] = I2->size[1];
  b_I2[2] = I2->size[2];
  emlrtSubAssignSizeCheckR2012b(&b_I2[0], 3, &c_I2->size[0], 3, &b_emlrtECI,
                                (emlrtCTX)sp);
  Wt3_tmp = I2->size[1] - 1;
  loop_ub = I2->size[2] - 1;
  i = c_I2->size[0] * c_I2->size[1] * c_I2->size[2];
  c_I2->size[0] = 1;
  c_I2->size[1] = I2->size[1];
  c_I2->size[2] = I2->size[2];
  emxEnsureCapacity_real_T(sp, c_I2, i, &c_emlrtRTEI);
  b_I2_data = c_I2->data;
  for (i = 0; i <= loop_ub; i++) {
    for (i1 = 0; i1 <= Wt3_tmp; i1++) {
      b_I2_data[i1 + c_I2->size[1] * i] =
          I2_data[(I2->size[0] * i1 + I2->size[0] * I2->size[1] * i) + 2];
    }
  }
  loop_ub = c_I2->size[2];
  for (i = 0; i < loop_ub; i++) {
    b_loop_ub = c_I2->size[1];
    for (i1 = 0; i1 < b_loop_ub; i1++) {
      I2_data[I2->size[0] * i1 + I2->size[0] * I2->size[1] * i] =
          b_I2_data[i1 + c_I2->size[1] * i];
    }
  }
  loop_ub = I2->size[1];
  b_loop_ub = I2->size[2];
  i = c_I2->size[0] * c_I2->size[1] * c_I2->size[2];
  c_I2->size[0] = 1;
  c_I2->size[1] = I2->size[1];
  c_I2->size[2] = I2->size[2];
  emxEnsureCapacity_real_T(sp, c_I2, i, &d_emlrtRTEI);
  b_I2_data = c_I2->data;
  for (i = 0; i < b_loop_ub; i++) {
    for (i1 = 0; i1 < loop_ub; i1++) {
      b_I2_data[i1 + c_I2->size[1] * i] =
          I2_data[(I2->size[0] * i1 + I2->size[0] * I2->size[1] * i) + 2];
    }
  }
  b_I2[0] = 1;
  b_I2[1] = I2->size[1];
  b_I2[2] = I2->size[2];
  emlrtSubAssignSizeCheckR2012b(&b_I2[0], 3, &c_I2->size[0], 3, &c_emlrtECI,
                                (emlrtCTX)sp);
  Wt3_tmp = I2->size[1] - 1;
  loop_ub = I2->size[2] - 1;
  i = c_I2->size[0] * c_I2->size[1] * c_I2->size[2];
  c_I2->size[0] = 1;
  c_I2->size[1] = I2->size[1];
  c_I2->size[2] = I2->size[2];
  emxEnsureCapacity_real_T(sp, c_I2, i, &d_emlrtRTEI);
  b_I2_data = c_I2->data;
  for (i = 0; i <= loop_ub; i++) {
    for (i1 = 0; i1 <= Wt3_tmp; i1++) {
      b_I2_data[i1 + c_I2->size[1] * i] =
          I2_data[(I2->size[0] * i1 + I2->size[0] * I2->size[1] * i) + 2];
    }
  }
  loop_ub = c_I2->size[2];
  for (i = 0; i < loop_ub; i++) {
    b_loop_ub = c_I2->size[1];
    for (i1 = 0; i1 < b_loop_ub; i1++) {
      I2_data[I2->size[0] * i1 + I2->size[0] * I2->size[1] * i] =
          b_I2_data[i1 + c_I2->size[1] * i];
    }
  }
  if (I2->size[0] < 1) {
    emlrtDynamicBoundsCheckR2012b(I2->size[0], 1, I2->size[0], &e_emlrtBCI,
                                  (emlrtCTX)sp);
  }
  if ((I2->size[0] - 2 < 1) || (I2->size[0] - 2 > I2->size[0])) {
    emlrtDynamicBoundsCheckR2012b(I2->size[0] - 2, 1, I2->size[0], &d_emlrtBCI,
                                  (emlrtCTX)sp);
  }
  loop_ub = I2->size[1];
  b_loop_ub = I2->size[2];
  i = c_I2->size[0] * c_I2->size[1] * c_I2->size[2];
  c_I2->size[0] = 1;
  c_I2->size[1] = I2->size[1];
  c_I2->size[2] = I2->size[2];
  emxEnsureCapacity_real_T(sp, c_I2, i, &e_emlrtRTEI);
  b_I2_data = c_I2->data;
  for (i = 0; i < b_loop_ub; i++) {
    for (i1 = 0; i1 < loop_ub; i1++) {
      b_I2_data[i1 + c_I2->size[1] * i] =
          I2_data[((I2->size[0] + I2->size[0] * i1) +
                   I2->size[0] * I2->size[1] * i) -
                  3];
    }
  }
  b_I2[0] = 1;
  b_I2[1] = I2->size[1];
  b_I2[2] = I2->size[2];
  emlrtSubAssignSizeCheckR2012b(&b_I2[0], 3, &c_I2->size[0], 3, &d_emlrtECI,
                                (emlrtCTX)sp);
  Wt3_tmp = I2->size[1] - 1;
  loop_ub = I2->size[2] - 1;
  d_I2 = I2->size[0] - 1;
  i = c_I2->size[0] * c_I2->size[1] * c_I2->size[2];
  c_I2->size[0] = 1;
  c_I2->size[1] = I2->size[1];
  c_I2->size[2] = I2->size[2];
  emxEnsureCapacity_real_T(sp, c_I2, i, &e_emlrtRTEI);
  b_I2_data = c_I2->data;
  for (i = 0; i <= loop_ub; i++) {
    for (i1 = 0; i1 <= Wt3_tmp; i1++) {
      b_I2_data[i1 + c_I2->size[1] * i] =
          I2_data[((I2->size[0] + I2->size[0] * i1) +
                   I2->size[0] * I2->size[1] * i) -
                  3];
    }
  }
  loop_ub = c_I2->size[2];
  for (i = 0; i < loop_ub; i++) {
    b_loop_ub = c_I2->size[1];
    for (i1 = 0; i1 < b_loop_ub; i1++) {
      I2_data[(d_I2 + I2->size[0] * i1) + I2->size[0] * I2->size[1] * i] =
          b_I2_data[i1 + c_I2->size[1] * i];
    }
  }
  if ((I2->size[0] - 1 < 1) || (I2->size[0] - 1 > I2->size[0])) {
    emlrtDynamicBoundsCheckR2012b(I2->size[0] - 1, 1, I2->size[0], &g_emlrtBCI,
                                  (emlrtCTX)sp);
  }
  if ((I2->size[0] - 2 < 1) || (I2->size[0] - 2 > I2->size[0])) {
    emlrtDynamicBoundsCheckR2012b(I2->size[0] - 2, 1, I2->size[0], &f_emlrtBCI,
                                  (emlrtCTX)sp);
  }
  loop_ub = I2->size[1];
  b_loop_ub = I2->size[2];
  i = c_I2->size[0] * c_I2->size[1] * c_I2->size[2];
  c_I2->size[0] = 1;
  c_I2->size[1] = I2->size[1];
  c_I2->size[2] = I2->size[2];
  emxEnsureCapacity_real_T(sp, c_I2, i, &f_emlrtRTEI);
  b_I2_data = c_I2->data;
  for (i = 0; i < b_loop_ub; i++) {
    for (i1 = 0; i1 < loop_ub; i1++) {
      b_I2_data[i1 + c_I2->size[1] * i] =
          I2_data[((I2->size[0] + I2->size[0] * i1) +
                   I2->size[0] * I2->size[1] * i) -
                  3];
    }
  }
  b_I2[0] = 1;
  b_I2[1] = I2->size[1];
  b_I2[2] = I2->size[2];
  emlrtSubAssignSizeCheckR2012b(&b_I2[0], 3, &c_I2->size[0], 3, &e_emlrtECI,
                                (emlrtCTX)sp);
  Wt3_tmp = I2->size[1] - 1;
  loop_ub = I2->size[2] - 1;
  d_I2 = I2->size[0] - 2;
  i = c_I2->size[0] * c_I2->size[1] * c_I2->size[2];
  c_I2->size[0] = 1;
  c_I2->size[1] = I2->size[1];
  c_I2->size[2] = I2->size[2];
  emxEnsureCapacity_real_T(sp, c_I2, i, &f_emlrtRTEI);
  b_I2_data = c_I2->data;
  for (i = 0; i <= loop_ub; i++) {
    for (i1 = 0; i1 <= Wt3_tmp; i1++) {
      b_I2_data[i1 + c_I2->size[1] * i] =
          I2_data[((I2->size[0] + I2->size[0] * i1) +
                   I2->size[0] * I2->size[1] * i) -
                  3];
    }
  }
  loop_ub = c_I2->size[2];
  for (i = 0; i < loop_ub; i++) {
    b_loop_ub = c_I2->size[1];
    for (i1 = 0; i1 < b_loop_ub; i1++) {
      I2_data[(d_I2 + I2->size[0] * i1) + I2->size[0] * I2->size[1] * i] =
          b_I2_data[i1 + c_I2->size[1] * i];
    }
  }
  emxFree_real_T(sp, &c_I2);
  emxInit_real_T(sp, &e_I2, 3, &g_emlrtRTEI);
  loop_ub = I2->size[0];
  b_loop_ub = I2->size[2];
  i = e_I2->size[0] * e_I2->size[1] * e_I2->size[2];
  e_I2->size[0] = I2->size[0];
  e_I2->size[1] = 1;
  e_I2->size[2] = I2->size[2];
  emxEnsureCapacity_real_T(sp, e_I2, i, &g_emlrtRTEI);
  b_I2_data = e_I2->data;
  for (i = 0; i < b_loop_ub; i++) {
    for (i1 = 0; i1 < loop_ub; i1++) {
      b_I2_data[i1 + e_I2->size[0] * i] =
          I2_data[(i1 + I2->size[0] * 2) + I2->size[0] * I2->size[1] * i];
    }
  }
  b_I2[0] = I2->size[0];
  b_I2[1] = 1;
  b_I2[2] = I2->size[2];
  emlrtSubAssignSizeCheckR2012b(&b_I2[0], 3, &e_I2->size[0], 3, &f_emlrtECI,
                                (emlrtCTX)sp);
  Wt3_tmp = I2->size[0] - 1;
  loop_ub = I2->size[2] - 1;
  i = e_I2->size[0] * e_I2->size[1] * e_I2->size[2];
  e_I2->size[0] = I2->size[0];
  e_I2->size[1] = 1;
  e_I2->size[2] = I2->size[2];
  emxEnsureCapacity_real_T(sp, e_I2, i, &g_emlrtRTEI);
  b_I2_data = e_I2->data;
  for (i = 0; i <= loop_ub; i++) {
    for (i1 = 0; i1 <= Wt3_tmp; i1++) {
      b_I2_data[i1 + e_I2->size[0] * i] =
          I2_data[(i1 + I2->size[0] * 2) + I2->size[0] * I2->size[1] * i];
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
  loop_ub = I2->size[0];
  b_loop_ub = I2->size[2];
  i = e_I2->size[0] * e_I2->size[1] * e_I2->size[2];
  e_I2->size[0] = I2->size[0];
  e_I2->size[1] = 1;
  e_I2->size[2] = I2->size[2];
  emxEnsureCapacity_real_T(sp, e_I2, i, &h_emlrtRTEI);
  b_I2_data = e_I2->data;
  for (i = 0; i < b_loop_ub; i++) {
    for (i1 = 0; i1 < loop_ub; i1++) {
      b_I2_data[i1 + e_I2->size[0] * i] =
          I2_data[(i1 + I2->size[0] * 2) + I2->size[0] * I2->size[1] * i];
    }
  }
  b_I2[0] = I2->size[0];
  b_I2[1] = 1;
  b_I2[2] = I2->size[2];
  emlrtSubAssignSizeCheckR2012b(&b_I2[0], 3, &e_I2->size[0], 3, &g_emlrtECI,
                                (emlrtCTX)sp);
  Wt3_tmp = I2->size[0] - 1;
  loop_ub = I2->size[2] - 1;
  i = e_I2->size[0] * e_I2->size[1] * e_I2->size[2];
  e_I2->size[0] = I2->size[0];
  e_I2->size[1] = 1;
  e_I2->size[2] = I2->size[2];
  emxEnsureCapacity_real_T(sp, e_I2, i, &h_emlrtRTEI);
  b_I2_data = e_I2->data;
  for (i = 0; i <= loop_ub; i++) {
    for (i1 = 0; i1 <= Wt3_tmp; i1++) {
      b_I2_data[i1 + e_I2->size[0] * i] =
          I2_data[(i1 + I2->size[0] * 2) + I2->size[0] * I2->size[1] * i];
    }
  }
  loop_ub = e_I2->size[2];
  for (i = 0; i < loop_ub; i++) {
    b_loop_ub = e_I2->size[0];
    for (i1 = 0; i1 < b_loop_ub; i1++) {
      I2_data[(i1 + I2->size[0]) + I2->size[0] * I2->size[1] * i] =
          b_I2_data[i1 + e_I2->size[0] * i];
    }
  }
  if (I2->size[1] < 1) {
    emlrtDynamicBoundsCheckR2012b(I2->size[1], 1, I2->size[1], &i_emlrtBCI,
                                  (emlrtCTX)sp);
  }
  if ((I2->size[1] - 2 < 1) || (I2->size[1] - 2 > I2->size[1])) {
    emlrtDynamicBoundsCheckR2012b(I2->size[1] - 2, 1, I2->size[1], &h_emlrtBCI,
                                  (emlrtCTX)sp);
  }
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
          I2_data[(i1 + I2->size[0] * (I2->size[1] - 3)) +
                  I2->size[0] * I2->size[1] * i];
    }
  }
  b_I2[0] = I2->size[0];
  b_I2[1] = 1;
  b_I2[2] = I2->size[2];
  emlrtSubAssignSizeCheckR2012b(&b_I2[0], 3, &e_I2->size[0], 3, &h_emlrtECI,
                                (emlrtCTX)sp);
  Wt3_tmp = I2->size[0] - 1;
  loop_ub = I2->size[2] - 1;
  d_I2 = I2->size[1] - 1;
  i = e_I2->size[0] * e_I2->size[1] * e_I2->size[2];
  e_I2->size[0] = I2->size[0];
  e_I2->size[1] = 1;
  e_I2->size[2] = I2->size[2];
  emxEnsureCapacity_real_T(sp, e_I2, i, &i_emlrtRTEI);
  b_I2_data = e_I2->data;
  for (i = 0; i <= loop_ub; i++) {
    for (i1 = 0; i1 <= Wt3_tmp; i1++) {
      b_I2_data[i1 + e_I2->size[0] * i] =
          I2_data[(i1 + I2->size[0] * (I2->size[1] - 3)) +
                  I2->size[0] * I2->size[1] * i];
    }
  }
  loop_ub = e_I2->size[2];
  for (i = 0; i < loop_ub; i++) {
    b_loop_ub = e_I2->size[0];
    for (i1 = 0; i1 < b_loop_ub; i1++) {
      I2_data[(i1 + I2->size[0] * d_I2) + I2->size[0] * I2->size[1] * i] =
          b_I2_data[i1 + e_I2->size[0] * i];
    }
  }
  if ((I2->size[1] - 1 < 1) || (I2->size[1] - 1 > I2->size[1])) {
    emlrtDynamicBoundsCheckR2012b(I2->size[1] - 1, 1, I2->size[1], &k_emlrtBCI,
                                  (emlrtCTX)sp);
  }
  if ((I2->size[1] - 2 < 1) || (I2->size[1] - 2 > I2->size[1])) {
    emlrtDynamicBoundsCheckR2012b(I2->size[1] - 2, 1, I2->size[1], &j_emlrtBCI,
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
          I2_data[(i1 + I2->size[0] * (I2->size[1] - 3)) +
                  I2->size[0] * I2->size[1] * i];
    }
  }
  b_I2[0] = I2->size[0];
  b_I2[1] = 1;
  b_I2[2] = I2->size[2];
  emlrtSubAssignSizeCheckR2012b(&b_I2[0], 3, &e_I2->size[0], 3, &i_emlrtECI,
                                (emlrtCTX)sp);
  Wt3_tmp = I2->size[0] - 1;
  loop_ub = I2->size[2] - 1;
  d_I2 = I2->size[1] - 2;
  i = e_I2->size[0] * e_I2->size[1] * e_I2->size[2];
  e_I2->size[0] = I2->size[0];
  e_I2->size[1] = 1;
  e_I2->size[2] = I2->size[2];
  emxEnsureCapacity_real_T(sp, e_I2, i, &j_emlrtRTEI);
  b_I2_data = e_I2->data;
  for (i = 0; i <= loop_ub; i++) {
    for (i1 = 0; i1 <= Wt3_tmp; i1++) {
      b_I2_data[i1 + e_I2->size[0] * i] =
          I2_data[(i1 + I2->size[0] * (I2->size[1] - 3)) +
                  I2->size[0] * I2->size[1] * i];
    }
  }
  loop_ub = e_I2->size[2];
  for (i = 0; i < loop_ub; i++) {
    b_loop_ub = e_I2->size[0];
    for (i1 = 0; i1 < b_loop_ub; i1++) {
      I2_data[(i1 + I2->size[0] * d_I2) + I2->size[0] * I2->size[1] * i] =
          b_I2_data[i1 + e_I2->size[0] * i];
    }
  }
  emxFree_real_T(sp, &e_I2);
  emxInit_real_T(sp, &f_I2, 2, &k_emlrtRTEI);
  g_I2[0] = I2->size[0];
  g_I2[1] = I2->size[1];
  iv[0] = I2->size[0];
  iv[1] = I2->size[1];
  emlrtSubAssignSizeCheckR2012b(&g_I2[0], 2, &iv[0], 2, &j_emlrtECI,
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
          I2_data[(i1 + I2->size[0] * i) + I2->size[0] * I2->size[1] * 2];
    }
  }
  loop_ub = f_I2->size[1];
  for (i = 0; i < loop_ub; i++) {
    b_loop_ub = f_I2->size[0];
    for (i1 = 0; i1 < b_loop_ub; i1++) {
      I2_data[i1 + I2->size[0] * i] = b_I2_data[i1 + f_I2->size[0] * i];
    }
  }
  g_I2[0] = I2->size[0];
  g_I2[1] = I2->size[1];
  iv[0] = I2->size[0];
  iv[1] = I2->size[1];
  emlrtSubAssignSizeCheckR2012b(&g_I2[0], 2, &iv[0], 2, &k_emlrtECI,
                                (emlrtCTX)sp);
  Wt3_tmp = I2->size[0] - 1;
  loop_ub = I2->size[1] - 1;
  i = f_I2->size[0] * f_I2->size[1];
  f_I2->size[0] = I2->size[0];
  f_I2->size[1] = I2->size[1];
  emxEnsureCapacity_real_T(sp, f_I2, i, &l_emlrtRTEI);
  b_I2_data = f_I2->data;
  for (i = 0; i <= loop_ub; i++) {
    for (i1 = 0; i1 <= Wt3_tmp; i1++) {
      b_I2_data[i1 + f_I2->size[0] * i] =
          I2_data[(i1 + I2->size[0] * i) + I2->size[0] * I2->size[1] * 2];
    }
  }
  loop_ub = f_I2->size[1];
  for (i = 0; i < loop_ub; i++) {
    b_loop_ub = f_I2->size[0];
    for (i1 = 0; i1 < b_loop_ub; i1++) {
      I2_data[(i1 + I2->size[0] * i) + I2->size[0] * I2->size[1]] =
          b_I2_data[i1 + f_I2->size[0] * i];
    }
  }
  if (I2->size[2] < 1) {
    emlrtDynamicBoundsCheckR2012b(I2->size[2], 1, I2->size[2], &m_emlrtBCI,
                                  (emlrtCTX)sp);
  }
  if ((I2->size[2] - 2 < 1) || (I2->size[2] - 2 > I2->size[2])) {
    emlrtDynamicBoundsCheckR2012b(I2->size[2] - 2, 1, I2->size[2], &l_emlrtBCI,
                                  (emlrtCTX)sp);
  }
  g_I2[0] = I2->size[0];
  g_I2[1] = I2->size[1];
  iv[0] = I2->size[0];
  iv[1] = I2->size[1];
  emlrtSubAssignSizeCheckR2012b(&g_I2[0], 2, &iv[0], 2, &l_emlrtECI,
                                (emlrtCTX)sp);
  Wt3_tmp = I2->size[0] - 1;
  loop_ub = I2->size[1] - 1;
  d_I2 = I2->size[2] - 1;
  i = f_I2->size[0] * f_I2->size[1];
  f_I2->size[0] = I2->size[0];
  f_I2->size[1] = I2->size[1];
  emxEnsureCapacity_real_T(sp, f_I2, i, &m_emlrtRTEI);
  b_I2_data = f_I2->data;
  for (i = 0; i <= loop_ub; i++) {
    for (i1 = 0; i1 <= Wt3_tmp; i1++) {
      b_I2_data[i1 + f_I2->size[0] * i] =
          I2_data[(i1 + I2->size[0] * i) +
                  I2->size[0] * I2->size[1] * (I2->size[2] - 3)];
    }
  }
  loop_ub = f_I2->size[1];
  for (i = 0; i < loop_ub; i++) {
    b_loop_ub = f_I2->size[0];
    for (i1 = 0; i1 < b_loop_ub; i1++) {
      I2_data[(i1 + I2->size[0] * i) + I2->size[0] * I2->size[1] * d_I2] =
          b_I2_data[i1 + f_I2->size[0] * i];
    }
  }
  if ((I2->size[2] - 1 < 1) || (I2->size[2] - 1 > I2->size[2])) {
    emlrtDynamicBoundsCheckR2012b(I2->size[2] - 1, 1, I2->size[2], &o_emlrtBCI,
                                  (emlrtCTX)sp);
  }
  if ((I2->size[2] - 2 < 1) || (I2->size[2] - 2 > I2->size[2])) {
    emlrtDynamicBoundsCheckR2012b(I2->size[2] - 2, 1, I2->size[2], &n_emlrtBCI,
                                  (emlrtCTX)sp);
  }
  g_I2[0] = I2->size[0];
  g_I2[1] = I2->size[1];
  iv[0] = I2->size[0];
  iv[1] = I2->size[1];
  emlrtSubAssignSizeCheckR2012b(&g_I2[0], 2, &iv[0], 2, &m_emlrtECI,
                                (emlrtCTX)sp);
  Wt3_tmp = I2->size[0] - 1;
  loop_ub = I2->size[1] - 1;
  d_I2 = I2->size[2] - 2;
  i = f_I2->size[0] * f_I2->size[1];
  f_I2->size[0] = I2->size[0];
  f_I2->size[1] = I2->size[1];
  emxEnsureCapacity_real_T(sp, f_I2, i, &n_emlrtRTEI);
  b_I2_data = f_I2->data;
  for (i = 0; i <= loop_ub; i++) {
    for (i1 = 0; i1 <= Wt3_tmp; i1++) {
      b_I2_data[i1 + f_I2->size[0] * i] =
          I2_data[(i1 + I2->size[0] * i) +
                  I2->size[0] * I2->size[1] * (I2->size[2] - 3)];
    }
  }
  loop_ub = f_I2->size[1];
  for (i = 0; i < loop_ub; i++) {
    b_loop_ub = f_I2->size[0];
    for (i1 = 0; i1 < b_loop_ub; i1++) {
      I2_data[(i1 + I2->size[0] * i) + I2->size[0] * I2->size[1] * d_I2] =
          b_I2_data[i1 + f_I2->size[0] * i];
    }
  }
  emxFree_real_T(sp, &f_I2);
  /* clear I2; */
  for (b_i = 0; b_i < newdim_idx_0; b_i++) {
    for (b_loop_ub = 0; b_loop_ub < newdim_idx_1; b_loop_ub++) {
      for (k = 0; k < newdim_idx_2; k++) {
        real_T A[125];
        i = 2 * b_i;
        i1 = 2 * b_loop_ub;
        i2 = 2 * k;
        for (i3 = 0; i3 < 5; i3++) {
          i4 = (int32_T)(((real_T)i2 + ((real_T)i3 + -2.0)) + 3.0);
          for (i5 = 0; i5 < 5; i5++) {
            Wt3_tmp = (int32_T)(((real_T)i1 + ((real_T)i5 + -2.0)) + 3.0);
            for (i6 = 0; i6 < 5; i6++) {
              d_I2 = (int32_T)(((real_T)i + ((real_T)i6 + -2.0)) + 3.0);
              if ((d_I2 < 1) || (d_I2 > I2->size[0])) {
                emlrtDynamicBoundsCheckR2012b(d_I2, 1, I2->size[0], &p_emlrtBCI,
                                              (emlrtCTX)sp);
              }
              if ((Wt3_tmp < 1) || (Wt3_tmp > I2->size[1])) {
                emlrtDynamicBoundsCheckR2012b(Wt3_tmp, 1, I2->size[1],
                                              &q_emlrtBCI, (emlrtCTX)sp);
              }
              if ((i4 < 1) || (i4 > I2->size[2])) {
                emlrtDynamicBoundsCheckR2012b(i4, 1, I2->size[2], &r_emlrtBCI,
                                              (emlrtCTX)sp);
              }
              loop_ub = (i6 + 5 * i5) + 25 * i3;
              A[loop_ub] = I2_data[((d_I2 + I2->size[0] * (Wt3_tmp - 1)) +
                                    I2->size[0] * I2->size[1] * (i4 - 1)) -
                                   1] *
                           Wt3[loop_ub];
            }
          }
        }
        if (b_i + 1 > IResult->size[0]) {
          emlrtDynamicBoundsCheckR2012b(b_i + 1, 1, IResult->size[0],
                                        &s_emlrtBCI, (emlrtCTX)sp);
        }
        if (b_loop_ub + 1 > IResult->size[1]) {
          emlrtDynamicBoundsCheckR2012b(b_loop_ub + 1, 1, IResult->size[1],
                                        &t_emlrtBCI, (emlrtCTX)sp);
        }
        if (k + 1 > IResult->size[2]) {
          emlrtDynamicBoundsCheckR2012b(k + 1, 1, IResult->size[2], &u_emlrtBCI,
                                        (emlrtCTX)sp);
        }
        IResult_data[(b_i + IResult->size[0] * b_loop_ub) +
                     IResult->size[0] * IResult->size[1] * k] = sumColumnB(A);
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

/* End of code generation (GPReduce.c) */
