// DGradient.c
// Gradient along a dimension
// Y = DGradient(X, Dim, Spacing, Method)
// INPUT:
//   X:   Real DOUBLE array.
//   Spacing: Scalar or vector of the length SIZE(X, Dim).
//        A scalar value is the distance between all points, while a vector
//        contains all coordinates, such that DIFF(Spacing) are the distances.
//        For equally spaced input a scalar Spacing is much faster.
//        Optional, default: 1.0
//   Dim: Dimension to operate on.
//        Optional, default: [] (1st non-singelton dimension).
//   Method: String, order of the applied method for unevenly spaced X:
//        '1stOrder', faster centered differences as in Matlab's GRADIENT.
//        '2ndOrder', 2nd order accurate centered differences.
//        On the edges forward and backward difference are used.
//        Optional, default: '1stOrder'.
//
// OUTPUT:
//   Y:   Gradient of X, same size as X.
//
// EXAMPLES:
//   t = cumsum(rand(1, 100)) + 0.01;  t = 2*pi * t ./ max(t);
//   x = sin(t);
//   dx1 = DGradient(x, t, 2, '1stOrder');
//   dx2 = DGradient(x, t, 2, '2ndOrder');
//   dx  = cos(t);          % Analytic solution
//   h = plot(t, dx, t, dx1, 'or', t, dx2, 'og');  axis('tight');
//   title('cos(x) and DGradient(sin(x))');
//   legend(h, {'analytic', '1st order', '2nd order'}, 'location', 'best');
//
// NOTES:
// - There are a lot of other derivation tools in the FEX. This function is
//   faster, e.g. 25% faster than dqdt.c and 10 to 16 times faster than Matlab's
//   GRADIENT. In addition it works with multi-dim arrays, on a speicifc
//   dimension only and can use a 2nd order method for unevenly spaced data.
// - This function does not use temporary memory for evenly spaced data and if
//   a single vector is processed. Otherwise the 1st-order method needs one and
//   the 2nd-order method 3 temporary vectors of the length of the processed
//   dimension.
// - Matlab's GRADIENT processes all dimensions ever, while DGradient operates on
//   the specified dimension only.
// - 1st order centered difference:
//     y(i) = (x(i+1) - x(i-1) / (s(i+1) - s(i-1))
// - 2nd order centered difference:
//     y(i) = ((x(i+1) * (s(i)-s(i-1)) / (s(i+1)-s(i))) -
//             (x(i-1) * (s(i+1)-s(i)) / (s(i)-s(i-1)))) / (s(i+1)-s(i-1))
//            + x(i) * (1.0 / (s(i)-s(i-1)) - 1.0 / (s(i+1)-s(i)))
//   For evenly spaced X, both methods reply equal values.
//
// COMPILE:
//   mex -O DGradient.c
// Consider C99 comments on Linux:
//   mex -O CFLAGS="\$CFLAGS -std=c99" DGradient.c
// Pre-compiled Mex: http://www.n-simon.de/mex
// Run the unit test uTest_DGradient after compiling.
//
// Tested: Matlab 6.5, 7.7, 7.8, WinXP, 32bit
//         Compiler: LCC2.4/3.8, BCC5.5, OWC1.8, MSVC2008
// Assumed Compatibility: higher Matlab versions, Mac, Linux, 64bit
// Author: Jan Simon, Heidelberg, (C) 2011 matlab.THISYEAR(a)nMINUSsimon.de
//
// See also GRADIENT, DIFF.
// FEX: central_diff (#12 Robert A. Canfield)
//      derivative (#28920, Scott McKinney)
//      movingslope (#16997, John D'Errico)
//      diffxy (#29312, Darren Rowland)
//      dqdt (#11965, Geoff Wawrzyniak)

// Todo: SSE2 instructions
//       Newton polynomials for 2nd order edges
//       Multi-threading

/*
% $JRev: R0d V:004 Sum:sHhGcnzMMNaA Date:02-Jan-2008 17:46:05 $
% $License: BSD $
% $File: Tools\Mex\Source\DGradient.c $
% History:
% 001: 30-Dec-2010 22:42, First version published under BSD license.
*/

// Includes:
#include "mex.h"
#include <stdlib.h>
#include <float.h>
#include <string.h>
#include <math.h>

// Assume 32 bit addressing for Matlab 6.5:
// See MEX option "compatibleArrayDims" for MEX in Matlab >= 7.7.
#ifndef MWSIZE_MAX
#define mwSize  int32_T               // Defined in tmwtypes.h
#define mwIndex int32_T
#define MWSIZE_MAX MAX_int32_T
#endif

// There is an undocumented method to create a shared data copy. This is much
// faster, if the replied object is not changed, because it does not duplicate
// the contents of the array in the memory.
mxArray *mxCreateSharedDataCopy(const mxArray *mx);
#define COPY_ARRAY mxCreateSharedDataCopy
// #define COPY_ARRAY mxDuplicateArray    // slower, but documented

// Disable the /fp:precise flag to increase the speed on MSVC compiler:
#ifdef _MSC_VER
#pragma float_control(except, off)    // disable exception semantics
#pragma float_control(precise, off)   // disable precise semantics
#pragma fp_contract(on)               // enable contractions
// #pragma fenv_access(off)           // disable fpu environment sensitivity
#endif

// Error messages do not contain the function name in Matlab 6.5! This is not
// necessary in Matlab 7, but it does not bother:
#define ERR_HEAD "*** DGradient[mex]: "
#define ERR_ID   "JSimon:DGradient:"

// Prototypes: -----------------------------------------------------------------
void CoreDim1Space1(double *X, const mwSize M, const mwSize nDX, double Space,
              double *Y);
void CoreDimNSpace1(double *X, const mwSize step, const mwSize nX,
              const mwSize nDX, double Space, double *Y);

void WrapSpaceN(double *X, const mwSize Step, const mwSize nX,
              const mwSize nDX, double *Factor, double *Y);
void GetFactor(double *Space, mwSize nDX, double *Factor);
void CoreDim1SpaceN(double *X, const mwSize M, const mwSize nDX,
              double *Space, double *Y);
// void CoreDimNSpaceN(double *X, const mwSize step, const mwSize nX,
//            const mwSize nDX, double *Space, double *Y);
void CoreDim1FactorN(double *X, const mwSize M, const mwSize nDX,
              double *Factor, double *Y);
void CoreDimNFactorN(double *X, const mwSize Step, const mwSize nX,
              const mwSize nDX, double *Factor, double *Y);

void WrapSpaceNOrder2(double *X, const mwSize Step, const mwSize nX,
              const mwSize nDX, double *Space, double *Y);
void GetFactorOrder2(double *Space, const mwSize nDX,
              double *A, double *B, double *C);
void CoreDim1SpaceNOrder2(double *X, const mwSize nX, const mwSize nDX,
              double *Space, double *Y);
void CoreDimNSpaceNOrder2(double *X, const mwSize Step, const mwSize nX,
              const mwSize nDX, double *Space, double *Y);
void CoreDim1FactorNOrder2(double *X, const mwSize nX, const mwSize nDX,
              double *A, double *B, double *C, double *Y);
void CoreDimNFactorNOrder2(double *X, const mwSize Step, const mwSize nX,
              const mwSize nDX, double *A, double *B, double *C, double *Y);

mwSize FirstNonSingeltonDim(const mwSize Xndim, const mwSize *Xdim);
mwSize GetStep(const mwSize *Xdim, const mwSize N);

// Main function ===============================================================
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
  double *X, *Y, Nd, *Space, UnitSpace = 1.0;
  mwSize nX, nDX, ndimX, N, Step, nSpace;
  const mwSize *dimX;
  int    Order2 = 0;
  
  // Check number and type of inputs and outputs: ------------------------------
  if (nrhs == 0 || nrhs > 4) {
     mexErrMsgIdAndTxt(ERR_ID   "BadNInput",
                       ERR_HEAD "1 or 4 inputs required.");
  }
  if (nlhs > 1) {
     mexErrMsgIdAndTxt(ERR_ID   "BadNOutput",
                       ERR_HEAD "1 output allowed.");
  }
  
  if (!mxIsDouble(prhs[0]) || mxIsComplex(prhs[0]) || mxIsSparse(prhs[0])) {
     mexErrMsgIdAndTxt(ERR_ID   "BadTypeInput1",
                       ERR_HEAD "Input must be a full real double array.");
  }
  
  // Pointers and dimension to input array: ------------------------------------
  X     = mxGetPr(prhs[0]);
  nX    = mxGetNumberOfElements(prhs[0]);
  ndimX = mxGetNumberOfDimensions(prhs[0]);
  dimX  = mxGetDimensions(prhs[0]);
  
  // Return fast on empty input matrix:
  if (nX == 0) {
     plhs[0] = COPY_ARRAY(prhs[0]);
     return;
  }
  
  // Get spacing, if defined: --------------------------------------------------
  if (nrhs < 2) {  // No 2nd input defined - scalar unit spacing:
     nSpace = 1;
     Space  = &UnitSpace;
     
  } else {         // Get pointer to spacing vector:
     if (!mxIsDouble(prhs[1])) {
        mexErrMsgIdAndTxt(ERR_ID   "BadTypeInput2",
                          ERR_HEAD "2nd input [Spacing] must be a DOUBLE.");
     }
     Space  = mxGetPr(prhs[1]);
     nSpace = mxGetNumberOfElements(prhs[1]);
     if (nSpace == 0) {
        nSpace = 1;
        Space  = &UnitSpace;
     }
  }
  
  // Determine dimension to operate on: ----------------------------------------
  if (nrhs < 3) {
     N    = FirstNonSingeltonDim(ndimX, dimX);  // Zero based
     Step = 1;
     nDX  = dimX[N];
     
  } else if (mxIsNumeric(prhs[2]))  {  // 3rd input used:
     switch (mxGetNumberOfElements(prhs[2])) {
        case 0:  // Use 1st non-singelton dim if 3rd input is []:
           N    = FirstNonSingeltonDim(ndimX, dimX);
           Step = 1;
           nDX  = dimX[N];
           break;
           
        case 1:  // Numerical scalar:
           Nd = mxGetScalar(prhs[2]);
           N  = (mwSize) Nd - 1;
           if (Nd < 1.0 || Nd != floor(Nd)) {
              mexErrMsgIdAndTxt(ERR_ID   "BadValueInput3",
                       ERR_HEAD "Dimension must be a positive integer scalar.");
           }
           
           if (N < ndimX) {
              Step = GetStep(dimX, N);
              nDX  = dimX[N];
           } else {
              // Treat imaginated trailing dimensions as singelton, as usual in
              // Matlab:
              Step = nX;
              nDX  = 1;
           }
           break;
           
        default:
           mexErrMsgIdAndTxt(ERR_ID   "BadSizeInput3",
                             ERR_HEAD "3rd input [Dim] must be scalar index.");
     }
     
  } else {  // 2nd input is not numeric:
     mexErrMsgIdAndTxt(ERR_ID   "BadTypeInput3",
                       ERR_HEAD "3rd input must be scalar index.");
  }
  
  // Check matching sizes of X and Spacing:
  if (nSpace != 1 && nSpace != nDX) {
     mexErrMsgIdAndTxt(ERR_ID   "BadSizeInput2",
             ERR_HEAD "2nd input [Spacing] does not match the dimensions.");
  }
  
  // Check 4th input: ----------------------------------------------------------
  if (nrhs >= 4) {
     // "2ndOrder", but accept everything starting with "2":
     if (mxIsChar(prhs[3]) && !mxIsEmpty(prhs[3])) {
        Order2 = (*(mxChar *) mxGetData(prhs[3]) == L'2');
     } else {
        mexErrMsgIdAndTxt(ERR_ID   "BadTypeInput4",
                          ERR_HEAD "4th input must be a string.");
     }
  }
  
  // Create output matrix: -----------------------------------------------------
  plhs[0] = mxCreateNumericArray(ndimX, dimX, mxDOUBLE_CLASS, mxREAL);
  Y      = mxGetPr(plhs[0]);

  // Reply ZEROS, if the length of the processed dimension is 1:
  if (nDX == 1) {
     return;
  }
  
  // Calculate the gradient: ---------------------------------------------------
  if (nSpace == 1) {         // Scalar spacing
     if (Step == 1) {        // Operate on 1st dimension
        CoreDim1Space1(X, nX, nDX, *Space, Y);
     } else {                // Step >= 1, operate on any dimension
        CoreDimNSpace1(X, Step, nX, nDX, *Space, Y);
     }

  } else if (Order2) {       // Spacing defined as vector, 2nd order method:
     if (nX == nDX) {        // Single vector only - dynamic spacing factors:
        CoreDim1SpaceNOrder2(X, nX, nDX, Space, Y);
     } else {
        WrapSpaceNOrder2(X, Step, nX, nDX, Space, Y);
     }
     
  } else {                   // Spacing defined as vector, 1st order method:
     if (nX == nDX) {        // Single vector only - dynamic spacing factors:
        CoreDim1SpaceN(X, nX, nDX, Space, Y);
     } else {
        WrapSpaceN(X, Step, nX, nDX, Space, Y);
     }
  }

  return;
}

// Subroutines: ================================================================
mwSize FirstNonSingeltonDim(const mwSize Xndim, const mwSize *Xdim)
{
  // Get first non-singelton dimension - zero based.
  
  mwSize N;
  
  for (N = 0; N < Xndim; N++) {
     if (Xdim[N] != 1) {
        return (N);
     }
  }
  
  return (0);  // Use the first dimension if all dims are 1
}

// =============================================================================
mwSize GetStep(const mwSize *Xdim, const mwSize N)
{
  // Get step size between elements of a subvector in the N'th dimension.
  // This is the product of the leading dimensions.
  
  const mwSize *XdimEnd, *XdimP;
  mwSize       Step;
  
  Step    = 1;
  XdimEnd = Xdim + N;
  for (XdimP = Xdim; XdimP < XdimEnd; Step *= *XdimP++) ; // empty loop
  
  return (Step);
}

// =============================================================================
void CoreDim1Space1(double *X, const mwSize nX, const mwSize nDX, double Space,
                    double *Y)
{
  // Operate on first dimension, scalar spacing, 1st order method.
  
  double x0, x1, x2, *Xf, *Xc, fac1, fac2;
  
  // Multiplication is faster than division:
  fac1 = 1.0 / Space;
  fac2 = 1.0 / (2.0 * Space);

  Xf = X + nX;                 // End of input array
  while (X < Xf) {
     Xc   = X + nDX;
     
     x0   = *X++;              // Forward difference:
     x1   = *X++;
     *Y++ = (x1 - x0) * fac1;
     
     while (X < Xc) {          // Central differences:
        x2   = *X++;
        *Y++ = (x2 - x0) * fac2;
        x0   = x1;
        x1   = x2;
     }
  
     *Y++ = (x1 - x0) * fac1;  // Backward difference
  }

  return;
}

// =============================================================================
void CoreDimNSpace1(double *X, const mwSize Step, const mwSize nX,
                    const mwSize nDX, double Space, double *Y)
{
  // Operate on any dimension, scalar spacing, 1st order method.
  // Column oriented approach: Process contiguous memory blocks of input and
  // output.
  
  double *Xf, *X1, *X2, *Xc, fac1, fac2;
  mwSize nDXStep;

  // Multiplication is faster than division:
  fac1 = 1.0 / Space;
  fac2 = 1.0 / (2.0 * Space);
  
  // Distance between first and last element of X in specified dim:
  nDXStep = nDX * Step;
  
  Xf = X + nX;          // End of the input array
  while (X < Xf) {
     X1 = X;            // Forward differences:
     X2 = X1 + Step;
     Xc = X2;
     while (X1 < Xc) {
        *Y++ = (*X2++ - *X1++) * fac1;
     }
     
     X1 = X;            // Central differences:
     Xc = X + nDXStep;
     while (X2 < Xc) {
        *Y++ = (*X2++ - *X1++) * fac2;
     }

     X2 = X1 + Step;    // Backward differences:
     while (X2 < Xc) {
        *Y++ = (*X2++ - *X1++) * fac1;
     }
     
     X = Xc;            // Move input pointer to the next chunk
  }
  
  return;
}

// =============================================================================
void WrapSpaceN(double *X, const mwSize Step, const mwSize nX, const mwSize nDX,
                double *Space, double *Y)
{
  // Call different methods depending of the dimensions ofthe input.
  // X has more than 1 vector. Therefore precalculating the spacing factors is
  // cheaper.
  
  double *Factor;
  
  // Precalculate spacing factors:
  if ((Factor = (double *) mxMalloc(nDX * sizeof(double))) == NULL) {
     mexErrMsgIdAndTxt(ERR_ID   "NoMemory",
                       ERR_HEAD "No memory for Factor vector.");
  }
  GetFactor(Space, nDX, Factor);
  
  if (Step == 1) {   // Operate on 1st dimension:
     CoreDim1FactorN(X, nX, nDX, Factor, Y);
  } else {           // Operate on any dimension:
     CoreDimNFactorN(X, Step, nX, nDX, Factor, Y);
  }
  
  mxFree(Factor);
     
  return;
}

// =============================================================================
void CoreDim1SpaceN(double *X, const mwSize nX, const mwSize nDX,
                    double *Space, double *Y)
{
  // Operate on the first dimension, spacing is a vector, order 1 method.
  // The spacing factors are calculated dynamically. This is efficient for
  // a single vector, but slower for matrices. No temporary vector is needed.
  
  double x0, x1, x2, *Xf, *Xg, *Sp, s0, s1, s2;
  
  Xf = X + nX;
  while (X < Xf) {
     Xg   = X + nDX;    // Forward difference
     x0   = *X++;
     x1   = *X++;
     Sp   = Space;
     s0   = *Sp++;
     s1   = *Sp++;
     *Y++ = (x1 - x0) / (s1 - s0);
     
     while (X < Xg) {   // Central differences
        x2   = *X++;
        s2   = *Sp++;
        *Y++ = (x2 - x0) / (s2 - s0);
        x0   = x1;
        x1   = x2;
        s0   = s1;
        s1   = s2;
     }
     
     *Y++ = (x1 - x0) / (s1 - s0);  // Backward difference
  }

  return;
}

// =============================================================================
void GetFactor(double *Space, mwSize nDX, double *F)
{
  // If more than one vector is processed, it is cheaper to calculate the spacing
  // factors once. This needs the memory for a temporary vector.
  // INPUT:
  //   Space: Pointer to DOUBLE vector from the inputs.
  //   nDX:   Length of the Space vector.
  // OUTPUT:
  //   F:     Factors, pointer to DOUBLE vector:
  //          [ 1/(S[1]-S[0]), 1/(S[i+1]-S[i-1]), 1/(S[nDX-1]-S[nDX-2]) ]
   
  double s0, s1, s2, *Ff;
  
  Ff   = F + nDX - 1;
  s0   = *Space++;
  s1   = *Space++;
  *F++ = 1.0 / (s1 - s0);
  
  while (F < Ff) {
     s2   = *Space++;
     *F++ = 1.0 / (s2 - s0);
     s0   = s1;
     s1   = s2;
  }
  
  *F = 1.0 / (s1 - s0);
  
  return;
}

// =============================================================================
void CoreDim1FactorN(double *X, const mwSize nX, const mwSize nDX,
                     double *Factor, double *Y)
{
  // Operate on first dimension, spacing is a vector, 1st order method.
  // The spacing factors are calculated before. This needs a temporary vector
  // of nDX elements. It is faster than the dynamically calculated spacing
  // factors.
  
  double x0, x1, x2, *Xf, *Xc, *Fp;
  
  Xf = X + nX;
  while (X < Xf) {
     Xc   = X + nDX;    // End of column
     x0   = *X++;       // Forward difference
     x1   = *X++;
     Fp   = Factor;
     *Y++ = (x1 - x0) * *Fp++;
     
     while (X < Xc) {   // Central differences
        x2   = *X++;
        *Y++ = (x2 - x0) * *Fp++;
        x0   = x1;
        x1   = x2;
     }
  
     *Y++ = (x1 - x0) * *Fp;  // Backward difference
  }

  return;
}

// =============================================================================
void CoreDimNFactorN(double *X, const mwSize Step, const mwSize nX,
                     const mwSize nDX, double *Factor, double *Y)
{
  // Operate on any dimension, spacing is a vector, 1st order method.
  // The spacing factors are calculated before. This needs a temporary vector
  // of nDX elements. This is faster than the dynamically calculated spacing
  // factors, if more than one vector is processed.
  
  double *Xf, *X1, *X2, *Xc, *Fp, *Ff;
  
  Ff = Factor + nDX - 1;  // Zero-based indexing!

  Xf = X + nX;            // End of the array
  while (X < Xf) {
     X1 = X;              // Forward differences:
     X2 = X1 + Step;      // Element of second column
     Xc = X2;             // End of first column
     Fp = Factor;
     while (X1 < Xc) {
        *Y++ = (*X2++ - *X1++) * *Fp;
     }
     
     X1  = X;             // Central differences:
     Xc += Step;
     while (++Fp < Ff) {
        Xc += Step;
        while (X2 < Xc) {
           *Y++ = (*X2++ - *X1++) * *Fp;
        }
     }
        
     X2 = X1 + Step;      // Backward differences:
     while (X2 < Xc) {
        *Y++ = (*X2++ - *X1++) * *Fp;
     }
     
     X = Xc;              // Move input pointer to the next chunk
  }
  
  return;
}

// =============================================================================
void WrapSpaceNOrder2(double *X, const mwSize Step, const mwSize nX,
                      const mwSize nDX, double *Space, double *Y)
{
  // Call different methods depending of the dimensions ofthe input.
  // X has more than one vector. Therefore it is cheaper to calculate the
  // spacing factors once only.
  
  double *A, *B, *C;

  // Precalculate spacing factors:
  A = (double *) mxMalloc(nDX * sizeof(double));
  B = (double *) mxMalloc(nDX * sizeof(double));
  C = (double *) mxMalloc(nDX * sizeof(double));
  
  if (A == NULL || B == NULL || C == NULL) {
     mexErrMsgIdAndTxt(ERR_ID   "NoMemory",
                       ERR_HEAD "No memory for Factor vectors.");
  }

  GetFactorOrder2(Space, nDX, A, B, C);
  
  if (Step == 1) {        // Operate on first dimension:
     //CoreDim1FactorNOrder2(X, nX, nDX, A, B, C, Y);
     CoreDim1SpaceNOrder2(X, nX, nDX, Space, Y);
  } else {                // Operate on any dimension:
     //CoreDimNFactorNOrder2(X, Step, nX, nDX, A, B, C, Y);
     CoreDimNSpaceNOrder2(X, Step, nX, nDX, Space, Y);
  }
  
  mxFree(A);
  mxFree(B);
  mxFree(C);
  
  return;
}

// =============================================================================
void GetFactorOrder2(double *Space, const mwSize nDX,
                     double *A, double *B, double *C)
{
  // Calculate spacing factors for 2nd order method.
  
  double s0, s1, s2, s10, s21, *Sf;
  
  Sf = Space + nDX;
  
  s0   = *Space++;
  s1   = *Space++;
  s10  = s1 - s0;
  *A++ = 1.0 / s10;
  B++;
  C++;
  
  while (Space < Sf) {
     s2   = *Space++;
     s21  = s2 - s1;
     *A++ = s10 / (s21 * (s2 - s0));
     *B++ = 1.0 / s10 - 1.0 / s21;
     *C++ = s21 / (s10 * (s2 - s0));
     s0   = s1;
     s1   = s2;
     s10  = s21;
  }
  
  *A = 1.0 / s10;
  
  return;
}

// =============================================================================
void CoreDim1SpaceNOrder2(double *X, const mwSize nX, const mwSize nDX,
                          double *Space, double *Y)
{
  // Operate on first dimension, spacing is a vector, 2nd order method.
  // For unevenly spaced data this algorithm is 2nd order accurate.
  // This is fast for a single vector, while for arrays with more dimensions
  // is is cheaper to calculate the spacing factors once externally.
  
  double x0, x1, x2, *Xf, *Xc, *Sp, s0, s1, s2, s10, s21;
  
  Xf = X + nX;
  while (X < Xf) {
     Xc   = X + nDX;    // Forward difference (same as for evenly spaced X)
     x0   = *X++;
     x1   = *X++;
     Sp   = Space;
     s0   = *Sp++;
     s1   = *Sp++;
     s10  = s1 - s0;
     *Y++ = (x1 - x0) / s10;
     
     while (X < Xc) {    // Central differences, 2nd order method
        x2   = *X++;
        s2   = *Sp++;
        s21  = s2 - s1;
        *Y++ = ((x2 * s10 / s21) - (x0 * s21 / s10)) / (s2 - s0) +
                x1 * (1.0 / s10 - 1.0 / s21);
        x0  = x1;
        x1  = x2;
        s0  = s1;
        s1  = s2;
        s10 = s21;
     }
              
     *Y++ = (x1 - x0) / s10;  // Backward difference
  }

  return;
}

// =============================================================================
void CoreDimNSpaceNOrder2(double *X, const mwSize Step, const mwSize nX,
                          const mwSize nDX, double *Space, double *Y)
{
  // Operate on any dimension, spacing is a vector, 2nd order method.
  // The spacing factors are calculated dynamically. This is about 50% slower
  // than calculating the spacing factors at first. I assume the number of
  // registers are exhausted. Therefore this method is useful only, if the
  // memory is nearly full.

  double *Xf, *X0, *X1, *X2, *Xc, *Sp, *Sb;
  register double a, b, c;
  double s0, s1, s2;

  Sb = Space + nDX;
  
  Xf = X + nX;            // End of the array
  while (X < Xf) {
     X0  = X;             // Forward differences for first column:
     X1  = X;
     X2  = X0 + Step;     // Element of second column
     Xc  = X2;            // End of first column
     Sp  = Space;
     s0  = *Sp++;
     s1  = *Sp++;
     c   = 1.0 / (s1 - s0);
     while (X1 < Xc) {
        *Y++ = (*X2++ - *X1++) * c;
     }

     Xc += Step;                 // Central differences:
     while (Sp < Sb) {
        s2  = *Sp++;
        a   = (s1 - s0) / ((s2 - s1) * (s2 - s0));
        b   = 1.0 / (s1 - s0) - 1.0 / (s2 - s1);
        c   = (s2 - s1) / ((s1 - s0) * (s2 - s0));
        Xc += Step;
        while (X2 < Xc) {
           *Y++ = *X2++ * a + *X1++ * b - *X0++ * c;
        }
        s0  = s1;
        s1  = s2;
     }
     
     c = 1.0 / (s1 - s0);       // Backward differences:
     while (X1 < Xc) {
        *Y++ = (*X1++ - *X0++) * c;
     }

     X = Xc;                    // Move input pointer to the next chunk
  }

  return;
}

// =============================================================================
void CoreDim1FactorNOrder2(double *X, const mwSize nX, const mwSize nDX,
                           double *A, double *B, double *C, double *Y)
{
  // Operate on first dimension, spacing is a vector, 2nd order method.
  // For unevenly spaced data this algorithm is 2nd order accurate.
  // This is fast for a single vector, while for arrays with more dimensions
  // is is cheaper to calculate the spacing factors once externally.
  
  double x0, x1, x2, *Xf, *Xc, *a, *b, *c;

  Xf = X + nX;
  while (X < Xf) {
     Xc = X + nDX;    // Forward difference (same as for evenly spaced X)
     x0 = *X++;
     x1 = *X++;
     a  = A;
     b  = B + 1;
     c  = C + 1;
     *Y++ = (x1 - x0) * *a++;
     
     while (X < Xc) {    // Central differences, 2nd order method
        x2   = *X++;
        *Y++ = x2 * *a++ + x1 * *b++ - x0 * *c++;
        x0   = x1;
        x1   = x2;
     }
              
     *Y++ = (x1 - x0) * *a;  // Backward difference
  }

  return;
}

// =============================================================================
void CoreDimNFactorNOrder2(double *X, const mwSize Step, const mwSize nX,
                          const mwSize nDX, double *A, double *B, double *C,
                          double *Y)
{
  // Operate on any dimension, spacing is a vector, 2nd order method.
  // The spacing factors are calculated externally once only.

  double *Xf, *X0, *X1, *X2, *Xc, *Af, *a, *b, *c;
  register double a_, b_, c_;
  
  Af = A + nDX - 1;
  
  Xf = X + nX;            // End of the array
  while (X < Xf) {
     X0  = X;             // Forward differences for first column:
     X1  = X;
     X2  = X0 + Step;     // Element of second column
     Xc  = X2;            // End of first column
     a   = A;
     b   = B + 1;
     c   = C + 1;
     a_  = *a;
     while (X1 < Xc) {
        *Y++ = (*X2++ - *X1++) * a_;
     }
     
     Xc += Step;          // Central differences:
     a++;
     while (a < Af) {
        Xc += Step;
        a_  = *a++;
        b_  = *b++;
        c_  = *c++;
        while (X2 < Xc) {
           *Y++ = *X2++ * a_ + *X1++ * b_ - *X0++ * c_;
        }
     }
     
     a_ = *a;            // Backward differences:
     while (X1 < Xc) {
        *Y++ = (*X1++ - *X0++) * a_;
     }

     X = Xc;              // Move input pointer to the next chunk
  }

  return;
}
