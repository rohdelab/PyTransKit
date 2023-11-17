@echo off
set MATLAB=C:\PROGRA~1\MATLAB\R2015a
set MATLAB_ARCH=win64
set MATLAB_BIN="C:\Program Files\MATLAB\R2015a\bin"
set ENTRYPOINT=mexFunction
set OUTDIR=.\
set LIB_NAME=GPExpand_mex
set MEX_NAME=GPExpand_mex
set MEX_EXT=.mexw64
call setEnv.bat
echo # Make settings for GPExpand > GPExpand_mex.mki
echo COMPILER=%COMPILER%>> GPExpand_mex.mki
echo COMPFLAGS=%COMPFLAGS%>> GPExpand_mex.mki
echo OPTIMFLAGS=%OPTIMFLAGS%>> GPExpand_mex.mki
echo DEBUGFLAGS=%DEBUGFLAGS%>> GPExpand_mex.mki
echo LINKER=%LINKER%>> GPExpand_mex.mki
echo LINKFLAGS=%LINKFLAGS%>> GPExpand_mex.mki
echo LINKOPTIMFLAGS=%LINKOPTIMFLAGS%>> GPExpand_mex.mki
echo LINKDEBUGFLAGS=%LINKDEBUGFLAGS%>> GPExpand_mex.mki
echo MATLAB_ARCH=%MATLAB_ARCH%>> GPExpand_mex.mki
echo BORLAND=%BORLAND%>> GPExpand_mex.mki
echo OMPFLAGS= >> GPExpand_mex.mki
echo OMPLINKFLAGS= >> GPExpand_mex.mki
echo EMC_COMPILER=msvcsdk>> GPExpand_mex.mki
echo EMC_CONFIG=optim>> GPExpand_mex.mki
"C:\Program Files\MATLAB\R2015a\bin\win64\gmake" -B -f GPExpand_mex.mk
