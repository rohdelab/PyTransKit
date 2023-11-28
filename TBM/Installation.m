%Shinjini Kundu (c) 2016
%Installation file for Transport-Based Morphometry 

%Go to the GP file in the TBM folder and run these lines of code
codegen GPReduce -args {coder.typeof(1,[Inf Inf Inf],[1 1 1])} -o GPReduce
codegen GPExpand -args {coder.typeof(1,[Inf Inf Inf],[1 1 1]), zeros(1,3)} -o GPExpand

%Go to the DGradient file in the TBM folder and run this line of code

%for windows and Mac:
mex -O DGradient.c

%for Linux
mex -O CFLAGS="\$CFLAGS -std=c99" DGradient.c