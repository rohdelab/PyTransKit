%Shinjini Kundu (c) 2015
%Transport-Based Morphometry

function [ new_lambda, flag ] = checkLocalMin( results, lambda )
%Checks whether the curl and MSE that the code converges to is too far from the global minimum. 
%if so, instructs the gradient descent to keep going. 
%Input:        results      results from the Haber3D code
%              lambda       parameter from Haber3D code
%Output:       new_lambda   new value for lambda
%              flag         indicates whether the solution appears to be a global minimum 



CURL = results.curl(end); 
MSE = results.MSE2(end); 
flag = 0; 

if CURL > 5
    flag = 1;
    new_lambda = lambda*1.5;
elseif MSE > 7*10^-3
    flag = 1;
    new_lambda = lambda/1.5;
end



end








