function [ d ] = Projection_metric( A,B )
%This function calculates the projection metric based on the principal
%angles between the columns of matrix A and B
%Input: 
%    A: Orthonormal matrix N*M
%    B: Orthonormal matrix N*M
%Output: 
%    d: Distnace between column spaces of A and B
%Author: 
%    Soheil Kolouri, modified by Shinjini Kundu
%Date: 
%    09/27/2012

[~,M] = size(A);
A = orth(double(A)); 
B = orth(double(B)); 
for i = 1:M
    A(:,i) = A(:,i)/norm(A(:,i)); 
    B(:,i) = B(:,i)/norm(B(:,i)); 
end
[~,S,~] = svd(A'*B); 
costheta = diag(S); 
d = sqrt(M-sum(costheta.^2)); 

end

