% Shinjini Kundu (c) 2014
% Transport-Based Morphometry project (TBM)

function [I_out]=gen_pdf(I_in,dc,sigma)

%Preprocessing of 3D image

%Inputs:     I_in       3D input image
%            dc         a positive constant
%            phi        kernel for filtering image
%
%Outputs:    I_out      3D output image

% %use this code for 2015 matlab version
if sigma ~=0
    I = mat2gray(imgaussfilt3(I_in,sigma)) + dc;
elseif sigma==0
    I = mat2gray(I_in) + dc; 
end
%%%%%%

% %else, can use this 2014 version of the code
%if sigma ~=0
    % [Xt,Yt,Zt]=meshgrid(-3*sigma:3*sigma,-3*sigma:3*sigma,-3*sigma:3*sigma);
    % phi = gaussian_bf(Xt,Yt,Zt,sigma); %normalized gaussian kernel in 3D
    % I = mat2gray(convn(I_in,phi,'same')) + dc;
%else
%    I = mat2gray(I_in) + dc; 
%end
%%%%%%

I_out = I./sum(I(:)); 

end


