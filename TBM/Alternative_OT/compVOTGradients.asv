%Shinjini VOT (c) 2015
%Transport-Based Morphometry

function [ f1t,f2t,f3t,I0_recon,Ierror,flag ] = compVOTGradients( f1,f2,f3,I0,I1,lambda,gamma )
%Computes the gradients required in VOT et al's variation optimization approach
%inputs:    f1,f2,f3        current deformation fields
%           I0,I1           original images
%           lambda          penalty for curl term
%           gamma           penalty for mass transport term 
%
%outputs:   f1t,f2t,f3t     gradients to update in the next iteration
%           flag            1 if current deformation is not diffeomorphic

[X,Y,Z] = meshgrid(1:size(f1,2),1:size(f1,1),1:size(f1,3)); 

[f1x,f1y,f1z]=gradient(f1);
[f2x,f2y,f2z]=gradient(f2);
[f3x,f3y,f3z]=gradient(f3);

[f1yx,f1yy,~] = gradient(f1y); 
[f1zx,~,f1zz] = gradient(f1z);
[f2xx,f2xy,~] = gradient(f2x); 
[~,f2zy,f2zz] = gradient(f2z);
[f3xx,~,f3xz] = gradient(f3x); 
[~,f3yy,f3yz] = gradient(f3y); 

detf = (f1x.*f2y.*f3z + f1y.*f2z.*f3x + f1z.*f2x.*f3y - f1x.*f2z.*f3y - f1y.*f2x.*f3z - f1z.*f2y.*f3x);

%check to make sure that the current deformation is diffeomorphic
if sum(detf(:)<0) ~=0 %if there are any nonzero values in the determinant,  
    flag = 1; 
    %fprintf('Warning: mapping is not diffeomorphic. There are %d negative values in the determinant \n', sum(detf(:)<0)  ); 
else
    flag = 0;
end

It=abs(interp3(I1,f1,f2,f3,'cubic',min(I1(:))));     %linear interpolation does not produce unwanted negative values
Ierror=detf.*It-I0;
[Itx,Ity,Itz]=gradient(It);
    
[g11x,~,~]=gradient((f2y.*f3z-f2z.*f3y).*Ierror.*It);
[~,g12y,~]=gradient(-(f2x.*f3z-f2z.*f3x).*Ierror.*It);
[~,~,g13z]=gradient((f2x.*f3y-f2y.*f3x).*Ierror.*It); 
    
[g21x,~,~]=gradient(-(f1y.*f3z-f1z.*f3y).*Ierror.*It);
[~,g22y,~]=gradient((f1x.*f3z-f1z.*f3x).*Ierror.*It);
[~,~,g23z]=gradient(-(f1x.*f3y-f1y.*f3x).*Ierror.*It); 
    
[g31x,~,~]=gradient((f1y.*f2z-f1z.*f2y).*Ierror.*It);
[~,g32y,~]=gradient(-(f1x.*f2z-f1z.*f2x).*Ierror.*It);
[~,~,g33z]=gradient((f1x.*f2y-f1y.*f2x).*Ierror.*It); 

divD1=g11x+g12y+g13z;
divD2=g21x+g22y+g23z;
divD3=g31x+g32y+g33z;

curlC1 = f2xy - f1yy - f1zz + f3xz;
curlC2 = f3yz - f2zz - f2xx + f1yx; 
curlC3 = f1zx - f3xx - f3yy + f2zy; 
    
f1t=detf.*Itx.*Ierror-divD1 + lambda*curlC1 - gamma*(X-f1).*I0;
f2t=detf.*Ity.*Ierror-divD2 + lambda*curlC2 - gamma*(Y-f2).*I0;
f3t=detf.*Itz.*Ierror-divD3 + lambda*curlC3 - gamma*(Z-f3).*I0;

%in order to keep with the assumptions necessary in deriving the equations,
%we need to zero out the directional derivative at the boundary

Z = padarray(ones([size(Ierror,1)-2,size(Ierror,2)-2,size(Ierror,3)-2]),[1,1,1]); 

f1t = f1t.*Z;
f2t = f2t.*Z; 
f3t = f3t.*Z; 

I0_recon = detf.*It; 
end

