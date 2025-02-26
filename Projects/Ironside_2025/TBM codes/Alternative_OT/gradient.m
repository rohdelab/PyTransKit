%Shinjini Kundu (c) 2015

function [dx,dy,dz] = gradient(I)
%wrapper code that uses DGradient for fast gradient computation

dy = DGradient(I,1,1,'2ndOrder'); 
dx = DGradient(I,1,2,'2ndOrder'); 
dz = DGradient(I,1,3,'2ndOrder'); 

end

