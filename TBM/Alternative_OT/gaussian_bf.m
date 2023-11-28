function [phi] = gaussian_bf(X,Y,Z,sigma)


phi = 1/(2*pi*sigma^2)*exp( - (X.^2 + Y.^2 + Z.^2)/(2*sigma^2) );
% phi_x = -phi.*X/(sigma^2);
% phi_y = -phi.*Y/(sigma^2);

sp=sum(phi(:));
phi=phi/sp;
% phi_x=fliplr(phi_x/sp);
% phi_y=flipud(phi_y/sp);