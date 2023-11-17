function showgrid(ux,uy,downsample,lim)

    if nargin<3; downsample = 1; end;
    
    sizex = size(ux,1);
    sizey = size(uy,2);
    
    ux  = ux(1:downsample:end, 1:downsample:end);
    uy  = uy(1:downsample:end, 1:downsample:end);
    
    if nargin<4; scale = 3;                     end; % Scale vector to show small ones
    if nargin<5; lim   = [0 sizex-1 0 sizey-1]; end; % Display whole image

    [y,x] = ndgrid((0:downsample:(sizex-1))+downsample/2, (0:downsample:(sizey-1))+downsample/2); % coordinate image
    z = zeros(size(x));
    mesh(x+ux,y+uy,z); view(2);
    daspect([1 1 1]);
    colormap([0 0 0])
    axis([lim(3) lim(4) lim(1) lim(2)] + .5 + [downsample 0 downsample 0]/2);      % which vector to show
    axis off;
    set(gca,'YDir','reverse');    
    
end