%Shinjini VOT (c) 2015
%wrapper code to run VOT method

function final_results = multVOT(I0,I1,lambda,tot,sigma,numScales,level,gamma)

%addpath /afs/.ece.cmu.edu/project/cbi/users/gustavor/Shinjini/Autism/TBM_features/DGradient; 
%addpath /afs/.ece.cmu.edu/project/cbi/users/gustavor/Shinjini/Autism/TBM_features/codegen/mex/GPExpand; 
%addpath /afs/.ece.cmu.edu/project/cbi/users/gustavor/Shinjini/Autism/TBM_features/codegen/mex/GPReduce;

%numScales = 4; %can set the number of multi-resolution scales for initialization
%lambda = 150; %50; %3.5*10^-12; %tradeoff between minimizing MSE and curl

%2. Initial potential field of curl-free map (the identity)
[~,~,K]=size(I1);
[sx,sy,sz] = size(I0); 

globalIter = 1; 

tic
for scale = numScales:-1:0 
    fprintf('Now starting scale %d \n', scale);
    I0_down = I0; 
    I1_down = I1; 
    
    if scale~=0
        for i = 1:scale
            [X_down1,~,~] = meshgrid(1:2^(i-1):sy,1:2^(i-1):sx,1:2^(i-1):sz); 
            I0_down = GPReduce(I0_down); 
            I1_down = GPReduce(I1_down);
            newdim = size(X_down1);
        end 
    end
    [X,Y,Z] = meshgrid(1:size(I0_down,2),1:size(I0_down,1),1:size(I0_down,3));
    if globalIter==1
        f0 = X; 
        g0 = Y; 
        h0 = Z; 
    end
    results = VOT3D(I0_down,I1_down,f0,g0,h0,lambda,tot,sigma,level,scale,gamma);
    if globalIter == numScales + 1
        t = toc
        final_results = results
        final_results.time = t;
        fprintf('the final curl is %d \n', results.curl(end)); 
        fprintf('the final MSE overall is %d \n', final_results.MSE3(end)); 
        fprintf('the final MSE in the tissue is %d \n', final_results.MSE1(end)); 
        fprintf('the final time it took was %d minutes \n', t/60); 
        figure; imagesc(results.I0_recon(:,:,round(K/2))); colorbar; title('morphed'); 
        figure; imagesc(results.I0(:,:,round(K/2))); colorbar; title('target');
        figure; imagesc(results.I1(:,:,round(K/2))); colorbar; title('source');
        return; 
    else
%         [ ~,~,~,~,~,flag ] = compVOTGradients( results.f1,results.f2,results.f3,zeros(size(I0_down)),zeros(size(I0_down)),0 );
%         if (flag)
%             fprintf('I am not diffeomorphic! \n');
%         else
%             fprintf('I am diffeomorphic \n'); 
%         end
        [X2,Y2,Z2] = meshgrid(1:size(X_down1,2),1:size(X_down1,1),1:size(X_down1,3));     
        f0 = 2*GPExpand(results.f1-X,newdim)+X2; 
        g0 = 2*GPExpand(results.f2-Y,newdim)+Y2; 
        h0 = 2*GPExpand(results.f3-Z,newdim)+Z2; 
        [ ~,~,~,~,~,flag ] = compVOTGradients( f0,g0,h0,zeros(size(f0)),zeros(size(f0)),0,gamma );
        sigma_f = 2; 
        if (flag)
            %fprintf('I am not diffeomorphic! \n');
            [Xt,Yt,Zt]=meshgrid(-3*sigma_f:3*sigma_f,-3*sigma_f:3*sigma_f,-3*sigma_f:3*sigma_f);
            phi = gaussian_bf(Xt,Yt,Zt,sigma_f); %normalized gaussian kernel in 3D
            f0 = 2*GPExpand(convn(results.f1-X,phi,'same'),newdim)+X2; 
            g0 = 2*GPExpand(convn(results.f2-Y,phi,'same'),newdim)+Y2; 
            h0 = 2*GPExpand(convn(results.f3-Z,phi,'same'),newdim)+Z2;
%         else
%             fprintf('I am diffeomorphic \n'); 
        end
        globalIter = globalIter + 1; 
    end
end


end