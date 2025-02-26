%Shinjini VOT (c) 2015
%VOT et al approach to solving for mass-preserving mapping


function [results] = VOT3D(I0,I1,f1,f2,f3,penalty,tot,sigma,level,scale,gamma)

%addpath /afs/.ece.cmu.edu/project/cbi/users/gustavor/Shinjini/Autism/TBM_features
%addpath /afs/.ece.cmu.edu/project/cbi/users/gustavor/Shinjini/Autism/TBM_features/DGradient

%step_size = 5*10^-5; %10^9; 
cutoff = 10^-4; 
cutoff0 = 2*10^-4; 
%sigma = 1;
DC_level = 0.1; %Create Gaussian kernel in 3D
%tot = 10^7; %integrates to total level
it = 0;
lambda = 0; 
p = 1; %generate plots

I0 = gen_pdf(I0,DC_level,sigma); 
I1 = gen_pdf(I1,DC_level,sigma);

[M,N,K]=size(I1);
[X,Y,Z]=meshgrid(1:N,1:M,1:K);
figure(1)

mask = ones(size(I0)); 
for i = 1:size(I0,3)
    mask(:,:,i) = im2bw(I0(:,:,i),min(I0(:)));
end

I0 = I0*tot; 
I1 = I1*tot; 

iter = 1;
converged = 0; 

results.f1 = f1; 
results.f2 = f2; 
results.f3 = f3; 

[C1,C2,C3] = curl(f1,f2,f3); 
C = mean(C1(:).^2 + C2(:).^2 + C3(:).^2);
results.curl = C; 


while(true)
    if ~p
        fprintf('Now on interation %d \n', iter); 
    end
    if iter ==1
        [ f1t,f2t,f3t,I0_recon,Ierror,flag ] = compVOTGradients( f1,f2,f3,I0,I1,lambda,gamma );
        err3(iter) = mean((Ierror(:)./I0(:)).^2); %relative MSE reported
        results.MSE3(iter) = err3; 
        results.mass(iter) = sum(sum(sum(((f1 - X).^2 + (f2 - Y).^2 + (f3 - Z).^2).*I0)));    
        err1(iter)=.5*sum(((Ierror(:)./I0(:)).*mask(:)).^2)/nnz(mask(:)); %in the area of the brain %numel(I0(:)); %relative MSE
        results.MSE1(iter) = err1; 
        results.I0_recon = I0_recon; results.I0 = I0; results.I1 = I1; 
        %step_size = (10^-2/10^scale)/(max(sqrt(f1(:).^2 + f2(:).^2 + f3(:).^2)));
        if scale==0 || scale>2
            step_size = (10^-(scale+2))/(max(sqrt(f1(:).^2 + f2(:).^2 + f3(:).^2)));
        else
            step_size = (10^-(scale+1))/(max(sqrt(f1(:).^2 + f2(:).^2 + f3(:).^2)));
        end
        if (flag)
            error('The initial deformation field is not diffeomorphic'); 
        else
            xk1_temp = f1-step_size*f1t; xk2_temp = f2-step_size*f2t; xk3_temp = f3-step_size*f3t;
            yk1_temp = xk1_temp; yk2_temp = xk2_temp; yk3_temp = xk3_temp;
            
            %check to make sure that the updated fields are diffeomorphic
            [~,~,~,~,~,flag] = compVOTGradients(yk1_temp,yk2_temp,yk3_temp,I0,I1,lambda,gamma);
            %if not diffeomorphic, need to take a smaller stepsize
            while(flag && ~converged)
                step_size = step_size/2; 
                if step_size < (10^-8) %if there is no stepsize that will enable a diffeomorphic deformation, you have converged
                    converged = 1;
                    step_size = 0;
                    results.f1 = f1; 
                    results.f2 = f2; 
                    results.f3 = f3;  
                end
                xk1_temp = f1-step_size*f1t; xk2_temp = f2-step_size*f2t; xk3_temp = f3-step_size*f3t;
                yk1_temp = xk1_temp; yk2_temp = xk2_temp; yk3_temp = xk3_temp;
                [~,~,~,~,~,flag] = compVOTGradients(yk1_temp,yk2_temp,yk3_temp,I0,I1,lambda,gamma);
            end
            xk1 = xk1_temp; xk2 = xk2_temp; xk3 = xk3_temp;
            yk1 = yk1_temp; yk2 = yk2_temp; yk3 = yk3_temp; 
            %fprintf('the stepsize is %d \n', step_size);
               
            yk1minus1 = zeros(size(I0)); yk2minus1 = zeros(size(I0)); yk3minus1 = zeros(size(I0)); 
            xk1minus1 = zeros(size(I0)); xk2minus1 = zeros(size(I0)); xk3minus1 = zeros(size(I0)); 
        end
    end
   
    
    if iter > 1
        [ f1t,f2t,f3t,I0_recon,Ierror ] = compVOTGradients( yk1minus1,yk2minus1,yk3minus1,I0,I1,lambda,gamma ); 
        xk1_temp = yk1minus1-step_size*f1t; xk2_temp = yk2minus1-step_size*f2t; xk3_temp = yk3minus1-step_size*f3t;
        yk1_temp = xk1_temp + (iter-2)/(iter+1)*(xk1_temp - xk1minus1); yk2_temp = xk2_temp + (iter-2)/(iter+1)*(xk2_temp - xk2minus1); yk3_temp = xk3_temp + (iter-2)/(iter+1)*(xk3_temp - xk3minus1);
        %step_size = (10^-2/10^scale)/(max(sqrt(f1(:).^2 + f2(:).^2 + f3(:).^2))); 
        if scale>2
            step_size = (10^-(scale+2.5))/(max(sqrt(f1(:).^2 + f2(:).^2 + f3(:).^2)));
        elseif scale ==0
            step_size = (10^-(scale+1.5))/(max(sqrt(f1(:).^2 + f2(:).^2 + f3(:).^2)));
        elseif scale < 2
            step_size = (10^-(scale+2.5))/(max(sqrt(f1(:).^2 + f2(:).^2 + f3(:).^2)));
        elseif scale ==2
            step_size = (10^-(scale+1.5))/(max(sqrt(f1(:).^2 + f2(:).^2 + f3(:).^2)));
        end
% % %         if scale > 3 || scale == 0
% % %             step_size = (10^-(scale+2))/(max(sqrt(f1(:).^2 + f2(:).^2 + f3(:).^2)));
% % %         elseif scale ==3 || scale ==1
% % %             step_size = (10^-(scale+1))/(max(sqrt(f1(:).^2 + f2(:).^2 + f3(:).^2)));
% % %         else
% % %             step_size = (10^-(scale+0.5))/(max(sqrt(f1(:).^2 + f2(:).^2 + f3(:).^2)));
% % %         end
         %check whether updated fields are diffeomorphic
        [~,~,~,~,~,flag] = compVOTGradients(yk1_temp,yk2_temp,yk3_temp,I0,I1,lambda,gamma);
        while (flag && ~converged)
            step_size = step_size/2; 
            if step_size < (10^-8) %if there is no stepsize that will enable a diffeomorphic deformation, you have converged
                converged = 1; 
                step_size = 0; 
                if iter==2
                    results.f1 = f1; 
                    results.f2 = f2; 
                    results.f3 = f3;
                end
            end
            xk1_temp = yk1minus1-step_size*f1t; xk2_temp = yk2minus1-step_size*f2t; xk3_temp = yk3minus1-step_size*f3t;
            yk1_temp = xk1_temp + (iter-2)/(iter+1)*(xk1_temp - xk1minus1); yk2_temp = xk2_temp + (iter-2)/(iter+1)*(xk2_temp - xk2minus1); yk3_temp = xk3_temp + (iter-2)/(iter+1)*(xk3_temp - xk3minus1);
            [~,~,~,~,~,flag] = compVOTGradients(yk1_temp,yk2_temp,yk3_temp,I0,I1,lambda,gamma);
        end
        xk1 = xk1_temp; xk2 = xk2_temp; xk3 = xk3_temp;
        yk1 = yk1_temp; yk2 = yk2_temp; yk3 = yk3_temp; 
       % fprintf('the stepsize is %d \n', step_size); 
    end
    
    if (~converged)
        yk1minus2 = yk1minus1; yk2minus2 = yk2minus1; yk3minus2 = yk3minus1; 
        yk1minus1 = yk1; yk2minus1 = yk2; yk3minus1 = yk3; 
        xk1minus1 = xk1; xk2minus1 = xk2; xk3minus1 = xk3; 
    end
    
   %plotting code
   if (p)
       subplot(221)
       %imshow(squeeze(sum(I0_recon,1)),[]); 
       imshow(I0_recon(:,:,round(K/2)),[]); %shows the middle section of the image
       title('$$det(D{\bf f})I_1({\bf f})$$','interpreter','latex','fontsize',20)
       freezeColors
       subplot(222)
       if iter==1
           showgrid(squeeze(X(:,:,round(K/2))-f1(:,:,round(K/2))),squeeze(Y(:,:,round(K/2))-f2(:,:,round(K/2))),3)
       else
           showgrid(squeeze(X(:,:,round(K/2))-yk1minus2(:,:,round(K/2))),squeeze(Y(:,:,round(K/2))-yk2minus2(:,:,round(K/2))),3)
        end
        title('$${\bf x}-{\bf f}({\bf x})$$','interpreter','latex','fontsize',20)
        subplot(2,2,3)
   end
    %%err(iter)=.5*sum(((Ierror(:)./I0(:))).^2)/numel(I0(:));
    err1(iter)=.5*sum(((Ierror(:)./I0(:)).*mask(:)).^2)/nnz(mask(:)); %in the area of the brain %numel(I0(:)); %relative MSE
    err2(iter)=.5*sum(((Ierror(:)./I0(:))).^2)/numel(I0(:)); %relative MSE
    err3(iter) = mean((Ierror(:)./I0(:)).^2); %relative MSE reported
    err4(iter) = 0.5*sum(Ierror(:).^2); %the data term in the update equation
    if err1(end)/err1(1) > level %when the MSE is reduced to a quarter of its original value, start penalizing the solution
        if it==0
            it = iter; 
        end
        lambda = penalty; %*(1.03)^(iter-it);
    end
    if (p)
        plot(err3,'linewidth',2)
        title('MSE: $$\frac{1}{2}\|det(D{\bf f})I_1({\bf f})-I_0\|^2$$','interpreter','latex','fontsize',20)
        grid on
        subplot(2,2,4)
    end
    mass(iter) = sum(sum(sum(((yk1minus2 - X).^2 + (yk2minus2 - Y).^2 + (yk3minus2 - Z).^2).*I0))); 
    if iter==1
        [C1,C2,C3] = curl(f1,f2,f3); 
    else
        [C1,C2,C3] = curl(yk1minus2,yk2minus2,yk3minus2); 
    end
    C = sum(C1(:).^2 + C2(:).^2 + C3(:).^2); %L2 norm
    errorcurl(iter)=0.5*C;
    objective = err4 + lambda*errorcurl; 
    %fprintf('The objective value is %d \n', objective(end)); 
    if iter > 50 && (objective(iter-1) < objective(iter))% || (errorcurl(iter)-errorcurl(iter-1))/errorcurl(iter) > 0.05) %prevents curl or objective value from shooting up at the end
        return
    end
    if p
        plot(errorcurl,'r','linewidth',2)
        title('Curl: $$\frac{1}{2}\|\nabla\times {\bf f}\|^2$$','interpreter','latex','fontsize',20)
        grid on
        drawnow
    end
    %%end of plotting code
    
    if (converged ||  iter>500 && (scale ==1) && (round(err3(iter))-round(err3(iter-1)))*10^6 ==0 || (scale ==0 &&(round(err3(iter)*10^3))/10^3 <= cutoff) || (scale~=0 && err3(iter) <= cutoff0) ) %|| (scale ==1 &&(round(err3(iter)*10^3))/10^3 <= cutoff0*10)
        return;
    end
    I0_recon = I0_recon./sum(I0_recon(:))*10^6;
    results.f1 = yk1minus2; 
    results.f2 = yk2minus2; 
    results.f3 = yk3minus2; 
    results.I0_recon = I0_recon;
    results.mass = mass; 
    results.MSE2 = err2; 
    results.MSE1 = err1;
    results.MSE3 = err3; 
    results.curl = 2*errorcurl/(numel(C)); 
    results.I0 = I0; 
    results.I1 = I1; 
    results.objective = objective;
    iter = iter + 1;
    

    
end
    

end 


%     [ f1t,f2t,f3t,I0_recon,Ierror ] = compVOTGradients( f1,f2,f3,I0,I1 ); 
%     
% %    step_size=.05/(max(sqrt(f1t(:).^2+f2t(:).^2+f3t(:).^2)));
%     f1=f1-step_size*f1t;
%     f2=f2-step_size*f2t;
%     f3=f3-step_size*f3t; 
      
