%Shinjini Kundu (c) 2016
%Transport-Based Morphometry analysis codes

function [ Thresh, error_subspace ] = Calculate_Alpha( FINAL_FEATS, EIGENV1, labels, Curveoption )
%This piece of code runs the PLDA function with different values of alpha
%and plots two error curves
%The projection metric distance of two consequent subspaces
%written originally by Soheil Kolouri, modified by Shinjini Kundu 

nPLDA = Curveoption.nPLDA; 
counter = 0; 

x = Curveoption.low:Curveoption.step:Curveoption.high; 

for i = 1:size(FINAL_FEATS,2)
    %load(char(strcat('eigenvector_',num2str(i))));
    VecPCA(:,i) = EIGENV1(:,i); 
end

VecPCA = double(VecPCA); 

for Alpha = x
    Alpha;
    counter = counter + 1; 
    [PLDA_directions] = PLDA(FINAL_FEATS', labels, Alpha, nPLDA); 
    Vec(:,:,counter) = VecPCA*PLDA_directions; 
    if counter > 1
        Vec(:,:,counter) = Vec(:,:,counter)*diag(sign(diag(Vec(:,:,counter)'*Vec(:,:,counter-1)))); 
        error_subspace(counter-1) = Projection_metric(Vec(:,:,counter),Vec(:,:,counter-1)); 
    end
end

%Choose alpha such that the absolute value of the relative change does not exceed 0.2%
cutoff = 0.5; %0.015; 
change = abs(gradient(double(error_subspace)));
ind = min(find(change*100 < cutoff)); 
 
%Calculate twice the half life of Alpha, 

% % % % %Fit an exponential to log(error_subspace)
% % % % f = @(a,b,x) log(a) - b*x; 
% % % % options = fitoptions('Method','LinearLeastSquares');
% % % % F_fitted = fit(x(1:end-1)',log(error_subspace)',f, ...
% % % %     'StartPoint', [1,x(1)], ...
% % % %     'Lower', [0,0], 'Robust', 'LAR');
% % % % coeff = coeffvalues(F_fitted); %Get coefficients of fitted function 
% % % % Thresh = log(2)*(1/coeff(2)); %Calculate twice the half life = 2(log(2)/b)

Thresh = x(ind);
%Thresh = 1; %default value 

figure; 
plot(x(1:end-1),error_subspace,'linewidth',2)
title({'Stability of subspace', 'with respect to \alpha'}, 'fontsize',24)
ylabel({'Projection metric between', 'two consequent subspaces'}, 'fontsize', 20); 
xlabel('\alpha','fontsize',20); 
grid on; 
set(gca,'fontsize',20); 
set(gca,'FontName','Times New Roman');
yL = get(gca, 'YLim');
hold on; 
line([Thresh Thresh],yL,'Color','r','linewidth',2);

% grid on; 
% plot(x,coeff(1)*exp(- coeff(2)*x),'k'); %plot the exponential


end

