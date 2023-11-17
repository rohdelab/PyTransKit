%Shinjini Kundu (c) 2014
%Discrimination of images by PLDA, visualization code

function [sensstrain, specstrain, sensstest, specstest, projtest, projtrain, sigmatest, sigmatrain, specificitytest, specificitytrain, sensitivitytest, sensitivitytrain, accuracytest, accuracytrain, v1, pvaluehisttest, pvaluehisttrain, histdiseasetest, histcontroltest, histdiseasetrain, histcontroltrain] = visualizePLDA2(I0, Z96_train, Z96_test, labelstrain, labelstest, A_mean, features_matrix, alpha, dir, EIGENV1)
%Inputs:   
%           Z96_train, Z96_test        reduced dimension feature matrices
%           labels                     labels (either -1 or +1)
%           A_mean                     feature mean of original matrix 
%           feature_matrix             original feature matrix
%           alpha                      penalty value for PLDA (needed when
%                                        visualizing discriminant direction)
%           dir                        which PLDA direction you want to
%                                        visualize (can only be one number), or w vector showing the direction to visualize


n_std = [-2 -1 0 1 2]; 
type = 'brain'; %will rotate images if brain data
NUM_SUBJECTS = numel(labelstrain);
NUM_TEST = numel(labelstest)
[M,N,K] = size(I0);
[X,Y,Z] = meshgrid(1:N, 1:M, 1:K); 

labelstest = double(labelstest);
labelstrain = double(labelstrain);
Curveoption.low = 0.01; %lower bound of alpha 
Curveoption.high = 100; % 75 upper bound of alpha
Curveoption.step = 7; % 5 step size of increaseing alpha
Curveoption.nPLDA = 3; %the dimension of subspaces to be compared
    
    
fprintf('Now calculating alpha \n'); 
%if isempty(alpha) && numel(dir)==1
alpha = Calculate_Alpha(Z96_train, EIGENV1, labelstrain, Curveoption); 
fprintf('The chosen alpha is %d \n', alpha); 
%end

%A = feature_matrix - repmat(A_mean,NUM_SUBJECTS,1); %mean-subtracted original feature matrix

fprintf('Now running PLDA \n'); 
v1 = 1
    [Vec, ~] = PLDA(Z96_train', labelstrain, alpha);
    v1 = real(Vec(:,1));


V1 = zeros(3*M*N*K,1); 
i = 1:size(Z96_train,2)
V1 = real(V1 + EIGENV1*v1(i)); 


projtest = Z96_test*v1;
sigmatest = std(projtest); 
projtrain = Z96_train*v1;
sigmatrain = std(projtrain);

%get p values
classifiedtest(projtest<0) = -1;
classifiedtest(projtest>0) = 1;
C = confusionmat(labelstest,classifiedtest);
acc = (C(1,1) + C(2,2))/sum(C(:)); 
if (acc < 0.5) 
    classifiedtest = classifiedtest*-1; 
end

[C, order] = confusionmat(classifiedtest,labelstest); 
posInd = find(order == -1); negInd = find(order == 1); 
        
sensitivitytest = C(posInd, posInd)/(C(posInd,posInd)+C(posInd,negInd)); 
specificitytest = C(negInd,negInd)/(C(negInd,negInd)+C(negInd,posInd)); 
accuracytest = (C(1,1) + C(2,2))/sum(C(:));

[h,pvaluehisttest] = ttest2(projtest(labelstest==1), projtest(labelstest==-1))

classifiedtrain(projtrain<0) = -1;
classifiedtrain(projtrain>0) = 1;
C = confusionmat(labelstrain,classifiedtrain);
acc = (C(1,1) + C(2,2))/sum(C(:)); 
if (acc < 0.5) 
    classifiedtrain = classifiedtrain*-1; 
end

[C, order] = confusionmat(classifiedtrain,labelstrain); 
posInd = find(order == -1); negInd = find(order == 1); 
        
sensitivitytrain = C(posInd, posInd)/(C(posInd,posInd)+C(posInd,negInd)); 
specificitytrain = C(negInd,negInd)/(C(negInd,negInd)+C(negInd,posInd)); 
accuracytrain = (C(1,1) + C(2,2))/sum(C(:));

[h,pvaluehisttrain] = ttest2(projtrain(labelstrain==1), projtrain(labelstrain==-1))
   
%create ROC curve

roc = unique(projtest);
classifiedtest(projtest<0) = -1;
classifiedtest(projtest>0) = 1;
[C, order] = confusionmat(labelstest,classifiedtest); 
posInd = find(order == -1); negInd = find(order == 1); 
acc = (C(1,1)+C(2,2))/sum(C(:)); 
if acc < 0.5
    mult = -1; 
else 
    mult = 1;
end
          
for i = 1:numel(roc)
    classifiedtest(projtest<roc(i)) = -1;
    classifiedtest(projtest>=roc(i)) = 1;
    [C, order] = confusionmat(labelstest,classifiedtest*mult); 
    posInd = find(order == -1); negInd = find(order == 1);
    senstest = C (posInd, posInd)/(C(posInd,posInd)+C(posInd,negInd)); 
    spectest = C (negInd,negInd)/(C(negInd,negInd)+C(negInd,posInd)); 
    accutest = (C(1,1) + C(2,2))/sum(C(:));
    specstest(i) = spectest; sensstest(i) = senstest; 
end

if sensstest(1) <0.5
    sensstest = flipud(sensstest)
    specstest = flipud(specstest)
end

roc = unique(projtrain);
classified(projtrain<0) = -1;
classified(projtrain>0) = 1;
[C, order] = confusionmat(labelstrain,classified); 
posInd = find(order == -1); negInd = find(order == 1); 
acc = (C(1,1)+C(2,2))/sum(C(:)); 
if acc < 0.5
    mult = -1; 
else 
    mult = 1;
end
          
for i = 1:numel(roc)
    classified(projtrain<roc(i)) = -1;
    classified(projtrain>=roc(i)) = 1;
    [C, order] = confusionmat(labelstrain,classified*mult); 
    posInd = find(order == -1); negInd = find(order == 1);
    senstrain = C (posInd, posInd)/(C(posInd,posInd)+C(posInd,negInd)); 
    spectrain = C (negInd,negInd)/(C(negInd,negInd)+C(negInd,posInd)); 
    accutrain = (C(1,1) + C(2,2))/sum(C(:));
    specstrain(i) = spectrain; sensstrain(i) = senstrain; 
end

if sensstrain(1) <0.5
    sensstrain = flipud(sensstrain)
    specstrain = flipud(specstrain)
end



figure; plot(1-specstrain,sensstrain,'LineWidth',3)
xlabel('False positive rate'); 
ylabel('True positive rate');
title('Classifier ROC Curve'); 
set(gca,'FontName','Times New Roman'); 
set(gca,'FontSize',18);
xlim([-0.05 1])
ylim([0 1.05])
x = 0:0.1:1; y = x; 
hold on; plot(x,y,'Color','k','linewidth',0.01,'linestyle','--');
clear x y
%grid on;


%create histogram and remaining figures

figure;
histdiseasetest = projtest(labelstest==1)
histcontroltest = projtest(labelstest==-1)
if mean(histdiseasetest) < mean(histcontroltest)
    histdiseasetest = histdiseasetest(:)*-1;
    histcontroltest = histcontroltest(:)*-1;
    V1 = V1(:)*-1;
end

histdiseasetrain = projtrain(labelstrain==1)
histcontroltrain = projtrain(labelstrain==-1)
if mean(histdiseasetrain) < mean(histcontroltrain)
    histdiseasetrain = histdiseasetrain(:)*-1;
    histcontroltrain = histcontroltrain(:)*-1;
end
xbtest = linspace(min(projtest),max(projtest),15);
xbtrain = linspace(min(projtrain),max(projtrain),15);
[nb_disease,xbtest]=hist(projtest(labelstest==-1),xbtest);
%bh=bar(xb,nb_disease./NUM_SUBJECTS*100);
%set(bh,'facecolor',[1 1 0]);
hold on; 
[nb_control,xbtest]=hist(projtest(labelstest==1),xbtest);
%bh=bar(xb,nb_control./NUM_SUBJECTS*100);
nb = [nb_disease' nb_control'];
bh = bar(xbtest,nb./NUM_SUBJECTS*100,1.3,'grouped');
%set(bh,'facecolor',[0 1 1]);
h_leg = legend('no growth','growth'); 
set(h_leg,'box','off');

center =0; %mean(proj); 

ax = gca; 
ax.XTick = [n_std(1)*sigmatest + center,n_std(2)*sigmatest+ center,center,n_std(4)*sigmatest+center,n_std(5)*sigmatest+center];
ax.XTickLabel = {strcat(num2str(n_std(1)),'\sigma'),strcat(num2str(n_std(2)),'\sigma'),'0',strcat(num2str(n_std(4)),'\sigma'),strcat(num2str(n_std(5)),'\sigma')};
set(gca,'fontsize',18); 
xlabel('Projection score')
ylabel('Percentage incidents'); 
set(gca, 'FontName','Times New Roman'); 

end
 


