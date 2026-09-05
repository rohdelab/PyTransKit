clc
clear all %first make sure you complete the installation as instructed by installation.m 
filename = ''; %input filename here 
cd(filename);
addpath(strcat(filename,'TBM/Alternative_OT')); 
addpath(strcat(filename,'TBM/DGradient'));
addpath(strcat(filename,'TBM/GP')); 


%% 1. Load the preprocessed files and convert to matlab

imagefolder = (strcat(filename,'')); %folder with nifti images, can be saved as '.nii.gz' or '.nii'
matfolder = (strcat(filename,'')); %folder to save matfiles  
addpath(strcat(filename1, '/Load_nii/'))
a = dir(fullfile(imagefolder,'*.nii')); %can also read in '.nii.gz', just change the suffix
n = numel(a);
numfiles = length(a); 
for i=1:numfiles
    file = (strcat(imagefolder, a(i).name));
    nii = load_nii(file); %calls the load_nii function
    [filepath,name,ext] = fileparts(file);
    newname = strsplit(name, '.'); %stores the name of the original file
    image = nii.img;
    image(image<0)=0; %sets any negative pixels to 0
    image = image+0.1; %adds a small positive to background pixels, this helps the TBM 
    image = image./sum(image(:))*10^6; %normalize all the images so the total value of the pixels adds to 10^6
    disp(sum(image(:))) %sanity check
    disp(min(image(:))) 
    save (strcat(matfolder, newname{1}, '.mat'), "image") %saves the .matfiles
end
%% Load the template and convert to matlab. 
%You may also wish to threshold the template image to create sharp edges using thresh_template function
nii = load_nii(strcat(filename, '')); % load your template file. This is the euclidean mean of the images.  
template = nii.img; 
%template = thresh_template(template) use this to threshold the template
template(template<0)=0; 
template = template+0.1;
template = template./sum(template(:))*10^6; %normalize the template using same method as above 
disp(sum(template(:))) %sanity check
disp(min(template(:)))
save (strcat(filename, ''), "template") 

%% 2. Run TBM forward transformation and save the transport maps 
 % The multVOT function takes the following input parameters: (I0,I1,lambda,tot,sigma,numScales,level,gamma) 
 % Lambda is the tradeoff between minimizing MSE and curl, the recommended parameter is 50
 % Tot is the total sum of the pixel values.
 % Sigma controls the smoothness of the image. Recommended parameter is 1
 % Recommended numScales is 3 or 4. 
 % Gamma penalizes the mass transport, recommended is 0.1. Making this larger can make it harder to converge.
 % Ringing in the descent of curl and MSE is due to a step size that is too high. Look for the variable step_size and decrease it from 0.01 of a voxel to 0.005 of a voxel.
 % The multVOT will have a hard time producing deformations at the edges of the image if they are too close to the border. Zero pad the images sufficiently to avoid problems with matching the edges. The tradeoff of low step size is time it takes to converge.
 % If slow to converge, downsample the images. The ideal size is between 128 x 128 x 128 and 256 x 256 x 256. 
 % The default level of convergence is MSE= 2×10^(-4), which is indicated by the variable cutoff the VOT3D function. This can be changed if the match of the images is not satisfactory based on visual appearance.
matfolder = (strcat(filename,'')) %folder where you saved the matfiles
transportfolder = (strcat(filename,'')); %folder to save the transport maps 
b = dir(fullfile(matfolder,'*.mat'));
n = numel(b);
numfiles1 = length(b); 
for i=1:numfiles1
    file1 = (strcat(matfolder, b(i).name))
    [filepath,name,ext] = fileparts(file1)
    newname = strsplit(name, '.')
    template = load(strcat(filename,'')) 
    I0 = template.template; %I0 is the Euclidean mean template. You are computing the optimal transport distance from the template to each image.
    source = load(file1)
    I1 = source.image; %I1 is the segmented image 
    results = multVOT(double(I0),double(I1),50,10^6,1,4,0.25,0.1); %Apply 3D linear optimal transportation.
    save(strcat(transportfolder, newname{1}, '.mat'), "results") %saving the results for each image 
end
%% 3. Perform train:test split and concatenate transport maps 
transportfolder = (strcat(filename,''))
template = load(strcat(filename,''));
I0 = template.template;
labels = load(strcat(filename, 'labels.mat')); %loading labels for pLDA classification analysis, these should be saved as 1 (disease), and -1 (no disease)
labels = labels.labels;
growth = load(strcat(filename, 'growth.mat')); %loading continuous y variable for CCA regression analysis. Comment out if not needed.
growth = growth.growth;
covariates = load(strcat(filename, 'covariates.mat')); %loading covariates if using demographic/clinical information
covariates = covariates.covariates;
[X,Y,Z] = meshgrid(1:size(I0,2),1:size(I0,1),1:size(I0,3)); 
c = dir(fullfile(transportfolder,'*.mat'));
disp(c)
n = numel(c);
numfiles2 = length(c) 
train = randsample(170, 103) %randomsampling enter the following: (total files, number for training)
a = 1
b = 1
n = 1 
m = 1
for i=1:numfiles2
    file1 = (strcat(transportfolder, c(i).name))
    load(file1);
    if any(train==i)
       featurestrain(a,:) = single([results.f1(:)-X(:); results.f2(:)-Y(:); results.f3(:)-Z(:)]); %concatenating the transport maps of the images, subtracts the template from each transport map
       labelstrain(n,:) = single(labels(i,:)) 
       growthtrain(n,:) = single(growth(i,:))
       covariatestrain(n,:) = single(covariates(i,:,:))
       a = a+1
       n = n+1
    elseif any(train~=1)
        featurestest(b,:) = single([results.f1(:)-X(:); results.f2(:)-Y(:); results.f3(:)-Z(:)]); %concatenating the transport maps of the images, subtracts the template from each transport map
        labelstest(m,:) = single(labels(i,:))
        growthtest(m,:) = single(growth(i,:))
        covariatestest(m,:) = single(covariates(i,:,:))
        m = m+1
        b = b+1
    end
end
%% 4. Run PCA to perform dimensionality reduction in the training and testing data 
%If you are using demographic/clinical covariates, you need to save additional eigenvectors to the PCA_decomp code. See the function for more information.
%For subsequent regressions, you will need the following variables: Z96_train, Z96_test, A_mean, EIGENV1
[Z96_train,Z96_test,A_mean,EIGENV, EIGENV1, Z_train,Z_test] = PCA_decomp(featurestrain, featurestest);
%If you are using demographic/clinical covariates these should be
%concatenated with Z96_train and Z96_test prior to subsequent regressions:
% Z96_train = [Z96_train, covariatestrain]
% Z96_test = [Z96_test, covariatestest]
%% 5. Run PLDA to perform classification
%Inputs:
%   I0 is the template image 
%   Z96_train, Z96_test are the dimensionality reduced data matrices from PCA
%   Labelstrain, labelstest are the labels (-1 or +1)
%   A_mean is the mean of all feature vectors
%   featurestrain are the original transport maps training data 
%   [] - leave blank to calculate the optimal alpha value 
%   EIGENV1 is the eigenvector matrix
%Outputs: 
%   sensstrain is training ROC curve X-axis
%   specstrain is training ROC curve y-axis
%   sensstest is testing ROC curve X-axis
%   specstest is testing ROC curve y-axis
%   projtest are the scores in the testing data - can be used to calculate
%   AUROC
%   projtrain are the scores in the training data
%   sigmatest is the sigma value in the testing dataset
%   sigmatrain is the sigma value in the training dataset 
%   specificitytest is the specificity in the testing dataset
%   specificitytrain is the specificity in the training dataset
%   sensitivitytest is the sensitivity in the testing dataset
%   sensitivitytrain is the sensitivity in the training dataset
%   accuracytest is the accuracy in the testing dataset
%   accuracytrain is the accuracy in the training dataset
%   v1 is the principal PLDA direction 
%   pvaluehisttest is the p-value for the separation of histogram means in the
%   testing dataset
%   pvaluehisttrain is the p-value for the separation of histogram means in the
%   training dataset
%   histdiseasetest is the histogram for the disease/true value in the testing dataset
%   histcontroltest is the histogram for the control/false value in the testing dataset
%   histdiseasetrain is the histogram for the disease/true value in the training dataset
%   histcontroltrain is the histogram for the control/false value in the
%   training dataset
[sensstrain, specstrain, sensstest, specstest, projtest, projtrain, sigmatest, sigmatrain, specificitytest, specificitytrain, sensitivitytest, sensitivitytrain, accuracytest, accuracytrain, v1, pvaluehisttest, pvaluehisttrain, histdiseasetest, histcontroltest, histdiseasetrain, histcontroltrain] = visualizePLDA2(I0, Z96_train, Z96_test, labelstrain, labelstest, A_mean, featurestrain, [], 1, EIGENV1)
%% 6. Create histogram plots
n_std = [-2 -1 0 1 2]
figure;
nb_disease = histfit2(histdiseasetest)
hold on; 
nb_control = histfit1(histcontroltest)
nb = [nb_disease' nb_control'];
h_leg = legend('','','',''); % enter group names here
set(h_leg,'box','off');
center =0;
ax = gca; 
%xlim([n_std(1)*sigmatest,n_std(5)*sigmatest+center])
ax.XTick = [n_std(1)*sigmatest + center,n_std(2)*sigmatest+ center,center,n_std(4)*sigmatest+center,n_std(5)*sigmatest+center];
ax.XTickLabel = {strcat(num2str(n_std(1)),'\sigma'),strcat(num2str(n_std(2)),'\sigma'),'0',strcat(num2str(n_std(4)),'\sigma'),strcat(num2str(n_std(5)),'\sigma')};
set(gca,'fontsize',14); 
xlabel('Projection score')
ylabel('Probability density'); 
set(gca, 'FontName','Arial'); 

%% 7. Creat ROC plots 
figure; plot(1-sensstest, specstest,'LineWidth',3)
xlabel('1-Specificity (%)'); 
ylabel('Sensitivity (%)');
title('ROC Curve'); 
set(gca,'FontName','Arial'); 
set(gca,'FontSize',14);
xlim([-0.05 1])
ylim([0 1.05])
x = 0:0.1:1; y = x; 
hold on; plot(x,y,'Color','k','linewidth',0.01,'linestyle','--');
clear x y
%% 8. Run CCA to perform regression analysis against a continuous variable
%Inputs: 
%   Z96_train, Z96_test are the dimensionality reduced data matrices from PCA
%   growthtrain, growthtest are the continous independent variables 
%   covariatestrain, covariatestest are co-variates/confounding variables
%   to adjust for, these should be normalised and scaled from 0 to 1 before
%   inclusion, this can be left [] if no covariates
%Outputs:
%results, containing 
%y is the continuous variable for the training dataset
%z is the continuous variable for the testing dataset
%wcorr is the most correlated direction 
%CC_train is the correlation co-efficient in the training dataset 
%CC_test is the correlation co-efficient in the testing dataset 
%train_inds is the number of training samples
%test_inds is the number of testing samples
% Xw_train is used for the regression plot for training data 
% Xw_test is used for the regression plot for testing data 
% lambda is the standard deviation for wcorr 
% ptrain is the pvalue for the training data
% ptest is the pvalue for the testing data
[ results ] = Regression(Z96_train, Z96_test, growthtrain, growthtest, covariatestrain, covariatestest)

%% 9. Create regression scatter plots 
figure; 
s = scatter(results.Xw_test, results.z,'filled'); %this can be changed to Xw_train and y for training data plots
hold on; 
s.MarkerFaceAlpha = '0.5';
s.MarkerEdgeColor = [0.8500 0.3250 0.0980]
s.MarkerFaceColor = [0.8500 0.3250 0.0980];
hold on
set(gca,'FontSize',14); 
set(gca,'FontName','Arial'); 
xlabel({'Projection score'}); 
ylabel({'\delta volume, mL'})%change ylabel according to continous variable
ylim([-20, 120])%change ylim to suit continous variable scale 
set(gca,'XTick',[-2*results.lambda, -results.lambda, 0, results.lambda, 2*results.lambda]);
set(gca,'XTickLabel',[{'-2\sigma','-\sigma','0','\sigma','2\sigma'}]); 
hold on; 
coefficientstest = polyfit([results.Xw_test], [results.z],1); 
line = coefficientstest(1).*[results.Xw_test] + coefficientstest(2); 
hold on; 
plot([results.Xw_test], line,'k','LineWidth',3);  %line of best fit computed using all the data
%% 10. Reconstruct images using inverse transformation
% Inputs:
% I0 is the template image
% v1 is the principal PLDA direction, this can be substituted for wcorr for
% the CCA direction
% A_mean is the mean of all feature vectors
% sigmatrain is the sigma value in the training dataset, this can be
% substituted for wcorr for the CCA direction
% EIGENV1 is the eigenvector matrix
% Outputs:
% Generates 5 images in series with less to more likelihood of the discriminant tested (i.e. less and more hematoma expansion) the third image should be equivalent in
% appearance to I0
% Images can be viewed in 3D using imshow3Dfull 
clear image
n_std = [-2 -1 0 1 2]; 
[M,N,K] = size(I0);
[X,Y,Z] = meshgrid(1:N, 1:M, 1:K); 
V1 = zeros(3*M*N*K,1); 
i = 1:size(EIGENV1,2);
V1 = real(V1 + EIGENV1*v1(i)); 
V1 = V1/norm(V1); 
fprintf('Now computing the images... \n'); 
for i = 1:length(n_std)
    n = n_std(i); 
    i 
    fprintf('the std dev of the PLDA dir in the feature space is %d \n', sigmatrain);
    fprintf('the norm of the matrix V1 is %d \n', norm(V1)); 
    
    disp = A_mean + n*sigmatrain*V1'; 
   
            sz = size(disp,2)/3;

            u = reshape(disp(1:sz),M,N,K);
            v = reshape(disp(sz+1:2*sz),M,N,K); 
            w = reshape(disp(2*sz+1:3*sz),M,N,K); 

            f = double(X + u); 
            g = double(Y + v); 
            h = double(Z + w); 

            fields{i,1} = u; 
            fields{i,2} = v; 
            fields{i,3} = w; 

            [dfdx,dfdy,dfdz] = gradient(f); %compute Jacobian map
            [dgdx,dgdy,dgdz] = gradient(g); 
            [dhdx,dhdy,dhdz] = gradient(h);

            %And Jacobian determinant |Du|
            D = (dfdx.*dgdy.*dhdz + dfdy.*dgdz.*dhdx + dfdz.*dgdx.*dhdy - dfdx.*dgdz.*dhdy - dfdy.*dgdx.*dhdz - dgdy.*dhdx.*dfdz); %determinant

            image{i} = inpaint_nans3(griddata(double(f),double(g),double(h),double((I0./D)),double(X),double(Y),double(Z))); 
            image{i}(image{i}>max(I0(:))) = max(I0(:)); 
            image{i}(image{i}<min(I0(:))) = min(I0(:)); 
            image{i} = imrotate((image{i}./sum(image{i}(:))*10^6),90); %rotates brain image data 
        end
   
%% 11. View three axial slice examples of image series in a panel 
tiledlayout(3,5, 'TileSpacing', 'none', 'Padding', 'none')
nexttile,imagesc(image{1}(:,:,40)); colormap gray; axis off
nexttile,imagesc(image{2}(:,:,40)); colormap gray; axis off
nexttile,imagesc(image{3}(:,:,40)); colormap gray; axis off
nexttile,imagesc(image{4}(:,:,40)); colormap gray; axis off
nexttile,imagesc(image{5}(:,:,40)); colormap gray; axis off
nexttile,imagesc(image{1}(:,:,50)); colormap gray; axis off
nexttile,imagesc(image{2}(:,:,50)); colormap gray; axis off
nexttile,imagesc(image{3}(:,:, 50)); colormap gray; axis off
nexttile,imagesc(image{4}(:,:,50)); colormap gray; axis off
nexttile,imagesc(image{5}(:,:,50)); colormap gray; axis off
nexttile,imagesc(image{1}(:,:,60)); colormap gray; axis off
nexttile,imagesc(image{2}(:,:,60)); colormap gray; axis off
nexttile,imagesc(image{3}(:,:,60)); colormap gray; axis off
nexttile,imagesc(image{4}(:,:,60)); colormap gray; axis off
nexttile,imagesc(image{5}(:,:,60)); colormap gray; axis off

%% 12. Run multiple iterations and train:test split and save the data 
%This can be modified for regression analyses by changing the labels to the
%continuous variable and changing the output to CCA/Regression 
%Covariates can also be included 
filename = '';
filename1 = '';
cd(filename);
transportfolder = (strcat(filename,''));
resultsfolder = (strcat(filename, ''));
c = dir(fullfile(transportfolder,'*.mat'));
template = load(strcat(filename,''))
labels = load(strcat(filename, 'labels.mat'))
labels = labels.labels
I0 = template.image;
numfiles2 = length(c); 
numtests = 1;
for j = 1:numtests
    train = randsample(numfiles2, 0.8*numfiles2);
    a = 1
    b = 1
    n = 1
    m = 1
    for i=1:numfiles2
        file1 = (strcat(transportfolder, c(i).name));
        load(file1);
        if any(train==i);
            featurestrain(a,:) = single([results.f1(:)-X(:); results.f2(:)-Y(:); results.f3(:)-Z(:)]); %concatenating the displacement fields of the images
            labelstrain(m,:) = double(labels(i,:));
            %covariatestrain(m,:) = double(covariates(m,:));
            a = a + 1;
            m = m + 1;
        elseif any(train~=1)
            featurestest(b,:) = single([results.f1(:)-X(:); results.f2(:)-Y(:); results.f3(:)-Z(:)]); %concatenating the displacement fields of the images
            labelstest(n,:) = double(labels(i,:));
            %covariatestest(n,:)= double(covariates(n,:));
            b = b + 1;
            n = n + 1;
        end
    end
  [Z96_train,Z96_test,A_mean,EIGENV, EIGENV1, Z_train,Z_test] = PCA_decomp(featurestrain, featurestest);
  %If you are using demographic/clinical covariates these should be
  %concatenated with Z96_train and Z96_test prior to subsequent regressions:
  % Z96_train = [Z96_train, covariatestrain]
  % Z96_test = [Z96_test, covariatestest
  [sensstrain, specstrain, sensstest, specstest, projtest, projtrain, sigmatest, sigmatrain, specificitytest, specificitytrain, sensitivitytest, sensitivitytrain, accuracytest, accuracytrain, v1, pvaluehisttest, pvaluehisttrain, histdiseasetest, histcontroltest, histdiseasetrain, histcontroltrain] = visualizePLDA2 (I0, Z96_train, Z96_test, labelstrain, labelstest, A_mean, featurestrain, [], 1, EIGENV1)
  senstrain{j,:} = sensstrain(:);
  spectrain{j,:} = specstrain(:);
  senstest{j,:} = sensstest(:);
  spectest{j,:} = specstest(:);
  projtrain1{j,:} = projtrain(:);
  projtest1{j,:} = projtest(:);
  sigmatest1{j,:} = sigmatest(:);
  sigmatrain1{j,:} = sigmatrain(:);
  sensitivitytest1{j,:} = sensitivitytest(:);
  specificitytest1{j,:} = specificitytest(:);
  sensitivitytrain1{j,:} = sensitivitytrain(:);
  specificitytrain1{j,:} = specificitytrain(:);
  accuracytest1{j,:} = accuracytest(:);
  accuracytrain1{j,:} = accuracytrain(:);
  v1test{j,:} = v1(:,:);
  labelstest1{j,:} = labelstest(:,:);
  pvaluehisttest1{j,:} = pvaluehisttest(:);
  histdiseasetest1{j,:} = histdiseasetest(:);
  histcontroltest1{j,:} = histcontroltest(:);
  labelstrain1{j,:} = labelstrain(:,:);
  pvaluehisttrain1{j,:} = pvaluehisttrain(:);
  histdiseasetrain1{j,:} = histdiseasetrain(:);
  histcontroltrain1{j,:} = histcontroltrain(:);
  save ((strcat(resultsfolder, 'results_',num2str(j))),'senstrain', 'spectrain', 'senstest', 'spectest', 'projtrain1', 'projtest1', 'sigmatest1', 'sigmatrain1', 'sensitivitytest1', 'specificitytest1', 'sensitivitytrain1', 'specificitytrain1', 'accuracytest1', 'accuracytrain1', 'v1test', 'labelstest1', 'labelstrain1', 'pvaluehisttest1', 'histdiseasetest1', 'histcontroltest1', 'pvaluehisttrain1', 'histdiseasetrain1', 'histcontroltrain1');
end


