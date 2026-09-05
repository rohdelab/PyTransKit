%Shinjini Kundu (c) 2016
%Note: when using covariates, check to make sure that scaling of each
%column is in the same order of magnitude as the other columns. If not,
%must whiten the covariates matrix (column mean 0 and standard deviation 1)

function [ results ] = Regression( matrixtrain, matrixtest, y, z, covariatestrain, covariatestest)
%Finds regression direction in the TBM space or image space that correlates 
%most with the independent variable of interest. Code also can adjust to 
%remove the effects of confounding variables
%inputs:   matrixtrain         Training data matrix, oriented NxD, where N is
%                              subjects (this is PCA-reduced
%                              data matrix).
%          matrixtest          Test data matrix, oriented NxD, where N is
%                              subjects (this is PCA-reduced
%                              data matrix).
%          y                   independent variable in column matrix for
%                              training
%          z                   independent variable in column matrix for
%                              testing
%          covariatestrain     matrices of confounding variables/covariates, oriented
%                              Nxd for training and testing
%          covariatestest       
%output:   wcorr               most correlated direction in the space
%          lambda              std of projection scores
%          CC_train            Pearson's correlation coefficient for
%                               training set
%          CC_test             Pearson's correlation coefficient for
%                               testing set
%          scores              projection scores
%          train_inds          indices used for training
%          test_inds           indices used for testing
%          ptrain              p value for training
%          ptest               pvalue for testing


Data = matrixtrain'; %orient matrix DxN
Datatest = matrixtest';
num = size(Data,2);
numtest = size(Datatest,2);
vizaxis = 'y'; %determine whether you want to visualize brains acros the projection score(y) or independent variable(x)

    %create testing and training indices    
    train_inds = 1:num; %randomly select 70% of samples for training
    test_inds = 1:numtest;

    if ~isempty(covariatestrain)
        v = y - covariatestrain/(covariatestrain'*covariatestrain)*covariatestrain'*y; 
        z = z - covariatestest/(covariatestest'*covariatestest)*covariatestest'*z; 
    elseif isempty(covariatestrain)
        v = y;
        z = z;
    end

    wcorr = Data(:,train_inds)*(v(train_inds)-mean(v(train_inds)))/sqrt((v(train_inds)-mean(v(train_inds)))'*Data(:,train_inds)'*Data(:,train_inds)*(v(train_inds)-mean(v(train_inds))));
    lambda = std(Data'*wcorr); 


    Xw_train=Data(:,train_inds)'*wcorr; %Projection of the dataset onto the most correlative direction 
    Xw_test=Datatest(:,test_inds)'*wcorr;
    
    scores(train_inds) = Xw_train; scores(test_inds) = Xw_test; 
    
   
    %Calculate the correlation coefficient
    
        [CC_train, ptrain] = corr(double(Xw_train),v(train_inds)); %computes Pearson's correlation coefficient
        [CC_test, ptest] = corr(double(Xw_test),z(test_inds)); 
        
        results.y = v;
        results.z = z;
        
results.wcorr = wcorr;
results.CC_train = CC_train; 
results.CC_test = CC_test; 
results.train_inds = train_inds; 
results.test_inds = test_inds;
results.Xw_train = Xw_train;
results.Xw_test = Xw_test;
results.lambda = lambda;
results.ptrain = ptrain;
results.ptest = ptest;




end