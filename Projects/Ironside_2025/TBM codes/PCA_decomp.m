function [Z96_train,Z96_test,A_mean,EIGENV,EIGENV1, Z_train,Z_test] = PCA_decomp(train_features, test_features)
%3. Perform PCA analysis on the training set and project the test set onto the training set. 

%Shinjini Kundu (c) 2014
%Uses the PCA trick described in the paper Cootes TF, Taylor CJ, Cooper DH,
%Graham J, Active Shape Models - Their Training and Application. Computer
%Vision and Image Understanidng 1995 61(1): 38-59

%Inputs:       train_features   training matrix 
%              test_features    test matrix
%Outputs:      returns eigenvectors and feature matrix

%%Information: If you have demographic or clinical covariates, adjust the variable cutoff2 accordingly, add to i the number of demographic/clinical covariates i.e. if there are four co-variates cutoff2 = i+4;

NUM_TRAIN = size(train_features,1); 
NUM_TEST = size(test_features,1); 

A_mean = mean(train_features,1); %mean of each training feature across all training subjects
A = (train_features - repmat(A_mean,NUM_TRAIN,1)); %treating the data vectors as if they were in columns in a D * N fashion, although features matrix is N x D, where A represents the matrix X'
B = (test_features - repmat(A_mean,NUM_TEST,1)); 

T = (A*A')./NUM_TRAIN; 
[V,D] = eig(T); %compute orthonoxrmal eigenvectors (V) and eigenvalues D such that Tv1 = dv1
%From eq. 41, A'*v1 is an eigenvector of S and has the same eigenvalue

V = fliplr(V); %order so that eigenvectors are in descreasing order
D = rot90(D,2); %order so that eigenvalues are in descending order

for i = 1:NUM_TRAIN
    eigenv = ((V(:,i)'*A)'./sqrt(D(i,i)*NUM_TRAIN)); %no need to make single and cause precision accuracy errors
    EIGENV(:,i) = eigenv; %column vectors of EIGENV 
    Z_train(:,i) = A*eigenv; 
    Z_test(:,i) = B*eigenv; 
end

Z_train = real(Z_train);
Z_test = real(Z_test); 

%project only onto components that capture 96% of variance
variances = (cumsum(sort(diag(D),'descend'))./sum(diag(D))); 
for i = 1:NUM_TRAIN
    if floor(variances(i)*100)/100 >= 0.96
        cutoff = i;
        cutoff2 = i; %% add to i the number of demographic/clinical covariates i.e. if there are four co-variates cutoff2 = i+4;
        break;
    end
end

for i = 1:cutoff2 
    eigenv = EIGENV(:,i); %column vectors of EIGENV
    EIGENV1(:,i) = eigenv;
end
for j = 1:cutoff
   eigenv1 = EIGENV(:,j);
   Z96_train(:,j) = A*eigenv1; 
   Z96_test(:,j) = B*eigenv1; 
end
end

