function [ Vec,eigval ] = PLDA(Data,Labels,Alpha,nPLDA)
%Penalized LDA 
%   Detailed explanation goes here
%Input:
%Data  = The data matrix each column represents one sample 
%Labels= Labels corresponding the data matrix
%Alpha = The weight of PCA in (LDA+Alpha*PCA);
%nPLDA = Number of PLDA directions
%Output: 
%PLDA:   Vector of PLDA directions
%
%Author: Soheil Kolouri
%Date:   09/26/2012
[nfeatures,nsamples]=size(Data);
if ~(exist('nPLDA'))
    if nfeatures<nsamples
        nPLDA=nfeatures;
    else
        nPLDA=nsamples;
    end
end
x=mean(Data')';
nclasses=unique(Labels);
SB=zeros(nfeatures,nfeatures);
SW=zeros(nfeatures,nfeatures);
for i=1:length(nclasses)
    Class{i}=Data(:,Labels==nclasses(i));
    disp(Class)
    mu(:,i)=mean((Class{i})')';
    Sb{i}=(mu(:,i)-x)*(mu(:,i)-x)';
    Sw{i}=zeros(nfeatures,nfeatures);
    for j=1:size(Class{i},2)
        Sw{i}=Sw{i}+(Class{i}(:,j)-mu(:,i))*(Class{i}(:,j)-mu(:,i))';
    end
    SB=SB+(1/nsamples)*size(Class{i},2)*Sb{i};
    SW=SW+(1/nsamples)*Sw{i};
end
ST=SB+SW;
SWNew=SW+(Alpha*nsamples)*eye(size(SW));
[eigvec,eigvalue]=eig(ST,SWNew);
eigvalue(isnan(eigvalue))=0;
eigvalue(abs(eigvalue)==inf)=1e10;
eigval = diag(eigvalue);
[eigval I] = sort((eigval),'descend');
Vec=real(eigvec(:,I(1:nPLDA)));

for i=1:size(Vec,2)
    Vec(:,i)=Vec(:,i)/sqrt(Vec(:,i)'*Vec(:,i));
end
end


