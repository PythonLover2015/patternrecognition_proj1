%Demissew Kessela
%ML Classifier
clear
clc
[traindata] = dlmread('iris_training.txt');
[testdata] = dlmread('iris_test.txt');
nf = size(traindata,2);% number of features + class column
d=nf-1; %dimensiones
nc = 3; % number of class
testdata_size = size(testdata,1);
traindata_size = size(traindata,1);%number samples in each class
%%
m = zeros(1,nc);
for n=1:nc
    for i=1:traindata_size
        if traindata(i,1)== n
            m(n) = m(n) + 1;
        end
    end
    %a(n,1+sum(m(1:n-1)):sum(m(1:n)),nf)= traindata(1+sum(m(1:n-1)):sum(m(1:n)),1:nf)
end
g =zeros(1,nc);
num_error = 0;
for k =1 :testdata_size
    for i=1:nc
    mu_t = mean(traindata(1+sum(m(1:i-1)):sum(m(1:i)),2:nf));
    sigma=cov(traindata(1+sum(m(1:i-1)):sum(m(1:i)),2:nf));
    W_t(:,:) = -0.5*inv(sigma);
    w_t = (sigma\mu_t')';
    w_io = -0.5*(mu_t/sigma)*mu_t' - 0.5*log(det(sigma));
    g(i) = testdata(k,2:nf)*W_t*testdata(k,2:nf)'+ w_t*testdata(k,2:nf)'+ w_io;
    end
    [M,I] = max(g);
    if (I) ~= testdata(k,1)
        num_error = num_error + 1;
    end
    g =zeros(1,nc);
    %here reset p
end
%%
classification_error = num_error*100/testdata_size;
fprintf('classification error    %.2f%%\n', classification_error);
fprintf('classification performance    %.2f%%\n', 100 - classification_error);
% find the pdf for each class and select the x that 
%gives the maximum px
%select max px of the classes
