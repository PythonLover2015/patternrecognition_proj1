function [ ] = mlestimator(filename,num_classes)
%return data read from file
%   filename: file name to which holds data
%   data : return data maxtrix
%   Assumes a text file with (category x1 x2 x3... ) format
%   st: total samples
%   nf: number of features/dimenstions
clc;
fileID = -1;
errmsg  ='';
%{
while fileID < 0
    filename = input('Open file: ', 's');
    [fileID, errmsg] = fopen(filename);
end   disp(errmsg)
%}
[fileID, errmsg] = fopen(filename);
disp(errmsg)
NUM_CLASSES = num_classes;
[data] = textread(filename);

c = size(data,1)/NUM_CLASSES;%number samples in each class
nf = size(data,2);% number of features + class column

mu_t = zeros(NUM_CLASSES,nf-1);
W_t = zeros(NUM_CLASSES*(nf -1),nf-1);
w_t = zeros(NUM_CLASSES,nf-1);
w_io=zeros(NUM_CLASSES,1);
%compute mean and variance for each class samples
%determine the linear machine parameters
%g_i(x)= x'*W_i*x + w_i'x + w_io
%W_i = -0.5*INV(sigma)
%w_i = INV(sigma)*mu_i
%W_io = -0.5*mu_i'*inv(sigma)*mu_i - 0.5ln(det(sigma_i)+ lnP(w_i)
for i =1:NUM_CLASSES
    mu_t(i,:) = mean(data(c*(i-1)+1:i*c,2:nf));
    %cov_t((nf-1)*(i-1)+1:i*(nf-1),:) = cov(data(c*(i-1)+1:i*c,2:nf));
    sigma = cov(data(c*(i-1)+1:i*c,2:nf));
    W_t((nf-1)*(i-1)+1:i*(nf-1),:) = -0.5*inv(sigma);
    w_t(i,:) = (sigma\mu_t(i,:)')';
    w_io(i) = -0.5*(mu_t(i,:)/sigma)*mu_t(i,:)' -0.5*log(det(sigma));
end
%% classifier
g = zeros(1,NUM_CLASSES);
test_data = textread('iris_training.txt');
test_data_size = size(test_data,1);
num_error = 0;
fprintf('sample from class         classified to class \n');
for j=1:test_data_size
    for i =1:NUM_CLASSES
        g(i) = test_data(j,2:nf)*W_t(1+(i-1)*(nf-1):i*(nf-1),:)*test_data(j,2:nf)' + w_t(i,:)*test_data(j,2:nf)'+ w_io(i);
    end
    [M,I] = max(g);
    if I ~= test_data(j,1)
        num_error = num_error + 1;
    end
     %display classification
    %fprintf('%d                          %d  \n',test_data(j,1),I);
end
hist(test_v);
classification_error = num_error*100/test_data_size;
fprintf('classification error    %.2f%%\n', classification_error);
fprintf('classification performance    %.2f%%\n', 100 - classification_error);
end

