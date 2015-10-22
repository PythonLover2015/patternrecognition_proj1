%Demissew Kessela
%KNN Classifier
clear
clc
[traindata] = dlmread('iris_training.txt');
[testdata] = dlmread('iris_test.txt');
nf = size(traindata,2);% number of features + class column
d=nf-1; %dimension
nc = 3; % number of classes
%%
testdata_size = size(testdata,1);
traindata_size = size(traindata,1);%number samples in each class
%%
K=6; %window size
num_error=0;
kn_dist = zeros(2,traindata_size);
for k =1 :testdata_size
    kn_dist(1,:) = traindata(:,1)';
    for i=1:nc
        for j =1:traindata_size
            if traindata(j,1)==i
                u = (testdata(k,2:end)- traindata(j,2:end));
                u = norm(u);
                kn_dist(2,j) =u;
            end
        end
    end
    %sorting and picking the most frequent class in the cluster
    B = sortrows(kn_dist',2)';
    C = B(:,1:K);
    max_v = mode(C,2);
    if max_v(1) ~= testdata(k,1)
        num_error = num_error + 1;
    end
    kn_dist = zeros(2,traindata_size);
    %}
    %here reset p
end
%%
classification_error = num_error*100/testdata_size;
fprintf('classification error    %.2f%%\n', classification_error);
fprintf('classification performance    %.2f%%\n', 100 - classification_error);
% find the pdf for each class and select the x that 
%gives the maximum px
%select max px of the classes
%}


