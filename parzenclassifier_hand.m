%Demissew Kessela
%Parzen Window Classifier
clear
clc
[traindata] = dlmread('zip_train_small.txt');
[testdata] = dlmread('zip_test.txt');
nf = size(traindata,2);% number of features + class column
d=nf-1; %dimensiones
nc = 10; % number of classes
%%
testdata_size = size(testdata,1);
traindata_size = size(traindata,1);%number samples in each class
%%
%determin number of samples per class
%order sample data if not ordered
%[v, o]= sort(testdata(:,1));
%testdata = sort(testdata(:,1));
h =0.5; %window size
p_phi_x =zeros(1,nc);
alph = 1/(sqrt(2*pi)*h^d);
num_error = 0;
for k =1 :testdata_size
    for i=1:nc
        for j =1:traindata_size
            if traindata(j,1)==i-1
                u = (testdata(k,2:end)- traindata(j,2:end));
                u = norm(u)/h;
                p_phi_x(i) =p_phi_x(i) +  alph*exp(-u^2/2);
            end
        end
    end
    [M,I] = max(p_phi_x);
    if (I-1) ~= testdata(k,1)
        num_error = num_error + 1;
    end
    p_phi_x =zeros(nc,1);
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


