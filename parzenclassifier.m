clc;
close all;
%parzen window
%[fileID, errmsg] = fopen(filename);
%disp(errmsg)
nc = 3; % number of classes
[traindata] = textread('wine_uci_test.txt');
[testdata] = textread('wine_uci_train.txt');
testdata_size = size(testdata,1);
c = size(traindata,1)/nc;%number samples in each class
nf = size(traindata,2);% number of features + class column
d=nf-1; %dimension
h = 3; %window size
p_phi_x =zeros(1,nc);
alph = 1/(2*pi*h^d);
num_error = 0;
for k =1 :size(testdata,1)
    for j = 1:nc
        for i =(j-1)*c +1 :j*c
            u = (testdata(k,2:end)- traindata(i,2:end))/h;
            u = norm(u);
            p_phi_x(j) =p_phi_x(j) +  1/alph*exp(-u^2/2);
        end
    end
    [M,I] = max(p_phi_x);
    if I ~= testdata(k,1)
        num_error = num_error + 1;
    end
    p_phi_x =zeros(nc,1);
    %here reset p
end

classification_error = num_error*100/testdata_size;
fprintf('classification error    %.2f%%\n', classification_error);
fprintf('classification performance    %.2f%%\n', 100 - classification_error);
% find the pdf for each class and select the x that 
%gives the maximum px
%select max px of the classes

