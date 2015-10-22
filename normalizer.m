function [x] = normalizer(x,d)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
%Normalization
mur = mean(x);
for i=1:d
  y = x(:,i+1) - mur(i+1);
  varx = var( x(:,i+1));
  x(:,i+1) = y/sqrt(varx);
end
end

