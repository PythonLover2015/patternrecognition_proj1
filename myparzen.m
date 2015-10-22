function [p_phi_x] = myparzen(h,alph,testdata, traindata )
%return parzenWindow weight for x
%   Detailed explanation goes here
        u = (testdata(k,2:end)- traindata(i,2:end))/h;
        u = norm(u);
        p_phi_x =p_phi_x+  1/alph*exp(-u^2/2);
end

