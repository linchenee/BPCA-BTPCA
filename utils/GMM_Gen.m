function w = GMM_Gen(Y, SNR, m, n)
% Generate the Gaussian mixture model (GMM) noise
% -----------------------------------------------
% version 1.0 - 05/01/2025
% Written by Lin Chen (lchen53@stevens.edu)

delta2 = 0.1;
ratio = 1000;
w = zeros(m,n);
Ps = norm(Y,'fro')^2/(m*n);
Pw = Ps/(10^(SNR/10));
sigma = sqrt(Pw);
sigma2 = sigma/sqrt((1-delta2)+delta2*ratio);
sigma1 = sqrt(ratio)*sigma2;
for i = 1:m
    for j = 1:n
        u = rand(1,1);
        if u >= delta2
            w(i,j) = sigma2 * randn(1,1);
        else
            w(i,j) = sigma1 * randn(1,1);
        end
    end
end