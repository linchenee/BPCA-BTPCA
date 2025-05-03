function [G, K, Wu, Wv] = est_subspace_tensor3(L, r1, tau)
% Estimate orthonormal bases of prior subspaces (used for the BTPCA method) 
% -----------------------------------------------------------
% Input: 
%  L:          an estimate of the ground-truth low-rank tensor
%  r1:         tubal rank of tensors G and K
%  tau:        threshold for singular values
% --------------------------------------------------------
% Output:
%  G and K:    orthonormal bases of prior subspaces
%  Wu and Wv:  weighting tensor
% ---------------------------------------------
% version 1.0 - 05/01/2025
% Written by Lin Chen (lchen53@stevens.edu)

[n1,n2,n3] = size(L);
n12 = min(n1,n2);
Lf = fft(L, [], 3);
Gf = zeros(n1,n12,n3);
Kf = zeros(n2,n12,n3);
Sf = zeros(r1*n3,1);
Sf_map = Sf;
for i = 1:n3    
 [Gf(:,:,i), temp, Kf(:,:,i)] = svd(Lf(:,:,i), 'econ');
 Sf((i-1)*r1 + 1:i*r1, 1) = diag(temp(1:r1, 1:r1));  
end
Gf = Gf(:, 1:r1, :);
Kf = Kf(:, 1:r1, :);

[id1,~] = find(Sf > tau);
[id2,~] = find(Sf <= tau & Sf > 0);
Sf_map(id1) = 1;
temp = Sf(id2); 
Sf_map(id2) = (temp - min(temp)) ./ (tau - min(temp));

Wuf = zeros(r1,r1,n3);
for i = 1:n3
 Wuf(:,:,i) = diag(Sf_map((i-1)*r1 + 1:i*r1, 1));
end

Wu = ifft(Wuf, [], 3);
Wv = Wu;
G = ifft(Gf, [], 3);
K = ifft(Kf, [], 3);
end