function [G, K] = est_subspace_tensor1(L, r1)
% Estimate orthonormal bases of prior subspaces (used for the M-TPCP method) 
% -----------------------------------------------------------
% Input: 
%  L:          an estimate of the ground-truth low-rank tensor
%  r1:         tubal rank of tensors G and K
% --------------------------------------------------------
% Output:
%  G and K:    orthonormal bases of prior subspaces
% ---------------------------------------------
% version 1.0 - 05/01/2025
% Written by Lin Chen (lchen53@stevens.edu)

[n1,n2,n3] = size(L);
n12 = min(n1,n2);
Lf = fft(L, [], 3);
Gf = zeros(n1,n12,n3);
Kf = zeros(n2,n12,n3);
for i = 1:n3    
 [Gf(:,:,i), ~, Kf(:,:,i)] = svd(Lf(:,:,i), 'econ');
end

Gf = Gf(:, 1:r1, :);
Kf = Kf(:, 1:r1, :);
G = ifft(Gf, [], 3);
K = ifft(Kf, [], 3);
end