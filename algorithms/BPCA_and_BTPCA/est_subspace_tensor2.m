function [G, K, Wu, Wv] = est_subspace_tensor2(L, r1, w)
% Estimate orthonormal bases of prior subspaces (used for the BTPCA method) 
% -----------------------------------------------------------
% Input: 
%  L:          an estimate of the ground-truth low-rank tensor
%  r1:         tubal rank of tensors G and K
%  w:          weighted scalar
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
for f = 1:n3    
 [Gf(:,:,f), ~, Kf(:,:,f)] = svd(Lf(:,:,f), 'econ');
end

Gf = Gf(:, 1:r1, :);
Kf = Kf(:, 1:r1, :);
Wuf = repmat(w*eye(r1), [1,1,n3]);
Wu = ifft(Wuf, [], 3);
Wv = Wu;
G = ifft(Gf, [], 3);
K = ifft(Kf, [], 3);
end