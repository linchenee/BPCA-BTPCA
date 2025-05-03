function [G, K, Wu, Wv] = est_subspace_matrix3(L, r1, tau)
% Estimate orthonormal bases of prior subspaces (used for the BPCA method) 
% -----------------------------------------------------------
% Input: 
%  L:          an estimate of the ground-truth low-rank matrix
%  r1:         rank of matrices G and K
%  tau:        threshold for singular values
% --------------------------------------------------------
% Output:
%  G and K:    orthonormal bases of prior subspaces
%  Wu and Wv:  weighting matrices
% ---------------------------------------------
% version 1.0 - 05/01/2025
% Written by Lin Chen (lchen53@stevens.edu)

[U, temp, V] = svd(L, 'econ');
G = U(:, 1:r1);
K = V(:, 1:r1);
S = diag(temp(1:r1, 1:r1));

[id1, ~] = find(S > tau);
[id2, ~] = find(S <= tau & S > 0);
S_map = zeros(r1,1);
S_map(id1) = 1;
temp = S(id2); 
S_map(id2) = (temp - min(temp)) ./ (tau - min(temp));

Wu = diag(S_map);
Wv = Wu;
end