function [G, K, Wu, Wv] = est_subspace_matrix2(L, r1, w)
% Estimate orthonormal bases of prior subspaces (used for the BPCA method) 
% -----------------------------------------------------------
% Input: 
%  L:          an estimate of the ground-truth low-rank matrix
%  r1:         rank of matrices G and K
%  w:          weighted scalar
% --------------------------------------------------------
% Output:
%  G and K:    orthonormal bases of prior subspaces
%  Wu and Wv:  weighting matrices
% ---------------------------------------------
% version 1.0 - 05/01/2025
% Written by Lin Chen (lchen53@stevens.edu)

[U, ~, V] = svd(L, 'econ');
G = U(:, 1:r1);
K = V(:, 1:r1);
Wu = w*eye(r1);
Wv = Wu;
end