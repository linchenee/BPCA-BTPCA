function [G, K] = est_subspace_matrix1(L, r1)
% Estimate orthonormal bases of prior subspaces (used for the M-PCP method) 
% -----------------------------------------------------------
% Input: 
%  L:          an estimate of the ground-truth low-rank matrix
%  r1:         rank of matrices G and K
% --------------------------------------------------------
% Output:
%  G and K:    orthonormal bases of prior subspaces
% ---------------------------------------------
% version 1.0 - 05/01/2025
% Written by Lin Chen (lchen53@stevens.edu)

[U, ~, V] = svd(L, 'econ');
G = U(:, 1:r1);
K = V(:, 1:r1);
end