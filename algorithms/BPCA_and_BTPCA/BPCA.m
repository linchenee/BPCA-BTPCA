function [L, S] = BPCA(M, G, K, Wu, Wv, lambda, mu, rho, mu_max, eps, iter_max)
% Boosting principal component analysis (BPCA) method in Eq. (2).
% -----------------------------------------------------------
% Input:
%  M:          noisy observation
%  G and K:    orthonormal bases of the prior subspaces
%  Wu and Wv:  weighting matrices
%  lambda:     regularization parameter
%  mu:         stepsize for dual variable updating in ADMM  
%  rho:        rho>=1, ratio that is used to increase mu
%  mu_max:     maximum stepsize
%  eps:        termination tolerance
%  iter_max:   maximum number of iterations
% --------------------------------------------------------
% Output:
%  L:          low-rank component
%  S:          sparse component
% ---------------------------------------------
% version 1.0 - 05/01/2025
% Written by Lin Chen (lchen53@stevens.edu)

[n1,n2] = size(M);   
Z_old = zeros(n1,n2);
S_old = Z_old;
N_old = S_old;
IGWuGT = eye(n1) - G * Wu * G';
IKWvKT = eye(n2) - K * Wv * K';
WuGT = Wu * G';
KWv = K * Wv;
cnt = 0;

while cnt <= iter_max
    cnt = cnt + 1;
    MSZ = M - S_old + Z_old/mu;
    temp = IGWuGT * MSZ;
    N = svt(temp * IKWvKT, 1/mu);
   
    A1T = WuGT * MSZ;
    A2 = temp * KWv;
    GA1TA2KT = G * A1T + A2 * K';

    S = shr(M - N - GA1TA2KT + Z_old/mu, lambda/mu);
    Z = Z_old + mu * (M - N - S - GA1TA2KT);
    
    if max(abs(N - N_old), [], 'all') < eps && ...
       max(abs(S - S_old), [], 'all') < eps && ...
       max(abs(M - N - S - GA1TA2KT), [], 'all') < eps
        fprintf('Iteration number of BPCA is %d\n', cnt);
        break;
    end

    N_old = N;
    S_old = S;
    Z_old = Z;
    GA1TA2KT_old = GA1TA2KT;
    mu = min(mu_max, mu * rho);
end
L = N_old + GA1TA2KT_old;
S = S_old;
end

function X_tr = svt(X, mu)
    [U, S, V] = svd(X, 'econ');
    s = diag(S) - mu;
    r = sum(s > 0);
    X_tr = U(:, 1:r) * diag(s(1:r)) * V(:, 1:r)';
end

function S_shr = shr(S, mu)
    S_shr = sign(S) .* max(abs(S) - mu, 0);
end