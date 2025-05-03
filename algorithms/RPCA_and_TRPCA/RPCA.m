function [L, S] = RPCA(M, lambda, mu, rho, mu_max, eps, iter_max)
% Robust principal component analysis (RPCA) method in Eq. (1.1) of [1].
% ---------------------------------------------------------------------------
% [1] E. J. Candès, X. Li, Y. Ma, and J. Wright, "Robust principal component
% analysis?" J. ACM, vol. 58, no. 3, pp. 1–37, 2011.
% ---------------------------------------------------------------------------
% Input:
%  M:          noisy observation
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

Y = zeros(size(M));
S = Y;
L = Y;
cnt = 0;
while cnt <= iter_max
    cnt = cnt + 1;
    L_next = svt(M - S + Y/mu, 1/mu);
    S_next = shr(M - L_next + Y/mu, lambda/mu);
    Y_next = Y + mu * (M - L_next - S_next);
    
    if max(abs(L_next - L), [], 'all') < eps && ...
       max(abs(S_next - S), [], 'all') < eps && ...
       max(abs(M - L_next - S_next), [], 'all') < eps
        fprintf('Iteration number of RPCA is %d\n', cnt);
        break;
    end

    L = L_next;
    S = S_next;
    Y = Y_next;
    mu = min(mu_max, mu * rho);
end
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