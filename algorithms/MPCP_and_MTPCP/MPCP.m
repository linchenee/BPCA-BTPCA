function [L, E] = MPCP(X, G, K, lambda, mu, rho, mu_max, eps, iter_max)
% Modified principal component pursuit (M-PCP) method, corresponding to a matrix version of the 
% modified tensor principal component pursuit (M-TPCP) method in Eq. (8) of [10].
% ------------------------------------------------------------------------------------------------------
% [10] F. Zhang, J. Wang, W. Wang, and C. Xu, "Low-tubal-rank plus sparse tensor recovery with prior 
% subspace information," IEEE Trans. Pattern Anal. Mach. Intell., vol. 43, no. 10, pp. 3492–3507, 2021.
% ------------------------------------------------------------------------------------------------------
% Input:
%  X:          noisy observation
%  G and K     orthonormal bases of the prior subspaces
%  lambda:     regularization parameter
%  mu:         stepsize for dual variable updating in ADMM  
%  rho:        rho>=1, ratio that is used to increase mu
%  mu_max:     maximum stepsize
%  eps:        termination tolerance
%  iter_max:   maximum number of iterations
% --------------------------------------------------------
% Output:
%  L:          low-rank component
%  E:          sparse component
% ---------------------------------------------
% version 1.0 - 05/01/2025
% Written by Lin Chen (lchen53@stevens.edu)

[n1,n2] = size(X);    
O_old = zeros(n1,n2);
E_old = O_old;
Lnew_old = O_old;
IGGT = eye(n1) - G * G';
IKKT = eye(n2) - K * K';
cnt = 0;
while cnt <= iter_max
    cnt = cnt + 1;
    XEO = X - E_old + O_old/mu;
    temp = IGGT * XEO;
    Lnew = svt(temp * IKKT, 1/mu);

    B1T = G' * XEO;
    B2 = temp * K;
    GB1TB2KT = G * B1T + B2 * K';
    
    E = shr(X - Lnew - GB1TB2KT + O_old/mu, lambda/mu);
    L = Lnew + GB1TB2KT;
    O = O_old + mu * (X - E - L);

    if max(abs(Lnew - Lnew_old), [], 'all') < eps && ...
       max(abs(E - E_old), [], 'all') < eps && ...
       max(abs(X - L - E), [], 'all') < eps
        fprintf('Iteration number of M-PCP is %d\n', cnt);
        break;
    end

    Lnew_old = Lnew;
    E_old = E;
    O_old = O;
    GB1TB2KT_old = GB1TB2KT;
    mu = min(mu_max, mu * rho);
end
L = Lnew_old + GB1TB2KT_old;
E = E_old;
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