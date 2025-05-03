function [L, S] = BTPCA(M, G, K, Wu, Wv, lambda, mu, rho, mu_max, eps, iter_max)
% Boosting tensor principal component analysis (BTPCA) method in Eq. (41).
% -----------------------------------------------------------
% Input:
%  M:          noisy observation
%  G and K:    orthonormal bases of the prior subspaces
%  Wu and Wv:  weighting tensors
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

[n1, n2, n3] = size(M);
half = round(n3/2);
even = (mod(n3,2) == 0);
Z_old = zeros(n1,n2,n3); 
S_old = Z_old;
N_old = M;
Gf = fft(G, [], 3);
Kf = fft(K, [], 3);
[IGWuGTf, IKWvKTf, WuGTf, KWvf] = tprod1(n1, n2, n3, even, half, Gf, Kf, Wu, Wv);
cnt = 0;
while cnt <= iter_max
    cnt = cnt + 1;
    MSZ = M - S_old + Z_old/mu;
    %% update A1T and A2
    [N_prox, A1Tf, A2f, GA1TA2KT] = tprod2(n3, even, half, IGWuGTf, IKWvKTf, MSZ, Kf, Gf, WuGTf, KWvf);
    N = prox_tnn(N_prox, 1/mu);

    S = shr(M - N - GA1TA2KT + Z_old/mu, lambda/mu);
    Z = Z_old + mu*(M - N - S - GA1TA2KT);

    if max(abs(N(:) - N_old(:))) < eps && ...
       max(abs(S(:) - S_old(:))) < eps && ...
       max(abs(M(:) - N(:) - S(:) - GA1TA2KT(:))) < eps
        fprintf('Iteration number of BTPCA is %d\n', cnt);
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

function S_shr = shr(S, mu)
    S_shr = sign(S) .* max(abs(S) - mu, 0);
end
