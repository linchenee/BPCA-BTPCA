function [L, E] = MTPCP(X, G, K, lambda, mu, rho, mu_max, eps, iter_max)
% Modified tensor principal component pursuit (M-TPCP) method in Eq. (8) of [10].
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

[n1, n2, n3] = size(X);
half = round(n3/2);
even = (mod(n3,2) == 0);
O_old = zeros(n1, n2, n3); 
E_old = O_old;
Lnew_old = X;
Gf = fft(G, [], 3);
Kf = fft(K, [], 3);
[IGGTf, IKKTf] = tprod3(n1, n2, n3, even, half, Gf, Kf);
cnt = 0;
while cnt <= iter_max
    cnt = cnt + 1;
    XEO = X - E_old + O_old/mu;
    %% update B1T and B2
    [Lnew_prox, B1Tf, B2f, GB1TB2KT] = tprod4(n3, even, half, IGGTf, IKKTf, XEO, Kf, Gf);
    Lnew = prox_tnn(Lnew_prox, 1/mu);

    E = shr(X - Lnew - GB1TB2KT + O_old/mu, lambda/mu);
    L = Lnew + GB1TB2KT;
    O = O_old + mu * (X - E - L);

    if max(abs(Lnew(:) - Lnew_old(:))) < eps && ...
       max(abs(E(:) - E_old(:))) < eps && ...
       max(abs(X(:) - L(:) - E(:))) < eps
        fprintf('Iteration number of M-TPCP is %d\n', cnt);
        fprintf('Iteration number of M-TPCP is %d\n', cnt);
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

function S_shr = shr(S, mu)
    S_shr = sign(S) .* max(abs(S) - mu, 0);
end
