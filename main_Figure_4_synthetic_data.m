clc
clear
close all
addpath(genpath('.\algorithms'));
addpath(genpath('.\utils'));

rng(0)

n1 = 50;
n2 = 50;

list_SNR = 1:11; % list_SNR: list of SNRs
list_r = 1:33; % list_r: list of rank parameters
list_rho_s = linspace(0.01,0.2,25); % list_rho_s: list of parameters "rho_s" (the sparsity of S0)
num_monte_carlo = 50;

w = 0.1375*(min(list_SNR-1,4)) + 0.4; % w: weight parameter for BPCA
lambda = 1 / sqrt(max([n1, n2])); % lambda: regularization parameter
mu = 1e-2; % mu: stepsize for dual variable updating in ADMM
rho = 1.1; % rho>=1: ratio that is used to increase mu
mu_max = 10^9; % mu_max: maximum stepsize
eps = 1e-9; % eps: termination tolerance
iter_max = 1e10; % iter_max: maximum number of iterations

for i_SNR = 1:numel(list_SNR)
    SNR = list_SNR(i_SNR);
    for i_r = 1:numel(list_r)
        r = list_r(i_r);
        parfor i_rho_s = 1:numel(list_rho_s)
            rho_s = list_rho_s(i_rho_s);
            fprintf('SNR: %d, rank of L0: %d, sparsity of S0: %g\n', [SNR, r, rho_s])
            for i_mc = 1 : num_monte_carlo
                L0 = (1/sqrt(n1 * n2)) * randn(n1, r) * randn(n2, r)'; % L0: ground-truth low-rank matrix
                [U, ~, V] = svd(L0, 'econ');
                U = U(:, 1:r);
                V = V(:, 1:r);
                G = GS_orth(awgn(U, SNR, 'measured')); % G: orthonormal basis of the prior subspace
                K = GS_orth(awgn(V, SNR, 'measured')); % K: orthonormal basis of the prior subspace

                mask_noise = (rand(size(L0)) < rho_s);
                index_noise = find(mask_noise);
                M = L0; % M=L0+S0: noisy observation
                M(index_noise) = sqrt(20 * (norm(L0, 'fro')^2 / (n1 * n2))) * (randn(numel(index_noise), 1));

                %% RPCA method
                L1 = RPCA(M, lambda, mu, rho, mu_max, eps, iter_max);
                RSE1(i_SNR, i_r, i_rho_s, i_mc) = norm(L1 - L0, 'fro')^2 / norm(L0, 'fro')^2;

                %% M-PCP method
                L2 = MPCP(M, G, K, lambda, mu, rho, mu_max, eps, iter_max);
                RSE2(i_SNR, i_r, i_rho_s, i_mc) = norm(L2 - L0, 'fro')^2 / norm(L0, 'fro')^2;
                
                %% BPCA method
                Wu = w(1, i_SNR) * eye(r);
                Wv = Wu;
                L3 = BPCA(M, G, K, Wu, Wv, lambda, mu, rho, mu_max, eps, iter_max);
                RSE3(i_SNR, i_r, i_rho_s, i_mc) = norm(L3 - L0, 'fro')^2 / norm(L0, 'fro')^2;
            end
        end
    end
end

save('resultFigure4.mat', 'RSE1', 'RSE2', 'RSE3');

% After saving 'resultFigure4.mat', run the Python file 'show_Figure4.py' 
% in the 'utils' folder to obtain Figures 4(a)-(c).

