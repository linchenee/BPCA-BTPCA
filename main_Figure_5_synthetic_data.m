clc
clear
close all
addpath(genpath('.\algorithms'));
addpath(genpath('.\utils'));

n1 = 50;
n2 = 50;

num_monte_carlo = 100;
list_r = 6:15; % list_r: list of rank parameters
list_mismatch = -5:10; % list_mismatch: list of dimensional mismatch degrees
SNR = 6;
rho_s = 0.06; % rho_s: the sparsity of S0

w = 0.95; % w: weight parameter for BPCA
lambda = 1 / sqrt(max(n1, n2)); % lambda: regularization parameter for RPCA, M-PCP, and BPCA
mu = 1e-2; % mu: stepsize for dual variable updating in ADMM
rho = 1.1; % rho>=1: ratio that is used to increase mu
mu_max = 10^9; % mu_max: maximum stepsize
eps = 1e-9; % eps: termination tolerance
iter_max = 1e10; % iter_max: maximum number of iterations

RSE1 = zeros(numel(list_r), num_monte_carlo);
RSE2 = zeros(numel(list_r), numel(list_mismatch), num_monte_carlo);
RSE3 = RSE2;

for i_r = 1:numel(list_r)
    r = list_r(i_r);
    for i_mc = 1 : num_monte_carlo
        rng(i_r*i_mc);
        fprintf('rank of L0: %d, trial: %g\n', [r, i_mc]);
        L0 = (1/sqrt(n1 * n2)) * randn(n1, r) * randn(n2, r)'; % L0: ground-truth low-rank matrix

        mask_noise = (rand(size(L0)) < rho_s);
        index_noise = find(mask_noise);
        M = L0; % M=L0+S0: noisy observation
        M(index_noise) = sqrt(20 * (norm(L0, 'fro')^2 / (n1 * n2))) * (randn(numel(index_noise), 1));

        %% RPCA method
        L1 = RPCA(M, lambda, mu, rho, mu_max, eps, iter_max);
        RSE1(i_r, i_mc) = norm(L1 - L0, 'fro')^2 / norm(L0, 'fro')^2;
        
        parfor i_mismatch = 1 : numel(list_mismatch)
        % for i_mismatch = 1 : numel(list_mismatch)
            mismatch = list_mismatch(i_mismatch);
            [U, S, V] = svd(L0, 'econ');
            U = U(:, 1:r + mismatch);
            V = V(:, 1:r + mismatch);
            G = GS_orth(awgn(U, SNR, 'measured')); % G: orthonormal basis of the prior subspace
            K = GS_orth(awgn(V, SNR, 'measured')); % K: orthonormal basis of the prior subspace

            %% MPCP method
            L2 = MPCP(M, G, K, lambda, mu, rho, mu_max, eps, iter_max);
            RSE2(i_r, i_mismatch, i_mc) = norm(L2 - L0, 'fro')^2 / norm(L0, 'fro')^2;
        
            %% BPCA method
            Wu = w * eye(r + mismatch);
            Wv = Wu;
            L3 = BPCA(M, G, K, Wu, Wv, lambda, mu, rho, mu_max, eps, iter_max);
            RSE3(i_r, i_mismatch, i_mc) = norm(L3 - L0, 'fro')^2 / norm(L0, 'fro')^2;
        end
    end
end

%% show results
success_bound = 5e-3;

figure(1)
temp = mean(RSE1(numel(list_r):-1:1, :) < success_bound, 2);
imagesc(repmat(temp,[1,numel(list_mismatch)])); colorbar;
xlabel('Dimensional mismatch degree ({\it{r}}-{\it{r}}_1)','fontsize',12,'FontName','Times new roman');
ylabel('Ground-truth rank (\itr)','fontsize',12,'FontName','Times new roman');
xlim([0.5, 16.5]);
xticks([1, 6, 11, 16]);
xticklabels({'-5', '0', '5', '10'});
ylim([0.5, 10.5]);
yticks([1, 4, 7, 10]);
yticklabels({'15', '12', '9', '6'});

figure(2);
imagesc(mean(RSE2(numel(list_r):-1:1, :, :) < success_bound, 3)); colorbar;
xlabel('Dimensional mismatch degree ({\it{r}}-{\it{r}}_1)','fontsize',12,'FontName','Times new roman');
ylabel('Ground-truth rank (\itr)','fontsize',12,'FontName','Times new roman');
xlim([0.5, 16.5]);
xticks([1, 6, 11, 16]);
xticklabels({'-5', '0', '5', '10'});
ylim([0.5, 10.5]);
yticks([1, 4, 7, 10]);
yticklabels({'15', '12', '9', '6'});

figure(3);
imagesc(mean(RSE3(numel(list_r):-1:1, :, :) < success_bound, 3)); colorbar;
xlabel('Dimensional mismatch degree ({\it{r}}-{\it{r}}_1)','fontsize',12,'FontName','Times new roman');
ylabel('Ground-truth rank (\itr)','fontsize',12,'FontName','Times new roman');
xlim([0.5, 16.5]);
xticks([1, 6, 11, 16]);
xticklabels({'-5', '0', '5', '10'});
ylim([0.5, 10.5]);
yticks([1, 4, 7, 10]);
yticklabels({'15', '12', '9', '6'});
