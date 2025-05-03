clc
clear
close all
addpath(genpath('.\algorithms'));
addpath(genpath('.\utils'));

rng(0)

n1 = 50;
n2 = 50;
num_monte_carlo = 100;
mismatch_max = 4; % mismatch_max: maximum of dimensional mismatch
r = 5; % r: rank of L0 
SNR = 20;
weight1 = 0.7; % weight1: weight parameter for BPCA
weight2 = 0.1; % weight2: weight parameter for BPCA

for i = 1 : num_monte_carlo
    rng(i);
    L0 = (1/sqrt(n1 * n2)) * randn(n1, r) * randn(n2, r)'; % L0: ground-truth low-rank matrix
    [U, ~, V] = svd(L0, 'econ');
    U = U(:,1:r);
    V = V(:,1:r);
    G = GS_orth(awgn(U, SNR, 'measured')); % G: orthonormal basis of the prior subspace
    K = GS_orth(awgn(V, SNR, 'measured')); % K: orthonormal basis of the prior subspace
    [G_temp,~] = qr(G);
    [K_temp,~] = qr(K);
    Wu = weight1*eye(r);
    Wv = weight1*eye(r);

    [inco_BPCA(i,1),ap0(i,1),ap1(i,1),ap2(i,1),gm1(i,1),gm2(i,1)] = incoh_BPCA_over(U, V, G, K, Wu, Wv, n1, n2);

    for j = 1:mismatch_max
        G_over = [G,G_temp(:, r+1:r+j)];
        K_over = [K,K_temp(:, r+1:r+j)];
        G_under = G(:,1:r-j);
        K_under = K(:,1:r-j);

        Wu_over = blkdiag(weight1*eye(r), weight2*eye(j));
        Wv_over = Wu_over;
        Wu_under = weight1*eye(r-j);
        Wv_under = Wu_under;

        [inco_RPCA_under(i,j),ap0_under(i,j),ap1_under(i,j),ap2_under(i,j),gm1_under(i,j),gm2_under(i,j)] = ...
            incoh_BPCA_under(U, V, G_under, K_under, Wu_under, Wv_under, n1, n2);
        check_u(i,j) = r*(sqrt(gm1_under(i,j))*ap1_under(i,j)+sqrt(gm2_under(i,j))*ap2_under(i,j))^2;

        [inco_BPCA_over(i,j),ap0_over(i,j),ap1_over(i,j),ap2_over(i,j),gm1_over(i,j),gm2_over(i,j)] = ...
            incoh_BPCA_over(U, V, G_over, K_over, Wu_over, Wv_over, n1, n2);

    end
end

temp1 = mean(ap0_under,1);
temp2 = mean(ap0_over,1);
temp3 = mean(ap0);
alpha0 = [temp1(4),temp1(3),temp1(2),temp1(1),temp3,temp2(1),temp2(2),temp2(3),temp2(4)];
fprintf('%0.2f %0.2f %0.2f %0.2f %0.2f %0.2f %0.2f %0.2f %0.2f\n',temp1(4),temp1(3),temp1(2),temp1(1),temp3,temp2(1),temp2(2),temp2(3),temp2(4));

temp1 = mean(ap1_under,1);
temp2 = mean(ap1_over,1);
temp3 = mean(ap1);
alpha1 = [temp1(4),temp1(3),temp1(2),temp1(1),temp3,temp2(1),temp2(2),temp2(3),temp2(4)];
fprintf('%0.2f %0.2f %0.2f %0.2f %0.2f %0.2f %0.2f %0.2f %0.2f\n',temp1(4),temp1(3),temp1(2),temp1(1),temp3,temp2(1),temp2(2),temp2(3),temp2(4));

temp1 = mean(ap2_under,1);
temp2 = mean(ap2_over,1);
temp3 = mean(ap2);
alpha2 = [temp1(4),temp1(3),temp1(2),temp1(1),temp3,temp2(1),temp2(2),temp2(3),temp2(4)];
fprintf('%0.2f %0.2f %0.2f %0.2f %0.2f %0.2f %0.2f %0.2f %0.2f\n',temp1(4),temp1(3),temp1(2),temp1(1),temp3,temp2(1),temp2(2),temp2(3),temp2(4));

temp1 = mean(gm1_under,1);
temp2 = mean(gm1_over,1);
temp3 = mean(gm1);
gamma1 = [temp1(4),temp1(3),temp1(2),temp1(1),temp3,temp2(1),temp2(2),temp2(3),temp2(4)];
fprintf('%0.2f %0.2f %0.2f %0.2f %0.2f %0.2f %0.2f %0.2f %0.2f\n',temp1(4),temp1(3),temp1(2),temp1(1),temp3,temp2(1),temp2(2),temp2(3),temp2(4));

temp1 = mean(gm2_under,1);
temp2 = mean(gm2_over,1);
temp3 = mean(gm2);
gamma2 = [temp1(4),temp1(3),temp1(2),temp1(1),temp3,temp2(1),temp2(2),temp2(3),temp2(4)];
fprintf('%0.2f %0.2f %0.2f %0.2f %0.2f %0.2f %0.2f %0.2f %0.2f\n',temp1(4),temp1(3),temp1(2),temp1(1),temp3,temp2(1),temp2(2),temp2(3),temp2(4));

temp1 = mean(inco_RPCA_under,1);
temp2 = mean(inco_BPCA_over,1);
temp3 = mean(inco_BPCA);
incoh = [temp1(4),temp1(3),temp1(2),temp1(1),temp3,temp2(1),temp2(2),temp2(3),temp2(4)];
fprintf('%0.2f %0.2f %0.2f %0.2f %0.2f %0.2f %0.2f %0.2f %0.2f\n',temp1(4),temp1(3),temp1(2),temp1(1),temp3,temp2(1),temp2(2),temp2(3),temp2(4));
