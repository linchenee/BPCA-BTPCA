clc
clear
close all
addpath(genpath('.\data'));
addpath(genpath('.\algorithms'));
addpath(genpath('.\utils'));

load('data_face_denoising.mat');

n1 = 96;
n2 = 84;
n3 = 64;
num = 31;

lambda1 = 1 / sqrt(max([n1*n2, n3])); % lambda1: regularization parameter for matrix-based methods
lambda2 = 1 / sqrt(n1*n3); % lambda2: regularization parameter for tensor-based methods
iter_max = 100; % iter_max: maximum number of iterations
mu = 1e-2; % mu: stepsize for dual variable updating in ADMM
mu_max = 10^9; % mu_max: maximum stepsize
rho = 1.1; % rho>=1: ratio that is used to increase mu
eps1 = 3e-3; % eps1: termination tolerance for matrix-based methods
eps2 = 1e-3; % eps2: termination tolerance for tensor-based methods

CR = 0.05; % corrupted ratio (CR)

PSNR1 = zeros(num,1);
PSNR2 = PSNR1;
PSNR3 = PSNR1;
PSNR4 = PSNR1;
PSNR5 = PSNR1;
PSNR6 = PSNR1;
PSNR7 = PSNR1;
PSNR8 = PSNR1;
PSNR9 = PSNR1;
PSNR10 = PSNR1;
PSNR11 = PSNR1;
PSNR12 = PSNR1;
PSNR13 = PSNR1;

parfor scene = 1:num
% for scene = 1:num
    fprintf('Scene ID = %d\n', scene);
    rng(scene);

    %% GT: ground-truth
    GT = GTs{scene};
    mask_noise = rand(size(GT)) < CR;

    %% M_3D and M_2D: noisy observations
    M_2D = GT;
    M_2D(mask_noise) = 0.5*double(rand(sum(mask_noise(:)),1)) + 0.5;
    M_3D = reshape(M_2D, [n1,n2,n3]);

    %% RPCA method
    L1 = RPCA(M_2D, lambda1, mu, rho, mu_max, eps1, iter_max);
    L1(L1 < 0) = 0;
    L1(L1 > 1) = 1;
    PSNR1(scene,1) = psnr(L1, GT);

    %% M-PCP method
    if CR == 0.05
        r1 = 23; % r1: set according to Table IV
    elseif CR == 0.1
        r1 = 16;
    elseif CR == 0.2
        r1 = 7;
    end
    [G, K] = est_subspace_matrix1(L1, r1);   
    L2 = MPCP(M_2D, G, K, lambda1, mu, rho, mu_max, eps1, iter_max);
    L2(L2 < 0) = 0;
    L2(L2 > 1) = 1;
    PSNR2(scene,1) = psnr(L2, GT);

    %% BPCA method
    if CR == 0.05
        r1 = 25; w = 0.99; % r1 and w: set according to Table IV
    elseif CR == 0.1
        r1 = 16; w = 0.98;
    elseif CR == 0.2
        r1 = 7; w = 0.97;
    end
    [G, K, Wu, Wv] = est_subspace_matrix2(L1, r1, w);
    L3 = BPCA(M_2D, G, K, Wu, Wv, lambda1, mu, rho, mu_max, eps1, iter_max);
    L3(L3 < 0) = 0;
    L3(L3 > 1) = 1;
    PSNR3(scene,1) = psnr(L3, GT);

    %% TRPCA method
    L4 = trpca_tnn(M_3D, lambda2, mu, rho, mu_max, eps2, iter_max);
    L4(L4 < 0) = 0;
    L4(L4 > 1) = 1;
    PSNR4(scene,1) = psnr(reshape(L4, [n1*n2,n3]), GT);

    %% HoRPCA-S method
    lambda      = 0.5/sqrt(max([n1,n2,n3]));
    mu1         = 5*std(M_3D(:));
    mu2         = 5*std(M_3D(:));
    max_iter    = 100;
    opt_tol     = 1e-3;
    L5  = tensor_rpca_adal_parfor(M_3D, lambda, mu1, mu2, max_iter, opt_tol);
    L5 = L5.data;
    L5(L5 < 0) = 0;
    L5(L5 > 1) = 1;
    PSNR5(scene,1) = psnr(reshape(L5,[n1*n2,n3]), GT);

    %% SNN-RPCA method
    alpha = 20*[1, 1, 1];
    L6 = trpca_snn_parfor(M_3D, alpha);
    L6(L6 < 0) = 0;
    L6(L6 > 1) = 1;
    PSNR6(scene,1) = psnr(reshape(L6,[n1*n2,n3]), GT);

    %% KBR-RPCA method
    beta    = 2.5*sqrt(max([n1,n2,n3]));
    gamma   = beta*100;
    MaxIter = 1000;
    lambda  = 1e4;
    mu0     = 10;
    tol     = 1e-5;
    rhos    = 1.05;
    L7 = KBR_RPCA_parfor(M_3D, beta, gamma, MaxIter, tol, mu0, rhos, lambda);
    L7(L7 < 0) = 0;
    L7(L7 > 1) = 1;
    PSNR7(scene,1) = psnr(reshape(L7,[n1*n2,n3]), GT);

    %% TTNN-RPCA method
    d1=4; d2=2; c1=4; c2=2; b1=3; b2=3; a1=2; a2=7;
    KA5D = CastImageAsKetAdjustable(M_3D,a1,a2,b1,b2,c1,c2,d1,d2,n3);
    lambda = 0.06;
    f = 2;
    gamma = 0.001;
    deta = 0.002;
    Z = TT_TRPCA(KA5D, lambda, f, gamma, deta, eps2, iter_max);
    L8 = CastKet2ImageAdjustable(Z,n1,n2,a1,a2,b1,b2,c1,c2,d1,d2,n3);
    L8(L8 < 0) = 0;
    L8(L8 > 1) = 1;
    PSNR8(scene,1) = psnr(reshape(L8,[n1*n2,n3]), GT);

    %% 3DCTV-RPCA method
    lambdaCTV  = 2*3/sqrt(n1*n2);
    L9 = ctv_rpca(M_3D, lambdaCTV);
    L9(L9 < 0) = 0;
    L9(L9 > 1) = 1;
    PSNR9(scene,1) = psnr(reshape(L9,[n1*n2,n3]), GT);

    %% t-CTV method
    rho1 = 1.25;
    directions = [1,2,3];
    L10 = TCTV_TRPCA_parfor(M_3D, rho1, directions);
    L10(L10 < 0) = 0;
    L10(L10 > 1) = 1;
    PSNR10(scene,1) = psnr(reshape(L10, [n1*n2,n3]), GT);

    %% M-TPCP method
    if CR == 0.05
        r1 = 25; % r1: set according to Table IV
    elseif CR == 0.1
        r1 = 16;
    elseif CR == 0.2
        r1 = 7;
    end
    [G, K] = est_subspace_tensor1(L4, r1);
    L11 = MTPCP(M_3D, G, K, lambda2, mu, rho, mu_max, eps2, iter_max);
    L11(L11 < 0) = 0;
    L11(L11 > 1) = 1;
    PSNR11(scene,1) = psnr(reshape(L11, [n1*n2,n3]), GT);

    %% BTPCA method
    if CR == 0.05
        r1 = 27; w = 0.98; % r1 and w: set according to Table IV
    elseif CR == 0.1
        r1 = 16; w = 0.98;
    elseif CR == 0.2
        r1 = 7; w = 0.98;
    end
    [G, K, Wu, Wv] = est_subspace_tensor2(L4, r1, w);
    L12 = BTPCA(M_3D, G, K, Wu, Wv, lambda2, mu, rho, mu_max, eps2, iter_max);
    L12(L12 < 0) = 0;
    L12(L12 > 1) = 1;
    PSNR12(scene,1) = psnr(reshape(L12, [n1*n2,n3]), GT);

    %% BTPCA-2 method
    if CR == 0.05
        r1 = 28; w = 0.98; % r1 and w: set according to Table IV
    elseif CR == 0.1
        r1 = 20; w = 0.96;
    elseif CR == 0.2
        r1 = 10; w = 0.94;
    end
    [G, K, Wu, Wv] = est_subspace_tensor2(L10, r1, w);
    L13 = BTPCA(M_3D, G, K, Wu, Wv, lambda2, mu, rho, mu_max, eps2, iter_max);
    L13(L13 < 0) = 0;
    L13(L13 > 1) = 1;
    PSNR13(scene,1) = psnr(reshape(L13, [n1*n2,n3]), GT);
end

fprintf('PSNR: RPCA = %.2f dB\n', mean(PSNR1));
fprintf('PSNR: M-PCP = %.2f dB\n', mean(PSNR2));
fprintf('PSNR: BPCA = %.2f dB\n', mean(PSNR3));
fprintf('PSNR: TRPCA = %.2f dB\n', mean(PSNR4));
fprintf('PSNR: HoRPCA-S = %.2f dB\n', mean(PSNR5));
fprintf('PSNR: SNN-RPCA = %.2f dB\n', mean(PSNR6));
fprintf('PSNR: KBR-RPCA = %.2f dB\n', mean(PSNR7));
fprintf('PSNR: TTNN-RPCA = %.2f dB\n', mean(PSNR8));
fprintf('PSNR: 3DCTV-RPCA = %.2f dB\n', mean(PSNR9));
fprintf('PSNR: t-CTV = %.2f dB\n', mean(PSNR10));
fprintf('PSNR: M-TPCP = %.2f dB\n', mean(PSNR11));
fprintf('PSNR: BTPCA = %.2f dB\n', mean(PSNR12));
fprintf('PSNR: BTPCA-2 = %.2f dB\n', mean(PSNR13));