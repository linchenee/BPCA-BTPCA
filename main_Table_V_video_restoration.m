clc
clear all
close all
addpath(genpath('.\data'));
addpath(genpath('.\algorithms'));
addpath(genpath('.\utils'));

load('videos_train.mat');
load('videos_test.mat');

[n1, n2, n3, num] = size(videos_test);
lambda1 = 1 / sqrt(n1*n2); % lambda1: regularization parameter for matrix-based methods
lambda2 = 1 / sqrt(n1*n3); % lambda2: regularization parameter for tensor-based methods
iter_max = 100; % iter_max: maximum number of iterations
mu = 1e-2; % mu: stepsize for dual variable updating in ADMM
mu_max = 10^9; % mu_max: maximum stepsize
rho = 1.1; % rho>=1: ratio that is used to increase mu
eps = 1e-3; % eps: termination tolerance

SNR = 10;  % signal-to-noise ratio (SNR)
MR = 0.05; % missing ratio (MR)

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
PSNR14 = PSNR1;
PSNR15 = PSNR1;

parfor scene = 1:num
% for scene = 1:num
    fprintf('Scene ID = %d\n', scene);
    rng(scene);

    %% video_test: test data
    video_test = double(reshape(squeeze(videos_test(:,:,:,scene)), [n1,n2,n3]))/255; 

    %% video2D_train and video3D_train: training data to estimate the prior subspace for BPCA-2 and BTPCA-3 methods
    video3D_train = double(reshape(squeeze(videos_train(:,:,:,scene)), [n1,n2,n3]))/255;
    video2D_train = reshape(video3D_train, [n1*n2,n3]);

    %% GT: ground-truth
    GT = reshape(video_test, [n1*n2,n3]); 

    %% M_3D and M_2D: noisy observations
    M_3D = zeros(n1,n2,n3);
    for j = 1:n3
        M_3D(:,:,j) = video_test(:,:,j) + GMM_Gen(video_test(:,:,j), SNR, n1, n2);
    end
    mask = rand(size(video_test)) < MR;
    M_3D(mask) = 0;   
    M_2D = reshape(M_3D, [n1*n2,n3]);

    %% RPCA method
    [L1, ~] = RPCA(M_2D, lambda1, mu, rho, mu_max, eps, iter_max);
    L1(L1 < 0) = 0;
    L1(L1 > 1) = 1;
    PSNR1(scene,1) = psnr(L1, GT);

    %% M-PCP method
    if SNR == 10 && MR == 0.05
        r1 = 10; % r1: set according to Table VI
    elseif SNR == 20 && MR == 0.05
        r1 = 15;
    elseif SNR == 10 && MR == 0.1
        r1 = 8;
    elseif SNR == 20 && MR == 0.1
        r1 = 9;
    end
    [G, K] = est_subspace_matrix1(L1, r1);
    L2 = MPCP(M_2D, G, K, lambda1, mu, rho, mu_max, eps, iter_max);
    L2(L2 < 0) = 0;
    L2(L2 > 1) = 1;
    PSNR2(scene,1) = psnr(L2, GT);

    %% BPCA method
    if SNR == 10 && MR == 0.05
        r1 = 13; tau = 0.5; % r1 and tau: set according to Table VI
    elseif SNR == 20 && MR == 0.05
        r1 = 15; tau = 0.03;
    elseif SNR == 10 && MR == 0.1
        r1 = 9; tau = 0.5;
    elseif SNR == 20 && MR == 0.1
        r1 = 10; tau = 0.2;
    end
    [G, K, Wu, Wv] = est_subspace_matrix3(L1, r1, tau);
    L3 = BPCA(M_2D, G, K, Wu, Wv, lambda1, mu, rho, mu_max, eps, iter_max);
    L3(L3 < 0) = 0;
    L3(L3 > 1) = 1;
    PSNR3(scene,1) = psnr(L3, GT);

    %% BPCA-2 method
    if SNR == 10 && MR == 0.05
        r1 = 11; tau = 3; % r1 and tau: set according to Table VI
    elseif SNR == 20 && MR == 0.05
        r1 = 13; tau = 2;
    elseif SNR == 10 && MR == 0.1
        r1 = 8; tau = 4;
    elseif SNR == 20 && MR == 0.1
        r1 = 7; tau = 3;
    end
    [G, K, Wu, Wv] = est_subspace_matrix3(video2D_train, r1, tau);
    L4 = BPCA(M_2D, G, K, Wu, Wv, lambda1, mu, rho, mu_max, eps, iter_max);
    L4(L4 < 0) = 0;
    L4(L4 > 1) = 1;
    PSNR4(scene,1) = psnr(L4, GT);

    %% TRPCA method
    [L5, ~] = trpca_tnn(M_3D, lambda2, mu, rho, mu_max, eps, iter_max);
    L5(L5 < 0) = 0;
    L5(L5 > 1) = 1;
    PSNR5(scene,1) = psnr(reshape(L5,[n1*n2,n3]), GT);   

    %% HoRPCA-S method
    lambda = 0.45/sqrt(max([n1,n2,n3]));
    mu1    = 500*std(M_3D(:));
    mu2    = 500*std(M_3D(:));
    if SNR == 10 && MR == 0.1
        max_iter = 100;
    else
        max_iter = 200;
    end
    L6 = tensor_rpca_adal_parfor(M_3D, lambda, mu1, mu2, max_iter, eps);
    L6 = L6.data;
    L6(L6 < 0) = 0;
    L6(L6 > 1) = 1;
    PSNR6(scene,1) = psnr(reshape(L6,[n1*n2,n3]), GT);

    %% SNN-RPCA method
    alpha = 15*[1, 1, 3];
    L7 = trpca_snn_parfor(M_3D, alpha);
    L7(L7 < 0) = 0;
    L7(L7 > 1) = 1;
    PSNR7(scene,1) = psnr(reshape(L7,[n1*n2,n3]), GT); 

    %% KBR-RPCA method
    beta    = 2.5*sqrt(max([n1,n2,n3]));
    gamma   = beta*100;
    MaxIter = 200;
    lambda  = 1e4;
    mu0      = 10;
    tol     = 1e-5;
    rhos    = 1.05;
    L8 = KBR_RPCA_parfor(M_3D, beta, gamma, MaxIter, tol, mu0, rhos, lambda);
    L8(L8 < 0) = 0;
    L8(L8 > 1) = 1;
    PSNR8(scene,1) = psnr(reshape(L8,[n1*n2,n3]), GT);

    %% TTNN-RPCA method
    d1=4; d2=2; c1=5; c2=2; b1=2; b2=5; a1=3; a2=5;
    KA5D = CastImageAsKetAdjustable(M_3D,a1,a2,b1,b2,c1,c2,d1,d2,n3);
    if SNR == 10
        lambda = 0.04;
    elseif SNR == 20
        lambda = 0.05;
    end
    f = 2;
    gamma = 0.001;
    deta = 0.002;
    Z = TT_TRPCA(KA5D, lambda, f, gamma, deta);
    L9 = CastKet2ImageAdjustable(Z,n1,n2,a1,a2,b1,b2,c1,c2,d1,d2,n3);
    L9(L9 < 0) = 0;
    L9(L9 > 1) = 1;
    PSNR9(scene,1) = psnr(reshape(L9,[n1*n2,n3]), GT); 

    % 3DCTV-RPCA method
    lambdaCTV  = 2*3/sqrt(n1*n2);
    L10 = ctv_rpca(M_3D, lambdaCTV);
    L10(L10 < 0) = 0;
    L10(L10 > 1) = 1;
    PSNR10(scene,1) = psnr(reshape(L10,[n1*n2,n3]), GT); 

    %% t-CTV method
    rho2 = 1.25;
    directions = [1,2,3];
    L11 = TCTV_TRPCA_parfor(M_3D, rho2, directions);
    L11(L11 < 0) = 0;
    L11(L11 > 1) = 1;
    PSNR11(scene,1) = psnr(reshape(L11,[n1*n2,n3]), GT); 

    %% M-TPCP method
    if SNR == 10 && MR == 0.05
        r1 = 3; % r1: set according to Table VI
    elseif SNR == 20 && MR == 0.05
        r1 = 11;
    elseif SNR == 10 && MR == 0.1
        r1 = 1;
    elseif SNR == 20 && MR == 0.1
        r1 = 7;
    end
    [G, K] = est_subspace_tensor1(L5, r1);
    L12 = MTPCP(M_3D, G, K, lambda2, mu, rho, mu_max, eps, iter_max);
    L12(L12 < 0) = 0;
    L12(L12 > 1) = 1;
    PSNR12(scene,1) = psnr(reshape(L12,[n1*n2,n3]), GT); 

    %% BTPCA method
    if SNR == 10 && MR == 0.05
        r1 = 46; tau = 6; % r1 and tau: set according to Table VI
    elseif SNR == 20 && MR == 0.05
        r1 = 55; tau = 2;
    elseif SNR == 10 && MR == 0.1
        r1 = 37; tau = 8;
    elseif SNR == 20 && MR == 0.1
        r1 = 37; tau = 4;
    end
    [G, K, Wu, Wv] = est_subspace_tensor3(L5, r1, tau);
    L13 = BTPCA(M_3D, G, K, Wu, Wv, lambda2, mu, rho, mu_max, eps, iter_max);
    L13(L13 < 0) = 0;
    L13(L13 > 1) = 1;
    PSNR13(scene,1) = psnr(reshape(L13,[n1*n2,n3]), GT); 

    %% BTPCA-2 method
    if SNR == 10 && MR == 0.05
        r1 = 86; tau = 4; % r1 and tau: set according to Table VI
    elseif SNR == 20 && MR == 0.05
        r1 = 95; tau = 1;
    elseif SNR == 10 && MR == 0.1
        r1 = 86; tau = 6;
    elseif SNR == 20 && MR == 0.1
        r1 = 95; tau = 2;
    end
    [G, K, Wu, Wv] = est_subspace_tensor3(L11, r1, tau);
    L14 = BTPCA(M_3D, G, K, Wu, Wv, lambda2, mu, rho, mu_max, eps, iter_max);
    L14(L14 < 0) = 0;
    L14(L14 > 1) = 1;
    PSNR14(scene,1) = psnr(reshape(L14,[n1*n2,n3]), GT); 

    %% BTPCA-3 method
    if SNR == 10 && MR == 0.05
        r1 = 50; tau = 8; % r1 and tau: set according to Table VI
    elseif SNR == 20 && MR == 0.05
        r1 = 85; tau = 6;
    elseif SNR == 10 && MR == 0.1
        r1 = 27; tau = 10;
    elseif SNR == 20 && MR == 0.1
        r1 = 40; tau = 8;
    end
    [G, K, Wu, Wv] = est_subspace_tensor3(video3D_train, r1, tau);
    L15 = BTPCA(M_3D, G, K, Wu, Wv, lambda2, mu, rho, mu_max, eps, iter_max);
    L15(L15 < 0) = 0;
    L15(L15 > 1) = 1;
    PSNR15(scene,1) = psnr(reshape(L15,[n1*n2,n3]), GT);
end

fprintf('PSNR: RPCA = %.2f dB\n', mean(PSNR1));
fprintf('PSNR: M-PCP = %.2f dB\n', mean(PSNR2));
fprintf('PSNR: BPCA = %.2f dB\n', mean(PSNR3));
fprintf('PSNR: BPCA-2 = %.2f dB\n', mean(PSNR4));
fprintf('PSNR: TRPCA = %.2f dB\n', mean(PSNR5));
fprintf('PSNR: HoRPCA-S = %.2f dB\n', mean(PSNR6));
fprintf('PSNR: SNN-RPCA = %.2f dB\n', mean(PSNR7));
fprintf('PSNR: KBR-RPCA = %.2f dB\n', mean(PSNR8));
fprintf('PSNR: TTNN-RPCA = %.2f dB\n', mean(PSNR9));
fprintf('PSNR: 3DCTV-RPCA = %.2f dB\n', mean(PSNR10));
fprintf('PSNR: t-CTV = %.2f dB\n', mean(PSNR11));
fprintf('PSNR: M-TPCP = %.2f dB\n', mean(PSNR12));
fprintf('PSNR: BTPCA = %.2f dB\n', mean(PSNR13));
fprintf('PSNR: BTPCA-2 = %.2f dB\n', mean(PSNR14));
fprintf('PSNR: BTPCA-3 = %.2f dB\n', mean(PSNR15));