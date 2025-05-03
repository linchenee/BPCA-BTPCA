clear all;clc
addpath(genpath('..\compete method\'))
h = 20;
w = 20;
band = 200;
show_band = 10;
r_s  = 0.05;
rho_s  = 0.1;
sigma  = 0.1;
lambda = 5;
smooth_flag = 1;
noise_mode  = 'U';% G, B, P, U
tic
[RLmat,RSmat,D] = generate_M(h,w,band,r_s,rho_s,sigma,lambda,smooth_flag,noise_mode);
noiseF = norm(D-RLmat-RSmat,'fro');
aver = std(D(:)-RLmat(:)-RSmat(:));
diff_v = diff3(RLmat,[h,w,band]);
ratio = length(find(abs(diff_v(1:2*h*w*band))>=0.001))/(h*w*band*2);
normR = norm(RLmat,'fro');
normS = norm(RSmat,'fro');
noise_data = reshape(D,[h,w,band]);
%% sqrt_spcp
[A_hat,E_hat,iter] = spcp_sqrt(D);
it = 1;
rellerr(it) = norm(A_hat-RLmat,'fro');
relserr(it) = norm(E_hat-RSmat,'fro');
%% sqrt_ctv_spcp
[res,E] =ctv_sqrt_spcp(noise_data);
A_hat = reshape(res,[h*w,band]);
it = 2;
rellerr(it) = norm(A_hat-RLmat,'fro');
relserr(it) = norm(E-RSmat,'fro'); 

%% spcp_alm
[A_hat,E_hat,~] = spcp_alm(D,sigma);
it = 3;
rellerr(it) = norm(A_hat-RLmat,'fro');
relserr(it) = norm(E_hat-RSmat,'fro');

%% ctv_spcp
[res,E] =ctv_alm_spcp(noise_data,sigma);
A_hat = reshape(res,[h*w,band]);
it = 4;
rellerr(it) = norm(A_hat-RLmat,'fro');
relserr(it) = norm(E-RSmat,'fro');
fprintf('given noise level = %f, real noise_level = %f\n',sigma,aver)
rellerr+relserr
toc