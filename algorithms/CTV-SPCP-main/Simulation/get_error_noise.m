%------------------------------------
%          read the data
%------------------------------------
clear all;clc
addpath(genpath('..\compete method\'))
addpath(genpath('..\ctv_spcp\'))
addpath(genpath('result\'))
addpath(genpath('generate_mechanism\'))
% set parameters
smooth_flag = 1;
noise_mode = 'P';
lambda = 1;
rho_s  = 0.10;
r_s   = 0.05;
sigma_list = 0.005:0.005:0.2;
len_s = length(sigma_list);
relL  = zeros(len_s,4);
relS  = zeros(len_s,4);
number = 20;
tic
for s_id = 1:len_s        
    fprintf('=========== rs_sequence = %d/%d ===========\n',s_id,len_s);
    h = 20;
    w = 20;
    band = 200;
    rellerr  = zeros(number,2);
    relserr  = zeros(number,2);
    rho_smat = zeros(number,1);
    sigma = sigma_list(s_id);
    for ind_out = 1:number
        [RLmat,RSmat,D] = generate_M(h,w,band,r_s,rho_s,sigma,lambda,smooth_flag,noise_mode);
        noiseF = norm(D-RLmat-RSmat,'fro');
        diff_v = diff3(RLmat,[h,w,band]);
        ratio = length(find(abs(diff_v(1:2*h*w*band))>=0.001))/(h*w*band*2);
        rho_smat(ind_out) = ratio;
        normR = norm(RLmat,'fro');
        normS = norm(RSmat,'fro');
        noise_data = reshape(D,[h,w,band]);
        %% sqrt_spcp
        [A_hat,E_hat,iter] = spcp_sqrt(D);
        it = 1;
        rellerr(ind_out,it) = norm(A_hat-RLmat,'fro');
        relserr(ind_out,it) = norm(E_hat-RSmat,'fro');
        %% sqrt_ctv_spcp
        [res,E] =ctv_sqrt_spcp(noise_data);
        A_hat = reshape(res,[h*w,band]);
        it = 2;
        rellerr(ind_out,it) = norm(A_hat-RLmat,'fro');
        relserr(ind_out,it) = norm(E-RSmat,'fro'); 
        
       %% spcp_alm
        [A_hat,E_hat,~] = spcp_alm(D,sigma);
        it = 3;
        rellerr(ind_out,it) = norm(A_hat-RLmat,'fro');
        relserr(ind_out,it) = norm(E_hat-RSmat,'fro');
        
        %% ctv_spcp
        [res,E] =ctv_alm_spcp(noise_data,sigma);
        A_hat = reshape(res,[h*w,band]);
        it = 4;
        rellerr(ind_out,it) = norm(A_hat-RLmat,'fro');
        relserr(ind_out,it) = norm(E-RSmat,'fro');
    end
    relL(s_id,:) = mean(rellerr);
    relS(s_id,:) = mean(relserr);
end
toc
vary_error_noise.sigma_list = sigma_list;
vary_error_noise.relS   = relS;
vary_error_noise.relL   = relL;
savename = ['result\vary_error_noise_',noise_mode,'_',num2str(lambda),'.mat'];
save(savename,'vary_error_noise');
