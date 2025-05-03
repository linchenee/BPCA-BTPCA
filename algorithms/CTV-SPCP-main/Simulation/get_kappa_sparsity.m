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
spar_l_list = [0.02:0.02:0.5,0.55:0.05:1,1.1:0.1:2];
len_r = length(spar_l_list);
sigma_s_list = [0.025,0.1,0.2];
len_g = length(sigma_s_list);
% define the vary variables
ratio_list = zeros(len_r,len_g);
spar_list  = zeros(len_r,len_g);
tic
for r_s_ind = 1:len_r
    for sigma_ind =1:len_g
        r_s   = spar_l_list(r_s_ind);
        sigma = sigma_s_list(sigma_ind);
        fprintf('=========== rs_sequence = %d/%d, std_sequence = %d/%d===========\n',r_s_ind,len_r,sigma_ind,len_g);
        number = 5;
        h = 20;
        w = 20;
        band = 200;
        rellerr  = zeros(number,2);
        relserr  = zeros(number,2);
        rho_smat = zeros(number,1);
        for ind_out = 1:number
            [RLmat,RSmat,D] = generate_M(h,w,band,r_s,rho_s,sigma,lambda,smooth_flag,noise_mode);
            noiseF = norm(D-RLmat-RSmat,'fro');
            diff_v = diff3(RLmat,[h,w,band]);
            ratio = length(find(abs(diff_v(1:2*h*w*band))>=0.001))/(h*w*band*2);
            rho_smat(ind_out) = ratio;
            normR = norm(RLmat,'fro');
            normS = norm(RSmat,'fro');
            noise_data = reshape(D,[h,w,band]);
            % sqrt_spcp
            [A_hat,E_hat,iter] = spcp_sqrt(D);
            it = 1;
            rellerr(ind_out,it) = norm(A_hat-RLmat,'fro');
            relserr(ind_out,it) = norm(E_hat-RSmat,'fro');
            % sqrt_ctv_spcp
            [res,E] =ctv_sqrt_spcp(noise_data);
            A_hat = reshape(res,[h*w,band]);
            it = 2;
            rellerr(ind_out,it) = norm(A_hat-RLmat,'fro');
            relserr(ind_out,it) = norm(E-RSmat,'fro');
        end
        error = rellerr+relserr;
        meanerr = mean(error);
        meanrho = mean(rho_smat);
        ratio =   meanerr(2)/meanerr(1);
        spar_list(r_s_ind,sigma_ind)=meanrho;
        ratio_list(r_s_ind,sigma_ind) = ratio;
    end
end
toc
function_kappa.spar_list  = spar_list;
function_kappa.ratio_list = ratio_list;
savename = ['result\function_kappa_',noise_mode,'_',num2str(lambda),'.mat'];
save(savename,'function_kappa');


