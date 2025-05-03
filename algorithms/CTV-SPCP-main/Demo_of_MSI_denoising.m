clear all;clc;
addpath(genpath('..\CTV-SPCP\'))
%% load data
hsi_name = 'pure_DCmall_small';
load([hsi_name,'.mat'])
clean_data       = Ori_H;
clean_data       = Normalize(clean_data);
[M,N,p]        = size(clean_data);

gaussian_level = 0.2;
sparse_level   = 0.0;
noise_data       = GetNoise(clean_data,gaussian_level,sparse_level);
D = reshape(noise_data,[M*N,p]);
it = 1;
[mpsnr(it),mssim(it),ergas(it)]=msqia(clean_data, noise_data);
%% RPCA
it = 2;
fprintf('======== RPCA  ========\n')
tic;
tmp = rpca_m(D);
RPCA_R = reshape(tmp,[M,N,p]);
time(it) = toc;
[mpsnr(it),mssim(it),ergas(it)]=msqia(clean_data, RPCA_R);
%% SPCP
it = 3;
fprintf('======== SPCP  ========\n')
tic;
spcp_alm(D,mean(gaussian_level));
SPCP_R = reshape(tmp,[M,N,p]);
time(it) = toc;
[mpsnr(it),mssim(it),ergas(it)]=msqia(clean_data, SPCP_R);

%% CTV-RPCA
it =4;
tic
fprintf('======== CTV-RPCA  ========\n')
CTV_R =ctv_rpca(noise_data);
time(it) = toc;
[mpsnr(it),mssim(it),ergas(it)]=msqia(clean_data, CTV_R);
%% CTV-SPCP
it = 5;
tic
fprintf('======== CTV-SPCP  ========\n')
CTV_SPCP_R =ctv_alm_spcp(noise_data,mean(gaussian_level));
time(it) = toc;
[mpsnr(it),mssim(it),ergas(it)]=msqia(clean_data, CTV_SPCP_R);

index = 36;
figure;
Y = WindowGig(clean_data(:,:,index),[0.6,0.43],[0.1,0.2],2.2,0);
subplot(2,3,1);imshow(Y,[]);title('Original: PSNR')
Y = WindowGig(noise_data(:,:,index),[0.6,0.43],[0.1,0.2],2.2,0);
subplot(2,3,2);imshow(Y,[]);title(['Noisy:',num2str(mpsnr(1))])
Y = WindowGig(RPCA_R(:,:,index),[0.6,0.43],[0.1,0.2],2.2,0);
subplot(2,3,3);imshow(Y,[]);title(['RPCA:',num2str(mpsnr(2))])
Y = WindowGig(SPCP_R(:,:,index),[0.6,0.43],[0.1,0.2],2.2,0);
subplot(2,3,4);imshow(Y,[]);title(['SPCP:',num2str(mpsnr(3))])
Y = WindowGig(CTV_R(:,:,index),[0.6,0.43],[0.1,0.2],2.2,0);
subplot(2,3,5);imshow(Y,[]);title(['CTV:',num2str(mpsnr(4))])
Y = WindowGig(CTV_SPCP_R(:,:,index),[0.56,0.43],[0.1,0.2],2.2,0);
subplot(2,3,6);imshow(Y,[]);title(['CTV-SPCP:',num2str(mpsnr(5))])