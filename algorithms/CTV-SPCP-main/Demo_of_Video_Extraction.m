clear all;clc
addpath(genpath('..\CTV-SPCP\'))
% data_list
data_name="airport";
weight = 13;
[original_data,gt_fore]=GetVideoMask(data_name);
[M,N,p]=size(original_data);
gaussian_level = 0.05;
InputTensor = GetNoise(original_data,gaussian_level,0);
InputMatrix       = reshape(InputTensor,[M*N,p]);


%% Input
it = 1;
auc(it) = MAUC(gt_fore,abs(InputTensor));
%% RPCA
it = 2;
fprintf('======== RPCA  ========\n')
tic;
[~,tmp] = rpca_m(InputMatrix);
RPCA_R = reshape(tmp,[M,N,p]);
time(it) = toc;
auc(it) = MAUC(gt_fore,abs(RPCA_R));
%% SPCP
it = 3;
fprintf('======== SPCP  ========\n')
[~,tmp] = spcp_sqrt(InputMatrix);
time(it) = toc;
SPCP_R = reshape(tmp,[M,N,p]);
time(it) = toc;
auc(it) = MAUC(gt_fore,abs(SPCP_R));
%% CTV
it =4;
lambda = 3/sqrt(M*N);
tic
fprintf('======== CTV-RPCA  ========\n')
[~,tmp] =ctv_rpca(InputTensor,lambda,weight);
time(it) = toc;
CTV_R = reshape(tmp,[M,N,p]);
auc(it) = MAUC(gt_fore,abs(CTV_R));
%% CTV-SPCP
it =5;
tic
fprintf('======== CTV-SPSP  ========\n')
[~,tmp] =ctv_sqrt_spcp(InputTensor,weight);
time(it) = toc;
CTV_S_R = reshape(tmp,[M,N,p]);
auc(it) = MAUC(gt_fore,abs(CTV_S_R));

figure;
index = 4;
subplot(2,3,1);imshow(abs(gt_fore(:,:,index)),[]);title('groundtruth')
subplot(2,3,2);imshow(abs(InputTensor(:,:,index)),[]);title('Observed')
subplot(2,3,3);imshow(abs(RPCA_R(:,:,index)),[]);title(['RPCA:',num2str(auc(2))])
subplot(2,3,4);imshow(abs(SPCP_R(:,:,index)),[]);title(['SPCP:',num2str(auc(3))])
subplot(2,3,5);imshow(abs(CTV_R(:,:,index)),[]);title(['CTV:',num2str(auc(4))])
subplot(2,3,6);imshow(abs(CTV_R(:,:,index)),[]);title(['CTV-SPCP:',num2str(auc(5))])