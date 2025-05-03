addpath(genpath('..\CTV-SPCP\'))
clear all;
clc;
load('fileNames.mat')
Dir    = 'data\DataSet\CAVE';
data_list = 1:32;
% mpsnr = zeros(1,12);
% mssim = zeros(1,12);
% ergas = zeros(1,12);
% time  = zeros(1,12);

mpsnr = zeros(1,32);
mssim = zeros(1,32);
ergas = zeros(1,32);
time  = zeros(1,32);
for data_id = 1:32
    Ori_H  =  LoadCAVE(fileNames{data_list(data_id)},Dir);
    gaussian_level = 0.2;
    sparse_level   = 0.2;
    clean_data     = Ori_H;
    clean_data     = Normalize(clean_data);
    [M,N,p]        = size(clean_data);
    noise_data     = GetNoise(clean_data,gaussian_level,sparse_level);
    [mpsnr(data_id),mssim(data_id),ergas(data_id)]=msqia(clean_data, noise_data);
end
