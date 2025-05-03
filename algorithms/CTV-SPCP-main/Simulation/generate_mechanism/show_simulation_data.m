clear all;clc
h = 20;
w = 20;
band = 200;
show_band = 10;
r_s  = 0.1;
rho_s  = 0.1;
sigma  = 0.01;
lambda = 1;
smooth_flag = 1;
noise_mode  = 'U';% G, B, P, U
[Lmat,Smat,Omat] = generate_M(h,w,band,r_s,rho_s,sigma,lambda,smooth_flag,noise_mode);
Nmat = Omat - Lmat - Smat;
aver = std(Nmat(:));
fprintf('given noise level = %f, real noise_level = %f\n',sigma,aver)
figure;
subplot(2,2,1);imshow(reshape(Lmat(:,show_band),[h,w]),[]);title('low-rank')
subplot(2,2,2);imshow(reshape(Smat(:,show_band),[h,w]),[]);title('sparse');
subplot(2,2,3);imshow(reshape(Omat(:,show_band),[h,w]),[]);title('observation');
subplot(2,2,4);imshow(reshape(Nmat(:,show_band),[h,w]),[]);title('noise');

