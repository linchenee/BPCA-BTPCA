clear all;clc
addpath(genpath('..\..\CTV-SPCP\'))
h = 20;
w = 20;
band = 200;
show_band = 10;
r_s  = 0.1;
rho_s  = 0.1;
sigma  = 0.1;
lambda = 5;
smooth_flag = 1;
noise_mode  = 'G';% G, B, P, U
tic
[RLmat,RSmat,~] = generate_M(h,w,band,r_s,rho_s,sigma,lambda,smooth_flag,noise_mode);
D = RLmat + RSmat;
normD = norm(RLmat,'fro');
normS = norm(RSmat,'fro');
noise_data = reshape(D,[h,w,band]);
repeated_time = 20;
%% rpca
err1 = 0;
for repeat_id = 1: repeated_time
    [A_hat,E_hat,iter] = rpca(D);
    tmp = norm(A_hat-RLmat,'fro')/normD + norm(E_hat-RSmat,'fro')/normS;
    err1 = err1+tmp/repeated_time;
end
number = 10;
err = zeros(number,1);
inter = 0.5;
for i=1:number
    lambda = 3*inter*i/sqrt(h*w);
    for repeat_id = 1: repeated_time 
        [output_image,E] = ctv_rpca(noise_data,lambda);
        A_hat2 = reshape(output_image,[h*w,band]);
        tmp = norm(A_hat2-RLmat,'fro')/normD + norm(E-RSmat,'fro')/normS;
        err(i) = err(i)+tmp/repeated_time;
    end
end
point = round(1/inter);
err(point) = (err(point) + err(point+1))/2;
toc
result.err = err;
result.err1 = err1;
result.inter = inter;
result.number = number;
savename = ['result\3dctv_','s_',num2str(rho_s),'_g_',num2str(sigma),'_h_',num2str(h),'_w_',num2str(w),'_rk_',num2str(r_s),'.mat'];
save(savename,'result');
figure;plot(inter:inter:inter*number,err(1:number),'r*-','LineWidth',2);
hold on;
plot(inter:inter:inter*number,err1*ones(number,1),'bp-','LineWidth',2);
xlabel('constant c of :$\lambda = c\sqrt{n_1}$','interpreter','latex');
ylabel('relative error','interpreter','latex');
legend(['3DCTV-RPCA'], ['RPCA'])
set(gca, 'Fontname', 'Times New Roman','FontSize',12);
