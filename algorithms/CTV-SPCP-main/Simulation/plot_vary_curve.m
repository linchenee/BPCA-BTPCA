addpath(genpath('..\Simulation\'))
%% casename has six cases: 
    % G_1, gasussian 
    % U_1, uniform 
    % P_1, possion with lambda =1
    % P_3, possion with lambda =3
    % P_5, possion with lambda =5
    % B_1, Blind Gaussian
casename='P_5';
%% plot kappa_s
kappa_name = ['function_kappa_',casename,'.mat'];
load(kappa_name)
figure;
id1=1;
plot(function_kappa.spar_list(:,id1),function_kappa.ratio_list(:,id1),'r-','LineWidth',2);
hold on; 
id2=2;
plot(function_kappa.spar_list(:,id2),function_kappa.ratio_list(:,id2),'g-','LineWidth',2);
hold on; 
id3=3;
plot(function_kappa.spar_list(:,id3),function_kappa.ratio_list(:,id3),'b-','LineWidth',2);
xlabel('sparsity of gradent map:$s$','interpreter','latex');
ylabel('decay rate $\kappa_s$','interpreter','latex');
legend(['\sigma',':','0.025'], ['\sigma',':','0.1'], ['\sigma',':','0.2'])
set(gca, 'Fontname', 'Times New Roman','FontSize',16);
%% plot noise
noise_name = ['vary_error_noise_',casename,'.mat'];
load(noise_name)
figure;plot(vary_error_noise.sigma_list,vary_error_noise.relL(:,1)+vary_error_noise.relS(:,1),'cp-');
hold on;
plot(vary_error_noise.sigma_list,vary_error_noise.relL(:,2)+vary_error_noise.relS(:,2),'ro-');
hold on;
plot(vary_error_noise.sigma_list,vary_error_noise.relL(:,3)+vary_error_noise.relS(:,3),'b*-');
hold on;
plot(vary_error_noise.sigma_list,vary_error_noise.relL(:,4)+vary_error_noise.relS(:,4),'gs-');
xlabel('\sigma');
ylabel('$\mbox{RMSE}$','interpreter','latex');
legend('$\sqrt{\mbox{PCP}}$','CTV-$\sqrt{\mbox{PCP}}$','SPCP','CTV-SPCP','interpreter','latex');
set(gca, 'Fontname', 'Times New Roman','FontSize',16);
%% plot size
size_name = ['vary_error_size_',casename,'.mat'];
load(size_name)
figure;plot(vary_error_size.h_list,vary_error_size.relL(:,1)+vary_error_size.relS(:,1),'cp-');
hold on;
plot(vary_error_size.h_list,vary_error_size.relL(:,2)+vary_error_size.relS(:,2),'ro-');
hold on;
plot(vary_error_size.h_list,vary_error_size.relL(:,3)+vary_error_size.relS(:,3),'b*-');
hold on;
plot(vary_error_size.h_list,vary_error_size.relL(:,4)+vary_error_size.relS(:,4),'gs-');
xlabel('$\sqrt{n_1}$','interpreter','latex');
ylabel('$\mbox{RMSE}$','interpreter','latex');
legend('$\sqrt{\mbox{PCP}}$','CTV-$\sqrt{\mbox{PCP}}$','SPCP','CTV-SPCP','interpreter','latex');
set(gca, 'Fontname', 'Times New Roman','FontSize',16);
