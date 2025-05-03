clc
clear
close all

rng(0)

n1 = 50;
n2 = 50;
num_monte_carlo = 1e3;
list_SNR = 5:5:120; % list_SNR: list of SNRs
r = 3; % r: rank(L0)=rank(G)=rank(K) 
rank_tol = 1e-10; % rank_tol: a tolerance parameter used to determine the rank of a matrix

inco_RPCA = zeros(numel(list_SNR), num_monte_carlo); % inco_RPCA: incoherence parameter of RPCA
inco_MPCP = inco_RPCA; % inco_MPCP: incoherence parameter of M-PCP
inco_BPCA = inco_RPCA; % inco_BPCA: incoherence parameter of BPCA
log_term_BPCA = inco_RPCA; % log_term_BPCA: logarithmic term of BPCA

w = 0.95; % w: weight parameter for BPCA
lambda = 1 / sqrt(max([n1, n2])); % lambda: regularization parameter for RPCA, M-PCP, and BPCA
mu = 1e-2; % mu: stepsize for dual variable updating in ADMM
rho = 1.1; % rho>=1: ratio that is used to increase mu
mu_max = 10^9; % mu_max: maximum stepsize
eps = 1e-9; % eps: termination tolerance

for i_SNR = 1:numel(list_SNR)
    SNR = list_SNR(i_SNR);
        fprintf('SNR: %d\n', SNR);
        parfor i_mc = 1 : num_monte_carlo
        % for i_mc = 1 : num_monte_carlo
            L0 = (1/sqrt(n1 * n2)) * randn(n1, r) * randn(n2, r)'; % L0: ground-truth low-rank matrix
            [U, ~, V] = svd(L0, 'econ');
            U = U(:,1:r);
            V = V(:,1:r);
            G = GS_orth(awgn(U, SNR, 'measured')); % G: orthonormal basis of the prior subspace
            K = GS_orth(awgn(V, SNR, 'measured')); % K: orthonormal basis of the prior subspace

            %% Calculate the incoherence parameter of RPCA
            inco_RPCA(i_SNR,i_mc) = incoh_RPCA(U, V, n1, n2, r);

            %% Calculate the incoherence parameter of M-PCP
            Lnew = (eye(n1) - G*G') * L0 * (eye(n2) - K*K');
            [Unew, Snew, Vnew] = svd(Lnew,'econ');
            Unew = Unew(:,1:rank(Snew, rank_tol));
            Vnew = Vnew(:,1:rank(Snew, rank_tol));
            inco_MPCP(i_SNR,i_mc) = incoh_MPCP(U, V, Unew, Vnew, n1, n2, r);
            
            %% Calculate the incoherence parameter of BPCA
            Wu = w * eye(r);
            Wv = Wu;
            [inco_BPCA(i_SNR,i_mc), alpha0, alpha1, alpha2] = incoh_BPCA_over(U, V, G, K, Wu, Wv, n1, n2);
            log_term_BPCA(i_SNR,i_mc) = max( log(n1*min(sqrt(alpha1^2 + alpha2^2),1) ), 1 );
        end
end

temp1 = mean(inco_RPCA,2)';
temp2 = mean(inco_MPCP,2)';
temp3 = mean(inco_BPCA,2)';
temp4 = ones(1,numel(list_SNR))*log(n1);
temp5 = mean(log_term_BPCA,2)';
temp6 = n1./(temp1.*(temp4.^2)); % calculated according to Table I
temp7 = n1./(temp2.*(temp4.^2)); % calculated according to Table I
temp8 = n1./(temp3.*(temp5.*temp4)); % calculated according to Table I

%% show results
figure(1);
plot(list_SNR, temp6,'g','LineWidth',2,'Marker','x','MarkerSize',7); hold on;
plot(list_SNR, temp7,'b','LineWidth',2,'Marker','o','MarkerSize',7); hold on;
plot(list_SNR, temp8,'r','LineWidth',2,'Marker','square','MarkerSize',7); hold on;
xlabel('SNR (dB)','fontsize',19,'FontName','Times new roman');
ylabel('Value','fontsize',19,'FontName','Times new roman');
set(gca, 'FontName', 'Times new roman', 'FontSize', 19);
h=legend('$\mathcal{O}(\rm{rank}(\textbf{L}_0))$ (RPCA)',...
    '$\mathcal{O}({\rm{rank}}(\textbf{L}_0))$ (M-PCP)',...
    '$\mathcal{O}({\rm{rank}}(\textbf{L}_0))$ (BPCA)');
set(h,'Interpreter','latex','fontsize',17.5,'FontName','Times new roman');

figure(2);
plot(list_SNR, temp4,'b','LineWidth',2,'Marker','o','MarkerSize',7); hold on;
plot(list_SNR, temp5,'r','LineWidth',2,'Marker','square','MarkerSize',7); hold on;
xlabel('SNR (dB)','fontsize',19,'FontName','Times new roman');
ylabel('Value','fontsize',19,'FontName','Times new roman');
set(gca, 'FontName', 'Times new roman', 'FontSize', 19);
h=legend('$\log(n_{(1)})$ (RPCA and M-PCP)','$\max\{ \log(n_{(1)}\min\{\!\sqrt{\!\alpha_1^2\!+\!\alpha_2^2},1\}),\!1\}$ (BPCA)');
set(h,'Interpreter','latex','fontsize',17.5,'FontName','Times new roman');

figure(3);
plot(list_SNR, temp1,'g','LineWidth',2,'Marker','x','MarkerSize',7); hold on;
plot(list_SNR, temp2,'b','LineWidth',2,'Marker','o','MarkerSize',7); hold on;
plot(list_SNR, temp3,'r','LineWidth',2,'Marker','square','MarkerSize',7); hold on;
xlabel('SNR (dB)','fontsize',19,'FontName','Times new roman');
ylabel('Value','fontsize',19,'FontName','Times new roman');
set(gca, 'FontName', 'Times new roman', 'FontSize', 19);
h=legend('$\mu$ (RPCA)','$\nu$ (M-PCP)','$\gamma$ (BPCA)');
set(h,'Interpreter','latex','fontsize',17.5,'FontName','Times new roman');
