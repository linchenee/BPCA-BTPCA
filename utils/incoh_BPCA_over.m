function [incoh_BPCA, alpha0, alpha1, alpha2, gamma1, gamma2] = incoh_BPCA_over(U, V, G, K, Wu, Wv, n1, n2)
% Calculate the incoherence parameter of BPCA in cases of r1>r and r1=r.
% Case 1: r1>r, the rank-overparameterized prior subspace 
% Case 2: r1=r, the dimension of the prior subspace is identical to that of the ground-truth subspace
% Please use the function "incoh_BPCA_under.m" in the case of r1<r.
% ---------------------------------------------------------------------------------------------------
% version 1.0 - 05/01/2025
% Written by Lin Chen (lchen53@stevens.edu)

r = size(U,2);    
r1 = size(G,2);
[Qu1, Ru1] = qr(U*U'*G,0);
[Qv1, Rv1] = qr(V*V'*K,0);
Qu1 = Qu1(:,1:r);
Qv1 = Qv1(:,1:r);
[Qu2, Ru2] = qr((eye(n1)-U*U')*G,0);
[Qv2, Rv2] = qr((eye(n2)-V*V')*K,0);

norm_u1 = zeros(1,n1);
norm_u2 = zeros(1,n1);
for i = 1:n1
 norm_u1(1,i) = norm(Qu1(i,:),'fro')^2;
 norm_u2(1,i) = norm(Qu2(i,:),'fro')^2;
end

norm_v1 = zeros(1,n2);
norm_v2 = zeros(1,n2);
for j = 1:n2
 norm_v1(1,j) = norm(Qv1(j,:),'fro')^2;
 norm_v2(1,j) = norm(Qv2(j,:),'fro')^2;
end

gamma1 = max( max(norm_u1)*n1/r,max(norm_v1)*n2/r );
gamma2 = max( [max(norm_u2)*n1/r,max(norm_v2)*n2/r] );

Ru = [Ru1; Ru2];
Rv = [Rv1; Rv2];
[Qu, Bu] = qr(eye(2*r1)-Ru*Wu*Ru');
[Qv, Bv] = qr(eye(2*r1)-Rv*Wv*Rv');

%% avoid negative numbers on the diagonal
for i = 1:2*r1
 if Bu(i,i) < 0
   Bu(i,:) = -Bu(i,:);
   Qu(:,i) = -Qu(:,i);
 end
 if Bv(i,i) < 0
   Bv(i,:) = -Bv(i,:);
   Qv(:,i) = -Qv(:,i);
 end
end

Bu1 = Bu(1:r, 1:r);
Bu2 = Bu(1:r1, r1+1:2*r1);
Bu3 = Bu(r1+1:2*r1, r1+1:2*r1);
Bv1 = Bv(1:r, 1:r);
Bv2 = Bv(1:r1, r1+1:2*r1);
Bv3 = Bv(r1+1:2*r1, r1+1:2*r1);

alpha0 = norm(Bu2)*norm(Bv2) + norm(eye(r1)-Bu3) + norm(eye(r1)-Bv3'); % alpha0: according to Eq.(11)
alpha1 = norm(Bu1)*norm(Bv1); % alpha1: according to Definition 2
alpha2 = norm(Bu2)*norm(Bv1) + norm(Bu1)*norm(Bv2); % alpha2: according to Definition 2

incoh_BPCA = gamma1*max(r*(sqrt(gamma1)*alpha1+sqrt(gamma2)*alpha2)^2,1); % incoh_BPCA (gamma): according to Definition 2
end