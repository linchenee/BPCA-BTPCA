function [incoh_MPCP] = incoh_MPCP(U, V, Unew, Vnew, n1, n2, r)
% Calculate the incoherence parameter of M-PCP
% -----------------------------------------------
% version 1.0 - 05/01/2025
% Written by Lin Chen (lchen53@stevens.edu)

norm_u = zeros(1,n1);
for i = 1:n1
 norm_u(1,i) = norm([U(i,:), Unew(i,:)],'fro')^2;
end
mu1 = max(norm_u)*n1/r;

norm_v = zeros(1,n2);
for j = 1:n2
 norm_v(1,j) = norm([V(j,:), Vnew(j,:)],'fro')^2;
end
mu2 = max(norm_v)*n2/r;

mu3 = max(max(abs(Unew*Vnew')))^2*n1*n2/r;

incoh_MPCP = max([mu1,mu2,mu3]);
end