%% solve the problem 
%              min_X ||D_x(X)||_*+||D_y(X)||_* +||D_z(X)||_* +\lambda||E||_1
%                                   s.t.  Y= X+E
%                          ===============================
%              min_X ||X1||_*+||X2||_* +||X3||_* +\lambda||E||_1
%                            s.t.  Y= X+E
%                                  D_x(X)=X1 
%                                  D_y(X)=X2 
%                                  D_z(X)=X3 
%                          ===============================                       
%         D is difference operator,T is difference tensor,T is known
%  ------------------------------------------------------------------------

function [ output_image,S] = ctv_rpca(oriData3_noise,lambda,weight)
tol     = 1e-4; %1e-6
maxIter = 100; %400
rho     = 1.25;
% max_mu  = 1e6;
% mu      = 1e-3;
[M,N,p] = size(oriData3_noise);
if nargin < 2
    lambda  = 3/sqrt(M*N);
end
if nargin < 3
    weight = 1;
end

sizeM   = size(oriData3_noise);
MM       = zeros(M*N,p) ;
for i=1:p
    bandp = oriData3_noise(:,:,i);
    MM(:,i)= bandp(:);
end
normD   = norm(MM,'fro');
% initialize
norm_two = lansvd(MM, 1, 'L');
norm_inf = norm( MM(:), inf) / lambda;
dual_norm = max(norm_two, norm_inf);

mu = 1.25/dual_norm;%1.25/norm_two % this one can be tuned
max_mu = mu * 1e7;
%% FFT setting
h               = sizeM(1);
w               = sizeM(2);
d               = sizeM(3);
%% 
Eny_x   = ( abs(psf2otf([+1; -1], [h,w,d])) ).^2  ;
Eny_y   = ( abs(psf2otf([+1, -1], [h,w,d])) ).^2  ;
Eny_z   = ( abs(psf2otf([+1, -1], [w,d,h])) ).^2  ;
Eny_z   =  permute(Eny_z, [3, 1 2]);
determ  =  Eny_x + Eny_y + Eny_z;
%% Initializing optimization variables
X              = MM;
S              = zeros(M*N,p);
%M1 =zeros(size(D));  % multiplier for D-X-E
M1 = MM / dual_norm;
M2 = M1;%zeros(size(D));  % multiplier for Dx_X-X1
M3 = M2;%zeros(size(D));  % multiplier for Dy_X-X2
M4 = M3;%zeros(size(D));  % multiplier for Dz_X-X3
% main loop
iter = 0;
tic
while iter<maxIter
    iter          = iter + 1;   
    %% -Updata X1,X2,X3
    [u,s,v] = svd(reshape(diff_x(X,sizeM),[M*N,p])+M2/mu,'econ');
    G1      = u*softthre(s,1/mu)*v';
    [u,s,v] = svd(reshape(diff_y(X,sizeM),[M*N,p])+M3/mu,'econ');
    G2      = u*softthre(s,1/mu)*v';
    [u,s,v] = svd(reshape(diff_z(X,sizeM),[M*N,p])+M4/mu,'econ');
    G3      = u*softthre(s,weight/mu)*v';
    %% -Updata X
    diffT_p  = diff_xT(mu*G1-M2,sizeM)+diff_yT(mu*G2-M3,sizeM);
    diffT_p  = diffT_p + diff_zT(mu*G3-M4,sizeM);
    numer1   = reshape( diffT_p + mu*(MM(:)-S(:)) + M1(:), sizeM);
    x        = real( ifftn( fftn(numer1) ./ (mu*determ + mu) ) );
    X        = reshape(x,[M*N,p]);
    %% -Update E
    S             = softthre(MM-X+M1/mu, lambda/mu);
%     E               = (M1+mu*(D-X))/(2*lambda+mu);% Gaussian noise
    %% stop criterion  
    leq1 = MM -X -S;
    leq2 = reshape(diff_x(X,sizeM),[M*N,p])- G1;
    leq3 = reshape(diff_y(X,sizeM),[M*N,p])- G2;
    leq4 = reshape(diff_z(X,sizeM),[M*N,p])- G3;
    stopC1 = norm(leq1,'fro')/normD;
    stopC2 = max(abs(leq2(:)));
    % stopC4 = norm(leq4,'fro')/normD;
    % if mod(iter,10)==0
    %     disp(['iter ' num2str(iter) ',mu=' num2str(mu,'%2.1e')  ...
    %             ',Y-X-E=' num2str(stopC1,'%2.3e') ',||DX-X1||=' num2str(stopC2,'%2.3e')...
    %             ',|DZ-X3|' num2str(stopC4,'%2.3e')]);
    % end
    if stopC1<tol && stopC2<tol
        fprintf('Iteration number of 3DCTV-RPCA is %d\n', iter);
        break;
    else
        M1 = M1 + mu*leq1;
        M2 = M2 + mu*leq2;
        M3 = M3 + mu*leq3;
        M4 = M4 + mu*leq4;
        mu = min(max_mu,mu*rho); 
    end 
%     load('Simu_indian.mat');
%     [mp(iter),sm(iter),er(iter)]=msqia(simu_indian,reshape(X,[M,N,p]));
end
% [u,s,v]= svd(D-E,'econ');
% diags = diag(s);
% for svp = 1:length(diags)
%     if sum(diags(1:svp))/sum(diags)>=0.995
%         break;
%     end
% end
% X = u(:,1:svp)*diag(diags(1:svp))*v(:,1:svp)';
output_image = reshape(X,[M,N,p]);
end