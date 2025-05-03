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


function [ output_image,E] =ctv_con_spcp(oriData3_noise,sigma,weight)
tol     = 1e-6;
maxIter = 1000;
rho     = 1.1;
% max_mu  = 1e6;
% mu      = 1e-3;
[M,N,p] = size(oriData3_noise);
lambda  = 3/sqrt(M*N);
eps = 1e-6;
if nargin < 2
    sigma = 0;
end
if nargin < 3
    weight = 1;
end
beta    = 6/((sqrt(M*N)+sqrt(p))*sigma+eps);
sizeD   = size(oriData3_noise);
D       = zeros(M*N,p) ;
for i=1:p
    bandp = oriData3_noise(:,:,i);
    D(:,i)= bandp(:);
end
normD   = norm(D,'fro');
% initialize
norm_two = lansvd(D, 1, 'L');
norm_inf = norm( D(:), inf) / lambda;
dual_norm = max(norm_two, norm_inf);

mu = 1.25/dual_norm;%1.25/norm_two % this one can be tuned
max_mu = mu * 1e7
%% FFT setting
h               = sizeD(1);
w               = sizeD(2);
d               = sizeD(3);
%% 
Eny_x   = ( abs(psf2otf([+1; -1], [h,w,d])) ).^2  ;
Eny_y   = ( abs(psf2otf([+1, -1], [h,w,d])) ).^2  ;
Eny_z   = ( abs(psf2otf([+1, -1], [w,d,h])) ).^2  ;
Eny_z   =  permute(Eny_z, [3, 1 2]);
determ  =  Eny_x + Eny_y + weight^2*Eny_z;
%% Initializing optimization variables
X1             = D;%randn(M*N,p);
X2             = X1;
X3             = X1;
E1             = zeros(M*N,p);
E2             = zeros(M*N,p);
G              = zeros(M*N,p);

M1 = zeros(size(D));  % multiplier for Dx_X-X1
M2 = zeros(size(D));  % multiplier for Dy_X-X2
M3 = zeros(size(D));  % multiplier for Dz_X-X3
M4 = D / dual_norm;
M5 = zeros(size(D));
M6 = zeros(size(D));
M7 = zeros(size(D));
% main loop
iter = 0;
tic
while iter<maxIter
    iter          = iter + 1;   
    %% solve the first block
    % -Updata G1,G2,G3
    [u,s,v] = svd(reshape(diff_x(X2,sizeD),[M*N,p])-M1/mu,'econ');
    G1      = u*softthre(s,1/mu)*v';
    [u,s,v] = svd(reshape(diff_y(X2,sizeD),[M*N,p])-M2/mu,'econ');
    G2      = u*softthre(s,1/mu)*v';
    [u,s,v] = svd(reshape(diff_z(X2,sizeD),[M*N,p])-M3/mu,'econ');
    G3      = u*softthre(s,1/mu)*v';
    disp(['the rank of X2 = ',num2str(rank(G2))]);
    % -Updata X1
    X1 = (X2+X3-(M5+M6)/mu)/(1+1);
    % -Update S1
    E1 = softthre(E2-M7/mu, lambda/mu);
    % -Update G
    G = (M4+mu*(D-X3-E2))/(beta+mu);% Gaussian noise
    %% solve the second block
    % -Update X2
    diffT_p  = diff_xT(mu*G1+M1,sizeD)+diff_yT(mu*G2+M2,sizeD);
    diffT_p  = diffT_p + weight*diff_zT(mu*G3+M3,sizeD);
    numer1   = reshape( diffT_p + mu*X1(:) + M5(:), sizeD);
    x        = real( ifftn( fftn(numer1) ./ (mu*determ + mu) ) );
    X2       = reshape(x,[M*N,p]);
    % -Update S2
    E2       = 1/3*(2*E1+D-G-X1 + (2*M7+M4-M6)/mu);
    X3       = 1/3*(2*X1+D-G-E1 + (2*M6+M4-M7)/mu);
    
    %% stop criterion  
    leq1 = G1 - reshape(diff_x(X2,sizeD),[M*N,p]);
    leq2 = G2 - reshape(diff_y(X2,sizeD),[M*N,p]);
    leq3 = G3 - reshape(diff_z(X2,sizeD),[M*N,p]);
    leq4 = D -X3 -E2 - G;
    leq5 = X1 - X2;
    leq6 = X1 - X3;
    leq7 = E1 - E2;
    stopC1 = norm(leq1,'fro')/normD;
    stopC4 = norm(leq4,'fro')/normD;
    stopC7 = norm(leq7,'fro')/normD;
    disp(['iter ' num2str(iter) ',mu=' num2str(mu,'%2.1e')  ...
            ',Y-X-E=' num2str(stopC4,'%2.3e') ',||DX-X1||=' num2str(stopC1,'%2.3e')...
            ',|DZ-X3|' num2str(stopC7,'%2.3e')]);
    if stopC1<tol && stopC4<tol && stopC7<tol
        break;
    else
        M1 = M1 + mu*leq1;
        M2 = M2 + mu*leq2;
        M3 = M3 + mu*leq3;
        M4 = M4 + mu*leq4;
        M5 = M5 + mu*leq5;
        M6 = M6 + mu*leq6;
        M7 = M7 + mu*leq7;
        mu = min(max_mu,mu*rho); 
    end 
end
E = (E1+E2)/2;
output_image = reshape((X1+X2+X3)/3,[M,N,p]);
%output_image = reshape(X3,[M,N,p]);
%output_image = reshape(D-E,[M,N,p]);
end