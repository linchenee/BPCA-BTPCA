function [Y] = tensor_rpca_adal_parfor(T, lambda, mu1, mu2, max_iter, opt_tol)
% Solve
%       min_{X,E} \sum_i ||X_(i)||_* + \lambda*||E||_1
%       s.t. X + E = T
% converse to
%       min_{X1,X2,...,XN,E} \sum_i ||Xi_(i)||_* + \lambda*||E||_1
%       s.t. Xi=Y,i=1,2,...,N
%            Y+E=T
% algorithm：交替投影增广拉格朗日，其中最小化Y=两个二范数的平方和，令其导数为零即可求解。

% data.T
% params
% X, V are cell arrays of tensors.
%
% Algorithm: ADAL

N = length( size(T) );
X = cell( 1, N );
U = cell( 1, N );
V = cell( 1, N+1 );
for i = 1:N
    X{i} = T;
    V{i} = zeros(size(T));
end
V{N+1} = zeros(size(T));
Y = T;

for iter = 1:max_iter
    % solve X_i's
    for i = 1:N
        [X{i}, junk, U{i}] = tensor_shrinkage( Y+mu2*V{i}, mu2, i );
    end
    
    % solve E
    P = T - Y + mu1*V{N+1};
    E = shrinkage_t( P, lambda*mu1 );
    
    % solve Y
    Yrhs = V{N+1} + (T - E)/mu1;
    Yrhs = Yrhs + ten_sum_all( X )/mu2 - ten_sum_all( V(1:N) );
    Yprev = Y;
    Y = Yrhs / (N/mu2 + 1/mu1);
    
    % compute optimality stats
    pres = 0;
    tdiff = cell( 1, N+1 );
    for i = 1:N
        tdiff{i} = X{i} - Y;
        pres = pres + norm( tdiff{i} )^2;
    end
    tdiff{N+1} = Y + E - T;
    pres = pres + norm(tdiff{N+1})^2;     %pres = sqrt(pres);
    
    pres = sqrt( pres / (norm(T(:))^2+N*norm(Y)^2) );
    Ydiff = Y-Yprev;
    dres = norm(Ydiff) / norm(Yprev(:));

    if max(pres, dres) < opt_tol
%     if pres < params.opt_tol
        fprintf('Iteration number of HoRPCA-S is %d\n', iter);
        break;
    end
    
    % update Lagrange multipliers
    for i = 1:N
        V{i} = V{i} - tdiff{i}/mu2;
    end
    V{N+1} = V{N+1} - tdiff{N+1}/mu1;
end
end
