function Noise = generate_N(mode, sigma, lambda, mat_size)
%% mode = 'G': i.i.d noise with sigma
%% mode = 'B': Non i.i.d noise with sigma_t in [0.5*sigma, 1.5*sigma]
%% mode = 'U': uniform distribution. unifrnd(-2*sigma,2*sigma,[m,n])
%% mode = 'P': l.possion(lambda) with l=sigma/(lambda+lambda^2), lambda \in [1,3,5]
if nargin < 2
    sigma = 0.1;
end
if nargin < 3
    lambda = 1;
end
if nargin < 4
    mat_size = [400,200];
end
m = mat_size(1);
n = mat_size(2);
if strcmp(mode,'B')
    sigma_list   = rand(m,n)*sigma+sigma/2;
    Gmat  = randn(m,n);
    Noise = sigma_list.*Gmat;
elseif strcmp(mode,'P')
    %scale = sigma/(sqrt(lambda+lambda^2));
    scale = sigma/sqrt(lambda);
    Noise = scale*(random('poisson',lambda,m,n)-lambda);
elseif strcmp(mode,'U')
    Noise = unifrnd(-sqrt(3)*sigma,sqrt(3)*sigma,[m,n]);
else
    Noise = sigma * randn(m,n);
end