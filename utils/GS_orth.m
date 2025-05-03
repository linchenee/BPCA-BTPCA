function Q = GS_orth(A)
% Perform the Gram-Schmidt orthogonalization
% -----------------------------------------------
% version 1.0 - 05/01/2025
% Written by Lin Chen (lchen53@stevens.edu)

[m, n] = size(A);
Q = zeros(m, n);
R = zeros(m, n);
for j = 1:n
    v = A(:, j);
    for i = 1:j-1
        R(i, j) = Q(:, i)' * v;
        v = v - R(i, j) * Q(:, i);
    end

    R(j, j) = norm(v);
    Q(:, j) = v ./ R(j, j);
end
end