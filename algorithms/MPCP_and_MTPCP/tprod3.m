function [IGGTf, IKKTf] = tprod3(n1, n2, n3, even, half, Gf, Kf)
% Perform tensor-tensor products (used for the M-TPCP method)
% -----------------------------------------------------------
% version 1.0 - 05/01/2025
% Written by Lin Chen (lchen53@stevens.edu)

I1 = zeros(n1,n1,n3);
I2 = zeros(n2,n2,n3);
I1(:,:,1) = eye(n1);
I2(:,:,1) = eye(n2);
I1f = fft(I1, [], 3);
I2f = fft(I2, [], 3);

IGGTf(:,:,1) = I1f(:,:,1) - Gf(:,:,1) * Gf(:,:,1)';
IKKTf(:,:,1) = I2f(:,:,1) - Kf(:,:,1) * Kf(:,:,1)';

for i = 2:half
 IGGTf(:,:,i) = I1f(:,:,i) - Gf(:,:,i) * Gf(:,:,i)';
 IKKTf(:,:,i) = I2f(:,:,i) - Kf(:,:,i) * Kf(:,:,i)';

 IGGTf(:,:,n3-i+2) = conj(IGGTf(:,:,i));
 IKKTf(:,:,n3-i+2) = conj(IKKTf(:,:,i));
end

if even
 IGGTf(:,:,half+1) = I1f(:,:,half+1) - Gf(:,:,half+1) * Gf(:,:,half+1)';
 IKKTf(:,:,half+1) = I2f(:,:,half+1) - Kf(:,:,half+1) * Kf(:,:,half+1)';
end
end