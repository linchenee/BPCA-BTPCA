function [IGWGTf, IKWKTf, WuGTf, KWvf] = tprod1(n1, n2, n3, even, half, Gf, Kf, Wu, Wv)
% Perform tensor-tensor products (used for the BTPCA method) 
% ----------------------------------------------------------
% version 1.0 - 05/01/2025
% Written by Lin Chen (lchen53@stevens.edu)

I1 = zeros(n1,n1,n3);
I2 = zeros(n2,n2,n3);
I1(:,:,1) = eye(n1);
I2(:,:,1) = eye(n2);
I1f = fft(I1, [], 3);
I2f = fft(I2, [], 3);
Wuf = fft(Wu, [], 3);
Wvf = fft(Wv, [], 3);

WuGTf(:,:,1) = Wuf(:,:,1) * Gf(:,:,1)';
KWvf(:,:,1) = Kf(:,:,1) * Wvf(:,:,1);
IGWGTf(:,:,1) = I1f(:,:,1) - Gf(:,:,1) * Wuf(:,:,1) * Gf(:,:,1)';
IKWKTf(:,:,1) = I2f(:,:,1) - Kf(:,:,1) * Wvf(:,:,1) * Kf(:,:,1)';

for i = 2:half
 WuGTf(:,:,i) = Wuf(:,:,i) * Gf(:,:,i)';
 KWvf(:,:,i) = Kf(:,:,i) * Wvf(:,:,i);
 IGWGTf(:,:,i) = I1f(:,:,i) - Gf(:,:,i) * Wuf(:,:,i) * Gf(:,:,i)';
 IKWKTf(:,:,i) = I2f(:,:,i) - Kf(:,:,i) * Wvf(:,:,i) * Kf(:,:,i)';

 WuGTf(:,:,n3-i+2) = conj(WuGTf(:,:,i));
 KWvf(:,:,n3-i+2) = conj(KWvf(:,:,i));
 IGWGTf(:,:,n3-i+2) = conj(IGWGTf(:,:,i));
 IKWKTf(:,:,n3-i+2) = conj(IKWKTf(:,:,i));
end

if even
 WuGTf(:,:,half+1) = Wuf(:,:,half+1) * Gf(:,:,half+1)';
 KWvf(:,:,half+1) = Kf(:,:,half+1) * Wvf(:,:,half+1);
 IGWGTf(:,:,half+1) = I1f(:,:,half+1) - Gf(:,:,half+1) * Wuf(:,:,half+1) * Gf(:,:,half+1)';
 IKWKTf(:,:,half+1) = I2f(:,:,half+1) - Kf(:,:,half+1) * Wvf(:,:,half+1) * Kf(:,:,half+1)';
end
end