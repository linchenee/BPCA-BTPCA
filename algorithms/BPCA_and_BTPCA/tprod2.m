function [N_prox, A1Tf, A2f, GA1TA2KT] = tprod2(n3, even, half, IGWuGTf, IKWvKTf, MSZ, Kf, Gf, WuGTf, KWvf)
% Perform tensor-tensor products (used for the BTPCA method)
% ----------------------------------------------------------
% version 1.0 - 05/01/2025
% Written by Lin Chen (lchen53@stevens.edu)

MSZf = fft(MSZ,[],3);
temp = IGWuGTf(:,:,1) * MSZf(:,:,1);

Nf_prox(:,:,1) = temp * IKWvKTf(:,:,1);
A1Tf(:,:,1) = WuGTf(:,:,1) * MSZf(:,:,1);
A2f(:,:,1) =  temp * KWvf(:,:,1);
GA1TA2KTf(:,:,1) = Gf(:,:,1) * A1Tf(:,:,1) + A2f(:,:,1) * Kf(:,:,1)';

for i = 2:half
 temp = IGWuGTf(:,:,i) * MSZf(:,:,i);

 Nf_prox(:,:,i) = temp * IKWvKTf(:,:,i);   
 A1Tf(:,:,i) = WuGTf(:,:,i) * MSZf(:,:,i);
 A2f(:,:,i) =  temp * KWvf(:,:,i);
 GA1TA2KTf(:,:,i) = Gf(:,:,i) * A1Tf(:,:,i) + A2f(:,:,i) * Kf(:,:,i)';
 
 Nf_prox(:,:,n3-i+2) = conj(Nf_prox(:,:,i));
 A1Tf(:,:,n3-i+2) = conj(A1Tf(:,:,i));
 A2f(:,:,n3-i+2) = conj(A2f(:,:,i));
 GA1TA2KTf(:,:,n3-i+2) = conj(GA1TA2KTf(:,:,i));
end

if even
 temp = IGWuGTf(:,:,half+1) * MSZf(:,:,half+1);

 Nf_prox(:,:,half+1) = temp * IKWvKTf(:,:,half+1);   
 A1Tf(:,:,half+1) = WuGTf(:,:,half+1) * MSZf(:,:,half+1);
 A2f(:,:,half+1) =  temp * KWvf(:,:,half+1);
 GA1TA2KTf(:,:,half+1) = Gf(:,:,half+1) * A1Tf(:,:,half+1) + A2f(:,:,half+1) * Kf(:,:,half+1)';
end

N_prox = ifft(Nf_prox, [], 3);
GA1TA2KT = ifft(GA1TA2KTf, [], 3);
end