function [Lnew_prox, B1Tf, B2f, GB1TB2KT] = tprod4(n3, even, half, IGGTf, IKKTf, XEO, Kf, Gf)
% Perform tensor-tensor products (used for the M-TPCP method)
% -----------------------------------------------------------
% version 1.0 - 05/01/2025
% Written by Lin Chen (lchen53@stevens.edu)

XEOf = fft(XEO,[],3);
temp = IGGTf(:,:,1) * XEOf(:,:,1);

Lnewf_prox(:,:,1) = temp * IKKTf(:,:,1);
B1Tf(:,:,1) = Gf(:,:,1)' * XEOf(:,:,1);
B2f(:,:,1) =  temp * Kf(:,:,1);
GB1TB2KTf(:,:,1) = Gf(:,:,1) * B1Tf(:,:,1) + B2f(:,:,1) * Kf(:,:,1)';

for i = 2:half
 temp = IGGTf(:,:,i) * XEOf(:,:,i);

 Lnewf_prox(:,:,i) = temp * IKKTf(:,:,i);   
 B1Tf(:,:,i) = Gf(:,:,i)' * XEOf(:,:,i);
 B2f(:,:,i) =  temp * Kf(:,:,i);
 GB1TB2KTf(:,:,i) = Gf(:,:,i) * B1Tf(:,:,i) + B2f(:,:,i) * Kf(:,:,i)';
 
 Lnewf_prox(:,:,n3-i+2) = conj(Lnewf_prox(:,:,i));
 B1Tf(:,:,n3-i+2) = conj(B1Tf(:,:,i));
 B2f(:,:,n3-i+2) = conj(B2f(:,:,i));
 GB1TB2KTf(:,:,n3-i+2) = conj(GB1TB2KTf(:,:,i));
end

if even
 temp = IGGTf(:,:,half+1) * XEOf(:,:,half+1);

 Lnewf_prox(:,:,half+1) = temp * IKKTf(:,:,half+1);   
 B1Tf(:,:,half+1) = Gf(:,:,half+1)' * XEOf(:,:,half+1);
 B2f(:,:,half+1) =  temp * Kf(:,:,half+1);
 GB1TB2KTf(:,:,half+1) = Gf(:,:,half+1) * B1Tf(:,:,half+1) + B2f(:,:,half+1) * Kf(:,:,half+1)';
end

Lnew_prox = ifft(Lnewf_prox, [], 3);
GB1TB2KT = ifft(GB1TB2KTf, [], 3);
end