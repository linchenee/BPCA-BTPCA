function [X] = CastImageAsKetAdjustable(M,a1,a2,b1,b2,c1,c2,d1,d2,frame)
    i=1;
    j=1;
    X = zeros(d1*d2,c1*c2,b1*b2,a1*a2,frame);
    
   for i4 = 1:d1*d2
    [i,j] = index4_ij(i,j,a1,a2,b1,b2,c1,c2,d1,i4);
    for i3 = 1:c1*c2
     [i,j] = index3_ij(i,j,a1,a2,b1,b2,c1,i3);
     for i2 = 1:b1*b2
       [i,j] = index2_ij(i,j,a1,a2,b1,i2);
       Mtemp = M(i:i+a1-1,j:j+a2-1,:);
       X(i4,i3,i2,:) = Mtemp(:);
%        fprintf('%d %d\n',i,j);
     end
    end
   end
end

function [iN,jN] = index2_ij(i0,j0,a1,a2,b1,k_N)
  if k_N==1
    iN = i0;
    jN = j0;   
  elseif (mod(k_N,b1)==1) && (k_N~=1)
    iN = i0-a1*(b1-1);
    jN = j0+a2;    
  else
    iN = i0+a1;
    jN = j0;    
  end  
end

function [iN,jN] = index3_ij(i0,j0,a1,a2,b1,b2,c1,k_N)
  if k_N==1
    iN = i0;
    jN = j0;   
  elseif (mod(k_N,c1)==1) && (k_N~=1)
    iN = i0-a1*(b1*c1-1);
    jN = j0+a2;    
  else
    iN = i0+a1;
    jN = j0-a2*(b2-1);    
  end  
end

function [iN,jN] = index4_ij(i0,j0,a1,a2,b1,b2,c1,c2,d1,k_N)
  if k_N==1
    iN = i0;
    jN = j0;   
  elseif (mod(k_N,d1)==1) && (k_N~=1)
    iN = i0-a1*(b1*c1*d1-1); %
    jN = j0+a2;    
  else
    iN = i0+a1;
    jN = j0-a2*(b2*c2-1);    %
  end  
end