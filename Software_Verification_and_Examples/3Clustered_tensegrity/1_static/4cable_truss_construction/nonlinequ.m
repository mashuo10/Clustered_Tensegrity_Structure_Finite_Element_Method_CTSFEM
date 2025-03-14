function X=nonlinequ(X0)
%solve nonlinear equilibrium equations
global E_bar A_bar l0_bar Ia Ib C w ne n
X=X0;
Xb=Ib'*X;
u=1e-4;
for i=1:1e3
Xa=Ia'*X;

l_bar=diag(sqrt(sum((reshape(X,3,[])*C').^2))); %bar length matrix
q_bar=E_bar*A_bar*(inv(l0_bar)-inv(l_bar));      %force density

K=kron(C'*q_bar*C,eye(3));                      %stiffness matrix
Kx=K*X;
Fp=w-Kx;                                       %unbalanced force
norm(Ia'*Fp);                                  %see the norm of unbalanced force
if norm(Ia'*Fp)<1e-4
    break 
end
%calculate stiffness matrix
dqdn=zeros(ne,3*n);
for j=1:ne
    dqdn(j,:)=X'*kron((C(j,:))'*C(j,:),eye(3));
end
dqdn=E_bar*A_bar*(l_bar)^-3*dqdn;

K_t=kron(C'*q_bar*C,eye(3))+...
    kron(C',eye(3))*diag(kron(C,eye(3))*X)*kron(eye(ne),ones(3,1))*dqdn;

K_taa=Ia'*K_t*Ia;

%modify the stiffness matrix
e=eig(K_taa);                       %刚度矩阵特征根
% eigv=sort(e);
% eigv=eigv(7);
lmd=min(e);                     %刚度矩阵最小特征根
if lmd>0
    Km=K_taa+u*eye(size(K_taa)); %修正的刚度矩阵
else
Km=K_taa+(abs(lmd)+u)*eye(size(K_taa)); 
end
dXa=Km\(Ia'*Fp);

% dXa=K_taa\(Ia'*Fp);
Xa=Xa+dXa;
X=[Ia';Ib']\[Xa;Xb];
end

A23=13;  %无用语句，打断用






 

