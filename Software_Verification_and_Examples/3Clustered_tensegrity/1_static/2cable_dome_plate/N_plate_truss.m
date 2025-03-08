function [N0,C_b,n_qp] =N_plate_truss(R,rate,p,h,H);
%UNTITLED 此处提供此函数的摘要
%   此处提供详细说明
% generate node in one unit
alpha=pi/p;
q1=round(R/(2*h));      %complexity in Radial edge
q2=round(2*R*sin(pi/p)/(2*h));    % complexity in bottom edge
N0=zeros(3,(2*q1+q2+1)*2*p);
N_ini=zeros(3,3*p);
N_ini(:,1:3)= R*  [-rate 0 H/R; 
        sin(alpha)-rate -cos(alpha) H/R;
        -sin(alpha)-rate -cos(alpha) H/R]';
N_ini(3,:)=H;
N0(:,1:q1+1)=kron(ones(1,q1+1),N_ini(:,1))+kron(linspace(0,1,q1+1),(N_ini(:,2)-N_ini(:,1)));
N0(:,q1+1:q1+q2+1)=kron(ones(1,q2+1),N_ini(:,2))+kron(linspace(0,1,q2+1),(N_ini(:,3)-N_ini(:,2)));
N0(:,q1+q2+1:2*q1+q2+1)=kron(ones(1,q1+1),N_ini(:,3))+kron(linspace(0,1,q1+1),(N_ini(:,1)-N_ini(:,3)));

N0(:,2*q1+q2+2:2*(2*q1+q2+1))=N0(:,1:2*q1+q2+1);
N0(3,2*q1+q2+2:2*(2*q1+q2+1))=H+h;

N_ini(3,:)=H+h;
T1=[cos(2*alpha) -sin(2*alpha) 0
    sin(2*alpha) cos(2*alpha) 0
    0 0 1]';
for i=2:p
N0(:,((2*q1+q2+1)*2)*(i-1)+1:((2*q1+q2+1)*2)*i)=T1^(i-1)*N0(:,1:(2*q1+q2+1)*2);
N_ini(:,3*i-2:3*i)=T1^(i-1)*N_ini(:,1:3);
end

% C_b_in = kron(ones(p,1),[1 2;2 3;3 1])+kron(3*[0:p-1]',ones(3,2));  % Bar 
C_b_in1=[[1:2*q1+q2+1;2*q1+q2+2:2*(2*q1+q2+1)]';[1:2*(2*q1+q2+1)-1;2:2*(2*q1+q2+1)]';[1:2*q1+q2;2*q1+q2+3:2*(2*q1+q2+1)]'];
C_b_in=kron(ones(p,1),C_b_in1)+kron(2*(2*q1+q2+1)*[0:p-1]',ones(size(C_b_in1)));
C_b = tenseg_ind2C(C_b_in,N0);

n_qp=mat2cell(N_ini(:),3*3*ones(1,p));
end