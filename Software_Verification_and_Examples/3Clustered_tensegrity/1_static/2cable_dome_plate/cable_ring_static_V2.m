function [V2_1,V2_2] = cable_ring_static_V2(rate1,R,p,h1,h2,h3)
%UNTITLED2 此处显示有关此函数的摘要
%   此处显示详细说明

r0=rate1*R;
% p=12;          %complexity for cable dome(Outer ring node)

% h1=0;       %中环高度
% h2=10;       %上环高度
% h3=-10;       %下环高度    

% generate node in one unit
beta1=pi/(p); beta2=4*pi/p;         %  两点间旋转量
beta3=2*pi/p;         %  整体角度

T1=[cos(beta1) -sin(beta1) 0
    sin(beta1) cos(beta1) 0
    0 0 1];            %内节点旋转量
T2=[cos(beta2) -sin(beta2) 0
    sin(beta2) cos(beta2) 0
    0 0 1];            %上下节点旋转量

T3=[cos(beta3) -sin(beta3) 0
    sin(beta3) cos(beta3) 0
    0 0 1];

N_1_0=[T1*[r0;0;h1]];       %内初节点
N_2_0=[T2*[R;0;h2]];      %上初节点
N_3_0=[T2*[R;0;h3]];      %下初节点

N_1=[];
N_2=[];
N_3=[];

for i=1:p  %内环节点
 N_1=[N_1,T3^(i-1)*N_1_0];
end
for i=0:p-1    %上环节点
 N_2=[N_2,T3^(i-1)*N_2_0];
end
for i=0:p-1    %下环节点
 N_3=[N_3,T3^(i-1)*N_3_0];
end

N=[N_3,N_1,N_2];  

C_b_in=[];

C_s_in=[[1:1:p]',[p+1:1:2*p]';[1:1:p]',[p+2:1:2*p,p+1]';
    [2*p+1:1:3*p]',[p+1:1:2*p]';[2*p+1:1:3*p]',[p+2:1:2*p,p+1]';
     [p+1:1:2*p]',[p+2:1:2*p,p+1]'];
 
C_b = tenseg_ind2C(C_b_in,N);  
C_s = tenseg_ind2C(C_s_in,N);   
C=[C_s;C_b];
[ne,nn]=size(C);        % ne:No.of element;nn:No.of node
% Plot the structure to make sure it looks right
% tenseg_plot(N,C_b,C_s);
% title('Cable dome');
%% %% Boundary constraints
% pinned_X=([1:1:2*p,2*p+1,3*p+1])'; pinned_Y=([1:1:2*p,2*p+1,3*p+1])'; pinned_Z=([1:1:2*p,2*p+1,3*p+1])';

pinned_X=([1:1:p,2*p+1:1:3*p])'; pinned_Y=([1:1:p,2*p+1:1:3*p])'; pinned_Z=([1:1:p,2*p+1:1:3*p])';
[Ia,Ib,a,b]=tenseg_boundary(pinned_X,pinned_Y,pinned_Z,nn);
%% %% Group/Clustered information 

% [gr] = cable_ring_gr(gr_num,p);
gr={[2*p+1:4*p]';[1:2*p]';[4*p+1:5*p]'};  % 上，下，环；一组

Gp=tenseg_str_gp3(gr,C);   %generate group matrix

S=Gp';                      % clustering matrix is group matrix

% tenseg_plot_CTS(N,C,[],S);
%% %% self-stress design
%Calculate equilibrium matrix and member length
[A_1a,A_1ag,A_2a,A_2ag,l,l_gp]=tenseg_equilibrium_matrix1(N,C,Gp,Ia);
A_1ac=A_1a*S';          %equilibrium matrix CTS
A_2ac=A_2a*S';          %equilibrium matrix CTS
l_c=S*l;                % length vector CTS

%SVD of equilibrium matrix
[U1,U2,V1,V2,S1]=tenseg_svd(A_1ag);

V2_1 = V2(1,:);    %斜索截预应力
V2_2 = V2(3,:);    %环索截预应力

end

