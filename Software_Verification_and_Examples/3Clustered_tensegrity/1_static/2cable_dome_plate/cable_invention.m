%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%A Clustered Cable Net(deployable)%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% /* This Source Code Form is subject to the terms of the Mozilla Public
% * License, v. 2.0. If a copy of the MPL was not distributed with this
% * file, You can obtain one at http://mozilla.org/MPL/2.0/.
%
% [1] structure design(define N, C, boundary constraints, clustering,
% calculate equilibrium matrix,
% group matrix,prestress mode, minimal mass design)
% [2] calculate tangent stiffness matrix, material
% stiffness, geometry stiffness,
% [3] dynamic simulation

%EXAMPLE
clc; clear all; close all;
% Global variable
[consti_data,Eb,Es,sigmab,sigmas,rho_b,rho_s]=material_lib('Steel_Q345','Steel_string');

pully =0;                      %pully influence 1 or 0
if pully 
[consti_data,Es,sigmas,rho_s]=material_lib_ring('Steel_string');
end

material{1}='linear_elastic'; % index for material properties: multielastic, plastic.
material{2}=0; % index for considering slack of string (1) for yes,(0) for no (for compare with ANSYS)

% cross section design cofficient
thick=6e-3;        % thickness of hollow bar
hollow_solid=0;          % use hollow bar or solid bar in minimal mass design (1)hollow (0)solid
c_b=0.1;           % coefficient of safty of bars 0.5
c_s=0.1;           % coefficient of safty of strings 0.3

% static analysis set
substep=1;                                     %荷载子步
lumped=0;               % use lumped matrix 1-yes,0-no
saveimg=0;              % save image or not (1) yes (0)no
savedata=1;             % save data or not (1) yes (0)no
savevideo=1;            % make video(1) or not(0)
gravity=0;              % consider gravity 1 for yes, 0 for no
% move_ground=0;          % for earthquake, use pinned nodes motion(1) or add inertia force in free node(0) 

%% 绘图

puc_ture = 0;   %是否存图,未指定分辨率

puc_ture_1 = 1;   %是否存图,指定分辨率
dpi = 600; 

folderPath = 'C:\Users\Owner\Desktop\数据分析-实验VS理论\0820.2';   %图像保存路径

%代码运行
pully1 = 0;  %满跨荷载索力变化 - 以组别
pully2 = 0;  %半跨荷载索力变化 - 以组别
pully3 = 0;  %满跨荷载位移变化 - 以组别
pully4 = 0;  %半跨荷载位移变化 - 以组别

pully5 = 0;  %以开合度作为指标 - 索力-半跨
pully6 = 0;  %以开合度作为指标 - 位移-半跨

pully7 = 1;  %刚度特征值分析 - 以组别
pully8 = 1;  %频率特征值分析 - 以组别

pully9 = 1;  %以开合度作为指标 - 刚度特征值
pully10 = 1;  %以预应力作为指标 - 刚度特征值

pully11 = 1;  %以开合度作为指标 - 频率特征值
pully12 = 1;  %以预应力作为指标 - 频率特征值

Line_Width =2;
Marker_Size =10;
%% %% N C of the structure

x_1 = [1 2 3 4];
y_1 = [150 200 250 300];   %目标预应力设计
coefficients1 = polyfit(x_1, y_1, 2);
ka1 = coefficients1(:,2); kb1 = coefficients1(:,3);
% x2 = y1;
% y2 = x1;   %目标预应力设计
% coefficients2 = polyfit(x2, y2, 2);
% ka2 = coefficients2(:,2); kb2 = coefficients2(:,3);

num_tume = 0.5 ;    %出图速度
num_1_1 = 0; num_2_1 = 0; num_1_1 = 0; num_3_1 = 0;



i_1 = 1:4;
for fd1 = ka1*i_1 +kb1  %环索内力设定值
    

    
gr_num_1=[1 2 3 6 12];      %同一个环，组数划分（1，2，3，4，6）；节点数需为其的整数倍数
for gr_num=gr_num_1

    
i_2 = 1:5;
for F1 = 20*i_2;   %节点竖向力（外荷载）; 分别为:20 40 60 80 100
    


rate1=0.3;       %内环半径与外环半径之比(初始）
Rate_1=0.7;
R=100;          %外环半径
r0=rate1*R;     %内环半径

p=12;          %complexity for cable dome(Outer ring node)；2的倍数

fd1=150;

h1=0;       %中环高度
h2=20;       %上环高度
h3=-h2;       %下环高度  


%截面直径为1.5mm
A_c1=1.0e-6*1.1605;
A_c2=1.0e-6*1.1605;
A_c3=1.0e-6*1.1605;
% generate node in one unit
% beta1=pi/p; beta2=4*pi/p;         %  两点间旋转量
beta1=pi/p; beta2=2*pi/p;         %  两点间旋转量
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

N_1_0=[T1*[r0;0;h1]];       %内初节点_1
N_2_0=[T2*[R;0;h2]];      %上初节点
N_3_0=[T2*[R;0;h3]];      %下初节点

N_1=[];N_2=[];N_3=[];


for i=1:p  %内环节点
 N_1=[N_1,T3^(i-1)*N_1_0];
end
for i=0:p-1    %上环节点
 N_2=[N_2,T3^(i-1)*N_2_0];
end
for i=0:p-1    %下环节点
 N_3=[N_3,T3^(i-1)*N_3_0];
end

N=[N_3,N_1,N_2];      %显示外框


C_b_in_1=[];

% C_s_in=[[1:1:p]',[p+1:1:2*p]';[1:1:p]',[p+2:1:2*p,p+1]';
%     [2*p+1:1:3*p]',[p+1:1:2*p]';[2*p+1:1:3*p]',[p+2:1:2*p,p+1]';
%      [p+1:1:2*p]',[p+2:1:2*p,p+1]'];
 
C_s_in=[[1:p]',[p+1:2*p]';[1:p]',[2*p,p+1:2*p-1]';
    [2*p+1:3*p]',[p+1:2*p]';[2*p+1:3*p]',[2*p,p+1:2*p-1]';
     [p+1:2*p]',[p+2:2*p,p+1]'];
 
 
C_b = tenseg_ind2C(C_b_in_1,N);  
C_s = tenseg_ind2C(C_s_in,N);   
C=[C_s;C_b];
[ne,nn]=size(C);        % ne:No.of element;nn:No.of node
% Plot the structure to make sure it looks right
tenseg_plot(N,C_b,C_s);
axis off;

% title('Cable dome');

%% 绘制连接杆
C_b_in_2=[[1:p]',[2:p,1]';[2*p+1:3*p]',[2*p+2:3*p,2*p+1]';[1:p]',[2*p+1:3*p]'];
C_b_2 = tenseg_ind2C(C_b_in_2,N); 

%% %% Boundary constraints
pinned_X=([1:1:p,2*p+1:1:3*p])'; pinned_Y=([1:1:p,2*p+1:1:3*p])'; pinned_Z=([1:1:p,2*p+1:1:3*p])';
[Ia,Ib,a,b]=tenseg_boundary(pinned_X,pinned_Y,pinned_Z,nn);
%% Plot the structure to make sure it looks right
fig_handle=figure('Position', [400, 400, 400, 300]);
%% 板壳结构
H=0;  %板壳结构的高度

[N0,C_b,n_qp] =N_plate(R,rate1,p,H+0.3);

% tenseg_plot(N0,C_b,[]);
% fig_handle=figure;
tenseg_plot_RBD(N0,C_b,[],fig_handle,[],[],[],[],n_qp);

tenseg_plot(N,[],C_s,fig_handle);
axis off;
view(180/p,25)
%% %% Group/Clustered information 
%generate group index

[gr] = cable_ring_gr(gr_num,p);

Gp=tenseg_str_gp3(gr,C);   %generate group matrix

S=Gp';                      % clustering matrix is group matrix

tenseg_plot_CTS(N,C,[],S,fig_handle);
% tenseg_plot(N,C_b_2,[],fig_handle); 
axis off;

folderPath1 = 'C:\Users\Owner\Desktop\数据分析-实验VS理论\模型图';   %图像保存路
str2 = string(['组',num2str(gr_num)]);  %创建字符串
chr2 = char(str2);    %创建字符
  if  0
    saveas(gcf,[chr2,'.png']);
    img = imread([chr2,'.png']);
    filename = fullfile(folderPath1, [chr2,'.png']);
    imwrite(img, filename);
  end
  
%% 板壳结构
% H=0;  %板壳结构的高度
% 
% [N0,C_b,n_qp] =N_plate(R,rate1,p,H+0.3);
% 
% % tenseg_plot(N0,C_b,[]);
% % fig_handle=figure;
% tenseg_plot_RBD(N0,C_b,[],fig_handle,[],[],[],[],n_qp);

%% %% self-stress design
%Calculate equilibrium matrix and member length
[A_1a,A_1ag,A_2a,A_2ag,l,l_gp]=tenseg_equilibrium_matrix1(N,C,Gp,Ia);
A_1ac=A_1a*S';          %equilibrium matrix CTS
A_2ac=A_2a*S';          %equilibrium matrix CTS9

l_c=S*l;                % length vector CTS
%SVD of equilibrium matrix
[U1,U2,V1,V2,S1]=tenseg_svd(A_1ag);
[V2_1,V2_2] = cable_ring_static_V2(rate1,R,p,h1,h2,h3);
[V2] = cable_ring_V2(gr_num,V2_1,V2_2,p);
%external force in equilibrium design
w0=zeros(numel(N),1); w0a=Ia'*w0;

%prestress design

switch gr_num
    case 1
index_gp=[3]; %一组
fd=[fd1];
    case 2
index_gp=[5,6]; %二组
fd=fd1*ones(2,1);
    case 3
index_gp=[1:3]; %三组
fd=fd1*ones(3,1);
    case 6
index_gp=[13:18];%六组
fd=fd1*ones(6,1);
    case 12
index_gp=[4*p+1:5*p];%六组
fd=fd1*ones(12,1);
end 

[q_gp,t_gp,q,t]=tenseg_prestress_design(Gp,l,l_gp,A_1ag,V2,w0a,index_gp,fd);    %prestress design

t_c=pinv(S')*t;
q_c=pinv(S')*q;
%% cross sectional design

index_b=find(l_c<0.1);              % index of bar in compression
index_s=setdiff(1:size(S,1),index_b);	% index of strings
[A_b,A_s,A_c,A,r_b,r_s,r_gp,radius,E_c,l0_c,rho,mass_c]=tenseg_minimass(t_c,l_c,eye(size(S,1)),sigmas,sigmab,Eb,Es,index_b,index_s,c_b,c_s,rho_b,rho_s,thick,hollow_solid);

E=S'*E_c;     %Young's modulus CTS

[A_c]=cable_ring_Ac(gr_num,p,A_c1,A_c2);   %各索段截面积
% [A_c]=cable_ring_Ac_2(gr_num,p,A_c1,A_c2,A_c3);   %各索段截面积

A=S'*A_c;     % Cross sectional area CTS

l0=(t+E.*A).\E.*A.*l;

mass=S'*rho.*A.*l0;

%% Step1: equilibrium calculation
% input data



c1=zeros(1,3*p);
c2=kron(ones(1,p),[0,0,1]);
F1_2=-[c1, F1*c2, c1];    %建立施加外荷载（环节点）坐标矩阵，满跨荷载
F1_3=-[c1, F1*kron(ones(1,p/2),[0,0,1]),F1*kron(zeros(1,p/2),[0,0,1]), c1];    %建立施加外荷载（环节点）坐标矩阵，半跨荷载

substep1=3;   %使 tenseg_cable_ring_1 函数内的 rate_0 有分布
substep = substep1;

%计算满跨荷载时的节点竖向位移
% ind_w=[]';w=[];
ind_w=[1:3*3*p]';w=[F1_2];
ind_dnb=[]; dnb0=[];
ind_dl0_c=[]; dl0_c=[];
[w_t,dnb_t,l0_ct,Ia_new,Ib_new]=tenseg_load_prestress(substep,ind_w,w,ind_dnb,dnb0,ind_dl0_c,dl0_c,l0_c,b,gravity,[0;9.8;0],C,mass);
[l0_ct1,t_c1,n_1,lm_k,lm_omega,m_k,m_omega] = ring_3_2(substep1,Rate_1,rate1,R,r0,p,h1,h2,h3,beta1,beta2,beta3,gr_num,fd1,A_c1,A_c2)
w_t_1 = w'*ones(1,substep);

data.N=N; data.C=C; data.ne=ne; data.nn=nn; data.Ia=Ia_new; data.Ib=Ib_new;data.S=S;
data.E=E_c; data.A=A_c; data.index_b=index_b; data.index_s=index_s;
data.consti_data=consti_data;   data.material=material; %constitue info
data.w_t=w_t_1;  % external force
data.dnb_t=dnb_t;% forced movement of pinned nodes
data.l0_t=l0_ct1;% forced movement of pinned nodes
data.substep=substep;    % substep

data_out=static_solver_CTS(data);

t_t=data_out.t_out;          %member force in every step n_t=data_out.n_out;          %nodal coordinate in every step
N_out=data_out.N_out;
K_out=data_out.K_out;
lmd_out=data_out.lmd_out;  %最小刚度特征值
d_out=data_out.d_out;  %刚度特征值
l0_out=data_out.l0_out;   %各步态下时的弦长长度
n_t=data_out.n_out;          %nodal coordinate in every step


num_1 = gr_num/gr_num + num_1_1;   %储存组别行数,同行为相同组别
num_2 = F1/F1 + num_2_1;   %储存外力列数,同列为相同外力

t_c1_1{num_1,num_2}=t_c1([7 19 25],:);    %取索件1，19，25；无外力
t_c1_2{num_1,num_2}=t_c1([1:5*p],:);    %取索件1~30；无外力
% t_c1_3{num_1,num_2}=t_1(1,:);    %只取索件1；无外力

l0_ct1_1{num_1,num_2}=l0_ct1;
t_t_1{num_1,num_2}=t_t([7 19 25],:);    %取索件7，19，25；有外力
lmd_out_1{num_1,num_2}=lmd_out;  %最小刚度特征值
d_out_1{num_1,num_2}=d_out;  %刚度特征值
n_t_1{num_1,num_2}=n_t(21,:);

% lm_k_1{num_1,num_3}=lm_k; %最小刚度
m_k_1{num_1,num_2}=m_k([1:10],:); %刚度特征值,无外力
% lm_omega_1{num_1,num_3}=lm_omega; %最小频率
m_omega_1{num_1,num_2}=m_omega([1:10],:); %频率特征值,无外力


%计算半跨荷载时的节点竖向位移

ind_w=[1:3*3*p]';w=[F1_3];
ind_dnb=[]; dnb0=[];
ind_dl0_c=[]; dl0_c=[];
[w_t,dnb_t,l0_ct,Ia_new,Ib_new]=tenseg_load_prestress(substep1,ind_w,w,ind_dnb,dnb0,ind_dl0_c,dl0_c,l0_c,b,gravity,[0;9.8;0],C,mass);

% [l0_ct1] = tenseg_cable_ring(rate1,R,r0,p,h1,h2,h3,beta1,beta2,beta3,gr_num,fd1,A_c1,A_c2,V2_1,V2_2,mun);
% [l0_ct1,t_1,lm_k,lm_omega,m_k,m_omega] = tenseg_cable_ring_1(substep1,Rate_1,rate1,R,r0,p,h1,h2,h3,beta1,beta2,beta3,gr_num,fd1,A_c1,A_c2);
% l0_ct=l0_ct1;
w_t_1 = w'*ones(1,substep);

data.N=N; data.C=C; data.ne=ne; data.nn=nn; data.Ia=Ia_new; data.Ib=Ib_new;data.S=S;
data.E=E_c; data.A=A_c; data.index_b=index_b; data.index_s=index_s;
data.consti_data=consti_data;   data.material=material; %constitue info
data.w_t=w_t_1;  % external force
data.dnb_t=dnb_t;% forced movement of pinned nodes
data.l0_t=l0_ct1;% forced movement of pinned nodes
data.substep=substep;    % substep

% data_out1=static_solver_CTS(data);
% % data_out{i}=equilibrium_solver_pinv(data);        %solve equilibrium using mNewton method

data_out=static_solver_CTS(data);

t_t=data_out.t_out;          %member force in every stepn_t=data_out.n_out;          %nodal coordinate in every step
N_out=data_out.N_out;
K_out=data_out.K_out; 
lmd_out=data_out.lmd_out;  %最小刚度特征值
d_out=data_out.d_out;  %刚度特征值
l0_out=data_out.l0_out;   %各步态下时的弦长长度
n_t=data_out.n_out;          %nodal coordinate in every step

t_t_2{num_1,num_2}=t_t([1:5*p],:);    %取索件1~30
t_t_3{num_1,num_2}=t_t([7 19 25],:);    %只取索件1

lmd_out_2{num_1,num_2}=lmd_out;
n_t_2{num_1,num_2}=n_t(3*(p+1):3:6*p,:);  %记录节点8，9，10，11的竖向位移
d_out_2{num_1,num_2}=d_out;

 num_2_1 = num_2_1 + 1;
 if num_2_1 >=numel(i_2)
    num_2_1 =0;
 end
end

 num_1_1 = gr_num/gr_num + num_1_1;
 if num_1_1 >=numel(gr_num_1)
    num_1_1 =0;
 end
 
end



%%  记录每组别数据


num_3 = num_3_1 + 1;

%满跨荷载
n_t_1_1{num_3} = n_t_1;
t_t_1_1{num_3} = t_t_1;   %索件7，19，25
m_k_1_1{num_3} = m_k_1;
m_omega_1_1{num_3} = m_omega_1;
t_c1_1_1{num_3} = t_c1_1;   %索件7，19，25
t_c1_2_1{num_3} = t_c1_2;   %索件1~30
% t_c1_3_1{num_3} = t_c1_3;
d_out_1_1{num_3} = d_out_1;

%半跨荷载
n_t_2_2{num_3} = n_t_2;
t_t_2_2{num_3} = t_t_2;   %索件1~30
d_out_2_2{num_3} = d_out_2;
t_t_3_3{num_3} = t_t_3;   %索件7，19，25


 num_3_1 = num_3_1 + 1;
 if num_3_1 >=numel(i_1)
    num_3_1 =0;
 end
 
 %% 数据导出
% 外：杆件分组情况  内：行：预应力变化； 列：外力变化

% 满跨荷载（位移，索力，刚度特征值，频率特征值，无外荷载下的索力）
n_t_1_1; t_t_1_1; m_k_1_1; m_omega_1_1; t_c1_1_1; t_c1_2_1; d_out_1_1;    %t_c1_3_1;

% 半跨荷载（位移，索力）
n_t_2_2; t_t_2_2; t_t_3_3; d_out_2_2;
end


%% 绘图 满跨荷载索力变化-以组别
% pully1 = 0;
if pully1
    
i_6 = 1:5;  %取各组别
for num_6 = i_6
    %行：各组别； 列：预应力 
    i_7 = 1:4;  %取各预应力
    for num_7 = i_7
%     Y{num_6,num_7} = [t_c1_1_1{num_7}{num_6,1}([1:3],:); t_t_1_1{num_7}{num_6,1}([1:3],:); t_t_1_1{num_7}{num_6,2}([1:3],:); t_t_1_1{num_7}{num_6,3}([1:3],:); t_t_1_1{num_7}{num_6,4}([1:3],:); t_t_1_1{num_7}{num_6.5}([1:3],:)];
    Y{num_6,num_7} = [t_c1_1_1{num_7}{num_6,1}([1:3],:); t_t_1_1{num_7}{num_6,1}([1:3],:); t_t_1_1{num_7}{num_6,2}([1:3],:); t_t_1_1{num_7}{num_6,3}([1:3],:); t_t_1_1{num_7}{num_6,4}([1:3],:); t_t_1_1{num_7}{num_6,5}([1:3],:)];

    end
end

x = [0 20*i_2];

i_8 = 1:3;  %加载步，即开合度情况
for num_8 = i_8
%    figure 
   
   for num_8 = num_8
     Op = linspace(rate1,Rate_1,substep1);
     open1 = Op(:,num_8); 
   end
  
   i_9 = 1:3;  % 三个循环，下斜索、上斜索、环索
   for num_9 = i_9
    num_9_1 = [1 4 7 10 13 16] + num_9 - 1;   %初始无荷载下的下斜索索力，依次为下斜索、上斜索、环索
   figure 

    switch num_9
    case 1
        open2 = '下斜索';
    case 2
        open2 = '上斜索';
    case 3
        open2 = '环索';
    end
      
      i_10 = 1:4;   %预应力
      for num_10 =i_10
%       set(gcf, 'Position', [200, 100, 1536, 864]);   % 设置Figure窗口的位置和大小
      set(gcf, 'Position', [200, 100, 800,600]);   % 设置Figure窗口的位置和大小
%       subplot(2,2,num_10)

%       y1 =  Y{1,num_10}(num_9_1,num_8);  y2 =  Y{2,num_10}(num_9_1,num_8);  y3 =  Y{3,num_10}(num_9_1,num_8);  y4 =  Y{4,num_10}(num_9_1,num_8);
            y1 =  Y{1,num_10}(num_9_1,num_8);  y2 =  Y{2,num_10}(num_9_1,num_8);  y3 =  Y{3,num_10}(num_9_1,num_8);  
            y4 =  Y{4,num_10}(num_9_1,num_8);  y5 =  Y{5,num_10}(num_9_1,num_8);

      plot(x,y1,'-ok','MarkerIndices',1:1:length(y1));%创建一个线图并每隔一个数据点显示一个标记
      hold on;
      plot(x, y2,'-+r','MarkerIndices',1:1:length(y2));%创建一个线图并每隔一个数据点显示一个标记
      hold on;
      plot(x, y3,'-*b','MarkerIndices',1:1:length(y3));%创建一个线图并每隔一个数据点显示一个标记
      hold on;
      plot(x, y4,'-^g','MarkerIndices',1:1:length(y4));%创建一个线图并每隔一个数据点显示一个标记
      hold on;
      plot(x, y5,'-.k','MarkerIndices',1:1:length(y5));%创建一个线图并每隔一个数据点显示一个标记
      hold off;
      grid on;
      
%       legend('组1','组2','组3','组6','组12','Location','best','Fontname','宋体','Fontsize',18);   %4条数据
%       xlabel(['\fontname{宋体}外力\fontname{Times New Roman}(N)'],'Fontsize',18);  %位移值
%       ylabel(['\fontname{宋体}索力\fontname{Times New Roman}(N)'],'Fontsize',18);  %力
      
      legend('Cluster-1','Cluster-2','Cluster-3','Cluster-6','Cluster-12','Location','best','Fontname','Times New Roman','Fontsize',18);   %4条数据
      xlabel(['\fontname{Times New Roman}External force\fontname{Times New Roman}(N)'],'Fontsize',20);  %位移值
      ylabel(['\fontname{Times New Roman}Member force\fontname{Times New Roman}(N)'],'Fontsize',20);  %力
set(gca,'FontName','Times New Roman','FontSize',20);

      %       title(['满跨荷载','(','预应力',num2str(ka1*num_10 +kb1),'，',char(open2),'，开合度',num2str(open1),')'],...
%              'Fontname','宋体','Fontsize',18);
      
      str2 = string(['满跨索力','(','预应力',num2str(ka1*num_10 +kb1),'，',char(open2),'，开合度',num2str(open1),')']);  %创建字符串
     chr2 = char(str2);    %创建字符
%      saveas(gcf,[chr2,'.png']);
  if  puc_ture
    saveas(gcf,[chr2,'.png']);
    img = imread([chr2,'.png']);
    filename = fullfile(folderPath, [chr2,'.png']);
    imwrite(img, filename);
  end
       pause(num_tume);

      end


   end
         
    
end

end

%% 绘图 半跨荷载索力变化-以组别
% pully1 = 0;
if pully2
    
i_6 = 1:5;  %取各组别
for num_6 = i_6
    %行：各组别； 列：预应力 
    i_7 = 1:4;  %取各预应力
    for num_7 = i_7
    Y{num_6,num_7} = [t_c1_2_1{num_7}{num_6,1}(:,:); t_t_2_2{num_7}{num_6,1}(:,:); t_t_2_2{num_7}{num_6,2}(:,:); t_t_2_2{num_7}{num_6,3}(:,:); t_t_2_2{num_7}{num_6,4}(:,:); t_t_2_2{num_7}{num_6,5}(:,:)];

    end
end

x = [0 20*i_2];

  %斜索索力绘制
i_8 = 1:3;  %加载步，即开合度情况
for num_8 = i_8
%    figure 
   
   for num_8 = num_8
     Op = linspace(rate1,Rate_1,substep1);
     open1 = Op(:,num_8); 
   end
  
   i_9 = 1:30;  % 从 索件1 循环到 索件30
   for num_9 = i_9
    num_9_1 = [1 31 61 91 121 151] + num_9 - 1;   %初始无荷载下的下斜索索力，依次为下斜索、上斜索、环索
   figure 

      if num_9 == num_9       % 1~12为下斜索；13~24 为上斜索；25~30为环索
%           open2 = ['索件',num2str(num_9)];
%       else num_9 > 12 , num_9 <= 24 
          open2 = ['索件',num2str(num_9)];

      end

      i_10 = 1:4;     %预应力
      for num_10 =i_10
%       set(gcf, 'Position', [200, 100, 1536, 864]);   % 设置Figure窗口的位置和大小
      set(gcf, 'Position', [200, 100, 800,600]);   % 设置Figure窗口的位置和大小

%       subplot(2,2,num_10)

      y1 =  Y{1,num_10}(num_9_1,num_8);  y2 =  Y{2,num_10}(num_9_1,num_8);  y3 =  Y{3,num_10}(num_9_1,num_8);  
      y4 =  Y{4,num_10}(num_9_1,num_8);  y5 =  Y{5,num_10}(num_9_1,num_8);
      
      plot(x,y1,'-ok','MarkerIndices',1:1:length(y1));%创建一个线图并每隔一个数据点显示一个标记
      hold on;
      plot(x, y2,'-+r','MarkerIndices',1:1:length(y2));%创建一个线图并每隔一个数据点显示一个标记
      hold on;
      plot(x, y3,'-*b','MarkerIndices',1:1:length(y3));%创建一个线图并每隔一个数据点显示一个标记
      hold on;
      plot(x, y4,'-^g','MarkerIndices',1:1:length(y4));%创建一个线图并每隔一个数据点显示一个标记
      hold on;
      plot(x, y5,'-.k','MarkerIndices',1:1:length(y5));%创建一个线图并每隔一个数据点显示一个标记
      hold off;
      grid on;
      
%       legend('组1','组2','组3','组6','组12','Location','best','Fontname','宋体','Fontsize',18);   %4条数据
%       xlabel(['\fontname{宋体}外力\fontname{Times New Roman}(N)'],'Fontsize',18);  %位移值
%       ylabel(['\fontname{宋体}索力\fontname{Times New Roman}(N)'],'Fontsize',18);  %力
      
      legend('Cluster-1','Cluster-2','Cluster-3','Cluster-6','Cluster-12','Location','best','Fontname','Times New Roman','Fontsize',18);   %4条数据
      xlabel(['\fontname{Times New Roman}External force\fontname{Times New Roman}(N)'],'Fontsize',20);  %位移值
      ylabel(['\fontname{Times New Roman}Member force\fontname{Times New Roman}(N)'],'Fontsize',20);  %力
set(gca,'FontName','Times New Roman','FontSize',20);

%       title(['半跨荷载','(','预应力',num2str(ka1*num_10 +kb1),'，',char(open2),'，开合度',num2str(open1),')'],...
%              'Fontname','宋体','Fontsize',18);

      str2 = string(['半跨索力','(','预应力',num2str(ka1*num_10 +kb1),'，',char(open2),'，开合度',num2str(open1),')']);  %创建字符串
      chr2 = char(str2);    %创建字符
%       saveas(gcf,[chr2,'.png']);
  if  puc_ture
    saveas(gcf,[chr2,'.png']);
    img = imread([chr2,'.png']);
    filename = fullfile(folderPath, [chr2,'.png']);
    imwrite(img, filename);
  end
        pause(num_tume);
      end



   end
         
    
end

end

%% 绘图 满跨荷载位移变化
% pully1 = 1;
if pully3
    
i_6 = 1:5;  %取各组别
for num_6 = i_6
    %外：%行：各组别； 列：预应力  ; 内：行：不同外力；列：开合度
    i_7 = 1:4;  %取各预应力
    for num_7 = i_7

    Y{num_6,num_7} = 1000*[n_t_1_1{num_7}{num_6,1}(1,:); n_t_1_1{num_7}{num_6,2}(1,:); n_t_1_1{num_7}{num_6,3}(1,:); n_t_1_1{num_7}{num_6,4}(1,:); n_t_1_1{num_7}{num_6,5}(1,:)];
    end
end

x = [0 20*i_2];

i_8 = 1:3;  %加载步，即开合度情况
for num_8 = i_8
%    figure 
   
   for num_8 = num_8
     Op = linspace(rate1,Rate_1,substep1);
     open1 = Op(:,num_8); 
   end
  
   figure 

      i_9 = 1:4;
      for num_9 =i_9
%       set(gcf, 'Position', [200, 100, 1536, 864]);   % 设置Figure窗口的位置和大小
      set(gcf, 'Position', [200, 100, 800,600]);   % 设置Figure窗口的位置和大小
%       subplot(2,2,num_9)

      y1 =  [0; Y{1,num_9}(:,num_8)];  y2 =  [0; Y{2,num_9}(:,num_8)];  y3 =  [0; Y{3,num_9}(:,num_8)];  
      y4 =  [0; Y{4,num_9}(:,num_8)];  y5 =  [0; Y{5,num_9}(:,num_8)];  
      
      plot(x,y1,'-ok','MarkerIndices',1:1:length(y1));%创建一个线图并每隔一个数据点显示一个标记
      hold on;
      plot(x, y2,'-+r','MarkerIndices',1:1:length(y2));%创建一个线图并每隔一个数据点显示一个标记
      hold on;
      plot(x, y3,'-*b','MarkerIndices',1:1:length(y3));%创建一个线图并每隔一个数据点显示一个标记
      hold on;
      plot(x, y4,'-^g','MarkerIndices',1:1:length(y4));%创建一个线图并每隔一个数据点显示一个标记
      hold on;
      plot(x, y5,'-.k','MarkerIndices',1:1:length(y5));%创建一个线图并每隔一个数据点显示一个标记
      hold off;
      grid on;
      
%       legend('组1','组2','组3','组6','组12','Location','best','Fontname','宋体','Fontsize',18);   %4条数据
%       xlabel(['\fontname{宋体}外力\fontname{Times New Roman}(N)'],'Fontsize',18);  %位移值
%       ylabel(['\fontname{宋体}位移\fontname{Times New Roman}(mm)'],'Fontsize',18);  %力
      
      legend('Cluster-1','Cluster-2','Cluster-3','Cluster-6','Cluster-12','Location','best','Fontname','Times New Roman','Fontsize',18);   %4条数据
      xlabel(['\fontname{Times New Roman}External force\fontname{Times New Roman}(N)'],'Fontsize',20);  %位移值
      ylabel(['\fontname{Times New Roman}Z-coordinate\fontname{Times New Roman}(mm)'],'Fontsize',20);  %力
set(gca,'FontName','Times New Roman','FontSize',20);

%       title(['满跨荷载','(','预应力',num2str(ka1*num_9 +kb1),'，开合度',num2str(open1),')'],...
%              'Fontname','宋体','Fontsize',18);

      str2 = string(['满跨位移','(','预应力',num2str(ka1*num_9 +kb1),'，开合度',num2str(open1),')']);  %创建字符串
     chr2 = char(str2);    %创建字符
%      saveas(gcf,[chr2,'.png']);
  if  puc_ture
    saveas(gcf,[chr2,'.png']);
    img = imread([chr2,'.png']);
    filename = fullfile(folderPath, [chr2,'.png']);
    imwrite(img, filename);
  end
     
     pause(num_tume);
      end

end
 
end
%% 绘图 半跨荷载位移变化
   % 只研究 8，9，10，11节点的位移情况
%  pully1 = 0;
if pully4
    
i_6 = 1:5;  %取各组别
for num_6 = i_6
    %外：%行：各组别； 列：预应力 ; 内：行：不同外力；列：开合度
    i_7 = 1:4;  %取各预应力
    for num_7 = i_7

    Y{num_6,num_7} = 1000*[n_t_2_2{num_7}{num_6,1}([1:6],:); n_t_2_2{num_7}{num_6,2}([1:6],:); n_t_2_2{num_7}{num_6,3}([1:6],:); n_t_2_2{num_7}{num_6,4}([1:6],:); n_t_2_2{num_7}{num_6,5}([1:6],:)];
    end
end

x = [0 20*i_2];

i_8 = 1:3;  %加载步，即开合度情况
for num_8 = i_8
%    figure 
   
   for num_8 = num_8
     Op = linspace(rate1,Rate_1,substep1);
     open1 = Op(:,num_8); 
   end
  
   i_9 = 1:6;  % 从节点7 循环到 节点12
   for num_9 = i_9
    num_9_1 = [1 7 13 19 25] + num_9 - 1; 
   figure 

    if num_9 == num_9
       open2 = num2str(num_9+6);
    end
    
      i_10 = 1:4;   %预应力
      for num_10 =i_10
%       set(gcf, 'Position', [200, 100, 1536, 864]);   % 设置Figure窗口的位置和大小
      set(gcf, 'Position', [200, 100, 800,600]);   % 设置Figure窗口的位置和大小
%       subplot(2,2,num_10)


      y1 =  [0; Y{1,num_10}(num_9_1,num_8)];  y2 =  [0; Y{2,num_10}(num_9_1,num_8)];  y3 =  [0; Y{3,num_10}(num_9_1,num_8)];  
      y4 =  [0; Y{4,num_10}(num_9_1,num_8)];  y5 =  [0; Y{5,num_10}(num_9_1,num_8)];

      plot(x,y1,'-ok','MarkerIndices',1:1:length(y1));%创建一个线图并每隔一个数据点显示一个标记
      hold on;
      plot(x, y2,'-+r','MarkerIndices',1:1:length(y2));%创建一个线图并每隔一个数据点显示一个标记
      hold on;
      plot(x, y3,'-*b','MarkerIndices',1:1:length(y3));%创建一个线图并每隔一个数据点显示一个标记
      hold on;
      plot(x, y4,'-^g','MarkerIndices',1:1:length(y4));%创建一个线图并每隔一个数据点显示一个标记
      plot(x, y5,'-.k','MarkerIndices',1:1:length(y5));%创建一个线图并每隔一个数据点显示一个标记
      hold off;
      hold off;
      grid on;
      
%       legend('组1','组2','组3','组6','组12','Location','best','Fontname','宋体','Fontsize',18);   %4条数据
%       xlabel(['\fontname{宋体}外力\fontname{Times New Roman}(N)'],'Fontsize',18);  %位移值
%       ylabel(['\fontname{宋体}位移\fontname{Times New Roman}(mm)'],'Fontsize',18);  %力
      
      legend('Cluster-1','Cluster-2','Cluster-3','Cluster-6','Cluster-12','Location','best','Fontname','Times New Roman','Fontsize',18);   %4条数据
      xlabel(['\fontname{Times New Roman}External force\fontname{Times New Roman}(N)'],'Fontsize',20);  %位移值
      ylabel(['\fontname{Times New Roman}Z-coordinate\fontname{Times New Roman}(mm)'],'Fontsize',20);  %力    
set(gca,'FontName','Times New Roman','FontSize',20);

      %       title(['半跨荷载','(','预应力',num2str(ka1*num_10+kb1),'，节点',char(open2),'，开合度',num2str(open1),')'],...
%              'Fontname','宋体','Fontsize',18);

      str2 = string(['半跨位移','(','预应力',num2str(ka1*num_10+kb1),'，节点',char(open2),'，开合度',num2str(open1),')']);  %创建字符串
      chr2 = char(str2);    %创建字符
%       saveas(gcf,[chr2,'.png']);
  if  puc_ture
    saveas(gcf,[chr2,'.png']);
    img = imread([chr2,'.png']);
    filename = fullfile(folderPath, [chr2,'.png']);
    imwrite(img, filename);
  end

      pause(num_tume);
      end
   end
         
end

end

%% 以开合度作为指标 - 索力 - 半跨荷载
  %半跨荷载 
t_c1_1_1;t_t_3_3;

% pully1 = 0;
if pully5
    
i_6 = 1:5;  %取各组别
for num_6 = i_6
    %行：各组别； 列：预应力 
    i_7 = 1:4;  %取各预应力
    for num_7 = i_7
    Y{num_6,num_7} = [t_c1_1_1{num_7}{num_6,1}([1:3],:); t_t_3_3{num_7}{num_6,1}([1:3],:); t_t_3_3{num_7}{num_6,2}([1:3],:); t_t_3_3{num_7}{num_6,3}([1:3],:); t_t_3_3{num_7}{num_6,4}([1:3],:); t_t_3_3{num_7}{num_6,5}([1:3],:)];
%     Y{num_6,num_7} = [t_c1_3_1{num_7}{num_6,1}(:,:); t_t_3_3{num_7}{num_6,1}(:,:); t_t_3_3{num_7}{num_6,2}(:,:); t_t_3_3{num_7}{num_6,3}(:,:); t_t_3_3{num_7}{num_6,4}(:,:); t_t_3_3{num_7}{num_6,5}(:,:)];

    end
end

x = [0 20*i_2];

i_8 = 1:4;  %预应力循环
for num_8 = i_8
%    figure 
   
   for num_8 = num_8   %预应力值
     open1 = 100*i_1 +100; 
   end
  
    i_9 = 1:3;   %下斜索、上斜索、环索
    for num_9 =i_9
          figure 

          num_9_1 = [1 4 7 10 13 16] + num_9 - 1;   %绘制下斜索、上斜索、环索

    switch num_9
    case 1
        open2 = '下斜索';
    case 2
        open2 = '上斜索';
    case 3
        open2 = '环索';
    end
    
   i_10 = 1:5;  % 组别循环
   for num_10 = i_10
%       set(gcf, 'Position', [200, 100, 1536, 864]);   % 设置Figure窗口的位置和大小
      set(gcf, 'Position', [200, 100, 800,600]);   % 设置Figure窗口的位置和大小
%       subplot(2,3,num_10)
      
   for num_10 = num_10    %组别
     open1 = gr_num_1(:,num_10); 
   end
   

      y1 =  Y{num_10,num_8}(num_9_1,1);  y2 =  Y{num_10,num_8}(num_9_1,2);  y3 =  Y{num_10,num_8}(num_9_1,3);  

      plot(x,y1,'-ok','MarkerIndices',1:1:length(y1));%创建一个线图并每隔一个数据点显示一个标记
      hold on;
      plot(x, y2,'-+r','MarkerIndices',1:1:length(y2));%创建一个线图并每隔一个数据点显示一个标记
      hold on;
      plot(x, y3,'-*b','MarkerIndices',1:1:length(y3));%创建一个线图并每隔一个数据点显示一个标记
      hold off;
      grid on;
      
      legend('ratio-0.4','ratio-0.55','ratio-0.7','Location','best','Fontname','宋体','Fontsize',18);   %4条数据
%       xlabel(['\fontname{宋体}外力\fontname{Times New Roman}(N)'],'Fontsize',18);  %位移值
%       ylabel(['\fontname{宋体}索力\fontname{Times New Roman}(N)'],'Fontsize',18);  %力

      xlabel(['\fontname{Times New Roman}External force\fontname{Times New Roman}(N)'],'Fontsize',20);  %位移值
      ylabel(['\fontname{Times New Roman}Member force\fontname{Times New Roman}(N)'],'Fontsize',20);  %力
set(gca,'FontName','Times New Roman','FontSize',20);
  
%       title(['半跨荷载','(','预应力',num2str(ka1*num_8 +kb1),'，组',num2str(open1),'，',num2str(open2),')'],...
%              'Fontname','宋体','Fontsize',18);
         

      str2 = string(['半跨索力','(','预应力',num2str(ka1*num_8 +kb1),'，组',num2str(open1),'，',num2str(open2),')']);  %创建字符串
     chr2 = char(str2);    %创建字符
%      saveas(gcf,[chr2,'.png']);
  if  puc_ture
    saveas(gcf,[chr2,'.png']);
    img = imread([chr2,'.png']);
    filename = fullfile(folderPath, [chr2,'.png']);
    imwrite(img, filename);
  end

     pause(num_tume);
      end
    end   
    
end

end

%% 以开合度作为指标 - 位移 - 半跨荷载
  %半跨荷载
n_t_2_2;

% pully1 = 0;
if pully6
    
i_6 = 1:5;  %取各组别
for num_6 = i_6
    %行：各组别； 列：预应力 
    i_7 = 1:4;  %取各预应力
    for num_7 = i_7
    Y{num_6,num_7} = 1000*[n_t_2_2{num_7}{num_6,1}([1:6],:); n_t_2_2{num_7}{num_6,2}([1:6],:); n_t_2_2{num_7}{num_6,3}([1:6],:); n_t_2_2{num_7}{num_6,4}([1:6],:); n_t_2_2{num_7}{num_6,5}([1:6],:)];

    end
end

x = [0 20*i_2];

i_8 = 1:4;  %预应力循环
for num_8 = i_8
%    figure 
   
   for num_8 = num_8   %预应力值
     open1 = 100*i_1 +100; 
   end
  
    i_9 = 1:6;   %节点7~12
    for num_9 =i_9
%           figure 

          num_9_1 = [1 7 13 19 25] + num_9 - 1;   %绘制下斜索、上斜索、环索

    
   i_10 = 1:5;  % 组别循环
   for num_10 = i_10
       figure
%       set(gcf, 'Position', [200, 100, 1536, 864]);   % 设置Figure窗口的位置和大小
      set(gcf, 'Position', [200, 100, 800,600]);   % 设置Figure窗口的位置和大小%       subplot(2,3,num_10)
      
   for num_10 = num_10    %组别
     open1 = gr_num_1(:,num_10); 
   end
   
      y1 =  [0; Y{num_10,num_8}(num_9_1,1)];  y2 =  [0; Y{num_10,num_8}(num_9_1,2)];  y3 =  [0; Y{num_10,num_8}(num_9_1,3)];  

      plot(x,y1,'-ok','MarkerIndices',1:1:length(y1));%创建一个线图并每隔一个数据点显示一个标记
      hold on;
      plot(x, y2,'-+r','MarkerIndices',1:1:length(y2));%创建一个线图并每隔一个数据点显示一个标记
      hold on;
      plot(x, y3,'-*b','MarkerIndices',1:1:length(y3));%创建一个线图并每隔一个数据点显示一个标记
      hold off;
      grid on;
      
      legend('ratio-0.4','ratio-0.55','ratio-0.7','Location','best','Fontname','宋体','Fontsize',18);   %4条数据
%       xlabel(['\fontname{宋体}外力\fontname{Times New Roman}(N)'],'Fontsize',18);  %位移值
%       ylabel(['\fontname{宋体}位移\fontname{Times New Roman}(mm)'],'Fontsize',18);  %力

      xlabel(['\fontname{Times New Roman}External force\fontname{Times New Roman}(N)'],'Fontsize',20);  %位移值
      ylabel(['\fontname{Times New Roman}Z-coordinate\fontname{Times New Roman}(mm)'],'Fontsize',20);  %力
set(gca,'FontName','Times New Roman','FontSize',20);

%       title(['半跨荷载','(','预应力',num2str(ka1*num_8 +kb1),'，组',num2str(open1),'，节点',num2str(num_9 + 6),')'],...
%              'Fontname','宋体','Fontsize',18);

      str2 = string(['半跨位移','(','预应力',num2str(ka1*num_8 +kb1),'，组',num2str(open1),'，节点',num2str(num_9 + 6),')']);  %创建字符串
     chr2 = char(str2);    %创建字符
%      saveas(gcf,[chr2,'.png']);
  if  puc_ture
    saveas(gcf,[chr2,'.png']);
    img = imread([chr2,'.png']);
    filename = fullfile(folderPath, [chr2,'.png']);
    imwrite(img, filename);
  end

     pause(num_tume);
   end

    end   
    
end

end

%% 绘图 刚度特征值分析
% pully1 = 0;
if pully7
    
m_k_1_1; num_9_1_1 = 0;
i_6 = 1:5;  %取各组别

x=1:10;  %取前10阶

% [num_h1,num_l1] = size(m_k);  %得到刚度特征值的行数和列数
% x = 1:num_h1;

x=1:10;  %取前10阶
num_h1=10;
for num_6 = i_6
    %外： 列：各预应力 ; 内：行：不同组别；列：开合度

    Y1{num_6} = [m_k_1_1{1}{num_6,1}(:,:); m_k_1_1{2}{num_6,2}(:,:); m_k_1_1{3}{num_6,3}(:,:); m_k_1_1{4}{num_6,4}(:,:)];

end

% [num_h1,num_l1] = size(m_k);  %得到刚度特征值的行数和列数
% x = 1:num_h1;

i_8 = 1:3;  %加载步，即开合度情况
for num_8 = i_8
 
   num_9_1_1 = 0;
   
   for num_8 = num_8
     Op = linspace(rate1,Rate_1,substep1);
     open1 = Op(:,num_8); 
   end
  

      i_9 = 1:4;   %提取不同预应力进行对比
      for num_9 =i_9
     
      figure 
      num_9_1 = [1:num_h1] + num_9_1_1;
%       set(gcf, 'Position', [200, 100, 1536, 864]);   % 设置Figure窗口的位置和大小
      set(gcf, 'Position', [200, 100, 800,600]);   % 设置Figure窗口的位置和大小%       subplot(2,2,num_9)


      y1 =  Y1{1}(num_9_1,num_8);  y2 =  Y1{2}(num_9_1,num_8);  y3 =  Y1{3}(num_9_1,num_8);
      y4 =  Y1{4}(num_9_1,num_8);  y5 =  Y1{5}(num_9_1,num_8);
      
plot(x, y1,'-o','color',[0/255 0/255 0/255],'MarkerIndices',1:1:length(y1), 'LineWidth', Line_Width, 'MarkerSize', Marker_Size);  hold on;
plot(x, y2,'-d','color',[108/255 163/255 15/255],'MarkerIndices',1:1:length(y2), 'LineWidth', Line_Width, 'MarkerSize', Marker_Size);  hold on;
plot(x, y3,'-*','color',[245/255 147/255 17/255],'MarkerIndices',1:1:length(y3), 'LineWidth', Line_Width, 'MarkerSize', Marker_Size);  hold on;
plot(x, y4,'-^','color',[250/255 67/255 67/255],'MarkerIndices',1:1:length(y4), 'LineWidth', Line_Width, 'MarkerSize', Marker_Size);  hold on;
plot(x, y5,'-p','color',[22/255 175/255 204/255],'MarkerIndices',1:1:length(y5), 'LineWidth', Line_Width, 'MarkerSize', Marker_Size);  hold off;
  grid on;
set(gca, 'GridLineStyle', '-', 'GridColor', [0, 0, 0], 'GridAlpha', 0.5, 'LineWidth', 0.8);
      
%       legend('组1','组2','组3','组6','组12','Location','best','Fontname','宋体','Fontsize',18);   %4条数据
%       xlabel(['\fontname{宋体}阶数'],'Fontsize',18);  %位移值
%       ylabel(['\fontname{宋体}刚度\fontname{Times New Roman}(×10^3 N/m)'],'Fontsize',18);  %力
      
      legend('Cluster-1','Cluster-2','Cluster-3','Cluster-6','Cluster-12','Location','best','Fontname','Times New Roman','Fontsize',18);   %4条数据
      xlabel(['\fontname{Times New Roman}Order'],'Fontsize',20);  %位移值
      ylabel(['\fontname{Times New Roman}Eigenvalue of Stiffness\fontname{Times New Roman}(N/m)'],'Fontsize',20);  %力
set(gca,'FontName','Times New Roman','FontSize',20);

%       title(['刚度特征值','(','预应力',num2str(ka1*num_9+kb1),'，开合度',num2str(open1),')'],...
%              'Fontname','宋体','Fontsize',18);
         
         num_9_1_1 = num_9_1_1 + num_h1;
         
      str2 = string(['刚度特征值','(','预应力',num2str(ka1*num_9+kb1),'，开合度',num2str(open1),')']);  %创建字符串
     chr2 = char(str2);    %创建字符
%      saveas(gcf,[chr2,'.png']);
   if puc_ture_1
      filename = fullfile(folderPath, [chr2,'.png']); % 构建完整的文件路径
     print(filename, '-dpng', ['-r' num2str(dpi)]);
   end
   
  if  puc_ture
    saveas(gcf,[chr2,'.png']);
    img = imread([chr2,'.png']);
    filename = fullfile(folderPath, [chr2,'.png']);
    imwrite(img, filename);
  end
 

%       str2 = string(['刚度特征值','(','预应力',num2str(ka1*num_9+kb1),'，开合度',num2str(open1),')']);  %创建字符串
%      chr2 = char(str2);    %创建字符
% %      saveas(gcf,[chr2,'.png']);
%   if  puc_ture
%     saveas(gcf,[chr2,'.png']);
%     img = imread([chr2,'.png']);
%     filename = fullfile(folderPath, [chr2,'.png']);
%     imwrite(img, filename);
%   end

     pause(num_tume);
      end

end

end

%% 绘图 频率特征值分析
% pully1 = 0;
if pully8
    
m_omega_1_1; num_9_1_1 = 0;
m_k_1_1; num_9_1_1 = 0;
i_6 = 1:5;  %取各组别
for num_6 = i_6
    %外： 列：各预应力 ; 内：行：不同组别；列：开合度

    Y1{num_6} = [m_omega_1_1{1}{num_6,1}(:,:); m_omega_1_1{2}{num_6,2}(:,:); m_omega_1_1{3}{num_6,3}(:,:); m_omega_1_1{4}{num_6,4}(:,:)];

end

% [num_h1,num_l1] = size(m_omega);  %得到刚度特征值的行数和列数
% x = 1:num_h1;

i_8 = 1:3;  %加载步，即开合度情况
for num_8 = i_8
 
   num_9_1_1 = 0;
   
   for num_8 = num_8
     Op = linspace(rate1,Rate_1,substep1);
     open1 = Op(:,num_8); 
   end
  

      i_9 = 1:4;   %提取不同预应力进行对比
      for num_9 =i_9

      figure 
      num_9_1 = [1:num_h1] + num_9_1_1;
%       set(gcf, 'Position', [200, 100, 1536, 864]);   % 设置Figure窗口的位置和大小
      set(gcf, 'Position', [200, 100, 800,600]);   % 设置Figure窗口的位置和大小%       subplot(2,2,num_9)


      y1 =  Y1{1}(num_9_1,num_8);  y2 =  Y1{2}(num_9_1,num_8);  y3 =  Y1{3}(num_9_1,num_8);
      y4 =  Y1{4}(num_9_1,num_8);  y5 =  Y1{5}(num_9_1,num_8);
      
plot(x, y1,'-o','color',[0/255 0/255 0/255],'MarkerIndices',1:1:length(y1), 'LineWidth', Line_Width, 'MarkerSize', Marker_Size);  hold on;
plot(x, y2,'-d','color',[108/255 163/255 15/255],'MarkerIndices',1:1:length(y2), 'LineWidth', Line_Width, 'MarkerSize', Marker_Size);  hold on;
plot(x, y3,'-*','color',[245/255 147/255 17/255],'MarkerIndices',1:1:length(y3), 'LineWidth', Line_Width, 'MarkerSize', Marker_Size);  hold on;
plot(x, y4,'-^','color',[250/255 67/255 67/255],'MarkerIndices',1:1:length(y4), 'LineWidth', Line_Width, 'MarkerSize', Marker_Size);  hold on; 
plot(x, y5,'-p','color',[22/255 175/255 204/255],'MarkerIndices',1:1:length(y5), 'LineWidth', Line_Width, 'MarkerSize', Marker_Size);  hold off;
      grid on
set(gca, 'GridLineStyle', '-', 'GridColor', [0, 0, 0], 'GridAlpha', 0.5, 'LineWidth', 0.8);
%       legend('组1','组2','组3','组6','组12','Location','best','Fontname','宋体','Fontsize',18);   %4条数据
%       xlabel(['\fontname{宋体}阶数'],'Fontsize',18);  %位移值
%       ylabel(['\fontname{宋体}频率\fontname{Times New Roman}(Hz)'],'Fontsize',18);  %力
      
     legend('Cluster-1','Cluster-2','Cluster-3','Cluster-6','Cluster-12','Location','best','Fontname','Times New Roman','Fontsize',18);   %4条数据
      xlabel(['\fontname{Times New Roman}Order'],'Fontsize',20);  %位移值
      ylabel(['\fontname{Times New Roman}Frequency\fontname{Times New Roman}(Hz)'],'Fontsize',20);  %力
set(gca,'FontName','Times New Roman','FontSize',20);

%       title(['频率特征值','(','预应力',num2str(ka1*num_9+kb1),'，开合度',num2str(open1),')'],...
%              'Fontname','宋体','Fontsize',18);
         
         num_9_1_1 = num_9_1_1 + num_h1;
      str2 = string(['频率特征值','(','预应力',num2str(ka1*num_9+kb1),'，开合度',num2str(open1),')']);  %创建字符串
     chr2 = char(str2);    %创建字符
%      saveas(gcf,[chr2,'.png']);
   if puc_ture_1
      filename = fullfile(folderPath, [chr2,'.png']); % 构建完整的文件路径
     print(filename, '-dpng', ['-r' num2str(dpi)]);
   end
   
  if  puc_ture
    saveas(gcf,[chr2,'.png']);
    img = imread([chr2,'.png']);
    filename = fullfile(folderPath, [chr2,'.png']);
    imwrite(img, filename);
  end

        
%       str2 = string(['频率特征值','(','预应力',num2str(ka1*num_9+kb1),'，开合度',num2str(open1),')']);  %创建字符串
%      chr2 = char(str2);    %创建字符
% %      saveas(gcf,[chr2,'.png']);
%   if  puc_ture
%     saveas(gcf,[chr2,'.png']);
%     img = imread([chr2,'.png']);
%     filename = fullfile(folderPath, [chr2,'.png']);
%     imwrite(img, filename);
%   end

     pause(num_tume);
      end

end
end


%% 绘图 以开合度作为指标 - 刚度特征值分析,以无外力进行分析
% pully1 = 0;
if pully9
    
m_k_1_1; num_9_1_1 = 0;
i_6 = 1:4;  %取各预应力
for num_6 = i_6
    %外： 列：各预应力 ; 内：行：阶数；列：每3列为一个组别

    Y1{num_6} = [m_k_1_1{num_6}{1,1}(:,:), m_k_1_1{num_6}{2,1}(:,:), m_k_1_1{num_6}{3,1}(:,:), m_k_1_1{num_6}{4,1}(:,:), m_k_1_1{num_6}{5,1}(:,:)];

end

% [num_h1,num_l1] = size(m_k);  %得到刚度特征值的行数(阶数)和列数
% 
% x = 1:num_h1;

i_8 = 1:4;  %4个不同预应力
for num_8 = i_8
    fd_1 = y_1(:,num_8);  %此图预应力
    
   num_9_1_1 = 0;
   
 
%    figure 

      i_9 = 1:5;   %提取不同组别进行对比
      for num_9 =i_9
          figure 
 switch num_9
     case 1; open1 = '组1';
     case 2; open1 = '组2';  
     case 3; open1 = '组3'; 
     case 4; open1 = '组6';
     case 5; open1 = '组12';
 end    
          
%       set(gcf, 'Position', [200, 100, 1536, 864]);   % 设置Figure窗口的位置和大小
      set(gcf, 'Position', [200, 100, 800,600]);   % 设置Figure窗口的位置和大小%       subplot(3,2,num_9)

      y1 =  Y1{1}(:,1 +num_9_1_1);  y2 =  Y1{1}(:,2 +num_9_1_1);  y3 =  Y1{1}(:,3 +num_9_1_1);

%       y1 =  Y1{1}(num_9_1,num_8);  y2 =  Y1{2}(num_9_1,num_8);  y3 =  Y1{3}(num_9_1,num_8);
%       y4 =  Y1{4}(num_9_1,num_8);  y5 =  Y1{5}(num_9_1,num_8);
      
plot(x, y1,'-o','color',[0/255 0/255 0/255],'MarkerIndices',1:1:length(y1), 'LineWidth', Line_Width, 'MarkerSize', Marker_Size);  hold on;
plot(x, y2,'-d','color',[108/255 163/255 15/255],'MarkerIndices',1:1:length(y2), 'LineWidth', Line_Width, 'MarkerSize', Marker_Size);  hold on; 
plot(x, y3,'-*','color',[245/255 147/255 17/255],'MarkerIndices',1:1:length(y3), 'LineWidth', Line_Width, 'MarkerSize', Marker_Size);  hold off;
      grid on;
set(gca, 'GridLineStyle', '-', 'GridColor', [0, 0, 0], 'GridAlpha', 0.5, 'LineWidth', 0.8);

      legend('ratio-0.4','ratio-0.55','ratio-0.7','Location','best','Fontname','Times New Roman','Fontsize',18);   %4条数据
%       xlabel(['\fontname{宋体}阶数'],'Fontsize',18);  %位移值
%       ylabel(['\fontname{宋体}刚度\fontname{Times New Roman}(×10^3 N/m)'],'Fontsize',18);  %力
      
      xlabel(['\fontname{Times New Roman}Order'],'Fontsize',20);  %位移值
      ylabel(['\fontname{Times New Roman}Eigenvalue of Stiffness\fontname{Times New Roman}(N/m)'],'Fontsize',20);  %力
set(gca,'FontName','Times New Roman','FontSize',20);

%       title(['刚度特征值','(','预应力',num2str(fd_1),',',num2str(open1),')'],...
%              'Fontname','宋体','Fontsize',18);
         
         num_9_1_1 = num_9_1_1 + 3;
      str2 = string(['刚度特征值','(','预应力',num2str(fd_1),num2str(open1),')']);  %创建字符串
     chr2 = char(str2);    %创建字符
%      saveas(gcf,[chr2,'.png']);
   if puc_ture_1
      filename = fullfile(folderPath, [chr2,'.png']); % 构建完整的文件路径
     print(filename, '-dpng', ['-r' num2str(dpi)]);
   end
   
  if  puc_ture
    saveas(gcf,[chr2,'.png']);
    img = imread([chr2,'.png']);
    filename = fullfile(folderPath, [chr2,'.png']);
    imwrite(img, filename);
  end


%       str2 = string(['刚度特征值','(','预应力',num2str(fd_1),',',num2str(open1),')']);  %创建字符串
%      chr2 = char(str2);    %创建字符
% %      saveas(gcf,[chr2,'.png']);
%   if  puc_ture
%     saveas(gcf,[chr2,'.png']);
%     img = imread([chr2,'.png']);
%     filename = fullfile(folderPath, [chr2,'.png']);
%     imwrite(img, filename);
%   end

     pause(num_tume);
      end

end

end

%% 绘图 以预应力作为指标 - 刚度特征值分析,以无外力进行分析
% pully1 = 0;
if pully10
    
m_k_1_1; num_9_1_1 = 0;
i_6 = 1:4;  %取各预应力
for num_6 = i_6
    %外： 列：各预应力 ; 内：行：阶数；列：每3列为一个组别

   Y1{num_6} = [m_k_1_1{num_6}{1,1}(:,:), m_k_1_1{num_6}{2,1}(:,:), m_k_1_1{num_6}{3,1}(:,:), m_k_1_1{num_6}{4,1}(:,:), m_k_1_1{num_6}{5,1}(:,:)];

end

% [num_h1,num_l1] = size(m_k);  %得到刚度特征值的行数(阶数)和列数
% 
% x = 1:num_h1;
% 
   num_9_1_1 = 0;
i_8 = 1:5;  %5个组别
for num_8 = i_8
%     fd_1 = y1(:,i_8);  %此图预应力
 switch num_8
     case 1; open1 = '组1';
     case 2; open1 = '组2';  
     case 3; open1 = '组3'; 
     case 4; open1 = '组6';
     case 5; open1 = '组12';
 end    

%    num_9_1_1 = 0;
    
%    figure 

      i_9 = 1:3;   %3个开合度
      for num_9 =i_9
          figure 
   for num_9 = num_9
     Op = linspace(rate1,Rate_1,substep1);
     open_1 = Op(:,num_9); 
   end 
          
%           num_9_1 = [1:num_h1] + num_9_1_1;
%       set(gcf, 'Position', [200, 100, 1536, 864]);   % 设置Figure窗口的位置和大小
      set(gcf, 'Position', [200, 100, 800,600]);   % 设置Figure窗口的位置和大小%       subplot(2,2,num_9)

%       y1 =  Y1{1}(:,1 +num_9_1_1);  y2 =  Y1{1}(:,2 +num_9_1_1);  y3 =  Y1{1}(:,3 +num_9_1_1);

      y1 =  Y1{1}(:,1 +num_9_1_1);  y2 =  Y1{2}(:,1 +num_9_1_1); 
      y3 =  Y1{3}(:,1 +num_9_1_1);  y4 =  Y1{4}(:,1 +num_9_1_1);
      
plot(x, y1,'-o','color',[0/255 0/255 0/255],'MarkerIndices',1:1:length(y1), 'LineWidth', Line_Width, 'MarkerSize', Marker_Size);  hold on;
plot(x, y2,'-d','color',[108/255 163/255 15/255],'MarkerIndices',1:1:length(y2), 'LineWidth', Line_Width, 'MarkerSize', Marker_Size);  hold on;
plot(x, y3,'-*','color',[245/255 147/255 17/255],'MarkerIndices',1:1:length(y3), 'LineWidth', Line_Width, 'MarkerSize', Marker_Size);  hold on;
plot(x, y4,'-^','color',[250/255 67/255 67/255],'MarkerIndices',1:1:length(y4), 'LineWidth', Line_Width, 'MarkerSize', Marker_Size);  hold off;
% plot(x, y5,'-p','color',[22/255 175/255 204/255],'MarkerIndices',1:1:length(y5), 'LineWidth', Line_Width, 'MarkerSize', Marker_Size);  hold off;
      grid on;
set(gca, 'GridLineStyle', '-', 'GridColor', [0, 0, 0], 'GridAlpha', 0.5, 'LineWidth', 0.8);
      legend('fd-150N','fd-200N','fd-250N','fd-300N','Location','best','Fontname','Times New Roman','Fontsize',18);   %4条数据
%       xlabel(['\fontname{宋体}阶数'],'Fontsize',18);  %位移值
%       ylabel(['\fontname{宋体}刚度\fontname{Times New Roman}(×10^3 N)'],'Fontsize',18);  %力
      
      xlabel(['\fontname{Times New Roman}Order'],'Fontsize',20);  %位移值
      ylabel(['\fontname{Times New Roman}Eigenvalue of Stiffness\fontname{Times New Roman}(N/m)'],'Fontsize',20);  %力
set(gca,'FontName','Times New Roman','FontSize',20);

%       title(['刚度特征值','(',num2str(open1),',','开合度',num2str(open_1),')'],...
%              'Fontname','宋体','Fontsize',18);
         
         num_9_1_1 = num_9_1_1 + 1;
      str2 = string(['刚度特征值','(',num2str(open1),')']);  %创建字符串
     chr2 = char(str2);    %创建字符
%      saveas(gcf,[chr2,'.png']);
   if puc_ture_1
      filename = fullfile(folderPath, [chr2,'.png']); % 构建完整的文件路径
     print(filename, '-dpng', ['-r' num2str(dpi)]);
   end
   
  if  puc_ture
    saveas(gcf,[chr2,'.png']);
    img = imread([chr2,'.png']);
    filename = fullfile(folderPath, [chr2,'.png']);
    imwrite(img, filename);
  end
  

%       str2 = string(['刚度特征值','(',num2str(open1),',','开合度',num2str(open_1),')']);  %创建字符串
%      chr2 = char(str2);    %创建字符
% %      saveas(gcf,[chr2,'.png']);
%   if  puc_ture
%     saveas(gcf,[chr2,'.png']);
%     img = imread([chr2,'.png']);
%     filename = fullfile(folderPath, [chr2,'.png']);
%     imwrite(img, filename);
%   end

     pause(num_tume);
      end

end

end

%% 绘图 以开合度作为指标 - 频率特征值分析,以无外力进行分析
% pully1 = 0;
if pully11
    
m_omega_1_1; num_9_1_1 = 0;
i_6 = 1:4;  %取各预应力
for num_6 = i_6
    %外： 列：各预应力 ; 内：行：阶数；列：每3列为一个组别

    Y1{num_6} = [m_omega_1_1{num_6}{1,1}(:,:), m_omega_1_1{num_6}{2,1}(:,:), m_omega_1_1{num_6}{3,1}(:,:), m_omega_1_1{num_6}{4,1}(:,:), m_omega_1_1{num_6}{5,1}(:,:)];

end

% [num_h1,num_l1] = size(m_omega);  %得到刚度特征值的行数(阶数)和列数
% 
% x = 1:num_h1;

i_8 = 1:4;  %4个不同预应力
for num_8 = i_8
    fd_1 = y_1(:,num_8);  %此图预应力
    
   num_9_1_1 = 0;
   
%    for num_8 = num_8
%      Op = linspace(rate1,Rate_1,substep1);
%      open1 = Op(:,num_8); 
%    end
  
%    figure 

      i_9 = 1:5;   %提取不同组别进行对比
      for num_9 =i_9
          figure 
 switch num_9
     case 1; open1 = '组1';
     case 2; open1 = '组2';  
     case 3; open1 = '组3'; 
     case 4; open1 = '组6';
     case 5; open1 = '组12';
 end    
          
%           num_9_1 = [1:num_h1] + num_9_1_1;
%       set(gcf, 'Position', [200, 100, 1536, 864]);   % 设置Figure窗口的位置和大小
      set(gcf, 'Position', [200, 100, 800,600]);   % 设置Figure窗口的位置和大小%       subplot(3,2,num_9)

      y1 =  Y1{1}(:,1 +num_9_1_1);  y2 =  Y1{1}(:,2 +num_9_1_1);  y3 =  Y1{1}(:,3 +num_9_1_1);

%       y1 =  Y1{1}(num_9_1,num_8);  y2 =  Y1{2}(num_9_1,num_8);  y3 =  Y1{3}(num_9_1,num_8);
%       y4 =  Y1{4}(num_9_1,num_8);  y5 =  Y1{5}(num_9_1,num_8);
      
plot(x, y1,'-o','color',[0/255 0/255 0/255],'MarkerIndices',1:1:length(y1), 'LineWidth', Line_Width, 'MarkerSize', Marker_Size);  hold on;
plot(x, y2,'-d','color',[108/255 163/255 15/255],'MarkerIndices',1:1:length(y2), 'LineWidth', Line_Width, 'MarkerSize', Marker_Size);  hold on;
plot(x, y3,'-*','color',[245/255 147/255 17/255],'MarkerIndices',1:1:length(y3), 'LineWidth', Line_Width, 'MarkerSize', Marker_Size);  hold off;
% plot(x, y4,'-^','color',[250/255 67/255 67/255],'MarkerIndices',1:1:length(y4), 'LineWidth', Line_Width, 'MarkerSize', Marker_Size);  hold on;
% plot(x, y5,'-p','color',[22/255 175/255 204/255],'MarkerIndices',1:1:length(y5), 'LineWidth', Line_Width, 'MarkerSize', Marker_Size);  hold off;
      grid on;
set(gca, 'GridLineStyle', '-', 'GridColor', [0, 0, 0], 'GridAlpha', 0.5, 'LineWidth', 0.8);
      legend('ratio-0.4','ratio-0.55','ratio-0.7','Location','best','Fontname','Times New Roman','Fontsize',18);   %4条数据
%       xlabel(['\fontname{宋体}阶数'],'Fontsize',18);  %位移值
%       ylabel(['\fontname{宋体}频率\fontname{Times New Roman}(Hz)'],'Fontsize',18);  %力
      
      xlabel(['\fontname{Times New Roman}Order'],'Fontsize',20);  %位移值
      ylabel(['\fontname{Times New Roman}Frequency\fontname{Times New Roman}(Hz)'],'Fontsize',20);  %力
set(gca,'FontName','Times New Roman','FontSize',20);

%       title(['频率特征值','(','预应力',num2str(fd_1),',',num2str(open1),')'],...
%              'Fontname','宋体','Fontsize',18);

         num_9_1_1 = num_9_1_1 + 3;
      str2 = string(['频率特征值','(','预应力',num2str(fd_1),',',num2str(open1),')']);  %创建字符串
     chr2 = char(str2);    %创建字符
%      saveas(gcf,[chr2,'.png']);
   if puc_ture_1
      filename = fullfile(folderPath, [chr2,'.png']); % 构建完整的文件路径
     print(filename, '-dpng', ['-r' num2str(dpi)]);
   end
   
  if  puc_ture
    saveas(gcf,[chr2,'.png']);
    img = imread([chr2,'.png']);
    filename = fullfile(folderPath, [chr2,'.png']);
    imwrite(img, filename);
  end 

%       str2 = string(['频率特征值','(','预应力',num2str(fd_1),',',num2str(open1),')']);  %创建字符串
%      chr2 = char(str2);    %创建字符
% %      saveas(gcf,[chr2,'.png']);
%   if  puc_ture
%     saveas(gcf,[chr2,'.png']);
%     img = imread([chr2,'.png']);
%     filename = fullfile(folderPath, [chr2,'.png']);
%     imwrite(img, filename);
%   end

     pause(num_tume);
      end

end

end

%% 绘图 以预应力作为指标 - 频率特征值分析,以无外力进行分析
% pully1 = 0;
if pully12
    
m_omega_1_1; num_9_1_1 = 0;
i_6 = 1:4;  %取各预应力
for num_6 = i_6
    %外： 列：各预应力 ; 内：行：阶数；列：每3列为一个组别

    Y1{num_6} = [m_omega_1_1{num_6}{1,1}(:,:), m_omega_1_1{num_6}{2,1}(:,:), m_omega_1_1{num_6}{3,1}(:,:), m_omega_1_1{num_6}{4,1}(:,:), m_omega_1_1{num_6}{5,1}(:,:)];

end

% [num_h1,num_l1] = size(m_omega);  %得到刚度特征值的行数(阶数)和列数
% 
% x = 1:num_h1;

   num_9_1_1 = 0;
i_8 = 1:5;  %5个组别
for num_8 = i_8
%     fd_1 = y1(:,i_8);  %此图预应力
 switch num_8
     case 1; open1 = '组1';
     case 2; open1 = '组2';  
     case 3; open1 = '组3'; 
     case 4; open1 = '组6';
     case 5; open1 = '组12';
 end    

%    num_9_1_1 = 0;
   
%    for num_8 = num_8
%      Op = linspace(rate1,Rate_1,substep1);
%      open1 = Op(:,num_8); 
%    end
  
   figure 

      i_9 = 1:3;   %3个开合度
      for num_9 =i_9
          figure 
   for num_9 = num_9
     Op = linspace(rate1,Rate_1,substep1);
     open_1 = Op(:,num_9); 
   end 
          
%           num_9_1 = [1:num_h1] + num_9_1_1;
%       set(gcf, 'Position', [200, 100, 1536, 864]);   % 设置Figure窗口的位置和大小
      set(gcf, 'Position', [200, 100, 800,600]);   % 设置Figure窗口的位置和大小
%       subplot(2,2,num_9)

%       y1 =  Y1{1}(:,1 +num_9_1_1);  y2 =  Y1{1}(:,2 +num_9_1_1);  y3 =  Y1{1}(:,3 +num_9_1_1);

      y1 =  Y1{1}(:,1 +num_9_1_1);  y2 =  Y1{2}(:,1 +num_9_1_1); 
      y3 =  Y1{3}(:,1 +num_9_1_1);  y4 =  Y1{4}(:,1 +num_9_1_1);
      
plot(x, y1,'-o','color',[0/255 0/255 0/255],'MarkerIndices',1:1:length(y1), 'LineWidth', Line_Width, 'MarkerSize', Marker_Size);  hold on;
plot(x, y2,'-d','color',[108/255 163/255 15/255],'MarkerIndices',1:1:length(y2), 'LineWidth', Line_Width, 'MarkerSize', Marker_Size);  hold on;
plot(x, y3,'-*','color',[245/255 147/255 17/255],'MarkerIndices',1:1:length(y3), 'LineWidth', Line_Width, 'MarkerSize', Marker_Size);  hold on;
plot(x, y4,'-^','color',[250/255 67/255 67/255],'MarkerIndices',1:1:length(y4), 'LineWidth', Line_Width, 'MarkerSize', Marker_Size);  hold off;
% plot(x, y5,'-p','color',[22/255 175/255 204/255],'MarkerIndices',1:1:length(y5), 'LineWidth', Line_Width, 'MarkerSize', Marker_Size);  hold off;
      grid on;
set(gca, 'GridLineStyle', '-', 'GridColor', [0, 0, 0], 'GridAlpha', 0.5, 'LineWidth', 0.8);
      legend('fd-150N','fd-200N','fd-250N','fd-300N','Location','best','Fontname','Times New Roman','Fontsize',18);   %4条数据
%       xlabel(['\fontname{宋体}阶数'],'Fontsize',18);  %位移值
%       ylabel(['\fontname{宋体}频率\fontname{Times New Roman}(Hz)'],'Fontsize',18);  %力
      
      xlabel(['\fontname{Times New Roman}Order'],'Fontsize',20);  %位移值
      ylabel(['\fontname{Times New Roman}Frequency\fontname{Times New Roman}(Hz)'],'Fontsize',20);  %力
set(gca,'FontName','Times New Roman','FontSize',20);

%       title(['频率特征值','(',num2str(open1),',','开合度',num2str(open_1),')'],...
%              'Fontname','宋体','Fontsize',18);
         
         num_9_1_1 = num_9_1_1 + 1;
      str2 = string(['频率特征值','(',num2str(open1),')']);  %创建字符串
     chr2 = char(str2);    %创建字符
%      saveas(gcf,[chr2,'.png']);
   if puc_ture_1
      filename = fullfile(folderPath, [chr2,'.png']); % 构建完整的文件路径
     print(filename, '-dpng', ['-r' num2str(dpi)]);
   end
   
  if  puc_ture
    saveas(gcf,[chr2,'.png']);
    img = imread([chr2,'.png']);
    filename = fullfile(folderPath, [chr2,'.png']);
    imwrite(img, filename);
  end

%       str2 = string(['频率特征值','(',num2str(open1),',','开合度',num2str(open_1),')']);  %创建字符串
%       chr2 = char(str2);    %创建字符
% %      saveas(gcf,[chr2,'.png']);
%   if  puc_ture
%     saveas(gcf,[chr2,'.png']);
%     img = imread([chr2,'.png']);
%     filename = fullfile(folderPath, [chr2,'.png']);
%     imwrite(img, filename);
%   end

     pause(num_tume);
      end

end

end