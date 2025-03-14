%%%%%%%%%%%%%%%%%%%椭圆形马鞍面索桁结构%%%%%%%%%%%%%%%%%%%
%%%%%Elliptical Saddle-Surface Cable-Grid Structure%%%%%%
%%%%%%%%%%本算例仅考虑均匀荷载%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc
clear all
warning off

%% material and force
%外侧为F1，最内侧为F5
load_case=1;    %1为正常使用极限状态；2为承载能力极限状态 

F1_dead=10.375*0.3;
F1_live=10.375*0.5;
F2_dead=13.825*0.3;
F2_live=13.825*0.5;
F3_dead=11.5*0.3;
F3_live=11.5*0.5;
F4_dead=7.425*0.3;
F4_live=7.425*0.5;
F5_dead=2.25*0.3;
F5_live=2.25*0.5;

if load_case==1
    F1=F1_dead+F1_live;
    F2=F2_dead+F2_live;
    F3=F3_dead+F3_live;
    F4=F4_dead+F4_live;
    F5=F3_dead+F5_live;
end

if load_case==2
    F1=1.3*F1_dead+1.5*F1_live;
    F2=1.3*F2_dead+1.5*F2_live;
    F3=1.3*F3_dead+1.5*F3_live;
    F4=1.3*F4_dead+1.5*F4_live;
    F5=1.3*F3_dead+1.5*F5_live;
end
    
Es=1.10e11;      %Elastic modulus of struct
Eb=2.06e11;      %Elastic modulus of bar
sigmas=1670e6;                      %同理 设定0.5的sigmas为容许拉应力，335MPa为容许压应力。  实际屈服为1650与345MPa
sigmab=-345e6;
sigmab_Des=-310e6;
Eps=sigmas/Es;   %allowed strain of strings
Epb=sigmab/Eb;   %allowed strain of strings

%% geometry 
% 椭圆的长轴和短轴
%最外圈
% La_ex=(110.5/2+43.984)/3;       %长半轴
% Lb_ex=(97/2+34.033)/3;      %短半轴
% 
% %最内圈
% La_in=(110.5/2)/3;      %长半轴
% Lb_in=(97/2)/3;      %短半轴


La_ex=(110.5/2+43.984);       %长半轴
Lb_ex=(97/2+34.033);      %短半轴

%最内圈
La_in=(110.5/2);      %长半轴
Lb_in=(97/2);      %短半轴




% 等分数
n = 40;

% 初始化x和y坐标
%最外圈
N1_x = zeros(1, n);
N1_y = zeros(1, n);

% 使用循环直接计算每个点的角度和坐标
for i = 1:n
     % 计算对应的x和y坐标
    N1_x(i) = Lb_ex * cos((pi/2) - (i - 1) * 9 * pi / 180);  % x坐标
    N1_y(i) = La_ex * sin((pi/2) - (i - 1) * 9 * pi / 180);  % y坐标
end

% 修正x值为0时的数值误差，使用round函数精确到小数点后10位
N1_x = round(N1_x, 10);  % 只保留10位小数，避免浮动
N1_y = round(N1_y, 10);  % 只保留10位小数，避免浮动

%最内圈
N6_x = zeros(1, n);
N6_y = zeros(1, n);

% 使用循环直接计算每个点的角度和坐标
for i = 1:n
     % 计算对应的x和y坐标
    N6_x(i) = Lb_in * cos((pi/2) - (i - 1) * 9 * pi / 180);  % x坐标
    N6_y(i) = La_in * sin((pi/2) - (i - 1) * 9 * pi / 180);  % y坐标
end

% 修正x值为0时的数值误差，使用round函数精确到小数点后10位
N6_x = round(N6_x, 10);  % 只保留10位小数，避免浮动
N6_y = round(N6_y, 10);  % 只保留10位小数，避免浮动

%2-5圈是按照1:2:2:2:1进行分段的
ratio2=7/8;
ratio3=5/8;
ratio4=3/8;
ratio5=1/8;
delta_Nx=N1_x-N6_x;
delta_Ny=N1_y-N6_y;

N2_x=N6_x+ratio2*delta_Nx;
N2_y=N6_y+ratio2*delta_Ny;

N3_x=N6_x+ratio3*delta_Nx;
N3_y=N6_y+ratio3*delta_Ny;

N4_x=N6_x+ratio4*delta_Nx;
N4_y=N6_y+ratio4*delta_Ny;

N5_x=N6_x+ratio5*delta_Nx;
N5_y=N6_y+ratio5*delta_Ny;

% N1_z_up_part=[15.000 	15.128 	15.594 	16.260 	17.003 	17.622 	18.178 	18.592 	18.842 	18.966 	19.009 18.966 18.842 18.592 18.178 17.622 17.003 16.260 15.594 15.128];
% N1_z_up=[N1_z_up_part N1_z_up_part];
% 
% N1_z_down_part=[14.000 	14.128 	14.594 	15.260 	16.003 	16.622 	17.178 	17.592 	17.842 	17.966 	18.009 17.966 	17.842 	17.592 	17.178 	16.622 	16.003 	15.260 	14.594 	14.128];
% N1_z_down=[N1_z_down_part N1_z_down_part];
% 
% N6_z_part=[15.859 	15.891 	15.982 	16.117 	16.276 	16.439 	16.590 	16.717 	16.812 	16.871 	16.891 16.871	16.812	16.717	16.59	16.439	16.276	16.117	15.982	15.891];
% N6_z=[N6_z_part N6_z_part];

N1_z_up_part=[42.529	42.912	44.31	46.309	48.537	50.394	52.064	53.305	54.054	54.426	54.555	54.426	54.054	53.305	52.064	50.394	48.537	46.309	44.31	42.912];
N1_z_up=[N1_z_up_part N1_z_up_part];

N1_z_down_part=[39.529	39.912	41.31	43.309	45.537	47.394	49.064	50.305	51.054	51.426	51.555	51.426	51.054	50.305	49.064	47.394	45.537	43.309	41.31	39.912];
N1_z_down=[N1_z_down_part N1_z_down_part];

N6_z_part=[45.107	45.203	45.476	45.881	46.358	46.846	47.299	47.68	47.966	48.142	48.203	48.142	47.966	47.68	47.299	46.846	46.358	45.881	45.476	45.203];
N6_z=[N6_z_part N6_z_part];

delta_Nz_up=N1_z_up-N6_z;
delta_Nz_down=N1_z_down-N6_z;

N2_z_up=N6_z+ratio2*delta_Nz_up;
N2_z_down=N6_z+ratio2*delta_Nz_down;

N3_z_up=N6_z+ratio3*delta_Nz_up;
N3_z_down=N6_z+ratio3*delta_Nz_down;

N4_z_up=N6_z+ratio4*delta_Nz_up;
N4_z_down=N6_z+ratio4*delta_Nz_down;

N5_z_up=N6_z+ratio5*delta_Nz_up;
N5_z_down=N6_z+ratio5*delta_Nz_down;

%矩阵组合
%1-40号为最外围约束上节点，41-80为下节点
%81-120为环索节点
%121-160, 161-200, 201-240, 241-280为上部径向索由外向内的节点
%281-320, 321-360, 361-400, 401-440为下部径向索由外向内的节点

Nx=[N1_x N1_x N6_x N2_x N3_x N4_x N5_x N2_x N3_x N4_x N5_x];
Ny=[N1_y N1_y N6_y N2_y N3_y N4_y N5_y N2_y N3_y N4_y N5_y];
Nz=[N1_z_up N1_z_down N6_z N2_z_up N3_z_up N4_z_up N5_z_up N2_z_down N3_z_down N4_z_down N5_z_down];

N=[Nx;Ny;Nz];

%% connectivity matrix of a unit
C_b_in_unit = [[121:280]', [281:440]'];  % Write down the connection of bars
C_b = tenseg_ind2C(C_b_in_unit,N); %get the matrix of relation through C_b_in
C_s_in_unit = [[81:120, 1:40, 121:280, 41:80, 281:440]', ...
               [82:120, 81, 121:280, 81:120, 281:440, 81:120]'];
C_s = tenseg_ind2C(C_s_in_unit,N);
tenseg_plot(N,C_b,C_s);
C = [C_b;C_s];
ne=size(C,1);%number of element
ne_c=size(C_b,1);%number of element
ne_s=size(C_s,1);%number of element

%通过节点坐标对结构构件的长度进行计算
%之前是通过手搓计算，这里通过NC方程进行计算
l_bar1=diag(sqrt(sum(N*C').^2)); 
l_vec1=diag(l_bar1);
% 此顺序为matlab中索杆的标号
%其中B1 1-40 B2 41-80  B3 81-120  B4 121-160 HS 161-200
%JS1 201-240 JS2 241-280 JS3 281-320 JS4 321-360 JS5 361-400
%XS1 401-440 XS2 441-480 XS3 481-520 XS4 521-560 XS5 561-600
c_rep_B1=[1:11];
c_rep_B2=[41:51];
c_rep_B3=[81:91];
c_rep_B4=[121:131];

c_rep_HS=[160+(1:10)];

c_rep_JS1=[160+(41:51)];
c_rep_JS2=[160+(81:91)];
c_rep_JS3=[160+(121:131)];
c_rep_JS4=[160+(161:171)];
c_rep_JS5=[160+(201:211)];

c_rep_XS1=[160+(241:251)];
c_rep_XS2=[160+(281:291)];
c_rep_XS3=[160+(321:331)];
c_rep_XS4=[160+(361:371)];
c_rep_XS5=[160+(401:411)];

c_rep_total=[c_rep_B1 c_rep_B2 c_rep_B3 c_rep_B4 c_rep_HS c_rep_JS1...
    c_rep_JS2 c_rep_JS3 c_rep_JS4 c_rep_JS5 c_rep_XS1 c_rep_XS2 ...
    c_rep_XS3 c_rep_XS4 c_rep_XS5]; 
%构件长度
B1=l_vec1(c_rep_B1);
B2=l_vec1(c_rep_B2);
B3=l_vec1(c_rep_B3);
B4=l_vec1(c_rep_B4);

HS=l_vec1(c_rep_HS);

JS1=l_vec1(c_rep_JS1);
JS2=l_vec1(c_rep_JS2);
JS3=l_vec1(c_rep_JS3);
JS4=l_vec1(c_rep_JS4);
JS5=l_vec1(c_rep_JS5);

XS1=l_vec1(c_rep_XS1);
XS2=l_vec1(c_rep_XS2);
XS3=l_vec1(c_rep_XS3);
XS4=l_vec1(c_rep_XS4);
XS5=l_vec1(c_rep_XS5);
%计算投影距离 作图用
N_xy=[Nx;Ny];
%1-40,41-81为最外围的节点投影.任取一组都可以
N1_xy=N_xy(:,41:80);
N6_xy=N_xy(:,81:120);
distances = sqrt((N1_xy(1, :) - N6_xy(1, :)).^2 + (N1_xy(2, :) - N6_xy(2, :)).^2);
distances=round(distances, 2); %保留小数点后2位
%%  design SVD, constrain,length and cross-area
%constrain
n=size(N,2);              %number of node
I=eye(3*n);
Ine=eye(ne,ne);           %element number index
b=[1:240];               %pinned Node 1-80
Ib=I(:,b);                %pinned nod index
a=setdiff(1:3*n,b);       %index of free node direction
Ia=I(:,a);      

%SVD
A=kron(C',eye(3))*diag(kron(C,eye(3))*N(:))*...
    kron(eye(ne),ones(3,1));  %equilibrium matrix
A_hat=Ia'*A;

%分组
% 主程序部分
Num = 40;  % 设定总数范围
increment_B1 = 0;   %1-40
gr_B1 = group_numbers_1(Num,increment_B1);  % 调用函数进行分组
increment_B2 = 40;   %41-80
gr_B2 = group_numbers_1(Num,increment_B2);  % 调用函数进行分组
increment_B3 = 80;   %81-120
gr_B3 = group_numbers_1(Num,increment_B3);  % 调用函数进行分组
increment_B4 = 120;   %121-160
gr_B4 = group_numbers_1(Num,increment_B4);  % 调用函数进行分组

increment_HS = 160;   %161-200
gr_HS = group_numbers_2(Num,increment_HS);  % 调用函数进行分组 group_numbers_2

increment_JS1 = 200;   %201-240
gr_JS1 = group_numbers_1(Num,increment_JS1);  % 调用函数进行分组
increment_JS2 = 240;   %241-280
gr_JS2 = group_numbers_1(Num,increment_JS2);  % 调用函数进行分组
increment_JS3 = 280;   %281-320
gr_JS3 = group_numbers_1(Num,increment_JS3);  % 调用函数进行分组
increment_JS4 = 320;   %321-360
gr_JS4 = group_numbers_1(Num,increment_JS4);  % 调用函数进行分组
increment_JS5 = 360;   %361-400
gr_JS5 = group_numbers_1(Num,increment_JS5);  % 调用函数进行分组

increment_XS1 = 400;   %401-440
gr_XS1 = group_numbers_1(Num,increment_XS1);  % 调用函数进行分组
increment_XS2 = 440;   %441-480
gr_XS2 = group_numbers_1(Num,increment_XS2);  % 调用函数进行分组
increment_XS3 = 480;   %481-520
gr_XS3 = group_numbers_1(Num,increment_XS3);  % 调用函数进行分组
increment_XS4 = 520;   %521-560
gr_XS4 = group_numbers_1(Num,increment_XS4);  % 调用函数进行分组
increment_XS5 = 560;   %561-600
gr_XS5 = group_numbers_1(Num,increment_XS5);  % 调用函数进行分组

gr=[gr_B1; gr_B2; gr_B3; gr_B4;gr_HS;gr_JS1;gr_JS2;gr_JS3;gr_JS4;gr_JS5;...
    gr_XS1;gr_XS2;gr_XS3;gr_XS4;gr_XS5];
Gp=tenseg_str_gp(gr,C);    %generate group matrix 

[U,S,V] = svd(A_hat);
r=rank(A_hat); %rank of (A_hat*Gp)
U1=U(:,1:r);U2=U(:,r+1:end);
S1=S(1:r,1:r);
V1=V(:,1:r);V2=V(:,r+1:end);
V2=-V2;

%% plot structure with member force

tenseg_plot_CTS(N,C,1:size(C_b_in_unit,1),Gp',[],[],[],[],[],V2);

%% construction analysis
