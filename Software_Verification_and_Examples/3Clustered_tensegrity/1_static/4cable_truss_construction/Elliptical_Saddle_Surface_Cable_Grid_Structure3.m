%%%%%%%%%%%%%%%%%%%椭圆形马鞍面索桁结构%%%%%%%%%%%%%%%%%%%
%%%%%Elliptical Saddle-Surface Cable-Grid Structure%%%%%%
%%%%%%%%%%本算例仅考虑均匀荷载%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc
clear all
warning off
global l m R p Eps f_bar E_bar A_bar l_bar l0_bar  X0 Ia Ib I Ine C Area w ne an bn cn n Ie Gp c_rep_total
%% material and force
%外侧为F1，最内侧为F5
%在27*33的基础上继续缩放
%钢丝绳面积触碰不到
sc=1/3;  %缩放系数
load_case=2;    %1为正常使用极限状态；2为承载能力极限状态 

F1_dead=10.375*1000*0.3*sc*sc;
F1_live=10.375*1000*0.5*sc*sc;
F2_dead=13.825*1000*0.3*sc*sc;
F2_live=13.825*1000*0.5*sc*sc;
F3_dead=11.5*1000*0.3*sc*sc;
F3_live=11.5*1000*0.5*sc*sc;
F4_dead=7.425*1000*0.3*sc*sc;
F4_live=7.425*1000*0.5*sc*sc;
F5_dead=2.25*1000*0.3*sc*sc;
F5_live=2.25*1000*0.5*sc*sc;

if load_case==1
    F1=F1_dead+F1_live;
    F2=F2_dead+F2_live;
    F3=F3_dead+F3_live;
    F4=F4_dead+F4_live;
    F5=F5_dead+F5_live;
end

if load_case==2
    F1=1.3*F1_dead+1.5*F1_live;
    F2=1.3*F2_dead+1.5*F2_live;
    F3=1.3*F3_dead+1.5*F3_live;
    F4=1.3*F4_dead+1.5*F4_live;
    F5=1.3*F5_dead+1.5*F5_live;
end
    
Es=1.10e11;      %Elastic modulus of struct
Eb=2.06e11;      %Elastic modulus of bar
sigmas=1670e6;                      %同理 设定0.5的sigmas为容许拉应力，335MPa为容许压应力。  实际屈服为1650与345MPa
sigmab=-345e6;
sigmab_Des=-310e6;
Eps=sigmas/Es;   %allowed strain of strings
Epb=sigmab/Eb;   %allowed strain of strings

%% geometry 
filename='\结构信息表.xlsx';
sheet1=3;
range1='B2:E761';
num_N=xlsread(filename,sheet1,range1);
range2='F2:L881';
num_force=xlsread(filename,sheet1,range2);
num_force=num_force(:,2);   

N_ini=[num_N(321:760,2:4);]';
%%%%%%%%%%%%%%%%%%%%%%%%%%%此顺序为matlab中索杆的标号%%%%%%%%%%%%%%
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
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

N_ini_x1_part1=N_ini(1,1:11);
N_ini_x1_part2=flip(N_ini_x1_part1(2:10));
N_ini_x1=[N_ini_x1_part1 N_ini_x1_part2 -N_ini_x1_part1 -N_ini_x1_part2];

N_ini_x2_part1=N_ini(1,121:131);
N_ini_x2_part2=flip(N_ini_x2_part1(2:10));
N_ini_x2=[N_ini_x2_part1 N_ini_x2_part2 -N_ini_x2_part1 -N_ini_x2_part2];

N_ini_x3_part1=N_ini(1,161:171);
N_ini_x3_part2=flip(N_ini_x3_part1(2:10));
N_ini_x3=[N_ini_x3_part1 N_ini_x3_part2 -N_ini_x3_part1 -N_ini_x3_part2];

N_ini_x4_part1=N_ini(1,201:211);
N_ini_x4_part2=flip(N_ini_x4_part1(2:10));
N_ini_x4=[N_ini_x4_part1 N_ini_x4_part2 -N_ini_x4_part1 -N_ini_x4_part2];

N_ini_x5_part1=N_ini(1,241:251);
N_ini_x5_part2=flip(N_ini_x5_part1(2:10));
N_ini_x5=[N_ini_x5_part1 N_ini_x5_part2 -N_ini_x5_part1 -N_ini_x5_part2];

N_ini_x6_part1=N_ini(1,81:91);
N_ini_x6_part2=flip(N_ini_x6_part1(2:10));
N_ini_x6=[N_ini_x6_part1 N_ini_x6_part2 -N_ini_x6_part1 -N_ini_x6_part2];



N_ini_y1_part1=N_ini(2,1:11)-N_ini(2,11);
N_ini_y1_part2=flip(N_ini_y1_part1(2:10));
N_ini_y1=[N_ini_y1_part1 -N_ini_y1_part2 -N_ini_y1_part1 N_ini_y1_part2];

N_ini_y2_part1=N_ini(2,121:131)-N_ini(2,131);
N_ini_y2_part2=flip(N_ini_y2_part1(2:10));
N_ini_y2=[N_ini_y2_part1 -N_ini_y2_part2 -N_ini_y2_part1 N_ini_y2_part2];

N_ini_y3_part1=N_ini(2,161:171)-N_ini(2,171);
N_ini_y3_part2=flip(N_ini_y3_part1(2:10));
N_ini_y3=[N_ini_y3_part1 -N_ini_y3_part2 -N_ini_y3_part1 N_ini_y3_part2];

N_ini_y4_part1=N_ini(2,201:211)-N_ini(2,211);
N_ini_y4_part2=flip(N_ini_y4_part1(2:10));
N_ini_y4=[N_ini_y4_part1 -N_ini_y4_part2 -N_ini_y4_part1 N_ini_y4_part2];

N_ini_y5_part1=N_ini(2,241:251)-N_ini(2,251);
N_ini_y5_part2=flip(N_ini_y5_part1(2:10));
N_ini_y5=[N_ini_y5_part1 -N_ini_y5_part2 -N_ini_y5_part1 N_ini_y5_part2];

N_ini_y6_part1=N_ini(2,81:91)-N_ini(2,91);
N_ini_y6_part2=flip(N_ini_y6_part1(2:10));
N_ini_y6=[N_ini_y6_part1 -N_ini_y6_part2 -N_ini_y6_part1 N_ini_y6_part2];



N_ini_z1_part1=N_ini(3,1:11);
N_ini_z1_part2=flip(N_ini_z1_part1(2:10));
N_ini_z1=[N_ini_z1_part1 N_ini_z1_part2 N_ini_z1_part1 N_ini_z1_part2];

N_ini_z2_part1=N_ini(3,121:131);
N_ini_z2_part2=flip(N_ini_z2_part1(2:10));
N_ini_z2=[N_ini_z2_part1 N_ini_z2_part2 N_ini_z2_part1 N_ini_z2_part2];

N_ini_z3_part1=N_ini(3,161:171);
N_ini_z3_part2=flip(N_ini_z3_part1(2:10));
N_ini_z3=[N_ini_z3_part1 N_ini_z3_part2 N_ini_z3_part1 N_ini_z3_part2];

N_ini_z4_part1=N_ini(3,201:211);
N_ini_z4_part2=flip(N_ini_z4_part1(2:10));
N_ini_z4=[N_ini_z4_part1 N_ini_z4_part2 N_ini_z4_part1 N_ini_z4_part2];

N_ini_z5_part1=N_ini(3,241:251);
N_ini_z5_part2=flip(N_ini_z5_part1(2:10));
N_ini_z5=[N_ini_z5_part1 N_ini_z5_part2 N_ini_z5_part1 N_ini_z5_part2];

N_ini_z6_part1=N_ini(3,81:91);
N_ini_z6_part2=flip(N_ini_z6_part1(2:10));
N_ini_z6=[N_ini_z6_part1 N_ini_z6_part2 N_ini_z6_part1 N_ini_z6_part2];


N_ini_z1_part1d=N_ini(3,41:51);
N_ini_z1_part2d=flip(N_ini_z1_part1d(2:10));
N_ini_z1d=[N_ini_z1_part1d N_ini_z1_part2d N_ini_z1_part1d N_ini_z1_part2d];

N_ini_z2_part1d=N_ini(3,281:291);
N_ini_z2_part2d=flip(N_ini_z2_part1d(2:10));
N_ini_z2d=[N_ini_z2_part1d N_ini_z2_part2d N_ini_z2_part1d N_ini_z2_part2d];

N_ini_z3_part1d=N_ini(3,321:331);
N_ini_z3_part2d=flip(N_ini_z3_part1d(2:10));
N_ini_z3d=[N_ini_z3_part1d N_ini_z3_part2d N_ini_z3_part1d N_ini_z3_part2d];

N_ini_z4_part1d=N_ini(3,361:371);
N_ini_z4_part2d=flip(N_ini_z4_part1d(2:10));
N_ini_z4d=[N_ini_z4_part1d N_ini_z4_part2d N_ini_z4_part1d N_ini_z4_part2d];

N_ini_z5_part1d=N_ini(3,401:411);
N_ini_z5_part2d=flip(N_ini_z5_part1d(2:10));
N_ini_z5d=[N_ini_z5_part1d N_ini_z5_part2d N_ini_z5_part1d N_ini_z5_part2d];





N_x=[N_ini_x1 N_ini_x1 N_ini_x6 N_ini_x2 N_ini_x3 N_ini_x4 N_ini_x5 N_ini_x2 N_ini_x3 N_ini_x4 N_ini_x5];
N_y=[N_ini_y1 N_ini_y1 N_ini_y6 N_ini_y2 N_ini_y3 N_ini_y4 N_ini_y5 N_ini_y2 N_ini_y3 N_ini_y4 N_ini_y5];
N_z=[N_ini_z1 N_ini_z1d N_ini_z6 N_ini_z2 N_ini_z3 N_ini_z4 N_ini_z5 N_ini_z2d N_ini_z3d N_ini_z4d N_ini_z5d];

N=[N_x;N_y;N_z]./3*sc;

N_81=N(:,81);
N_82=N(:,82);
N_120=N(:,120);
HS1=norm(N_81 -N_82);

HS40=norm(N_120 -N_81);

%确定ANSYS中杆件顺序
%%%%%%%%%%%%%%%%%%%%%%%%%%此顺序为Ansys中杆件单元的顺序，并非matlab顺序%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
c_HSA=[1:40];
c_HSB=[(1:40)+1*40];
c_HSC=[(1:40)+2*40];
c_HSD=[(1:40)+3*40];
c_HSE=[(1:40)+4*40];
c_HSF=[(1:40)+5*40];
c_HSG=[(1:40)+6*40];
c_HSH=[(1:40)+7*40];

c_JS1=[320+5*(1:40)-4];
c_JS2=[320+5*(1:40)-3];
c_JS3=[320+5*(1:40)-2];
c_JS4=[320+5*(1:40)-1];
c_JS5=[320+5*(1:40)-0];

c_XS1=[520+5*(1:40)-4];
c_XS2=[520+5*(1:40)-3];
c_XS3=[520+5*(1:40)-2];
c_XS4=[520+5*(1:40)-1];
c_XS5=[520+5*(1:40)-0];

c_B1=[721:760];
c_B2=[761:800];
c_B3=[801:840];
c_B4=[841:880];
force_HSA=num_force(c_HSA);
force_HSB=num_force(c_HSB);
force_HSC=num_force(c_HSC);
force_HSD=num_force(c_HSD);
force_HSE=num_force(c_HSE);
force_HSF=num_force(c_HSF);
force_HSG=num_force(c_HSG);
force_HSH=num_force(c_HSH);

force_HS_total=[force_HSA force_HSB force_HSC force_HSD force_HSE force_HSF force_HSG force_HSH];
force_HS=(1/8).*sum(force_HS_total,2);   %40个索段 每个索段的力

force_JS1=num_force(c_JS1);
force_JS2=num_force(c_JS2);
force_JS3=num_force(c_JS3);
force_JS4=num_force(c_JS4);
force_JS5=num_force(c_JS5);

force_XS1=num_force(c_XS1);
force_XS2=num_force(c_XS2);
force_XS3=num_force(c_XS3);
force_XS4=num_force(c_XS4);
force_XS5=num_force(c_XS5);

force_B1=num_force(c_B1);
force_B2=num_force(c_B2);
force_B3=num_force(c_B3);
force_B4=num_force(c_B4);

%这里形成了自己的顺序
t2_pre=[force_B1;force_B2;force_B3;force_B4;force_HS;force_JS1;force_JS2;force_JS3;force_JS4;...
    force_JS5;force_XS1;force_XS2;force_XS3;force_XS4;force_XS5];







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
l_bar1=diag(sqrt(sum((N*C').^2))); 
l_vec1=diag(l_bar1);

B1=l_vec1(c_rep_B1);
B2=l_vec1(c_rep_B2);
B3=l_vec1(c_rep_B3);
B4=l_vec1(c_rep_B4);

B_rep=[B1'; B2'; B3' ;B4'];

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
Nx=N(1,:);
Ny=N(2,:);
N_xy=[Nx;Ny];
%1-40,41-81为最外围的节点投影.任取一组都可以
N1_xy=N_xy(:,41:80);
N6_xy=N_xy(:,81:120);
distances = sqrt((N1_xy(1, :) - N6_xy(1, :)).^2 + (N1_xy(2, :) - N6_xy(2, :)).^2);
distances=round(distances, 2); %保留小数点后2位

% B1=[N(3,121:160)-N(3,281:320)]';
% B2=[N(3,161:200)-N(3,321:360)]';
% B3=[N(3,201:240)-N(3,361:400)]';
% B4=[N(3,241:280)-N(3,401:440)]';
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
Aa=Ia'*A ;
%% 分组
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
%%svd
[U,S,V] = svd(A_hat*Gp);
r=rank(A_hat*Gp); %rank of (A_hat*Gp)
U1=U(:,1:r);U2=U(:,r+1:end);
S1=S(1:r,1:r);
V1=V(:,1:r);V2=V(:,r+1:end);

Z=0.29;
V2=t2_pre./l_vec1*sc;
K2=V2*Z;
%% 加载
wp=[kron(ones(1,80),[0 0 0]) kron(ones(1,40),[0 0 -F5])  kron(ones(1,40),[0 0 -F1]) ...
    kron(ones(1,40),[0 0 -F2]) kron(ones(1,40),[0 0 -F3]) kron(ones(1,40),[0 0 -F4])...
    kron(ones(1,160),[0 0 -0])]';
w =wp;
wpa=Ia'*wp;
K1=pinv(Aa)*wpa;

qf=(K1+K2); %Aq=F,q=(A+)F+V2Z,F=0 截面设计下的力密度
t=l_bar1*qf;  %此时的t是外力与预应力共同作用下的内力，用该力来确定截面面积，保证强度
t2=l_bar1*K2;

t_rep=t(c_rep_total);                            %取每个代表杆件的内力值
t2_rep=t2(c_rep_total);

maxHS = max(t2_rep(45:54));
maxJS1= max(t2_rep(11*(5:9)));
maxJS2= max(t2_rep(11*(5:9)+1));
maxJS3= max(t2_rep(11*(5:9)+2));
maxJS4= max(t2_rep(11*(5:9)+3));
maxJS5= max(t2_rep(11*(5:9)+4));
maxJS6= max(t2_rep(11*(5:9)+5));
maxJS7= max(t2_rep(11*(5:9)+6));
maxJS8= max(t2_rep(11*(5:9)+7));
maxJS9= max(t2_rep(11*(5:9)+8));
maxJS10= max(t2_rep(11*(5:9)+9));
maxJS11= max(t2_rep(11*(5:9)+10));
% maxJS1 = max(Area_des(55:65));
% maxJS2 = max(Area_des(66:76));
% maxJS3 = max(Area_des(77:87));
% maxJS4 = max(Area_des(88:98));
% maxJS5 = max(Area_des(99:109));
maxXS1= max(t2_rep(11*(10:14)));
maxXS2= max(t2_rep(11*(10:14)+1));
maxXS3= max(t2_rep(11*(10:14)+2));
maxXS4= max(t2_rep(11*(10:14)+3));
maxXS5= max(t2_rep(11*(10:14)+4));
maxXS6= max(t2_rep(11*(10:14)+5));
maxXS7= max(t2_rep(11*(10:14)+6));
maxXS8= max(t2_rep(11*(10:14)+7));
maxXS9= max(t2_rep(11*(10:14)+8));
maxXS10= max(t2_rep(11*(10:14)+9));
maxXS11= max(t2_rep(11*(10:14)+10));
t2_rep_kN=t2_rep./1e3;
t2_rep_max = [maxHS;maxJS1;maxJS2;maxJS3;maxJS4;maxJS5;maxJS6;maxJS7;maxJS8;maxJS9;maxJS10;maxJS11;...
    maxXS1;maxXS2;maxXS3;maxXS4;maxXS5;maxXS6;maxXS7;maxXS8;maxXS9;maxXS10;maxXS11];

c_first_rep=[45,55:65,110:120];
t2_rep_first_kN=[t2_rep(c_first_rep)./1e3]';
%% 面积设计
sigma_des=[ones(44,1)*sigmab_Des; ones(120,1)*sigmas*0.5];          %取每个代表杆件的应力设计值.1-3为杆件，4-11为索
Area_des=abs(t_rep./sigma_des);           %取每个代表杆件的内力值, 可计算预应力和荷载态
% 第 1 到第 11 行的最大值
maxB1 = max(Area_des(1:11));  %B1的代表构件序号，1/4跨度的榀数
maxB2 = max(Area_des(12:22));
maxB3 = max(Area_des(23:33));
maxB4 = max(Area_des(34:44));

maxB=max(Area_des(1:44));
maxHS = max(Area_des(45:54));
%这里上部下部的脊索斜索通常，选取一个面积即可，所以按径向设计
%用JS1-11，XS1-11代替
maxJS1= max(Area_des(11*(5:9)));
maxJS2= max(Area_des(11*(5:9)+1));
maxJS3= max(Area_des(11*(5:9)+2));
maxJS4= max(Area_des(11*(5:9)+3));
maxJS5= max(Area_des(11*(5:9)+4));
maxJS6= max(Area_des(11*(5:9)+5));
maxJS7= max(Area_des(11*(5:9)+6));
maxJS8= max(Area_des(11*(5:9)+7));
maxJS9= max(Area_des(11*(5:9)+8));
maxJS10= max(Area_des(11*(5:9)+9));
maxJS11= max(Area_des(11*(5:9)+10));
% maxJS1 = max(Area_des(55:65));
% maxJS2 = max(Area_des(66:76));
% maxJS3 = max(Area_des(77:87));
% maxJS4 = max(Area_des(88:98));
% maxJS5 = max(Area_des(99:109));
maxXS1= max(Area_des(11*(10:14)));
maxXS2= max(Area_des(11*(10:14)+1));
maxXS3= max(Area_des(11*(10:14)+2));
maxXS4= max(Area_des(11*(10:14)+3));
maxXS5= max(Area_des(11*(10:14)+4));
maxXS6= max(Area_des(11*(10:14)+5));
maxXS7= max(Area_des(11*(10:14)+6));
maxXS8= max(Area_des(11*(10:14)+7));
maxXS9= max(Area_des(11*(10:14)+8));
maxXS10= max(Area_des(11*(10:14)+9));
maxXS11= max(Area_des(11*(10:14)+10));
% maxXS1 = max(Area_des(110:120));
% maxXS2 = max(Area_des(121:131));
% maxXS3 = max(Area_des(132:142));
% maxXS4 = max(Area_des(143:153));
% maxXS5 = max(Area_des(154:164));


% 将最大值存储到一个新的 2x1 向量
Area_des_rep = [maxB;maxHS;maxJS1;maxJS2;maxJS3;maxJS4;maxJS5;maxJS6;maxJS7;maxJS8;maxJS9;maxJS10;maxJS11;...
    maxXS1;maxXS2;maxXS3;maxXS4;maxXS5;maxXS6;maxXS7;maxXS8;maxXS9;maxXS10;maxXS11];
%% 按照上式计算出的数值，以及规范中的标准截面面积，进行截面选取

Area_B=216.77e-6; %Area of bar element 26(3)

% Area_HS=5894.21e-6; %6*46
Area_HS=4911.85e-6;   %4*46
Area_JS=742.81e-6; %40
Area_XS1_6=818.95e-6; %2*42
Area_XS7_9=742.81e-6; %2*40
Area_XS10_11=742.81e-6; %40
% 直径选取
D0_s=1000.*[0.046  0.040*ones(1,11) 0.042*ones(1,6) 0.040*ones(1,3) 0.040*ones(1,2)]' ;    %这里采用的应是钢丝绳的公称直径 单位mm
%%  (design Kt in Rt matrix)
E_bar=diag([Eb*ones(ne_c,1); Es*ones(ne_s,1)]);
num_HS=40;
num_JS=40*5;
num_XS1_6=(2+5*4)*5; %1好2根，2-5号有4*5根，每根5段
num_XS7_9=(3*4)*5; 
num_XS10_11=(2+4)*5; 

% A_bar=diag([Area_B*ones(ne_c,1); Area_HS*ones(num_HS,1);Area_JS*ones(num_JS,1);...
%     Area_XS1_6*ones(num_XS1_6,1);Area_XS7_9*ones(num_XS7_9,1);Area_XS10_11*ones(num_XS10_11,1)]);
Area_XS_vec=[Area_XS1_6*ones(1+5,1);Area_XS7_9*ones(3,1);Area_XS10_11*ones(1+2,1);Area_XS7_9*ones(3,1);Area_XS1_6*ones(5,1)];
A_bar=diag([Area_B*ones(ne_c,1); Area_HS*ones(num_HS,1);Area_JS*ones(num_JS,1);...
     kron(ones(2*5,1),Area_XS_vec)]);   %按照杆件顺序重新分配面积顺序

Area_hat=A_bar;                   %矩阵表达形式
Area_diag=diag(Area_hat);
Area_rep=Area_diag(c_rep_total,:);            %向量表达形式
faiB=0.85;
  %参考E:\学习\博士生\博士生课题\[0]小论文\
               % 第五篇 基于数据驱动神经网络拟合的弹性模量时变退化曲线规律\素材
%              \Levy索穹顶算例-60m跨度\索穹顶部分信息
fais=1;

fai_hat=diag([ones(160,1)*faiB;ones(440,1)*fais]);

% %  计算下料长度l0
l0_vec=diag(E_bar*A_bar*l_bar1)./(t+diag(E_bar*A_bar));
% l0_vec=diag(E_bar*A_bar*l_bar1)./(l_bar1*K2+diag(E_bar*A_bar))
l0_vec_rep=[l0_vec(c_rep_total)]';
l0_bar=diag(l0_vec);
l0_inv=inv(l0_bar);

X0=N(:);                                  %Rearrange in column
X=nonlinequ(X0);   %更新X
l_bar=diag(sqrt(sum((reshape(X,3,[])*C').^2))); %bar length matrix
l_vec=diag(l_bar);
l_vec_rep=l_vec(c_rep_total);
l_inv=inv(l_bar);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%计算无荷载时的中心竖杆节点坐标位置
w =[kron(ones(1,440),[0 0 0])]';
X0=N(:);                                  %Rearrange in column
X1=nonlinequ(X0);                          %更新X 预应力态下的节点坐标
w=wp;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% X_Dleta=reshape(X1-X,3,[])
X1_reshape1=reshape(X1,3,[]);   %X1为不考虑荷载的成形态下的节点z坐标
ef=[81:440];
X1_z=X1_reshape1(3,ef);
c_81_291=[81 86 91 121 126 131 281 286 291];
X1_z_81_291=X1_z(c_81_291);
X_reshape1=reshape(X,3,[]);   %X为考虑荷载的成形态下的节点z坐标
X_z=X_reshape1(3,ef);
delta_X=X1_z-X_z;
span=2*N(2,1);  %2倍的长轴半径
X_Alo=span./250 ;
check_x1=(X_Alo)-(X1_z-X_z);   %检查是否满足1/250的要求
min_check_x1=min(check_x1);

l0_bar2=diag(sqrt(sum((reshape(X1,3,[])*C').^2))); %bar length matrix
l0_vec2=diag(l0_bar2);
check_l0=l0_vec2-l0_vec;
%由于是ANSYS模型倒过来的，L0有一些误差，因此这里用新的L0

% l0_vec=l0_vec2;
% l0_vec_rep=[l0_vec(c_rep_total)]';
% l0_bar=diag(l0_vec);
% l0_inv=inv(l0_bar);

q_bar=E_bar*A_bar*(inv(l0_bar)-inv(l_bar));      %force density
f_bar=diag(l_bar*q_bar);
f_bar_rep=f_bar(c_rep_total);
qf2=diag(q_bar);

sigma_hat=(inv(Area_hat*fai_hat)*l_bar*q_bar);
sigma=diag(sigma_hat);
sigma_MPa=sigma./1e6;

sigma_rep=sigma(c_rep_total);
sigma_rep_MPa=sigma_rep./1e6;

sigma_rep2=t2_rep./Area_rep;
sigma_rep_MPa2=sigma_rep2./1e6;

check_sigma=sigma_rep_MPa-sigma_rep_MPa2;
K_t=tenseg_stiff_matx(X);
K_taa=Ia'*K_t*Ia;
K_taa=(K_taa+K_taa').*0.5;

eig(Ia'*K_t*Ia);
%%  design Rt matrix
Cn=E_bar*A_bar*l0_inv;
Bl=(A*l_inv)';   %此处的A是对力密度的平衡矩阵，Bl矩阵是对f的平衡矩阵的转置，所以要有一个l的转化
Bla=(Ia'*A*l_inv)';
Rt=eye(ne,ne)-Cn*Bla*pinv(K_taa)*Bla';  %eye(ne,ne)=I
%% design Rx matrix including Rx_A and Rx_L
Rx_L= -pinv(K_taa)*Bla'*Cn;
Rx_A= -pinv(K_taa)*Bla'*sigma_hat;
%%  design Rt_* matrix including Rt_A and Rt_L
Rt_L=Rt*Cn;
Rt_A=Rt*sigma_hat;
%% 初始误差分析




%双因素时变分析
% [l_delta_total,sigma_Mpa_total,X_total,At_total_mm2,dLt_sigma_total,dAt_sigma_total,dAx_total,dLx_total]=CreandCor_calculation_ESSCGS (sigma,D0_s,A_bar,E_bar,l0_vec,fai_hat,N,span,X1,X,load_case);
a23=231;
sigma_Mpa_total_rep=sigma_Mpa_total(:,c_rep_total);
c=c_rep_total;  %取代表杆件的编号，每一类型取一个
c1=c(1:44);                    %仅取杆单元
c2=c(45:164);                    %仅取索单元
l_delta_total_mm_rep=l_delta_total(:,c2-160)*1e3;

num_ct1=160;
num_ct2=440;
num_ct=600;
ct2=[161:600];

sigma_total=sigma_Mpa_total.*1e6;
%由面积和长度改变量引起的共同的位移和内力的改变量
dAx_total_mm=dAx_total*1000;
dLx_total_mm=dLx_total*1000;
dx_total_mm=dAx_total_mm+dLx_total_mm;
dt_total_sigma=dLt_sigma_total+dAt_sigma_total;

%删除全0行
%all(a==0, 2) 2代表检测每一行是否为全零元素，如果不全为0，则返回0，全为0则返回1
sigma_total(all(sigma_total==0,2),:)=[];
sigma_Mpa_total(all(sigma_Mpa_total==0,2),:)=[];
        %X_total第一行为0行，后期补充为未蠕变的坐标
sigma_Mpa_total_rep=sigma_Mpa_total(:,c);
At_total_mm2(all(At_total_mm2==0,2),:)=[];


At_total_mm2_rep=At_total_mm2(:,c);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 计算结构中各钢索的最小松弛内力
distances_rep=[distances(1:11)]';
JS1cos=(1/8).*distances_rep./JS1;
JS2cos=(2/8).*distances_rep./JS2;
JS3cos=(2/8).*distances_rep./JS3;
JS4cos=(2/8).*distances_rep./JS4;
JS5cos=(1/8).*distances_rep./JS5;

XS1cos=(1/8).*distances_rep./XS1;
XS2cos=(2/8).*distances_rep./XS2;
XS3cos=(2/8).*distances_rep./XS3;
XS4cos=(2/8).*distances_rep./XS4;
XS5cos=(1/8).*distances_rep./XS5;

HS1cos=1;


At_s_total_mm2=At_total_mm2(:,c2);


cos_s=[HS1cos*ones(1,10) JS1cos' JS2cos' JS3cos'...
    JS4cos' JS5cos'   XS1cos'  XS2cos'...
      XS3cos'    XS4cos'  XS5cos' ];  %各类索与地面的夹角
wg=10.*7850*(l0_vec(c2))'.*At_s_total_mm2./1e6;  %各类索的自重 7850是kg/m3，所以wg单位是N


Fs=20.*wg.* cos_s; %松弛失效下的最小内力   Fs=20*wg*cosa
Fs_sigma_MPa=Fs./At_s_total_mm2;  %是一个常量，任取一行均可
Fs_sigma_MPa_vec=Fs_sigma_MPa(1,:);

%% xr3为变形失效模式下的可靠约束置信点 x3为最优目标点
gf=[81:440];              %g_freedom    
% X_Dleta=reshape(X1-X,3,[])
X1_reshape=reshape(X1,3,[]);   %X1为不考虑荷载的成形态下的节点z坐标
% X1_rep=X1_reshape(3,g)'

X1_total=X1_reshape(3,gf)';


% X_reshape=reshape(X,3,[])     %X考虑荷载的成形态下的节点z坐标
% X_rep=X_reshape(3,g)'
X_z=reshape(X,3,[]);
X_total(1,:)=X_z(3,gf);
X_total(all(X_total==0,2),:)=[];  
%代表节点编号
e_rep_HS=[1:11];
e_rep_N2up=[41:51];
e_rep_N3up=[81:91];
e_rep_N4up=[121:131];
e_rep_N5up=[161:171];
e_rep_N2d=[201:211];
e_rep_N3d=[241:251];
e_rep_N4d=[281:291];
e_rep_N5d=[321:331];

e=[e_rep_HS e_rep_N2up e_rep_N3up e_rep_N4up e_rep_N5up e_rep_N2d e_rep_N3d e_rep_N4d e_rep_N5d];     %自由节点中的代表节点顺序号


X_total_rep=X_total(:,e);

n_X=size(X_total,1);
% %创建相应的矩阵
X1_rep=[X1_total(e)]';
X1_matrix=(kron(ones(1,n_X),X1_total))';

Xa_Dle=X1_matrix-X_total;
X_Alo=span./250  ;                  %按照跨度的1/250对结构的挠度进行控制
check_x=X_Alo-Xa_Dle;
check_x_mm=check_x.*1000;
check_x_mm_rep=check_x_mm(:,e);

u_limit=check_x(1,:);   % 单位m  共有6类代表节点类型，因此为6*1的向量

%% 模糊综合评价
%%%%%%% 确定权重 %%%%%%%%%
%1-4行分别为24小时，30d,90d,180d的影响，不足一年，因此舍弃
dAx_mm2=abs(dAx_total_mm(5:end,:));
dLx_mm2=abs(dLx_total_mm(5:end,:));

dLt_sigma2=abs(dLt_sigma_total(5:end,:));
dAt_sigma2=abs(dAt_sigma_total(5:end,:));
% 建立正常使用极限状态的dA与dL影响的权重
X_dA=sum(dAx_mm2,2);   %sum(A,2)是令A矩阵的每一行的元素相加，成为一列
X_dL=sum(dLx_mm2,2); 
X_d=[X_dA X_dL]';
X_d_normalize=(normalize(X_d,'norm',1))';   %normalize函数以1范数归一化，计算每一列的归一化，所以要2次转置
wdAx=X_d_normalize(:,1);
wdLx=X_d_normalize(:,2);
%建立dA与dL内部的影响权重大小（二级权重）
dAx_normalize=(normalize(dAx_mm2','norm',1))';
dLx_normalize=(normalize(dLx_mm2','norm',1))';
dAx_normalize(isnan(dAx_normalize))=0;     %isnan(A)找出A矩阵中NaN元素的位置

% 建立承载能力极限状态的dA与dL影响的权重
sigma_dA=sum(dAt_sigma2,2);
sigma_dL=sum(dLt_sigma2,2);
sigma_d=[sigma_dA sigma_dL]';
sigma_d_normalize=(normalize(sigma_d,'norm',1))';   %normalize函数以1范数归一化，计算每一列的归一化，所以要2次转置

wdAt_sigma=sigma_d_normalize(:,1);
wdLt_sigma=sigma_d_normalize(:,2);
%建立dA与dL内部的影响权重大小（二级权重）
dAt_sigma_normalize=(normalize(dAt_sigma2','norm',1))';
dLt_sigma_normalize=(normalize(dLt_sigma2','norm',1))';
dAt_sigma_normalize(isnan(dAt_sigma_normalize))=0;     %isnan(A)找出A矩阵中NaN元素的位置
%%%%%%%%%%%%%% 确定t年下节点和构件的分值，对结构的使用性和安全性整体进行评价 %%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%正常使用极限状态下的评价%%%%%%%%%%%%%%
%%%%建立评语集{安全 较安全 一般 较差 差}，对应评分分别为{9,7,5,3,1}
% u_limit_rep=u_limit(:,e);
% check_x_rep=check_x(:,e);
check_x2=check_x(6:end,:);
span_u=u_limit./5;
nf=size(span_u,2);
te=size(check_x2,1);

for k1=1:te
    %确定构件的分值分布
     for k2=1:nf
     if check_x2(k1,k2)>u_limit(k2)-span_u(k2)
        value_x(k1,k2)=9;
     else 
         if check_x2(k1,k2)>u_limit(k2)-2.*span_u(k2)
             value_x(k1,k2)=7;
         else 
            if check_x2(k1,k2)>u_limit(k2)-3.*span_u(k2)
                value_x(k1,k2)=5;  
            else 
              if check_x2(k1,k2)>u_limit(k2)-4.*span_u(k2)
                  value_x(k1,k2)=3; 
              else 
                  value_x(k1,k2)=1; 
              end
            end
         end
     end
   end
end
a111="节点二级";

%%%%%%%%%%%确定结构的分值分布%%%%%%%%%%%%%%%
%%%%%%%%%确定正常使用极限状态的分值%%%%%%%%%

for k1=1:te
    value_x_dA(k1)=dAx_normalize(k1,:)*(value_x(k1,:))';
    value_x_dL(k1)=dLx_normalize(k1,:)*(value_x(k1,:))';
end
    value_x_d=[value_x_dA; value_x_dL];                       %dA与dL下的结构评分(正常使用极限状态)
    
for k1=1:te   
   value_x_total(k1)=X_d_normalize(k1,:)*value_x_d(:,k1);     %最终结构评分(正常使用极限状态)
end 
a111="节点";

%%%%%%%%%%%%%%%承载能力极限状态下的评价%%%%%%%%%%%%%%
%%%%建立评语集{安全 较安全 一般 较差 差}，对应评分分别为{9,7,5,3,1}
%%%%%屈服条件%%%%%%
% sigma_Mpa_total2=abs(sigma_Mpa_total(5:end,:));
sigma_Mpa_total2=abs(sigma_Mpa_total(6:end,:));
%
sigma_Mpa_initial=abs(sigma_Mpa_total(1,:));
sigma_des2=abs([ones(1,160)*sigmab_Des ones(1,440)*sigmas*0.5]./1e6); 
sigma_yield_limit=sigma_des2-sigma_Mpa_initial;
span_sigma1=sigma_yield_limit./5;

for k1=1:te
%确定构件的分值分布
     for k2=1:ne
     if sigma_Mpa_total2(k1,k2)<sigma_Mpa_initial(k2)+span_sigma1(k2)
        value_sigma1(k1,k2)=9;
     else 
         if sigma_Mpa_total2(k1,k2)<sigma_Mpa_initial(k2)+2.*span_sigma1(k2)
             value_sigma1(k1,k2)=7;
         else 
            if sigma_Mpa_total2(k1,k2)<sigma_Mpa_initial(k2)+3.*span_sigma1(k2)
                value_sigma1(k1,k2)=5;  
            else 
              if sigma_Mpa_total2(k1,k2)<sigma_Mpa_initial(k2)+4.*span_sigma1(k2)
                  value_sigma1(k1,k2)=3; 
              else 
                  value_sigma1(k1,k2)=1; 
              end
            end
         end
     end
   end
end
a111="屈服二级";
%%%%%%%%%%%%结构整体评分(屈服条件)%%%%%%%%%%%%%%%

for k1=1:te
    value_sigma1_dA(k1)=dAt_sigma_normalize(k1,:)*(value_sigma1(k1,:))';
    value_sigma1_dL(k1)=dLt_sigma_normalize(k1,:)*(value_sigma1(k1,:))';
end
    value_sigma1_d=[value_sigma1_dA; value_sigma1_dL];                   %dA与dL下的结构评分(屈服极限)
    
for k1=1:te   
   value_sigma1_total(k1)=sigma_d_normalize(k1,:)*value_sigma1_d(:,k1);  %最终结构评分(屈服极限)
end 
a111="屈服";


%%%%%松弛条件%%%%%%
sigma_des3=[ones(1,600)*0];  
sigma_relax_limit=sigma_Mpa_initial-sigma_des3;
span_sigma2=sigma_relax_limit./5;

for k1=1:te
    %确定构件的分值分布
     for k2=1:ne
     if sigma_Mpa_total2(k1,k2)>sigma_Mpa_initial(k2)-span_sigma2(k2)
        value_sigma2(k1,k2)=9;
     else 
         if sigma_Mpa_total2(k1,k2)>sigma_Mpa_initial(k2)-2*span_sigma2(k2)
             value_sigma2(k1,k2)=7;
         else 
            if sigma_Mpa_total2(k1,k2)>sigma_Mpa_initial(k2)-3*span_sigma2(k2)
                value_sigma2(k1,k2)=5;  
            else 
              if sigma_Mpa_total2(k1,k2)>sigma_Mpa_initial(k2)-4*span_sigma2(k2)
                  value_sigma2(k1,k2)=3; 
              else 
                  value_sigma2(k1,k2)=1; 
              end
            end
         end
     end
   end
end
a111="松弛二级";
%%%%%%%%%%%%结构整体评分(松弛条件)%%%%%%%%%%%%%%%
for k1=1:te
    value_sigma1_dA2(k1)=dAt_sigma_normalize(k1,:)*(value_sigma2(k1,:))';
    value_sigma1_dL2(k1)=dLt_sigma_normalize(k1,:)*(value_sigma2(k1,:))';
end
    value_sigma1_d2=[value_sigma1_dA2; value_sigma1_dL2];                   %dA与dL下的结构评分(松弛极限)
    
for k1=1:te   
   value_sigma2_total(k1)=sigma_d_normalize(k1,:)*value_sigma1_d2(:,k1);  %最终结构评分(松弛极限)
end 
a111="松弛";


%%%%%由于承载能力极限状态，屈服条件与松弛条件是2种互斥的承载能力极限，因此取两者的最小值
for k1=1:te   
   value_sigma3_total(k1)=min(value_sigma1_total(k1),value_sigma2_total(k1));  %最终结构评分(松弛极限与屈服极限的最小值)
end 
a111="松弛与屈服最小值";

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%  真实值计算
%%以上数据存在误差，年限较长时误差变大，因此可以将比例分配到实际的位移和应力变化后，重新评估
%dt与dx后面几年的累计误差较大，因此通过比例系数反算一个比较真实作为输出
%计算出每一年下，每个节点位移或构件内力，在dA或dL下的归一化比值及真实的每年节点位移及构件内力即可算出
%计算得到每年的位移变化矩阵
X_total_2=X_total(6:end,:);
X_peryear=zeros(size(X_total_2,1),size(X_total_2,2));
X_peryear(1,:)=-(X_total(6,:)-X_total(1,:));
for i=2:size(X_total_2,1)
    X_peryear(i,:)=-(X_total_2(i,:)-X_total_2(i-1,:));
end


%sigma_Mpa_total2在上文中用过，为绝对值的6行之后数据，这里采用不包含绝对值的
sigma_Mpa_total3=sigma_Mpa_total(6:end,:);
sigma_Mpa_peryear=zeros(size(sigma_Mpa_total3,1),size(sigma_Mpa_total3,2));
sigma_Mpa_peryear(1,:)=sigma_Mpa_total(6,:)-sigma_Mpa_total(1,:);
for i=2:size(sigma_Mpa_total3,1)
    sigma_Mpa_peryear(i,:)=sigma_Mpa_total3(i,:)-sigma_Mpa_total3(i-1,:);
end

%X
%计算每年的不同节点dAdL影响比例系数
%dAx占比proportion
dAx_prop=dAx_normalize.*X_d_normalize(:,1);     %X_d_normalize第一列均为面积对位移的影响系数，第二列为长度 
%dLx占比proportion
dLx_prop=dLx_normalize.*X_d_normalize(:,2); 
%两者相加为总得因素，C=A+B，A/C和B/C就是归一化的占比结果
dx_prop_total=dAx_prop+dLx_prop;
dAx_prop_norm=dAx_prop./dx_prop_total;            %每一个节点对dA，dL的影响归一化
dLx_prop_norm=dLx_prop./dx_prop_total;
check_prop=dAx_prop_norm+dLx_prop_norm;

%计算所属位移
dAx_true=dAx_prop_norm.*X_peryear;
dLx_true=dLx_prop_norm.*X_peryear;

dAx_true_mm=dAx_true*1000;
dLx_true_mm=dLx_true*1000;

dAx_true_mm_rep=dAx_true_mm(:,e);
dLx_true_mm_rep=dLx_true_mm(:,e);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%计算真实位移随时间的累加位移
dAx_total_true_mm(1,:)=dAx_true_mm(1,:);
for i=2:size(dAx_true_mm,1)
    dAx_total_true_mm(i,:)=dAx_total_true_mm(i-1,:)+dAx_true_mm(i,:);
end

dLx_total_true_mm(1,:)=dLx_true_mm(1,:);
for i=2:size(dLx_true_mm,1)
    dLx_total_true_mm(i,:)=dLx_total_true_mm(i-1,:)+dLx_true_mm(i,:);
end

dAx_total_true_mm_rep=dAx_total_true_mm(:,e);
dLx_total_true_mm_rep=dLx_total_true_mm(:,e);

dx_total_true_mm=dAx_total_true_mm+dLx_total_true_mm;
dx_total_true_mm_rep=dx_total_true_mm(:,e);
%sigma
%计算每年的不同构件dAdL影响比例系数
%dAt_sigma占比proportion
dAt_sigma_prop=dAt_sigma_normalize.*sigma_d_normalize(:,1);     %sigma_d_normalize第一列均为面积对应力的影响系数，第二列为长度 
%dLt_sigma占比proportion
dLt_sigma_prop=dLt_sigma_normalize.*sigma_d_normalize(:,2); 

%两者相加为总得因素，C=A+B，A/C和B/C就是归一化的占比结果
dt_sigma_prop_total=dAt_sigma_prop+dLt_sigma_prop;

dAt_sigma_prop_norm=dAt_sigma_prop./dt_sigma_prop_total;
dLt_sigma_prop_norm=dLt_sigma_prop./dt_sigma_prop_total;

check_prop2=dAt_sigma_prop_norm+dLt_sigma_prop_norm;

%计算所属应力
dAt_sigma_true=dAt_sigma_prop_norm.*sigma_Mpa_peryear;
dLt_sigma_true=dLt_sigma_prop_norm.*sigma_Mpa_peryear;

dAt_sigma_true_rep=dAt_sigma_true(:,c);
dLt_sigma_true_rep=dLt_sigma_true(:,c);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%计算真实应力随时间的累加位移
dAt_sigma_total_true(1,:)=dAt_sigma_true(1,:);
for i=2:size(dAx_true_mm,1)
    dAt_sigma_total_true(i,:)=dAt_sigma_total_true(i-1,:)+dAt_sigma_true(i,:);
end

dLt_sigma_total_true(1,:)=dLt_sigma_true(1,:);
for i=2:size(dAx_true_mm,1)
    dLt_sigma_total_true(i,:)=dLt_sigma_total_true(i-1,:)+dLt_sigma_true(i,:);
end


dAt_sigma_total_true_rep=dAt_sigma_total_true(:,c);
dLt_sigma_total_true_rep=dLt_sigma_total_true(:,c);

dt_total_true_mm=dAt_sigma_total_true+dLt_sigma_total_true;
dt_total_true_mm_rep=dt_total_true_mm(:,c);
% 建立正常使用极限状态的dA与dL影响的权重
%dA与dL影响权重大小（一级权重）
X_dA_true=sum(dAx_total_true_mm,2);   %sum(A,2)是令A矩阵的每一行的元素相加，成为一列
X_dL_true=sum(dLx_total_true_mm,2); 
X_d_true=[X_dA_true X_dL_true]';
X_d_true_normalize=(normalize(X_d_true,'norm',1))';   %normalize函数以1范数归一化，计算每一列的归一化，所以要2次转置

%建立dA与dL内部的影响权重大小（二级权重）
dAx_true_normalize=(normalize(dAx_total_true_mm','norm',1))';
dLx_true_normalize=(normalize(dLx_total_true_mm','norm',1))';
dAx_true_normalize(isnan(dAx_true_normalize))=0;     %isnan(A)找出A矩阵中NaN元素的位置

% 建立承载能力极限状态的dA与dL影响的权重
%dA与dL影响权重大小（一级权重）
sigma_dA_true=sum(abs(dAt_sigma_total_true),2);
sigma_dL_true=sum(abs(dLt_sigma_total_true),2);
sigma_d_ture=[sigma_dA_true sigma_dL_true]';
sigma_d_true_normalize=(normalize(sigma_d_ture,'norm',1))';   %normalize函数以1范数归一化，计算每一列的归一化，所以要2次转置

%建立dA与dL内部的影响权重大小（二级权重）
dAt_sigma_true_normalize=(normalize(abs(dAt_sigma_total_true)','norm',1))';
dLt_sigma_true_normalize=(normalize(abs(dLt_sigma_total_true)','norm',1))';
dAt_sigma_true_normalize(isnan(dAt_sigma_true_normalize))=0;     %isnan(A)找出A矩阵中NaN元素的位置

%%%%%%%%%%%%%% 确定t年下节点和构件的分值，对结构的使用性和安全性整体进行评价 %%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%正常使用极限状态下的评价%%%%%%%%%%%%%%
%%%%建立评语集{安全 较安全 一般 较差 差}，对应评分分别为{9,7,5,3,1}
% check_x2=check_x(6:end,:);
% span_u=u_limit./5;
% nf=size(span_u,2);
% te=size(check_x2,1);

for k1=1:te
    %确定构件的分值分布
     for k2=1:nf
     if check_x2(k1,k2)>u_limit(k2)-span_u(k2)
        value_x(k1,k2)=9;
     else 
         if check_x2(k1,k2)>u_limit(k2)-2.*span_u(k2)
             value_x(k1,k2)=7;
         else 
            if check_x2(k1,k2)>u_limit(k2)-3.*span_u(k2)
                value_x(k1,k2)=5;  
            else 
              if check_x2(k1,k2)>u_limit(k2)-4.*span_u(k2)
                  value_x(k1,k2)=3; 
              else 
                  value_x(k1,k2)=1; 
              end
            end
         end
     end
   end
end
a222="节点二级";

%%%%%%%%%%%确定结构的分值分布%%%%%%%%%%%%%%%
%%%%%%%%%确定正常使用极限状态的分值%%%%%%%%%

for k1=1:te
    value_x_dA_true(k1)=dAx_true_normalize(k1,:)*(value_x(k1,:))';
    value_x_dL_true(k1)=dLx_true_normalize(k1,:)*(value_x(k1,:))';
end
    value_x_d_true=[value_x_dA_true; value_x_dL_true];                       %dA与dL下的结构评分(正常使用极限状态)
    
for k1=1:te   
   value_x_true_total(k1)=X_d_true_normalize(k1,:)*value_x_d_true(:,k1);     %最终结构评分(正常使用极限状态)
end 
value_x_d_true=value_x_d_true';                %形成竖向向量
value_x_true_total=value_x_true_total';        %形成竖向向量
a222="节点";

value_x_rep=value_x(:,e);
%%%%%%%%%%%%%%%承载能力极限状态下的评价%%%%%%%%%%%%%%
%%%%建立评语集{安全 较安全 一般 较差 差}，对应评分分别为{9,7,5,3,1}
%%%%%屈服条件%%%%%%
% sigma_Mpa_total2=abs(sigma_Mpa_total(6:end,:));
% sigma_Mpa_initial=abs(sigma_Mpa_total(1,:));
% sigma_des2=abs([ones(1,num_ct1)*sigmab_Des ones(1,num_ct2)*sigmas*0.5]./1e6); 
% sigma_yield_limit=sigma_des2-sigma_Mpa_initial;
% span_sigma1=sigma_yield_limit./5;

for k1=1:te
%确定构件的分值分布
     for k2=1:ne
     if sigma_Mpa_total2(k1,k2)<sigma_Mpa_initial(k2)+span_sigma1(k2)
        value_sigma1(k1,k2)=9;
     else 
         if sigma_Mpa_total2(k1,k2)<sigma_Mpa_initial(k2)+2.*span_sigma1(k2)
             value_sigma1(k1,k2)=7;
         else 
            if sigma_Mpa_total2(k1,k2)<sigma_Mpa_initial(k2)+3.*span_sigma1(k2)
                value_sigma1(k1,k2)=5;  
            else 
              if sigma_Mpa_total2(k1,k2)<sigma_Mpa_initial(k2)+4.*span_sigma1(k2)
                  value_sigma1(k1,k2)=3; 
              else 
                  value_sigma1(k1,k2)=1; 
              end
            end
         end
     end
   end
end
a222="屈服二级";
%%%%%%%%%%%%结构整体评分(屈服条件)%%%%%%%%%%%%%%%

for k1=1:te
    value_sigma1_dA_true(k1)=dAt_sigma_true_normalize(k1,:)*(value_sigma1(k1,:))';
    value_sigma1_dL_true(k1)=dLt_sigma_true_normalize(k1,:)*(value_sigma1(k1,:))';
end
    value_sigma1_d_true=[value_sigma1_dA_true; value_sigma1_dL_true];                   %dA与dL下的结构评分(屈服极限)
    
for k1=1:te   
   value_sigma1_true_total(k1)=sigma_d_true_normalize(k1,:)*value_sigma1_d_true(:,k1);  %最终结构评分(屈服极限)
end 

value_sigma1_d_true=value_sigma1_d_true' ;               %形成竖向向量
value_sigma1_true_total=value_sigma1_true_total'  ;      %形成竖向向量
a222="屈服";

value_sigma1_rep=value_sigma1(:,c);
%%%%%松弛条件%%%%%%
% sigma_des3=[ones(1,num_ct)*0];  
% sigma_relax_limit=sigma_Mpa_initial-sigma_des3;
% span_sigma2=sigma_relax_limit./5;

for k1=1:te
    %确定构件的分值分布
     for k2=1:ne
     if sigma_Mpa_total2(k1,k2)>sigma_Mpa_initial(k2)-span_sigma2(k2)
        value_sigma2(k1,k2)=9;
     else 
         if sigma_Mpa_total2(k1,k2)>sigma_Mpa_initial(k2)-2*span_sigma2(k2)
             value_sigma2(k1,k2)=7;
         else 
            if sigma_Mpa_total2(k1,k2)>sigma_Mpa_initial(k2)-3*span_sigma2(k2)
                value_sigma2(k1,k2)=5;  
            else 
              if sigma_Mpa_total2(k1,k2)>sigma_Mpa_initial(k2)-4*span_sigma2(k2)
                  value_sigma2(k1,k2)=3; 
              else 
                  value_sigma2(k1,k2)=1; 
              end
            end
         end
     end
   end
end
a222="松弛二级";
%%%%%%%%%%%%结构整体评分(松弛条件)%%%%%%%%%%%%%%%
for k1=1:te
    value_sigma1_dA2_true(k1)=dAt_sigma_true_normalize(k1,:)*(value_sigma2(k1,:))';
    value_sigma1_dL2_true(k1)=dLt_sigma_true_normalize(k1,:)*(value_sigma2(k1,:))';
end
    value_sigma2_d_true=[value_sigma1_dA2_true; value_sigma1_dL2_true];                   %dA与dL下的结构评分(松弛极限)
    
for k1=1:te   
   value_sigma2_true_total(k1)=sigma_d_true_normalize(k1,:)*value_sigma2_d_true(:,k1);  %最终结构评分(松弛极限)
end 
a222="松弛";

value_sigma2_rep=value_sigma2(:,c);

value_sigma2_d_true=value_sigma2_d_true';
value_sigma2_true_total=value_sigma2_true_total';
%%%%%由于承载能力极限状态，屈服条件与松弛条件是2种互斥的承载能力极限，因此取两者的最小值
for k1=1:te   
   value_sigma3_true_total(k1)=min(value_sigma1_true_total(k1),value_sigma2_true_total(k1));  %最终结构评分(松弛极限与屈服极限的最小值)
end 
a222="松弛与屈服最小值";
