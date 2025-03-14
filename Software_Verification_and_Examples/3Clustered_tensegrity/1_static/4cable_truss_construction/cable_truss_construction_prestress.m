%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%Cable truss construction_prestressing%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% /* This Source Code Form is subject to the terms of the Mozilla Public
% * License, v. 2.0. If a copy of the MPL was not distributed with this
% * file, You can obtain one at http://mozilla.org/MPL/2.0/.
%
% 

%EXAMPLE
clc; clear all; close all;
% Global variable
[consti_data,Eb,Es,sigmab,sigmas,rho_b,rho_s]=material_lib('Steel_Q345','Steel_string');
material{1}='linear_elastic'; % index for material properties: multielastic, plastic.
material{2}=0; % index for considering slack of string (1) for yes,(0) for no (for compare with ANSYS)

% cross section design cofficient
thick=6e-3;        % thickness of hollow bar
hollow_solid=0;          % use hollow bar or solid bar in minimal mass design (1)hollow (0)solid
c_b=0.1;           % coefficient of safty of bars 0.5
c_s=0.1;           % coefficient of safty of strings 0.3

% static analysis set
substep=5;                                     %荷载子步
lumped=0;               % use lumped matrix 1-yes,0-no
saveimg=0;              % save image or not (1) yes (0)no
savedata=1;             % save data or not (1) yes (0)no
savevideo=1;            % make video(1) or not(0)
gravity=1;              % consider gravity 1 for yes, 0 for no
% move_ground=0;          % for earthquake, use pinned nodes motion(1) or add inertia force in free node(0) 
%% %% N C of the structure
filename='\结构信息表.xlsx';
sheet1=3;
range1='B2:E761';
num_N=xlsread(filename,sheet1,range1);
range2='F2:L881';
num_force=xlsread(filename,sheet1,range2);
num_force=num_force(:,2);   

N=[num_N(321:760,2:4);]';


% connectivity matrix 
C_b_in_unit = [[121:280]', [281:440]'];  % Write down the connection of bars
C_b = tenseg_ind2C(C_b_in_unit,N); %get the matrix of relation through C_b_in
C_s_in_unit = [[81:120, 1:40, 121:280, 41:80, 281:440]', ...
               [82:120, 81, 121:280, 81:120, 281:440, 81:120]'];
C_s = tenseg_ind2C(C_s_in_unit,N);
tenseg_plot(N,C_b,C_s);
C = [C_b;C_s];
ne=size(C,1);%number of element

nb=size(C_b,1);
[ne,nn]=size(C);        % ne:No.of element;nn:No.of node
%% Plot the structure to make sure it looks right
fig_handle=figure
tenseg_plot(N,C_b,C_s,fig_handle);

tenseg_plot(N,[],C);

%% %% Group/Clustered information 
%generate group index
% gr=[];
%gr={[1:p,2*p+1:3*p]',[p+1:2*p,3*p+1:4*p]',[4*p+1:5*p]',[5*p+1:6*p]};
gr={};  % outer diagonal, inner diagonal, inner hoop
Gp=tenseg_str_gp3(gr,C);    %generate group matrix
% S=eye(ne);                  % no clustering matrix
S=Gp';                      % clustering matrix is group matrix

[nec,ne]=size(S);
tenseg_plot_CTS(N,C,[],S);




%% %% Boundary constraints
pinned_X=(1:80)'; pinned_Y=(1:80)'; pinned_Z=(1:80)';
[Ia,Ib,a,b]=tenseg_boundary(pinned_X,pinned_Y,pinned_Z,nn);



%% %% self-stress design
%Calculate equilibrium matrix and member length
[A_1a,A_1ag,A_2a,A_2ag,l,l_gp]=tenseg_equilibrium_matrix1(N,C,Gp,Ia);
A_1ac=A_1a*S';          %equilibrium matrix CTS
A_2ac=A_2a*S';          %equilibrium matrix CTS
l_c=S*l;                % length vector CTS
%SVD of equilibrium matrix
[U1,U2,V1,V2,S1]=tenseg_svd(A_2ag);

%external force in equilibrium design
w0=zeros(numel(N),1); w0a=Ia'*w0;

%prestress design
index_gp=[161]; % number of groups with designed force

fd=[300e4];                       % force in bar is given as -1000

% [q_gp,t_c,q,t]=tenseg_prestress_design(Gp,l,l_gp,A_1ag,V1(:,end),w0a,index_gp,fd);    %prestress design
[t_c,t]=tenseg_prestress_design2(Gp,l,l_gp,A_2ag,V2,w0a,index_gp,fd);    %prestress design

% 
% t_c=1e4*ones(nec,1);
% % t_c=1e7*[1;1;0.1];
% t=S'*t_c;
% plot prestress
tenseg_plot_CTS2(N,C,1:nb,S,[],[],[],[],[],t);
tenseg_plot_CTS2(N,C,1:nb,S,[],[],[],[],[],-1e4*V2);

%% rest length design

index_b=find(t_c<0);              % index of bar in compression
index_s=setdiff(1:size(S,1),index_b);	% index of strings
[A_b,A_s,A_c,A,r_b,r_s,r_gp,radius,E_c,l0_c,rho,mass_c]=tenseg_minimass(t_c,l_c,eye(size(S,1)),sigmas,sigmab,Eb,Es,index_b,index_s,c_b,c_s,rho_b,rho_s,thick,hollow_solid);
%% cross sectional design
% A_c=(6e-3)^2*ones(nec,1);
% E_c=Es*ones(nec,1);
E=S'*E_c;     %Young's modulus CTS
A=S'*A_c;     % Cross sectional area CTS

% l0=(t+E.*A).\E.*A.*l;

l0=S'*l0_c;
mass=S'*rho.*A.*l0;
% % Plot the structure with radius
% R3Ddata.Bradius=interp1([min(radius),max(radius)],[0.03,.1],r_b);
% R3Ddata.Sradius=interp1([min(radius),max(radius)],[0.03,.1],r_s);
% R3Ddata.Nradius=0.1*ones(nn,1);
% tenseg_plot(N,C_b,C_s,[],[],[],'Double layer prism',R3Ddata);

%% tangent stiffness matrix
% [Kt_aa,Kg_aa,Ke_aa,K_mode,k]=tenseg_stiff_CTS(Ia,C,S,q,A_1a,E_c,A_c,l_c);
[Kt_aa,Kg_aa,Ke_aa,K_mode,k]=tenseg_stiff_CTS3(Ia,C,S,t_c,A_2a,E_c,A_c,l0,l);
% plot the mode shape of tangent stiffness matrix
num_plt=1:4;

% plot_mode2(K_mode,k,N,Ia,C_b,C_s,l,'tangent stiffness matrix',...
%     'Order of Eigenvalue','Eigenvalue of Stiffness (N/m)','N/m',num_plt,0.9,saveimg,3);
if 0
plot_mode_CTS2(K_mode,k,N,Ia,C,1:nb,S,l,'tangent stiffness matrix',...
    'Order of Eigenvalue','Eigenvalue of Stiffness (N/m)','N/m',num_plt,0.9,saveimg,3);
end
%% input file of ANSYS
% ansys_input_gp(N,C,A_gp,t_gp,b,Eb,Es,rho_b,rho_s,Gp,index_s,find(t_gp>0),'tower');

%% mass matrix and damping matrix
M=tenseg_mass_matrix(mass,C,lumped); % generate mass matrix
% damping matrix
d=0;     %damping coefficient
D=d*2*max(sqrt(mass.*E.*A./l0))*eye(3*nn);    %critical damping

%% mode analysis
[V_mode,D1] = eig(Kt_aa,Ia'*M*Ia);         % calculate vibration mode
w_2=diag(D1);                                    % eigen value of 
% sort the mode
[w_2_sort,I]=sort(w_2);
V_mode_sort=V_mode(:,I);

omega=real(sqrt(w_2_sort))/2/pi;                   % frequency in Hz

% plot_mode2(V_mode_sort,omega,N,Ia,C_b,C_s,l,'natrual vibration',...
%     'Order of Vibration Mode','Frequency (Hz)','Hz',num_plt,0.8,saveimg,3);
if 0
plot_mode_CTS2(V_mode_sort,omega,N,Ia,C,1:nb,S,l,'natrual vibration',...
    'Order of Vibration Mode','Frequency (Hz)','Hz',num_plt,0.9,saveimg,3);
end



%% external force, forced motion of nodes, shrink of strings
% calculate external force and 
substep=100;
ind_w=[];w=[];
ind_dnb=[]; dnb0=[];

ind_dl0_c=[401:440];dl0_c=[0.2*ones(40,1)];              % extend bottom cables 

[w_t,dnb_t,l0_ct,Ia_new,Ib_new]=tenseg_load_prestress(substep,ind_w,w,ind_dnb,dnb0,ind_dl0_c,dl0_c,l0,b,gravity,[0;0;9.8],C,mass);
% % rest length
% dl0_i=zeros(size(l02));
% dl0_i(ind_dl0_c)=dl0_c;
% dl0_t=dl0_i*[linspace(0,0.1,round(0.5*substep)),linspace(0.1,1,substep-round(0.5*substep))];
% l0_ct=dl0_t+l02*linspace(1,1,substep);

dnb_d_t=zeros(size(dnb_t));         % boundary speed
dnb_dd_t=zeros(size(dnb_t));         % boundary acceleration

%% Step1: equilibrium calculation
% input data
data.N=N; data.C=C; data.ne=ne; data.nn=nn; data.Ia=Ia_new; data.Ib=Ib_new;data.S=S;
data.E=E_c; data.A=A; data.index_b=index_b; data.index_s=index_s;
data.consti_data=consti_data;   data.material=material; %constitue info
data.w_t=w_t;  % external force
data.dnb_t=dnb_t;% forced movement of pinned nodes
data.l0_t=l0_ct;% forced movement of pinned nodes
data.substep=substep;    % substep
data.InitialLoadFactor=1e-3;
data.MaxIcr=1000;


%% nonlinear analysis
if 1
data_out1=static_solver_CTS(data);
t_t1=data_out1.t_out;          %member force in every step
n_t1=data_out1.n_out;          %nodal coordinate in every step
% N_out1=data_out1.N_out;
t_c_t=pinv(S')*t_t1;



%% plot member force 
tenseg_plot_result(1:substep,flip(t_c_t([161;241;441],:),2),{'环索', '上径向索', '下径向索'},{'Substep','Force / N'},'plot_member_force.png',saveimg);

%% Plot nodal coordinate curve X Y
tenseg_plot_result(1:substep,flip((n_t1([81*3-2:81*3,111*3-2:111*3],:)-kron(ones(1,substep),n_t1([81*3-2:81*3,111*3-2:111*3],end))),2),...
    {'长轴环节点-X','长轴环节点Y','长轴环节点Z','短轴环节点-X','短轴环节点Y','短轴环节点Z'},{'Substep','Displacement /m)'},'plot_coordinate.png',saveimg);



end







return

















%% 
data_out1=static_solver_CTS2(data);

t_t1=data_out1.t_out;          %member force in every step
n_t1=data_out1.n_out;          %nodal coordinate in every step
% N_out1=data_out1.N_out;
lmd_his=data_out1.lmd_his;
t_c_t=pinv(S2')*t_t1;
num_icr=sum(lmd_his~=0);            % maximum increasment step
 N_out1=reshape(n_t1(:,num_icr),3,[]);
%% plot member force 
tenseg_plot_result(lmd_his(1:num_icr),t_c_t(:,1:num_icr),{'ODC', 'IDC', 'HC'},{'Substep','Force / N'},'plot_member_force.png',saveimg);

%% Plot nodal coordinate curve X Y
tenseg_plot_result(lmd_his,n_t1([3*3-2],:),{'3X'},{'Substep','Coordinate /m)'},'plot_coordinate.png',saveimg);

%% make video of the dynamic
name=['cable_trus_prestress'];

tenseg_video_CTS(flip(n_t1,2),C,1:nb,S,[],[],[],[],[],[],t_t1,[],substep,10,name,savevideo)



%% Plot final configuration
% tenseg_plot_catenary( reshape(n_t(:,end),3,[]),C_b,C_s,[],[],[0,0],[],[],l0_ct(index_s,end))
% tenseg_plot( reshape(n_t(:,end),3,[]),C_b,C_s,[],[],[])
 j=linspace(0.01,1,6);
for i=1:numel(j)

   [~, num]=min(abs(j(i)-lmd_his)); %find the num of increasment step
   name1=['牵引索长度为',num2str(j(i)*10),'m'];
%  tenseg_plot( reshape(n_t(:,num),3,[]),C_b,C_s,[],[],[]);
tenseg_plot_CTS2(reshape(n_t1(:,num),3,[]),C2,1:nb,S2,[],[],[],name1,[],t_t1(:,num));
%  axis off;
grid on
end

% plot nodal displacement
for i=1:numel(j)
   [~, num]=min(abs(j(i)-lmd_his)); %find the num of increasment step
   name1=['牵引索长度为',num2str(j(i)*10),'m'];
N_disp=sqrt(sum((N2-reshape(n_t1(:,num),3,[])).^2,1));
tenseg_plot_CTS2(reshape(n_t1(:,num),3,[]),C2,1:nb,S2,[],[],[],name1,[],[],N_disp');
%  axis off;
grid on
end

% plot without countor
for i=1:numel(j)
    [~, num]=min(abs(j(i)-lmd_his)); %find the num of increasment step
    N_out_temp=reshape(n_t1(:,num),3,[]);
    name1=['牵引索长度为',num2str(j(i)*10),'m'];
    N_disp=sqrt(sum((N2-N_out_temp).^2,1));
    tenseg_plot(N_out_temp,C_b2,C_s2,[],[],[],name1);

    S =  N_out_temp*C_s2(end-80+1:end,:)';
    string_start_nodes = zeros(3,size(S,2));
    string_end_nodes = zeros(3,size(S,2));
    for k = 1:size(S,2)
        string_start_nodes(:,k) =  N_out_temp(:,C_s2(end-80+k,:)==-1);
        string_end_nodes(:,k) =  N_out_temp(:,C_s2(end-80+k,:)==1);
    end
    hold on
    quiver3(string_start_nodes(1,:),string_start_nodes(2,:),string_start_nodes(3,:),S(1,:),S(2,:),S(3,:),'green.','Autoscale','off','LineWidth',2);
    
    set(gcf, 'Position', [10 10 800 600]);
    %  axis off;
grid on
end

% plot 2D
 j=linspace(0.01,1,4);
Cs2_2D=[C_s2;zeros(1,size(C_s2,2))];         %add extra string
changzhou=0;    %1 changzhou, 0 duanzhou 
if changzhou                
Cs2_2D(end,[81,101])=[-1,1];
else
Cs2_2D(end,[91,111])=[-1,1];
end
for i=1:numel(j)
    [~, num]=min(abs(j(i)-lmd_his)); %find the num of increasment step
    N_out_temp=reshape(n_t1(:,num),3,[]);
    name1=['牵引索长度为',num2str(j(i)*10),'m'];
    N_disp=sqrt(sum((N2-N_out_temp).^2,1));
    tenseg_plot(N_out_temp,C_b2,Cs2_2D,[],[],[]);
% tenseg_plot(N_out_temp,[],C2,[],[],[],name1);

    S =  N_out_temp*C_s2(end-80+1:end,:)';
    string_start_nodes = zeros(3,size(S,2));
    string_end_nodes = zeros(3,size(S,2));
    for k = 1:size(S,2)
        string_start_nodes(:,k) =  N_out_temp(:,C_s2(end-80+k,:)==-1);
        string_end_nodes(:,k) =  N_out_temp(:,C_s2(end-80+k,:)==1);
    end
    hold on
    quiver3(string_start_nodes(1,:),string_start_nodes(2,:),string_start_nodes(3,:),S(1,:),S(2,:),S(3,:),'green.','Autoscale','off','LineWidth',2);
    
    set(gcf, 'Position', [10 10 800 600]);
    %  axis off;
    if changzhou
        axis([-1,1,-100,100,10,60]);
    view([90,0]);
    else
        axis([-100,100,-1,1,10,60]);
    view([0,0]);
    end
grid on
end



%% make video of the dynamic
name=['cable_trus_r'];
 j=linspace(0.01,1,50);
 num=zeros(50,1);
for i=1:numel(j)
   [~, num(i)]=min(abs(j(i)-lmd_his)); %find the num of increasment step
end
tenseg_video_CTS(flip(n_t1(:,num),2),C2,1:nb,S2,[],[],[],[],[],[],t_t1(:,num),[],numel(j),10,name,savevideo)

%% save output data
if savedata==1
    save (['cable_truss_prestress','.mat']);
end


return;

%% dynamics relaxation method

%dynamic analysis set
dt=1e-4;               % time step in dynamic simulation
auto_dt=1;              % use(1 or 0) auto time step, converengency is guaranteed if used
tf=10;                   % final time of dynamic simulation
out_dt=1e-4;            % output data interval(approximately, not exatly)

% time step
if auto_dt
dt=pi/(8*max(omega)); 	% time step dt is 1/8 of the smallest period, guarantee convergence in solving ODE
end
tspan=0:dt:tf;
tspan1=0:dt:tf/2;
out_tspan=interp1(tspan,tspan,0:out_dt:tf, 'nearest','extrap');  % output data time span


% give initial speed of free coordinates
n0a_d=zeros(numel(a2),1);                    %initial speed in X direction

data.n0a_d=n0a_d;        %initial speed of free coordinates
data.M=M;data.D=D2;
data.rho=rho_s;
data.tf=tf;data.dt=dt;data.tspan=tspan;data.out_tspan=out_tspan;
data.dnb_t=dnb_t; data.dnb_d_t=dnb_d_t;  data.dnb_dd_t=dnb_dd_t; % forced movement of pinned nodes

data_out1=DR_solver_CTS(data);


t_t1=data_out1.t_out;          %member force in every step
n_t1=data_out1.n_out;          %nodal coordinate in every step
N_out1=data_out1.N_out;
t_c_t=pinv(S2')*t_t1;
%% plot member force 
tenseg_plot_result(1:substep,t_c_t,{'ODC', 'IDC', 'HC'},{'Substep','Force / N'},'plot_member_force.png',saveimg);

%% Plot nodal coordinate curve X Y
tenseg_plot_result(1:substep,n_t1([3*3-2],:),{'3X'},{'Substep','Coordinate /m)'},'plot_coordinate.png',saveimg);

%% Plot final configuration
% tenseg_plot_catenary( reshape(n_t(:,end),3,[]),C_b,C_s,[],[],[0,0],[],[],l0_ct(index_s,end))
% tenseg_plot( reshape(n_t(:,end),3,[]),C_b,C_s,[],[],[])
 j=linspace(0.01,1,3);
for i=1:3
    num=ceil(j(i)*size(n_t1,2));
%  tenseg_plot( reshape(n_t(:,num),3,[]),C_b,C_s,[],[],[]);
tenseg_plot_CTS2(reshape(n_t1(:,num),3,[]),C2,1:nb,S2,[],[],[],[],[],t_t1(:,num));
 axis off;
end



