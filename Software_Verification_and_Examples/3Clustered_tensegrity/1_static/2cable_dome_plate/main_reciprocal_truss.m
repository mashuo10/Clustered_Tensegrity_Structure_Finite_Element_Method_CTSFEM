%A cable supported reciprocal truss retractable roof%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% /* This Source Code Form is subject to the terms of the Mozilla Public
% * License, v. 2.0. If a copy of the MPL was not distributed with this
% * file, You can obtain one at http://mozilla.org/MPL/2.0/.
%
% [1] structure design(calculate equilibrium matrix,
% group matrix,prestress mode, minimal mass design)
% [2] modal analysis(calculate tangent stiffness matrix, material
% stiffness, geometry stiffness, generalized eigenvalue analysis)
% [3] dynamic simulation

%EXAMPLE
clc; clear all; close all;
% Global variable
[consti_data,Eb,Es,sigmab,sigmas,rho_b,rho_s]=material_lib('Steel_Q345','Steel_string');
material{1}='linear_elastic'; % index for material properties: multielastic, plastic.
material{2}=1; % index for considering slack of string (1) for yes,(0) for no (for compare with ANSYS)

% cross section design cofficient
thick=6e-3;        % thickness of hollow bar
hollow_solid=0;          % use hollow bar or solid bar in minimal mass design (1)hollow (0)solid
c_b=0.1;           % coefficient of safty of bars 0.5
c_s=0.1;           % coefficient of safty of strings 0.3

% dynamic analysis set
amplitude=50;            % amplitude of external force of ground motion 
period=0.5;             %period of seismic

dt=0.001;               % time step in dynamic simulation
auto_dt=0;              % use(1 or 0) auto time step, converengency is guaranteed if used
tf=2;                   % final time of dynamic simulation
out_dt=0.02;            % output data interval(approximately, not exatly)
lumped=0;               % use lumped matrix 1-yes,0-no
saveimg=0;              % save image or not (1) yes (0)no
savedata=1;             % save data or not (1) yes (0)no
savevideo=1;            % make video(1) or not(0)
gravity=0;              % consider gravity 1 for yes, 0 for no
% move_ground=0;          % for earthquake, use pinned nodes motion(1) or add inertia force in free node(0) 

%% N C of the structure
% Manually specify node positions of double layer prism.
R=50;          %radius
p=6;          %complexity for cable dome
H=25;
high=0.08*R;    % height of the truss
rate=0.5;
explod_rate=0*[1.37,1.4];      % explode rate(1) for plate,(2)for cable
[N0,C_b,n_qp] =N_plate_truss(R,rate,p,high,H+0.3+2*explod_rate(1)*H);

tenseg_plot(N0,C_b,[]);
fig_handle=figure;
num_node=size(N0,2)/2/p;
% tenseg_plot_RBD(N0,C_b,[],fig_handle,[],[],[],[],n_qp);
tenseg_plot_RBD(N0,C_b,[],fig_handle,1:num_node:size(N0,2),[],[],[],n_qp);
axis off
view([0,25]);

if 1
% plot shell
dx=0.1; %ration for circular part
max_rds=1*sqrt(2+2*sin(pi/p));            %maximum radius for track
x=[linspace(R*cos(pi/p),max_rds*R,10),max_rds*R+dx*R*sin(linspace(0,2*pi/3,6)),linspace(max_rds*R+dx*R*sin(2*pi/3),0.9*max_rds*R,5)];
z=[linspace(H,H,10),H-dx*R+dx*R*cos(linspace(0,2*pi/3,6)),linspace(H-dx*R+dx*R*cos(2*pi/3),0,5)]';
r=x;
th=linspace(0,2*pi,100);
[RR,Th]=meshgrid(r,th);
[X,Y] = pol2cart(Th,RR);
Z=(z*ones(size(th)))';


ss=surf(X,Y,Z,'FaceAlpha',0.9);
% ss.EdgeColor = 'none';
ss.FaceColor=	'#EDB120';
ss.FaceLighting='gouraud';

view(3);
% surf(X,Y,Z)

%plot cable dome
h=1e-10*2*R;   %hight of the dome
m=1;   %number of circle of the vertical bars
beta=15*pi/180*ones(m,1);    %all angle of diagonal string
% [N,C_b,C_s,C] =generat_cable_dome(R,p,m,h,beta);

[N2,C_b2,C_s2,C2] =N_cable_dome_plate(R,rate,p,m,h,beta);
N2(3,:)=N2(3,:)+H+explod_rate(2)*H;
tenseg_plot(N2,C_b2,C_s2,fig_handle);
axis off

% plot track
alpha=pi/p;
N_r=zeros(3,2*p);
N_r(:,1:2)=  [-(R*(1+sin(alpha))) -R* cos(alpha) H;
        R*sin(alpha) -R*cos(alpha) H]';
T1=[cos(2*alpha) -sin(2*alpha) 0
    sin(2*alpha) cos(2*alpha) 0
    0 0 1];
for i=2:p
N_r(:,2*i-1:2*i)=T1^(i-1)*N_r(:,1:2);
end

C_b_in_r = [1:2:2*p-1;2:2:2*p]';  % Bar 
% C_b_r = tenseg_ind2C(C_b_in_r,N_r);
% tenseg_plot_dash(N_r,C_b_r,[],fig_handle);
line([N_r(1,C_b_in_r(:,1));N_r(1,C_b_in_r(:,2))],[N_r(2,C_b_in_r(:,1));N_r(2,C_b_in_r(:,2))],...
    [N_r(3,C_b_in_r(:,1));N_r(3,C_b_in_r(:,2))],'Linestyle','-.','Color','b','LineWidth',2)

axis off
view([0,16]);
end
%% different deployment rate
 
num=3;
rate_n=linspace(0,0.9,num);
for i=1:num
    rate=rate_n(i);

[N0,C_b,n_qp] =N_plate_truss(R,rate,p,high,H+0.3+2*explod_rate(1)*H);
[N2,C_b2,C_s2,C2] =N_cable_dome_plate(R,rate,p,m,h,beta);   % coordinate of cable dome
N2(3,:)=N2(3,:)+H;

if 0    %if split the structure in different level
    dh=1.5;
N2(3,:)=N2(3,:)+dh*H;        % enhance cable dome
[N0,C_b,n_qp] =N_plate(R,rate,p,H+2*dh*H);     %nodal coordinate of plate
end

% tenseg_plot(N0,C_b,[]);
fig_handle=figure;
hold on;

if 1  %PLOT SHELL
ss=surf(X,Y,Z,'FaceAlpha',0.9);
% ss.EdgeColor = 'none';
ss.FaceColor=	'#EDB120';  %yellow
% ss.FaceColor=	'#808080';%brown
ss.FaceAlpha=0.8;
end

if 1    %plot cable dome
tenseg_plot(N2,C_b2,C_s2,fig_handle);

end

if 1    %plot plate
    tenseg_plot_RBD(N0,C_b,[],fig_handle,1:num_node:size(N0,2),[],[],[],n_qp);

end

axis off
axis(100*[-1 1 -1 1]);
% view(3);
% view(0,17);% to plot split pic
view(0,35);

% plot railway

line([N_r(1,C_b_in_r(:,1));N_r(1,C_b_in_r(:,2))],[N_r(2,C_b_in_r(:,1));N_r(2,C_b_in_r(:,2))],...
    [N_r(3,C_b_in_r(:,1));N_r(3,C_b_in_r(:,2))],'Linestyle',':','Color','b','LineWidth',3)

fig_handle.Position = [0 0 900 600];
end


return

%%




name=['plate_trajectory'];
% % tenseg_video(n_t,C_b,C_s,[],min(substep,50),name,savevideo,R3Ddata);
% tenseg_video_slack(n_t,C_b,C_s,l0_ct,index_s,[],[],[],min(substep,50),name,savevideo,material{2})
tenseg_video(n_t,C_b,C_s,[],50,name,savevideo,material{2});
