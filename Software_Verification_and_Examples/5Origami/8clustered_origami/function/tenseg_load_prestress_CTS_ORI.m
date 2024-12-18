function [w_t,l0_t,theta_0t]=tenseg_load_prestress_CTS_ORI(tspan,ind_w,w,type,ind_dl0_c,dl0_c,l0,ind_theta_0,dtheta_0,theta_0,gravity,acc,C,M_p,p)
% /* This Source Code Form is subject to the terms of the Mozilla Public
% * License, v. 2.0. If a copy of the MPL was not distributed with this
% * file, You can obtain one at http://mozilla.org/MPL/2.0/.
%
% This function output the external force, nodal displacement, and
% shrink of rest length
%
% Inputs:
%   tspan: time step in iteration of dynamics
%   ind_w: index of nodal coordinate in external force(a vector);
%   w0: external force vector;
%   ind_l0: index of the element with changing rest length(a vector);
%   dl0: difference of rest length;
%	l0: rest length 
%   gravity:considering gravity or not(1 yes;0 no)
%   acc: acceleration vector:[0;0;9.8]
%	C: connectivity matrix
%   mass: mass vector of all elements
%
% Outputs:
%   w_t: time history of external force
%   l0_t: time history of rest length

substep=numel(tspan);
% p=numel(dl0_c);
% substep_1 = floor(substep/p);
%% external force
%gravity force vector
G=(gravity)*-M_p*kron(ones(size(C,2),1),acc);
%initialize force 
w0=zeros(size(G,1),1); %zero external force
w0(ind_w)=w;  %force exerted on bottom nodes
w_t=zeros(numel(G),substep);

switch type
    case 'impluse'
        w_t(:,find(tspan<0.05))=w0*20;        % impluse load in c_index
        w_t=w_t+G*ones(size(tspan));            % add gravity force
    case 'step'
        w_t=w0*ones(size(tspan));        % load in c_index
        w_t=w_t+G*ones(size(tspan));            % add gravity force
    case 'ramp'
        w_t=w0*linspace(0,1,numel(tspan));  % load in ind_w
        w_t=w_t+G*ones(size(tspan));            % add gravity force
end


% %% nodal displacement
% b_new=sort(unique([b;ind_dn]));
% a_new=setdiff(1:size(G,1),b_new);  %index of free node direction
% I=eye(size(G,1));
% Ia=I(:,a_new);  %free node index
% Ib=I(:,b_new);  %pinned nod index
% dn=zeros(size(b_new));
% 
% d3nn=zeros(3*numel(G),1);
% d3nn(ind_dn)=dn0;
% dn=d3nn(b_new);
% % dn(ind_dn)=dn0;
% dnb_t=dn*linspace(0,1,substep);

%% rest length
%% rest length
if p == 1 

    dl0_i=zeros(size(l0));
    dl0_i(ind_dl0_c)=dl0_c;
    dl0_t=dl0_i*linspace(0,1,substep);
    l0_t=dl0_t+l0*linspace(1,1,substep);

else
    substep=substep_1*p;
    dl0_i=zeros(size(l0));
    
    [ne,nn]=size(l0);
    num_1 = [0];
    l0_t = zeros(ne,substep);
    for i=1:p
    
       dl0_i=zeros(size(l0));
       dl0_i(ind_dl0_c(i)) = dl0_c(:,i);
       dl0_t{i}=dl0_i*linspace(0,1,substep/p);
       l0_t(:,[1+num_1:substep_1+num_1]) = cell2mat(dl0_t(i));
       num_1= num_1+substep_1;
    end
    
     num_3 = [0];
      for i=1:p-1
    
          num_2 = l0_t(ind_dl0_c(i),substep_1+num_3);
          l0_t(ind_dl0_c(i),[substep_1+1+num_3:end])=num_2*ones(1,substep-substep_1-num_3);
         num_3= num_3+substep_1;
      end
    
      l0_t=l0_t+l0*linspace(1,1,substep);
end

%% inital angle
dtheta0_i=zeros(size(theta_0));
dtheta0_i(ind_theta_0)=dtheta_0;
dtheta_0t=dtheta0_i*linspace(0,1,substep);
theta_0t=dtheta_0t+theta_0*linspace(1,1,substep);

end
% 
% dnb_t=zeros(numel(b),substep);     %move boundary nodes
% l0_t=zeros(size(C,1),substep);     %move boundary nodes
% 
% w_t(ind_w,:)=w0*linspace(0,1,substep);  % load in c_index
% w_t=w_t+G*ones(1,substep);            % add gravity force
% 
% 
% 
% dz_a_t=[];
% 
% switch type
%     case 'impluse'
%         w_t(c_index,find(tspan<0.05))=amplitude*20;        % impluse load in c_index
%         w_t=w_t+G*ones(size(tspan));            % add gravity force
%     case 'step'
%         w_t(c_index,:)=amplitude;        % load in c_index
%         w_t=w_t+G*ones(size(tspan));            % add gravity force
%     case 'ramp'
%         w_t(c_index,:)=amplitude*ones(numel(c_index),1)*linspace(0,1,size(tspan,2));  % load in c_index
%         w_t=w_t+G*ones(size(tspan));            % add gravity force
%     
%     
%     
%     
%     
%     case 'vib_force'
%         dz_d_t=-amplitude/(2*pi/period)^2*sin(2*pi/period*tspan);    % displacement of ground motion (time serises)
%         dz_v_t=-amplitude/(2*pi/period)*cos(2*pi/period*tspan);    % velocity of ground motion (time serises)
%         dz_a_t=amplitude*sin(2*pi/period*tspan);    % acceleration of ground motion (time serises)
%         
%         w_0=-0.5*kron(abs(C)'*mass,[1;1;1]*dz_a_t);        %load in c_index
%         w_t(c_index,:)=w_0(c_index,:);
%         w_t=w_t+G*ones(size(tspan));            % add gravity force
% 
%         %    [dz_d_t,dz_v_t,dz_a_t]=ground_motion(amplitude,period,tspan,1,0,0);
%         %    [w_t,dnb_t,dnb_d_t,dnb_dd_t]=tenseg_earthquake(G,C,mass,b,dz_d_t,dz_v_t,dz_a_t,move_ground);
%         
%     case 'vib_nodes'
%         w_t=w_t+G*ones(size(tspan));            % add gravity force
%         % displacement, velocity, acceleration of ground
%         dz_d_t=-amplitude/(2*pi/period)^2*sin(2*pi/period*tspan);    % displacement of ground motion (time serises)
%         dz_v_t=-amplitude/(2*pi/period)*cos(2*pi/period*tspan);    % velocity of ground motion (time serises)
%         dz_a_t=amplitude*sin(2*pi/period*tspan);    % acceleration of ground motion (time serises)
%         
%         dnb_t0=kron(ones(numel(b),1),[1;1;1]*dz_d_t);              %move boundary nodes
%         dnb_d_t0=kron(ones(numel(b),1),[1;1;1]*dz_v_t);    %velocity of moved boundary nodes
%         dnb_dd_t0=kron(ones(numel(b),1),[1;1;1]*dz_a_t);   %acceleration of moved boundary nodes
%         
%         dnb_t(c_index,:)=dnb_t0(c_index,:);
%         dnb_d_t(c_index,:)=dnb_d_t0(c_index,:);
%         dnb_dd_t(c_index,:)=dnb_dd_t0(c_index,:);
%         
% 
%     case 'self_define'
% end


