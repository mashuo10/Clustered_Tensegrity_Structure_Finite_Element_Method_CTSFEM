function data_out=DR_solver_CTS(data)
%solve nonlinear equilibrium equations using modified Newton method
%converge to stable equilibrium, considering substep, for CTS( including
%TTS)

global E A l0 Ia Ib C S w ne nb na dXa f_int l_int
% minimize total energy? (1: use, 0: not use) it's time consuming
use_energy=1;

%% input data
C=data.C;
ne=data.ne;
Ia=data.Ia;
Ib=data.Ib;
S=data.S;
index_b=data.index_b;
index_s=data.index_s;
substep=data.substep;
E0=data.E;
consti_data=data.consti_data;
material=data.material;
A=data.A;
% w0=data.w;
if  isfield(data,'w_t')
    if size(data.w_t,2)==substep
        w_t=data.w_t;
    elseif size(data.w_t,2)==1
        w_t=data.w_t*linspace(0,1,substep);
    end
else
    w_t=linspace(0,0,substep);
end

% dXb=data.dXb;
if  isfield(data,'dnb_t')
    if size(data.dnb_t,2)==substep
        dnb_t=data.dnb_t;
    elseif size(data.dnb_t,2)==1
        dnb_t=data.dnb_t*linspace(0,1,substep);
    end
else
    dnb_t=linspace(0,0,substep);
end
% l0_0=data.l0;
if size(data.l0_t,2)==substep
    l0_t=data.l0_t;
elseif size(data.l0_t,2)==1
    l0_t=data.l0_t*linspace(1,1,substep);
end

if  isfield(data,'subsubstep')
    subsubstep=data.subsubstep;
else
    subsubstep=30;          %default ssubstep
end

n0=data.N(:);
data_out=data;     %initialize output data
data_out.E_out=E0*ones(1,substep);
%% Dynamics relaxation method 
global l l_c f f_c n f_int l_int n_d stress strain  l0_c

Ia=data.Ia;
n0=data.N(:);
tspan=data.tspan;
n0a_d=data.n0a_d;    %initial speed of free coordinates
M=data.M;

%% dynamic iteration

% initial value
n0a=Ia'*n0;
% n0a_d=zeros(size(n0a));
Y0a=[n0a;n0a_d];


neq = length(Y0a);
num_tspan = length(tspan);
Y = zeros(neq,num_tspan);
F = zeros(neq,4);


odefun=@tenseg_dyn_x_xdot_CTS;
Y(:,1) = Y0a;
Ene=zeros(num_tspan,1);         % energy in each step
h = diff(tspan);
num_spd0=0;
for i = 2:num_tspan
    ti = tspan(i-1);
    
    hi = h(i-1);
    yi = Y(:,i-1);
    F(:,1) = feval(odefun,ti,yi,data);
    Ene(i-1)=n_d'*M*n_d;
    F(:,2) = feval(odefun,ti+0.5*hi,yi+0.5*hi*F(:,1),data);
    F(:,3) = feval(odefun,ti+0.5*hi,yi+0.5*hi*F(:,2),data);
    F(:,4) = feval(odefun,tspan(i),yi+hi*F(:,3),data);
    Y(:,i) = yi + (hi/6)*(F(:,1) + 2*F(:,2) + 2*F(:,3) + F(:,4));
    if  i>2
        if Ene(i-1)<Ene(i-2)&&Ene(i-1)~=0
        Y(neq/2+1:neq,i)=zeros(neq/2,1);        %set speed equal to zero
        num_spd0=num_spd0+1;
        end
    end
    if num_spd0==3
        break;
    end
end

na_end=Y(1:neq/2,i-1);


%% Statics equilibrium solver

nb0=Ib'*n0;           %pinned node
E=E0;
% lamda=linspace(0,1,substep);    %coefficient for substep
num_slack=ne*zeros(substep+1,1);    %num of string slack
na=na_end;
cont=2;
 u=1e-1;
for k=1:substep
    w=w_t(:,k);               %external force
    nb=nb0+dnb_t(:,k);         %forced node displacement
    l0=l0_t(:,k);         %forced enlongation of string
    disp(k);
   
    
    n=[Ia';Ib']\[na;nb];
    l=sqrt(sum((reshape(n,3,[])*C').^2))'; %bar length
    l_c=S*l;
    strain=(l_c-l0)./l0;        %strain of member
    [E,sigma]=stress_strain(consti_data,index_b,index_s,strain,material);
    t_c=sigma.*A;         %member force
    t=S'*t_c;
    q_c=t_c./l_c;
    q=t./l;      %reculate force density
    
    l_int=l;   f_int=t;
    
    for i=1:1e3
        n=[Ia';Ib']\[na;nb];
        l=sqrt(sum((reshape(n,3,[])*C').^2))'; %bar length
        l_c=S*l;
        %         q=E.*A.*(1./l0-1./l);      %force density
        strain=(l_c-l0)./l0;        %strain of member
        [E,sigma]=stress_strain(consti_data,index_b,index_s,strain,material);
        t_c=sigma.*A;         %member force
        t=S'*t_c;
        q_c=t_c./l_c;
        q=t./l;      %reculate force density
        
        q_bar=diag(q);
        
        K=kron(C'*q_bar*C,eye(3));                      %stiffness matrix
        Fp=w-K*n;                                       %unbalanced force
        Fp_a=Ia'*Fp;                                 %see the norm of unbalanced force
        F_norm=norm(Fp_a);
        disp(F_norm)
        if norm(Fp_a)<1e-5
            break
        end
        N=reshape(n,3,[]);
        H=N*C';
       Cell_H=mat2cell(H,3,ones(1,size(H,2)));          % transfer matrix H into a cell: Cell_H

A_2a=Ia'*kron(C',eye(3))*blkdiag(Cell_H{:})*diag(l.^-1);     % equilibrium matrix
 A_2ac=A_2a*S';    
% tangent stiffness matrix
Kg_aa=Ia'*K*Ia-A_2a*q_bar*A_2a';
Ke_aa=A_2ac*diag(E.*A./l0)*A_2ac';
K_taa=Kg_aa+(Ke_aa+Ke_aa')/2;       % this is to 

        if 1            % modify stiffness matrix or not
        %modify the stiffness matrix
        [V_mode,D]=eig(K_taa);                       %刚度矩阵特征根
        d=diag(D);                            %eigen value
        lmd=min(d);                     %刚度矩阵最小特征根
        if lmd>0
            Km=K_taa+u*eye(size(K_taa)); %修正的刚度矩阵
        else
            Km=K_taa+(abs(lmd)+u)*eye(size(K_taa));
        end
        else
            Km=K_taa;
        end
        dXa=Km\Fp_a;
%          dXa=(lmd*eye(size(Km)))\Fp_a;
        x=1;
        % line search
        if (use_energy==1)&(F_norm>1e4)           
            opt=optimset('TolX',1e-5);
            [x,V]=fminbnd(@energy_CTS,0,1e1,opt);
        end
        na=na+x*dXa;
    end
    %
    %     % change youngs mudulus if string slack
    %     strain=(l-l0)./l0;        %strain of member
    %     [E,stress]=stress_strain(consti_data,index_b,index_s,strain,material);
    % %     [E,sigma]=stress_strain(consti_data,index_b,index_s,strain,slack,plastic);
    %     f=stress.*A;         %member force
    %     q=f./l;      %reculate force density
    %     num_slack(k+1)=numel(find(E==0));
    %        % if string slack, recalculate with more steps
    %     if num_slack(k+1)>num_slack(k)
    %         p_s=k-1;
    %           p_e=k;
    %         [E,f,q] = nonlinear_solver(data,Xb0,w_t,dXb_t,l0_t,data_out.E_out(:,k-1),p_s,p_e,subsubstep,material);
    %     end
    %     num_slack(k+1)=numel(find(E==0));
    %
    %     if min(E)==0
    %         if cont<2
    %             [d_sort,idx]=sort(d);               %sorted eigenvalue
    %             D_sort=diag(d_sort);                  %sorted eigenvalue matrix
    %             V_mode_sort=V_mode(:,idx);              %sorted eigenvector
    %             index_bk=find(d_sort<1e-5);             %index for buckling mode
    %             cont=cont+1;
    %             Xa=Xa+0.0*min(l)*real(mean(V_mode_sort(:,index_bk),2));    %add unstable mode if needed
    %         end
    %     end
    
    
    %     if slack
    %         if sum(q_i(index_s)<1e-6)
    %             index_slack=find(q_i(index_s)<0);
    %             index_string_slack=index_s(index_slack);       %slack stings'number
    %             % change youngs mudulus of slack string E_ss=0
    %             E=E0;
    %             E(index_string_slack)=0;
    %             q=E.*A.*(1./l0-1./l);      %reculate force density
    %             q_bar=diag(q);
    %
    %             %give initial error in coordinate, prevent unstable solution
    %             if cont<3
    %             [d_sort,idx]=sort(d);               %sorted eigenvalue
    %             D_sort=diag(d_sort);                  %sorted eigenvalue matrix
    %             V_mode_sort=V_mode(:,idx);              %sorted eigenvector
    %             index_bk=find(d_sort<1e-5);             %index for buckling mode
    %             cont=cont+1;
    %             end
    %             Xa=Xa+0*min(l)*real(mean(V_mode_sort(:,index_bk),2));    %add unstable mode if needed
    %         else
    %             E=E0;              %use initial young's muldus
    %         end
    %     end
    
    
    %% output data
    
    data_out.N_out{k}=reshape(n,3,[]);
    data_out.n_out(:,k)=n;
    %     data_out.l_out(:,k)=l;
    %     data_out.q_out(:,k)=q;
    %     data_out.E_out(:,k)=E;
    data_out.t_out(:,k)=t;      %member force
    % data_out.V{k}=energy_cal(data_out);
    data_out.Fpn_out(k)=norm(Ia'*Fp);
end
data_out.E=E;
data_out.N=reshape(n,3,[]);









