function data_out=static_solver_CTS(data)
%solve nonlinear equilibrium equations using modified Newton method
%converge to stable equilibrium, considering substep, for CTS( including
%TTS)

global E A l0 Ia Ib C S w ne Xb Xa dXa f_int l_int
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
        dXb_t=data.dnb_t;
    elseif size(data.dnb_t,2)==1
        dXb_t=data.dnb_t*linspace(0,1,substep);
    end
else
    dXb_t=linspace(0,0,substep);
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

delta_w=w_t(:,end)-w_t(:,1);                %total load
delta_nb=dXb_t(:,end)-dXb_t(:,1);
delta_l0=l0_t(:,end)-l0_t(:,1);

% F=w_t(:,end)



X0=data.N(:);
data_out=data;     %initialize output data
data_out.E_out=E0*ones(1,substep);


%% calculate equilibrium
X=X0;               %initialize configuration
Xb0=Ib'*X;           %pinned node
E=E0;
% lamda=linspace(0,1,substep);    %coefficient for substep
num_slack=ne*zeros(substep+1,1);    %num of string slack
Xa0=Ia'*X;
Xa=Xa0;
cont=2;
 u=1e-1;
 tol = 1e-6; MaxIter = 30; 


 MaxIcr = data.MaxIcr;                   
    b_lambda = data.InitialLoadFactor;          
    Xa_his = zeros(numel(Xa),MaxIcr);  
    Xa_his(:,1)=Xa; 
    FreeDofs = find(sum(Ia,2));
    lmd = 0; icrm = 0; MUL = [Xa,Xa];
    lmd_his = zeros(MaxIcr,1);
    data_out.t_out=zeros(ne,MaxIcr);        %output member force
    data_out.l_out=zeros(ne,MaxIcr);                % member length
    data_out.n_out=zeros(numel(X),MaxIcr);                % nodal coordinate
    data_out.lmd_his=lmd_his ;

    while icrm<MaxIcr && lmd<=1 
        icrm = icrm+1;
        iter = 0; err = 1;
        fprintf('icrm = %d, lambda = %6.4f\n',icrm,lmd);
 
        w=w_t(:,1)+delta_w;
        Xb=Xb0+lmd*delta_nb;         %forced node displacement
        l0=l0_t(:,1)+lmd*delta_l0;         %forced enlongation of string

        while err>tol && iter<MaxIter
            iter = iter+1;
%% equilibrium & tangent stiffness matrix

            X=[Ia';Ib']\[Xa;Xb];
            l=sqrt(sum((reshape(X,3,[])*C').^2))'; %bar length
            l_c=S*l;

            % member force (of truss)
            %         q=E.*A.*(1./l0-1./l);      %force density
            strain=(l_c-l0)./l0;        %strain of member
            [E,sigma]=stress_strain(consti_data,index_b,index_s,strain,material);
            t_c=sigma.*A;         %member force
            t=S'*t_c;
            q_c=t_c./l_c;
            q=t./l;      %reculate force density
            q_bar=diag(q);

            K=kron(C'*q_bar*C,eye(3));                      %stiffness matrix
            Fp=w-K*X;                                       %unbalanced force
            Fp_a=Ia'*Fp;                                 %see the norm of unbalanced force
            F_norm=norm(Fp_a);
            disp(F_norm)
            if norm(Fp_a)<1e-4
                break
            end
            N=reshape(X,3,[]);
            H=N*C';
            Cell_H=mat2cell(H,3,ones(1,size(H,2)));          % transfer matrix H into a cell: Cell_H

            A_2a=Ia'*kron(C',eye(3))*blkdiag(Cell_H{:})*diag(l.^-1);     % equilibrium matrix
            A_2ac=A_2a*S';
            % tangent stiffness matrix
            Kg_aa=Ia'*K*Ia-A_2a*q_bar*A_2a';
            Ke_aa=A_2ac*diag(E.*A./l0)*A_2ac';
            K_taa=Kg_aa+(Ke_aa+Ke_aa')/2;       % this is to


%             [IF,K] = GlobalK_fast_ver(U,Node,truss,angles);
            
        if 1            % modify stiffness matrix or not
        %modify the stiffness matrix
        [V_mode,D]=eig(K_taa);                       %刚度矩阵特征根
        d=diag(D);                            %eigen value
        lmd_d=min(d);                     %刚度矩阵最小特征根
        if lmd_d>0
            Km=K_taa+u*eye(size(K_taa)); %修正的刚度矩阵
        else
            Km=K_taa+(abs(lmd_d)+u)*eye(size(K_taa));
        end
        else
            Km=K_taa;
        end

        dXa=Km\Fp_a;

          x=1;
        % line search
        if (use_energy==1)&(F_norm>1e5)           
            opt=optimset('TolX',1e-5);
            [x,V]=fminbnd(@energy_CTS,0,1e1,opt);
        end
        Xa=Xa+x*dXa;

  
            err = norm(dXa);

            fprintf('    iter = %d, err = %6.4f, dlambda = %6.4f\n',iter,err,b_lambda);
            if err > 1e8, disp('Divergence!'); break; end
        end

        if iter>28|err > 1e8
            b_lambda = b_lambda/2;
            disp('Reduce constraint radius...')
            icrm = icrm-1;
            Xa = Xa_his(:,max(icrm,1));  % restore displacement
            lmd = lmd_his(max(icrm,1));   % restore load
            
        elseif iter<8
            disp('Increase constraint radius...')
            b_lambda = b_lambda*1.5;
            Xa_his(:,icrm) = Xa;
            lmd_his(icrm) = lmd;
            data_out.t_out(:,icrm)=t_c;      %member force
            data_out.l_out(:,icrm)=l_c;      % length of bars
            data_out.n_out(:,icrm)=X;
            
        else
            Xa_his(:,icrm) = Xa;
            lmd_his(icrm) = lmd;
            data_out.t_out(:,icrm)=t_c;      %member force
            data_out.l_out(:,icrm)=l_c;      % length of bars
            data_out.n_out(:,icrm)=X;
            
        end
        lmd=lmd+b_lambda;

    end
        data_out.lmd_his=lmd_his;
end


   








