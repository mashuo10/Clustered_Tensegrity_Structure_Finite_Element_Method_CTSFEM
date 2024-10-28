function [fig_out] = tenseg_plot_CTS( N,C,index_b,S,fig_handle,highlight_nodes,view_vec, PlotTitle, R3Ddata,lb_ele,lb_nod,bound)
% [fig_out] = TENSEG_PLOT( N,C_b,C_s,fig_handle,highlight_nodes,view_vec )
% creates a rough visualization figure for a given tensegrity structure
%
% Inputs:
%	N: node matrix (3 x n array for n nodes)
%	C_b:connectivity matrix
%   S: clustered matrix
%   index_b: index for bar elements
%	fig_handle (optional): figure in which to plot
%	highlight_nodes (optional): node(s) to highlight when plotting (vector
%		containing node numbers)
%	view_vec (optional): plotting view direction (see view())
%   PlotTitle (optional): Title of the plot, default: ''
%   R3Ddata (optional): Structure with the radius of objects for 3D plots
%        [].Bradius: Radius of bars [# bars x 1]
%        [].Sradius: Radius of strings [# strings x 1]
%        [].Nradius: Radius of node spheres [# nodes x 1]
%   lb_ele: label of elements
%   lb_nod: label of nodes
%
% Outputs:
%	fig_out: figure handle in which plot was made
%
% Example:
%	tenseg_plot(N,C_b,C_b)

% Handle optional arguments

%% Object size options (for line plots)
BarWidth = 3; % Width of bar lines
StringWidth = 1.5; % Width of string lines
NodeSize = 12; % Size of node marker

%% Labeling options
% Write labels? (1: show, 0: suppress)
LabelNodes = 1;
LabelEle=0;
Color_CTS=0;   % 1 color by clustered info, 0 color by lb_ele
Label_colorbar=1;  % 1 colorbar, 2 legned, 0 nothing,

FontBars = 15; % Font of bar labels
FontStrings = 10; % Font of string labels
FontNodes = 12; % Font of node labels


FractionDistance = 0.005; % Distance between object and label (relative to overall axis size)

%% 3D plot options
nsurfpatches = 6; % Number of surface patches in 3D plots
BarSurfColor = [0.2, 0.2, 0.6];
StringSurfColor = [0.9, 0.1, 0.1];
LightAmbientStrength = 0.7;% [0,1], 0.3 is Matlab's default

%%
switch nargin
    case 6
        view_vec = [];
    case 5
        view_vec = [];
        highlight_nodes = [];
    case 4
        view_vec = [];
        highlight_nodes = [];
        fig_handle = [];
end

if nargin < 8 % Empty Plot Title
    PlotTitle = '';
end
if nargin < 9 % No 3D object data
    R3Ddata = [];
end
if nargin < 10 % lb_ele
    lb_ele = [];
end
if nargin < 11 % lb_nod
    lb_nod = [];
end
if nargin < 12 % lb_nod
   bound = [];
end
% Get min and max X,Y,Z coordinates contained in Nhist
min_x = min(N(1,:,:)); max_x = max(N(1,:,:));
min_y = min(N(2,:,:)); max_y = max(N(2,:,:));
min_z = min(N(3,:,:)); max_z = max(N(3,:,:));

%%
if ~isempty(R3Ddata) % Extending axis to account for radii of objects
    maxradius = 0;
    if isfield(R3Ddata,'Bradius')
        maxradius = max(maxradius,max(R3Ddata.Bradius));
    end
    if isfield(R3Ddata,'Sradius')
        maxradius = max(maxradius,max(R3Ddata.Sradius));
    end
    if isfield(R3Ddata,'Nradius')
        maxradius = max(maxradius,max(R3Ddata.Nradius));
    end
    min_x = min_x - maxradius; max_x = max_x + maxradius;
    min_y = min_y - maxradius; max_y = max_y + maxradius;
    min_z = min_z - maxradius; max_z = max_z + maxradius;
end

% Get difference between min and max values for each axis
diff_x = max_x-min_x;
diff_y = max_y-min_y;
diff_z = max_z-min_z;

% Get distance for labels
dist_x = FractionDistance * diff_x;
dist_y = FractionDistance * diff_y;
dist_z = FractionDistance * diff_z;
%%
if isempty(lb_ele)
    Color_CTS=1;   % 1 color by clustered info, 0 color by lb_ele
Label_colorbar=0;  % 1 colorbar, 2 legned, 0 nothing,
end
%% Open specified figure or create new one
if isempty(fig_handle)
    fig_out = figure;
else
    fig_out = figure(fig_handle);
end


%% cluster information & color
index_s=setdiff(1:size(C,1),index_b);
C_b=C(index_b,:);
C_s=C(index_s,:);

[n_clu,n_e]=size(S);    %n_clu number of clustered members, n_e No. of elements
for i=1:n_clu
    index_clu{i}=find(S(i,:));
end
vec_bar=zeros(n_e,1);
vec_bar(index_b)=1;
n_clu_b=find(S*vec_bar>0);      % groups corresponding to bars
n_clu_s=find(S*vec_bar==0);
Pplot=[];
if Color_CTS==1
% color_b=[zeros(1,numel(n_clu_b));zeros(1,numel(n_clu_b));linspace(0,1,numel(n_clu_b))]';
color_b=[zeros(1,numel(n_clu_b));zeros(1,numel(n_clu_b));linspace(0,0,numel(n_clu_b))]';
% color_b=winter(numel(n_clu_b));
sort_b=reshape(reshape(1:2*ceil(numel(n_clu_b)/2),[],2)',[],1);%this is to reorder the color
sort_b=sort_b(1:numel(n_clu_b));
color_b=color_b(sort_b,:);
color_s=[linspace(1,0,numel(n_clu_s));linspace(0,1,numel(n_clu_s));linspace(0,0,numel(n_clu_s))]';  %pink
% color_s=[linspace(1,0,numel(n_clu_s));linspace(0,1,numel(n_clu_s));linspace(0,0,numel(n_clu_s))]';
% color_s=autumn(numel(n_clu_s));

sort_s=reshape(reshape(1:2*ceil(numel(n_clu_s)/2),[],2)',[],1);%this is to reorder the color
sort_s=sort_s(1:numel(n_clu_s));
color_s=color_s(sort_s,:);
else
         lb_ele_clu=diag(sum(S,2))\S*lb_ele;
    [SIGMA_1,Ib]=sort(lb_ele_clu);
[SIGMA_2,Ia,Ia2]=uniquetol(SIGMA_1,1e-1);
% cc=jet(length(SIGMA));
cc=jet(100);%    color vector
color_clu=zeros(size(S,1),3);
if isempty(bound)
    bound=[min(lb_ele_clu),max(lb_ele_clu)];
end
for k=1:length(SIGMA_2)
    l = find(lb_ele_clu==SIGMA_2(k));
     cc2=1+ceil((SIGMA_2(k)-min(bound))/(max(bound)-min(bound))*99);
    
l=Ib(find(Ia2==k));
    
    color_clu(l,:)=ones(length(l),1)*cc(cc2,:);

end
    color_b=color_clu(n_clu_b,:);
    color_s=color_clu(n_clu_s,:);    
end
% index_b=ismember(C_b,C,'row')

% %% plot bars and strings
% for k=1:length(SIGMA)
%     l = find(lb_ele_clu==SIGMA(k));
%        
%     color_clu(l,:)=ones(length(l),1)*cc(k,:);
%     for i=l(1):l(end)
%         for j=1:numel(index_clu{n_clu_b(i)})
%                 nod1=find(C(index_clu{n_clu_b(i)}(j),:)==-1);
%                 nod2=find(C(index_clu{n_clu_b(i)}(j),:)==1);
%                 PP=line([N(1,nod1),N(1,nod2)],[N(2,nod1),N(2,nod2)],[N(3,nod1),N(3,nod2)],'linewidth',BarWidth,'color',color_b(i,:),'linestyle','-');
%                 hold on
%             end
%              Pplot=[Pplot,PP]; 
%                AA{1,i}=[num2str(lb_ele_clu(n_clu_b(i)),4)];
%     end
% 
% end





%% plot bars
if ~isempty(C_b)
    Hb = N*C_b';
bar_start_nodes = zeros(3,size(Hb,2));
bar_end_nodes = zeros(3,size(Hb,2));
for j = 1:size(Hb,2)
    bar_start_nodes(:,j) = N(:,C_b(j,:)==-1);
    bar_end_nodes(:,j) = N(:,C_b(j,:)==1);
end
    if ~isempty(R3Ddata) && isfield(R3Ddata,'Bradius') % 3D plot
        for j = 1:size(C_b,1)
            [LatFace, UpFace, DwFace] = PlotCylinderObject(bar_start_nodes(:,j),bar_end_nodes(:,j),...
                R3Ddata.Bradius(j),nsurfpatches); % Lateral, upper, and down surface of cylinder representing a bar
            
            surf(LatFace.x, LatFace.y, LatFace.z, 'MeshStyle','row','FaceColor',BarSurfColor, ...
                'FaceLighting','gouraud', 'AmbientStrength',LightAmbientStrength);
            hold on
            surf(UpFace.x, UpFace.y, UpFace.z, 'MeshStyle','row','FaceColor',BarSurfColor, ...
                'FaceLighting','gouraud', 'AmbientStrength',LightAmbientStrength);
            hold on
            surf(DwFace.x, DwFace.y, DwFace.z, 'MeshStyle','row','FaceColor',BarSurfColor, ...
                'FaceLighting','gouraud', 'AmbientStrength',LightAmbientStrength);
            hold on
        end
    else
        for i=1:numel(n_clu_b)
            for j=1:numel(index_clu{n_clu_b(i)})
                nod1=find(C(index_clu{n_clu_b(i)}(j),:)==-1);
                nod2=find(C(index_clu{n_clu_b(i)}(j),:)==1);
                PP=line([N(1,nod1),N(1,nod2)],[N(2,nod1),N(2,nod2)],[N(3,nod1),N(3,nod2)],'linewidth',BarWidth,'color',color_b(i,:),'linestyle','-');
                hold on
            end
             Pplot=[Pplot,PP]; 

        end
    end
end



%% plot strings
if ~isempty(C_s)
    Hs = N*C_s';
string_start_nodes = zeros(3,size(Hs,2));
string_end_nodes = zeros(3,size(Hs,2));
for j = 1:size(Hs,2)
    string_start_nodes(:,j) = N(:,C_s(j,:)==-1);
    string_end_nodes(:,j) = N(:,C_s(j,:)==1);
end
    if ~isempty(R3Ddata) && isfield(R3Ddata,'Sradius') % 3D plot
        for j = 1:size(C_s,1)
            [LatFace, UpFace, DwFace] = PlotCylinderObject(string_start_nodes(:,j),string_end_nodes(:,j),...
                R3Ddata.Sradius(j),nsurfpatches); % Lateral, upper, and down surface of cylinder representing a string
            
            surf(LatFace.x, LatFace.y, LatFace.z, 'MeshStyle','row','FaceColor',StringSurfColor, ...
                'FaceLighting','gouraud', 'AmbientStrength',LightAmbientStrength);
            hold on
            surf(UpFace.x, UpFace.y, UpFace.z, 'MeshStyle','row','FaceColor',StringSurfColor, ...
                'FaceLighting','gouraud', 'AmbientStrength',LightAmbientStrength);
            hold on
            surf(DwFace.x, DwFace.y, DwFace.z, 'MeshStyle','row','FaceColor',StringSurfColor, ...
                'FaceLighting','gouraud', 'AmbientStrength',LightAmbientStrength);
            hold on
        end
    else
        for i=1:numel(n_clu_s)
            for j=1:numel(index_clu{n_clu_s(i)})
                nod1=find(C(index_clu{n_clu_s(i)}(j),:)==-1);
                nod2=find(C(index_clu{n_clu_s(i)}(j),:)==1);
                PP=line([N(1,nod1),N(1,nod2)],[N(2,nod1),N(2,nod2)],[N(3,nod1),N(3,nod2)],'linewidth',StringWidth,'color',color_s(i,:),'linestyle','-');
                hold on
            end
                         Pplot=[Pplot,PP]; 

        end
    end
end

%% legend
switch Label_colorbar
    case 2
if Color_CTS==0    % plot element information
[~,Ic]=sort([n_clu_b;n_clu_s]);
Pplot_c=Pplot(Ic);
Pplot_b=Pplot_c(Ib);
Pplot_a=Pplot_b(Ia);

label=cell(1,length(SIGMA_2));
for i=1:length(SIGMA_2)
    label{1,i}=num2str(SIGMA_2(i),2);
end
legend(Pplot_a,label,'AutoUpdate','off');
end
%% colorbar
    case 1
uu=linspace(min(bound),max(bound),8);
ticklabel=cell(1,8);
for i=1:8
    ticklabel{i}=num2str(uu(i),2);
end
colorbar('Ticks',linspace(0,1,8),...
    'TickLabels',ticklabel,'fontsize',12);
colormap jet;
end
%% lable element number and value

if LabelEle == 1
    for i=1:n_e
        if ~isempty(lb_ele)
            text(mean(N(1,find(C(i,:))))+ dist_x, ...
                mean(N(2,find(C(i,:))))+ dist_y, ...
                mean(N(3,find(C(i,:))))+ dist_z, ...
                [num2str(lb_ele(i),2)], 'FontSize',FontBars, 'Color', 'k')
        else
            text(mean(N(1,find(C(i,:))))+ dist_x, ...
                mean(N(2,find(C(i,:))))+ dist_y, ...
                mean(N(3,find(C(i,:))))+ dist_z, ...
                num2str(i), 'FontSize',FontBars, 'Color', 'k')
        end
    end
end







%% Plot node markers
if ~isempty(R3Ddata) && isfield(R3Ddata,'Nradius') % 3D plot
    for i = 1:size(N,2)
        [xnod,ynod,znod] = sphere(nsurfpatches);
        xnod = R3Ddata.Nradius(i)*xnod + N(1,i);
        ynod = R3Ddata.Nradius(i)*ynod + N(2,i);
        znod = R3Ddata.Nradius(i)*znod + N(3,i);
        surf(xnod,ynod,znod,...
            'FaceColor','k','EdgeColor','none','FaceLighting','gouraud',...
            'AmbientStrength',LightAmbientStrength);
        hold on
    end
else % Normal plot
    plot3(N(1,:),N(2,:),N(3,:),'black.','MarkerSize',NodeSize)
    axis equal
end

% Write node labels
if LabelNodes == 1
    for i = 1:size(N,2)
        text(N(1,i) + dist_x, N(2,i) + dist_y, N(3,i) + dist_z,...
            num2str(i), 'FontSize', FontNodes, 'Color', 'b')
    end
end

% Highlight specified nodes if applicable
for j=1:numel(highlight_nodes)
    node_index = highlight_nodes(j);
    plot3(N(1,node_index),N(2,node_index),N(3,node_index),'rd','MarkerSize',8,'MarkerFaceColor','black')
    axis equal
end

% Modify plot display
grid off
axis equal
if isempty(view_vec)
    [~, view_vec_derived] = tenseg_axisview(N,R3Ddata);
    view_vec = view_vec_derived;
end
view(view_vec)

xlabel('x')
ylabel('y')
zlabel('z')
title(PlotTitle)
set(gca,'fontsize', 15,'linewidth',1.15)
set(gca,'ticklength',1.2*get(gca,'ticklength'))

if ~isempty(R3Ddata) % 3D object changes
    camlight('headlight');
    camproj('perspective');
end