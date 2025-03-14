function tenseg_video_CTS(n_t,C,index_b,S,axislim,fig_handle,highlight_nodes,view_vec, PlotTitle, R3Ddata,lb_ele_t,lb_nod_t,num_pic,time,name,savevideo)
% /* This Source Code Form is subject to the terms of the Mozilla Public
% * License, v. 2.0. If a copy of the MPL was not distributed with this
% * file, You can obtain one at http://mozilla.org/MPL/2.0/.
%
% This function make video for dynamic simulation.
%
% Inputs:
%   n_t: time history of tensegrity configuration
%   axislim: limit of axis, like [-20,20,-15,15,0,70], or use []
%   num: number of pictures in the video
%   name: name of the file
%   savevideo: save video or not
% Outputs:
%	make the video
%%

% automatic axislim
if isempty(axislim)
    %     axislim=='auto'
    nx_t=n_t(1:3:end); min_x=min(min(nx_t));max_x=max(max(nx_t));d_x=max_x-min_x+5e-2;
    ny_t=n_t(2:3:end); min_y=min(min(ny_t));max_y=max(max(ny_t));d_y=max_y-min_y+5e-2;
    nz_t=n_t(3:3:end); min_z=min(min(nz_t));max_z=max(max(nz_t));d_z=max_z-min_z+5e-1;
    
    d=max(max(n_t))-min(min(n_t));
    axislim=[min_x-0.2*d_x,max_x+0.2*d_x,min_y-0.2*d_y,max_y+0.2*d_y,min_z-0.2*d_z,max_z+0.2*d_z];
end


if savevideo==1
    figure(99);
    %     set(gcf,'Position',get(0,'ScreenSize')); % full screen
    for p = 1:floor(size(n_t,2)/num_pic):size(n_t,2)
        N = reshape(n_t(:,p),3,[]);
        lb_ele=lb_ele_t(:,p);
        if isempty(lb_nod_t)
            lb_nod=[];
        else
        lb_nod=lb_nod_t(:,p);
        end

%         tenseg_plot_CTS(N,C,index_b,S,99,highlight_nodes,view_vec, PlotTitle, R3Ddata,lb_ele,lb_nod,[min(lb_ele_t),max(lb_ele_t)]);
tenseg_plot_CTS2(N,C,index_b,S,99,highlight_nodes,view_vec, PlotTitle, R3Ddata,lb_ele,lb_nod,[min(lb_ele_t),max(lb_ele_t)]);
        set(gcf,'color','w');
        axis(axislim);
        tenseg_savegif_forever_CTS(name,time/num_pic);
       clf;
    end
    close
end
end
