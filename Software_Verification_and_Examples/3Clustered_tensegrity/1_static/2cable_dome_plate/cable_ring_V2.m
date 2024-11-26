function [V2] = cable_ring_V2(gr_num,V2_1,V2_2,p)
%UNTITLED2 此处显示有关此函数的摘要
%   此处显示详细说明
switch gr_num
    case 1
        V2=[V2_1;V2_1;V2_2];  % 上，下，环；一组
        
    case 2

        V2=[V2_2*ones(1,2),V2_1*ones(1,4)]';
        
    case 3

        V2=[V2_2*ones(1,3),V2_1*ones(1,6)]';
        
    case 4

        V2=[V2_1*ones(1,4),V2_2*ones(1,4),V2_1*ones(1,4)]';
        
    case 6
         
        V2=[V2_1*ones(1,12),V2_2*ones(1,6)]';
        
    case 12
        
        V2=[V2_1*ones(1,2*p),V2_1*ones(1,2*p),V2_2*ones(1,p)]';

end
end

