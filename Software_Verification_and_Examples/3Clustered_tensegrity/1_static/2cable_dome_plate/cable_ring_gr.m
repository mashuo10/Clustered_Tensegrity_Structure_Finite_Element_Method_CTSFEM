function [gr] = cable_ring_gr(gr_num,p)
%UNTITLED 此处显示有关此函数的摘要
%   此处显示详细说明
%   用于创建单各环的组数


switch gr_num
    case 1
        gr={[2*p+1:4*p]';[1:2*p]';[4*p+1:5*p]'};  % 上，下，环；一组

%         gr={[4*p+1:5*p]';[2*p+1:4*p]';[1:2*p]'};  % 上，下，环；一组
    case 2
        gr_fz=[zeros(1,p/2),p/2*ones(1,p/2)];  %赋值
        gr_xxs1=[kron(ones(1,p/2),[1,2])+2*kron(0:(p-1)/2,[1,1])+gr_fz]';
        gr_xxs2=[kron(ones(1,p/2),[p/2+1,p/2+2])+2*kron(0:(p-1)/2,[1,1])+gr_fz]';
        gr_sxs1=[kron(ones(1,p/2),[2*p+1,2*p+2])+2*kron(0:(p-1)/2,[1,1])+gr_fz]';
        gr_sxs2=[kron(ones(1,p/2),[5*p/2+1,5*p/2+2])+2*kron(0:(p-1)/2,[1,1])+gr_fz]';
        gr_hs1= [4*p+1:9*p/2]';
        gr_hs2=[9*p/2+1:5*p]';
        gr={gr_xxs1;gr_xxs2;gr_sxs1;gr_sxs2;gr_hs1;gr_hs2}; %两组; 环，上，下
        
    case 3
        gr_fz=[zeros(1,p/3),2*p/3*ones(1,p/3)];  %赋值
        gr_xxs1=[kron(ones(1,p/3),[1,2])+2*kron(0:(p-1)/3,[1,1])+gr_fz]';
        gr_xxs2=[kron(ones(1,p/3),[p/3+1,p/3+2])+2*kron(0:(p-1)/3,[1,1])+gr_fz]';
        gr_xxs3=[kron(ones(1,p/3),[2*p/3+1,2*p/3+2])+2*kron(0:(p-1)/3,[1,1])+gr_fz]';
        gr_sxs1=[kron(ones(1,p/3),[2*p+1,2*p+2])+2*kron(0:(p-1)/3,[1,1])+gr_fz]';
        gr_sxs2=[kron(ones(1,p/3),[7*p/3+1,7*p/3+2])+2*kron(0:(p-1)/3,[1,1])+gr_fz]';
        gr_sxs3=[kron(ones(1,p/3),[8*p/3+1,8*p/3+2])+2*kron(0:(p-1)/3,[1,1])+gr_fz]';
        gr_hs1= [4*p+1:13*p/3]';
        gr_hs2=[13*p/3+1:14*p/3]';
        gr_hs3=[14*p/3+1:5*p]';
        gr={gr_hs1;gr_hs2;gr_hs3;gr_sxs1;gr_sxs2;gr_sxs3;gr_xxs1;gr_xxs2;gr_xxs3}; %三组;环，上, 下，

    case 4
        gr_fz=[zeros(1,p/4),3*p/4*ones(1,p/4)];  %赋值
        gr_xxs1=[kron(ones(1,p/4),[1,2])+2*kron(0:(p-1)/4,[1,1])+gr_fz]';
        gr_xxs2=[kron(ones(1,p/4),[p/4+1,p/4+2])+2*kron(0:(p-1)/4,[1,1])+gr_fz]';
        gr_xxs3=[kron(ones(1,p/4),[2*p/4+1,2*p/4+2])+2*kron(0:(p-1)/4,[1,1])+gr_fz]';
        gr_xxs4=[kron(ones(1,p/4),[3*p/4+1,3*p/4+2])+2*kron(0:(p-1)/4,[1,1])+gr_fz]';
        gr_sxs1=[kron(ones(1,p/4),[2*p+1,2*p+2])+2*kron(0:(p-1)/4,[1,1])+gr_fz]';
        gr_sxs2=[kron(ones(1,p/4),[9*p/4+1,9*p/4+2])+2*kron(0:(p-1)/4,[1,1])+gr_fz]';
        gr_sxs3=[kron(ones(1,p/4),[10*p/4+1,10*p/4+2])+2*kron(0:(p-1)/4,[1,1])+gr_fz]';
        gr_sxs4=[kron(ones(1,p/4),[11*p/4+1,11*p/4+2])+2*kron(0:(p-1)/4,[1,1])+gr_fz]';
        gr_hs1= [4*p+1:17*p/4]';
        gr_hs2=[17*p/4+1:18*p/4]';
        gr_hs3=[18*p/4+1:19*p/4]';
        gr_hs4=[19*p/4+1:5*p]';
        gr={gr_xxs1;gr_xxs2;gr_xxs3;gr_xxs4;gr_hs1;gr_hs2;gr_hs3;gr_hs4;gr_sxs1;gr_sxs2;gr_sxs3;gr_sxs4}; %四组; 下，环，上

    case 6
        gr_fz=[zeros(1,p/6),5*p/6*ones(1,p/6)];  %赋值
        gr_xxs1=[kron(ones(1,p/6),[1,2])+2*kron(0:(p-1)/6,[1,1])+gr_fz]';
        gr_xxs2=[kron(ones(1,p/6),[p/6+1,p/6+2])+2*kron(0:(p-1)/6,[1,1])+gr_fz]';
        gr_xxs3=[kron(ones(1,p/6),[2*p/6+1,2*p/6+2])+2*kron(0:(p-1)/6,[1,1])+gr_fz]';
        gr_xxs4=[kron(ones(1,p/6),[3*p/6+1,3*p/6+2])+2*kron(0:(p-1)/6,[1,1])+gr_fz]';
        gr_xxs5=[kron(ones(1,p/6),[4*p/6+1,4*p/6+2])+2*kron(0:(p-1)/6,[1,1])+gr_fz]';
        gr_xxs6=[kron(ones(1,p/6),[5*p/6+1,5*p/6+2])+2*kron(0:(p-1)/6,[1,1])+gr_fz]';
       
        gr_sxs1=[kron(ones(1,p/6),[2*p+1,2*p+2])+2*kron(0:(p-1)/6,[1,1])+gr_fz]';
        gr_sxs2=[kron(ones(1,p/6),[13*p/6+1,13*p/6+2])+2*kron(0:(p-1)/6,[1,1])+gr_fz]';
        gr_sxs3=[kron(ones(1,p/6),[14*p/6+1,14*p/6+2])+2*kron(0:(p-1)/6,[1,1])+gr_fz]';
        gr_sxs4=[kron(ones(1,p/6),[15*p/6+1,15*p/6+2])+2*kron(0:(p-1)/6,[1,1])+gr_fz]';
        gr_sxs5=[kron(ones(1,p/6),[16*p/6+1,16*p/6+2])+2*kron(0:(p-1)/6,[1,1])+gr_fz]';
        gr_sxs6=[kron(ones(1,p/6),[17*p/6+1,17*p/6+2])+2*kron(0:(p-1)/6,[1,1])+gr_fz]';
        
        gr_hs1= [4*p+1:25*p/6]';
        gr_hs2=[25*p/6+1:26*p/6]';
        gr_hs3=[26*p/6+1:27*p/6]';
        gr_hs4=[27*p/6+1:28*p/6]';
        gr_hs5=[28*p/6+1:29*p/6]';
        gr_hs6=[29*p/6+1:5*p]';
        gr={gr_xxs1;gr_xxs2;gr_xxs3;gr_xxs4;gr_xxs5;gr_xxs6;
            gr_sxs1;gr_sxs2;gr_sxs3;gr_sxs4;gr_sxs5;gr_sxs6;
             gr_hs1;gr_hs2;gr_hs3;gr_hs4;gr_hs5;gr_hs6}; %六组; 下，上，环
         
       case 12  
       
%          matrix1 = [1:2*p]';  %生成 p*1 的矩阵,下斜索
%          matrix2 = [2*p+1:4*p]';  %生成 p*1 的矩阵，上斜索
%          matrix3 = [4*p+1:5*p]';  %生成 p*1 的矩阵，环索
%          matrix = [matrix1;matrix2;matrix3];
%          gr = num2cell(matrix);  %生成 p*1 的单元数组,赋值到 [gr]




        gr_fz=[zeros(1,p/12),11*p/12*ones(1,p/12)];  %赋值
        gr_fz_1=[24,24+11*p/12*ones(1,p/12)];  %赋值

        gr_xxs1=[kron(ones(1,p/12),[1,2])+2*kron(0:(p-1)/12,[1,1])+gr_fz]';
        gr_xxs2=[kron(ones(1,p/12),[p/12+1,p/12+2])+2*kron(0:(p-1)/12,[1,1])+gr_fz]';
        gr_xxs3=[kron(ones(1,p/12),[2*p/12+1,2*p/12+2])+2*kron(0:(p-1)/12,[1,1])+gr_fz]';
        gr_xxs4=[kron(ones(1,p/12),[3*p/12+1,3*p/12+2])+2*kron(0:(p-1)/12,[1,1])+gr_fz]';
        gr_xxs5=[kron(ones(1,p/12),[4*p/12+1,4*p/12+2])+2*kron(0:(p-1)/12,[1,1])+gr_fz]';
        gr_xxs6=[kron(ones(1,p/12),[5*p/12+1,5*p/12+2])+2*kron(0:(p-1)/12,[1,1])+gr_fz]';
        gr_xxs7=[kron(ones(1,p/12),[6*p/12+1,6*p/12+2])+2*kron(0:(p-1)/12,[1,1])+gr_fz]';
        gr_xxs8=[kron(ones(1,p/12),[7*p/12+1,7*p/12+2])+2*kron(0:(p-1)/12,[1,1])+gr_fz]';
        gr_xxs9=[kron(ones(1,p/12),[8*p/12+1,8*p/12+2])+2*kron(0:(p-1)/12,[1,1])+gr_fz]';
        gr_xxs10=[kron(ones(1,p/12),[9*p/12+1,9*p/12+2])+2*kron(0:(p-1)/12,[1,1])+gr_fz]';
        gr_xxs11=[kron(ones(1,p/12),[10*p/12+1,10*p/12+2])+2*kron(0:(p-1)/12,[1,1])+gr_fz]';
        gr_xxs12=[kron(ones(1,p/12),[11*p/12+1,11*p/12+2])+2*kron(0:(p-1)/12,[1,1])+gr_fz]';

        gr_sxs1=[kron(ones(1,p/12),[1,2])+2*kron(0:(p-1)/12,[1,1])+gr_fz_1]';
        gr_sxs2=[kron(ones(1,p/12),[p/12+1,p/12+2])+2*kron(0:(p-1)/12,[1,1])+gr_fz_1]';
        gr_sxs3=[kron(ones(1,p/12),[2*p/12+1,2*p/12+2])+2*kron(0:(p-1)/12,[1,1])+gr_fz_1]';
        gr_sxs4=[kron(ones(1,p/12),[3*p/12+1,3*p/12+2])+2*kron(0:(p-1)/12,[1,1])+gr_fz_1]';
        gr_sxs5=[kron(ones(1,p/12),[4*p/12+1,4*p/12+2])+2*kron(0:(p-1)/12,[1,1])+gr_fz_1]';
        gr_sxs6=[kron(ones(1,p/12),[5*p/12+1,5*p/12+2])+2*kron(0:(p-1)/12,[1,1])+gr_fz_1]';
        gr_sxs7=[kron(ones(1,p/12),[6*p/12+1,6*p/12+2])+2*kron(0:(p-1)/12,[1,1])+gr_fz_1]';
        gr_sxs8=[kron(ones(1,p/12),[7*p/12+1,7*p/12+2])+2*kron(0:(p-1)/12,[1,1])+gr_fz_1]';
        gr_sxs9=[kron(ones(1,p/12),[8*p/12+1,8*p/12+2])+2*kron(0:(p-1)/12,[1,1])+gr_fz_1]';
        gr_sxs10=[kron(ones(1,p/12),[9*p/12+1,9*p/12+2])+2*kron(0:(p-1)/12,[1,1])+gr_fz_1]';
        gr_sxs11=[kron(ones(1,p/12),[10*p/12+1,10*p/12+2])+2*kron(0:(p-1)/12,[1,1])+gr_fz_1]';
        gr_sxs12=[kron(ones(1,p/12),[11*p/12+1,11*p/12+2])+2*kron(0:(p-1)/12,[1,1])+gr_fz_1]';

        gr_hs1= [4*p+1]';
        gr_hs2=[4*p+2]';
        gr_hs3=[4*p+3]';
        gr_hs4=[4*p+4]';
        gr_hs5=[4*p+5]';
        gr_hs6=[4*p+6]';
        gr_hs7=[4*p+7]';
        gr_hs8=[4*p+8]';
        gr_hs9=[4*p+9]';
        gr_hs10=[4*p+10]';
        gr_hs11=[4*p+11]';
        gr_hs12=[5*p]';
        gr={gr_xxs1;gr_xxs2;gr_xxs3;gr_xxs4;gr_xxs5;gr_xxs6;gr_xxs7;gr_xxs8;gr_xxs9;gr_xxs10;gr_xxs11;gr_xxs12;
            gr_sxs1;gr_sxs2;gr_sxs3;gr_sxs4;gr_sxs5;gr_sxs6;gr_sxs7;gr_sxs8;gr_sxs9;gr_sxs10;gr_sxs11;gr_sxs12;
             gr_hs1;gr_hs2;gr_hs3;gr_hs4;gr_hs5;gr_hs6;gr_hs7;gr_hs8;gr_hs9;gr_hs10;gr_hs11;gr_hs12}; %六组; 下，上，环

         
end
end

