clc
clear
close all;
[X,Y] = meshgrid(1:0.5:10,1:20);
Z = 1000*(sin(X) + cos(Y));
C = X.*Y;
surf(X,Y,Z,C)
colorbar
figure
surf(X,Y,Z,C,'FaceColor','interp')