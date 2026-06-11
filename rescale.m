function y = rescale(u)
th1in = u(1);
th2in = u(2);

scale1 = 4.881666285191120;
scale2 = 4.932936598764020;

% eq1 = 3.796622000341846/scale1 *2*pi;
% eq2 = 1.210150569834988/scale2 *2*pi+pi;
%down up
% eq1 = 3.806622000341846/scale1 *2*pi;
% eq2 = 1.189150569834988/scale2 *2*pi + pi + 2*pi/360*7.5;
deg_to_add = 13;
eq_deg = 9.5;
%up up
eq1 = 3.806622000341846/scale1 *2*pi+ pi - 2*pi/360*deg_to_add;
eq2 = 1.189150569834988/scale2 *2*pi - 2*pi/360*eq_deg + 2*pi/360*deg_to_add; %low 7


th1 = mod(th1in/scale1*2*pi - eq1 +pi, 2*pi) -pi;
th2 = mod(th2in/scale2*2*pi - eq2 +pi, 2*pi) -pi;

y = [th1
     th2];
