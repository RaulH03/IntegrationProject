clear;clc;

A = [0.00000000e+00  0.00000000e+00  1.00000000e+00  0.00000000e+00
    0.00000000e+00  0.00000000e+00  0.00000000e+00  1.00000000e+00
    -7.97618129e+00  1.64569311e+01 -9.87998251e+01 -2.28807820e-03
    -8.43378039e+00  1.25554363e+02 -1.04190900e+02 -2.79559122e-01];
B = [   0.
     0.
    -459.04651634
    -485.38233599];

C = eye(4);

D = [0;
  0;0;0];
h = 0.01;
s=tf('s');
sysc = ss(A, B, C, D);
sysd = c2d(ss(A,B,C,D),h);



M = 2.4; % bound on the hinv norm
wb1 = 0.3; % desired bandwith
A = 1/10000; % attinuation of system
Wp11 = (s/M+wb1)/(s+wb1*A);
Wp = [Wp11,0,0,0;
     0,Wp11,0,0;
     0,0,Wp11,0;
     0,0,0,Wp11];
Wu22 = ((5*10^-3)*s^2 + (7*10^-4) * s + (5*10^-5))/(s^2 + s*14*10^-4 + 10^-6);
Wu = [Wu22];
Wt = [];



G = sysc;
size(G)
%%% defining the block diagram %%%
G_sim = G;
G_sim.u = 'u';
G_sim.y = 'y_plant';

Wu_sim = Wu;
Wu_sim.u = 'u';
Wu_sim.y = 'z2';

%y_sim = sumblk('y_meas = y_plant + y_dis',4);

%e_sim = sumblk('y_e = y_plant', 4); % No reference, no negative feedback

Wp_sim = Wp;
Wp_sim.u = 'y_plant';
Wp_sim.y = 'z1';

P_sim = connect(G_sim, Wu_sim , Wp_sim, e_sim, {'u'},{ 'z1';'z2';'y_plant'});
P_sim2 = minreal(P_sim);
size(P_sim2)

%%% Synthesising the controller %%%
[K, CL, GAM, INFO] = hinfsyn(P_sim, 2, 1);
%S = eye(2) / (eye(2)+G*K);
%N_norm = lft(P_sim2,K);
%mu_s = norm(N_norm,inf)
%INFO;
K;

%K_sim = minreal(K);
%K_sim.u = 'y_e';
%K_sim.y = 'u';