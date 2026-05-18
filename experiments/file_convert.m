T = load("idinput_dt005_amp02.mat");
D = table(T.ans);
writetable(D,'idinput_dt005_amp02.csv')