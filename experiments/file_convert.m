T = load("sin_amp02.mat");
D = table(T.ans);
writetable(D,'sin_amp02.csv')