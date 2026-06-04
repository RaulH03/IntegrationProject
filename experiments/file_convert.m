T = load("chirp_05_10_4_amp02.mat");
D = table(T.ans);
writetable(D,'chirp_05_10_4_amp02.csv')