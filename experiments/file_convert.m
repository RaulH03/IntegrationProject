T = load("chirp44_amp02_dt001.mat");
D = table(T.ans);
writetable(D,'chirp44_amp02_dt001.csv')