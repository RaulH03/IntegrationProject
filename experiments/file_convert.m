T = load("NEW_chirp_amp015_dt001.mat");
D = table(T.ans);
writetable(D,'NEW_chirp_amp015_dt001.csv')