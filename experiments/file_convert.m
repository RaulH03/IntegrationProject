T = load("zero_input_upright.mat");
D = table(T.ans);
writetable(D,'zero_input_upright.csv')