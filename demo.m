% Demonstration of how to use the ALFASVMLib
% ------
% Author     : Xiao, Huang
% Mail       : xiaohu@in.tum.de
% Institute  : Technical University of Munich
% Department : Computer Science
%
% Author     : Biggio, Battista
% Mail       : battista.biggio@diee.unica.it
% Institute  : University of Cagliari
% Department : Electronic and Electrical Eng., DIEE
%
% Copyright 2014-07 Huang Xiao and Battista Biggio.
%

addpath(genpath(pwd));

% get the dataset, training sampels 500, test samples 500;
load ./datasets/circle.mat;  %circle.mat
cvp = cvpartition(size(data,1), 'holdout', 0.8);
x_tr = data(cvp.training, 1:end-1);
y_tr = data(cvp.training, end);
x_ts = data(cvp.test, 1:end-1);
y_ts = data(cvp.test, end);

%%
% evaluation performance by accuracy;
perf = CPerfEval();
perf.setCriterion('accuracy');

mySVM = CClassifierSVM('rbf');
mySVM.verbose = 1;
params = {'C', 'gamma'};
values = {2.^[-5,3], 2.^[-5,3]};
best = mySVM.crossval(y_tr, x_tr, ...
                      params, values, 'cperf', 'accuracy', 'kfold', 5);
best_c = best{1};       % best C for SVM
best_g = best{2};       % best gamma for SVM
fprintf('Cross validation for SVM: best_C=%f, best_gamma=%f\n', mySVM.C, mySVM.gamma);
mySVM.verbose = 0;
%%
% training SVM and classify on test set
mySVM.train(y_tr, x_tr);
fprintf('SVM trained on %d training samples.\n', length(y_tr));
y_tt = mySVM.classify(x_ts);
acc = perf.performance(y_ts, y_tt);
err = 100*(1-acc);
fprintf('SVM Accuracy on predicting %d test samples is %f.\n', length(y_ts), acc);
figure(1)
mySVM.plot(y_tr, x_tr, 'xLabel',['Accuracy: ', num2str(acc)], 'background', true, 'title', 'SVC Boundary');
%%
% start label flip by various adversarial attacks
% initiate all the attackers
flip_rate = 0.1;            % Default flip rate, you can change it by explicitly set it.
atk_alfa = CAttackerSVMAlfa(mySVM.kType, 'C', best_c, 'gamma', best_g);
atk_alfa.flip_rate = flip_rate;
atk_alfacr = CAttackerSVMAlfaCr(mySVM.kType, 'C', best_c, 'gamma', best_g);
atk_alfatilt = CAttackerSVMAlfaTilt(mySVM.kType, 'C', best_c, 'gamma', best_g);
atk_far = CAttackerSVMDist('far', 'ker', 'rbf', 'C', best_c, 'gamma', best_g);
atk_near = CAttackerSVMDist('near', 'ker', 'rbf', 'C', best_c, 'gamma', best_g);
atk_rand = CAttackerSVMDist('rand', 'ker', 'rbf', 'C', best_c, 'gamma', best_g);
atk_corr = CAttackerSVMCorrClusters(mySVM.kType, 'C', best_c, 'gamma', best_g);
% Example of attacking process: 
% ALFA attack
% 1. First you need to train on the targeting training set to set up SVM
%    properties assuming that we knew defender's behaviors
atk_alfa.train(y_tr, x_tr); 
% 2. Then we flip labels by certain method, e.g., ALFA, ALFA_CR, ...
atk_alfa.flipLabels(y_tr, x_tr);
% 3. After flipping, the attacker object contains SVM properties based on
%    the contaminated dataset. Then we use the 'twisted' model 
atk_alfa_yt = atk_alfa.classify(x_ts);
% 4. Test the error rate on test set after flipping
atk_alfa_yt_err = 100*(1-perf.performance(y_ts, atk_alfa_yt));

% ALFACr attack
atk_alfacr.train(y_tr, x_tr); 
atk_alfacr.flipLabels(y_tr, x_tr);
atk_alfacr_yt = atk_alfacr.classify(x_ts);
atk_alfacr_yt_err = 100*(1-perf.performance(y_ts, atk_alfacr_yt));

% ALFATilt attack
atk_alfatilt.train(y_tr, x_tr); 
atk_alfatilt.flipLabels(y_tr, x_tr);
atk_alfatilt_yt = atk_alfatilt.classify(x_ts);
atk_alfatilt_yt_err = 100*(1-perf.performance(y_ts, atk_alfatilt_yt));

% far first attack
atk_far.train(y_tr, x_tr); 
atk_far.flipLabels(y_tr, x_tr);
atk_far_yt = atk_far.classify(x_ts);
atk_far_yt_err = 100*(1-perf.performance(y_ts, atk_far_yt));

% nearest first attack
atk_near.train(y_tr, x_tr); 
atk_near.flipLabels(y_tr, x_tr);
atk_near_yt = atk_near.classify(x_ts);
atk_near_yt_err = 100*(1-perf.performance(y_ts, atk_near_yt));

% random attack
atk_rand.train(y_tr, x_tr); 
atk_rand.flipLabels(y_tr, x_tr);
atk_rand_yt = atk_rand.classify(x_ts);
atk_rand_yt_err = 100*(1-perf.performance(y_ts, atk_rand_yt));

% correlated cluster attack
atk_corr.train(y_tr, x_tr); 
atk_corr.flipLabels(y_tr, x_tr);
atk_corr_yt = atk_corr.classify(x_ts);
atk_corr_yt_err = 100*(1-perf.performance(y_ts, atk_corr_yt));

%%
% plotting results
pt=8; bg=true; svsize=6; subplots=8;
subplot(2,4,1); 
mySVM.plot(y_tr,x_tr,'pointSize',pt,'svSize',svsize,'title','Original SVM','xLabel',['Error: ',num2str(err)],'background',bg);
subplot(2,4,2); 
atk_alfa.plot(x_tr,'pointSize',pt,'svSize',svsize,'title','ALFA','xLabel',['Error: ',num2str(atk_alfa_yt_err)],'background',bg);
subplot(2,4,3); 
atk_alfacr.plot(x_tr,'pointSize',pt,'svSize',svsize,'title','ALFA-Cr','xLabel',['Error: ',num2str(atk_alfacr_yt_err)],'background',bg);
subplot(2,4,4); 
atk_alfatilt.plot(x_tr,'pointSize',pt,'svSize',svsize,'title','ALFA-Tilt','xLabel',['Error: ',num2str(atk_alfatilt_yt_err)],'background',bg);
subplot(2,4,5); 
atk_far.plot(x_tr,'pointSize',pt,'svSize',svsize,'title','Far first','xLabel',['Error: ',num2str(atk_far_yt_err)],'background',bg);
subplot(2,4,6); 
atk_near.plot(x_tr,'pointSize',pt,'svSize',svsize,'title','Near first','xLabel',['Error: ',num2str(atk_near_yt_err)],'background',bg);
subplot(2,4,7); 
atk_rand.plot(x_tr,'pointSize',pt,'svSize',svsize,'title','Random','xLabel',['Error: ',num2str(atk_rand_yt_err)],'background',bg);
subplot(2,4,8); 
atk_corr.plot(x_tr,'pointSize',pt,'svSize',svsize,'title','Correlated Clusters','xLabel',['Error: ',num2str(atk_corr_yt_err)],'background',bg);


