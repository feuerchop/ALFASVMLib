% Script for experiment in Sec. 4.1 of the paper [1]:
%
% [1] Xiao, Huang, Battista Biggio, Blaine Nelson, Han Xiao, Claudia Eckert, and Fabio Roli. 
%    "Support Vector Machines under Adversarial Label Contamination."
%    Journal of Neurocomputing, Special Issue on Advances in Learning with Label Noise, 2014.
%
% You can use this script to test the performances of differnet adversarial
% label flip attacks on SVMs, against two toy data sets. Please refer to
% the paper for more details.
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

function oneClass_exp1(pattern, ker)
    data_root = './datasets/';
    load([data_root, pattern]);
    X = data(:, 1:end-1);
    Y = data(:, end);
    cvp = cvpartition(length(Y), 'holdout', 0.8);
    Xtr = X(cvp.training, :);
    Ytr = Y(cvp.training, :);
    Xtt = X(cvp.test, :);
    Ytt = Y(cvp.test, :);
    perf = CPerfEval();
    perf.setCriterion('accuracy');
    % plot original data
    ax=subplot(191); 
    hold(ax, 'on');
    axis(ax, 'square');
    box(ax, 'on');
    set(ax, 'XTick', [], 'YTick', []);
    plot(ax, X(Y==1, 1), X(Y==1, 2), 'r.','MarkerSize',8);
    plot(ax, X(Y==-1, 1), X(Y==-1, 2), 'b.','MarkerSize',8);
    title(ax, 'Original Data')
    % plot SVM classifier on untainted data
    mySVM = CClassifierSVM(ker, 'C', 1, 'gamma', 0.5, 'one_class', true);
    mySVM.train(Ytr, Xtr);
    yc = mySVM.classify(Xtt);
    err = 1-perf.performance(Ytt, yc);
    ax=subplot(192);
    hold(ax, 'on');
    axis(ax, 'square');
    box(ax, 'on');
    set(ax, 'XTick', [], 'YTick', []);
    mySVM.plot(Ytr, Xtr, 'pointSize', 0, ...
                         'svSize',0, ...
                         'background', 1, ...
                         'title', [num2str(100*err),'%'], ...
                         'xLabel', mySVM.name);
    % start ALFA attack and plot tainted SVM   
    myAlfa = CAttackerSVMAlfa(ker, 'C', 1, 'gamma', 0.5, 'one_class', true);
    myAlfa.train(Ytr, Xtr);
    myAlfa.flipLabels(Ytr, Xtr);
    yc = myAlfa.classify(Xtt);
    err = 1-perf.performance(Ytt, yc);
    ax=subplot(193);
    hold(ax, 'on');
    axis(ax, 'square');
    box(ax, 'on');
    set(ax, 'XTick', [], 'YTick', []);
    myAlfa.plot(Xtr, 'pointSize', 0, ...
                         'svSize',0, ...
                         'background', 1, ...
                         'title', [num2str(100*err),'%'], ...
                         'xLabel', myAlfa.name);
               
end

% e.g., paper_exp1('parabolic', 'linear'), paper_exp1('linear', 'rbf')