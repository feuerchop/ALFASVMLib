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

function paper_exp1(pattern, ker)
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
    mySVM = CClassifierSVM(ker, 'C', 1, 'gamma', 0.5);
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
    myAlfa = CAttackerSVMAlfa(ker, 'C', 1, 'gamma', 0.5);
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
    % start ALFACr attack and plot tainted SVM   
    myAlfaCr = CAttackerSVMAlfaCr(ker, 'C', 1, 'gamma', 0.5);
    myAlfaCr.train(Ytr, Xtr);
    myAlfaCr.flipLabels(Ytr, Xtr);
    yc = myAlfaCr.classify(Xtt);
    err = 1-perf.performance(Ytt, yc);
    ax=subplot(194);
    hold(ax, 'on');
    axis(ax, 'square');
    box(ax, 'on');
    set(ax, 'XTick', [], 'YTick', []);
    myAlfaCr.plot(Xtr, 'pointSize', 0, ...
                         'svSize',0, ...
                         'background', 1, ...
                         'title', [num2str(100*err),'%'], ...
                         'xLabel', myAlfaCr.name);
    % start ALFATilt attack and plot tainted SVM   
    myAlfaTilt = CAttackerSVMAlfaTilt(ker, 'C', 1, 'gamma', 0.5);
    myAlfaTilt.train(Ytr, Xtr);
    myAlfaTilt.flipLabels(Ytr, Xtr);
    yc = myAlfaTilt.classify(Xtt);
    err = 1-perf.performance(Ytt, yc);
    ax=subplot(195);
    hold(ax, 'on');
    axis(ax, 'square');
    box(ax, 'on');
    set(ax, 'XTick', [], 'YTick', []);
    myAlfaTilt.plot(Xtr, 'pointSize', 0, ...
                         'svSize',0, ...
                         'background', 1, ...
                         'title', [num2str(100*err),'%'], ...
                         'xLabel', myAlfaTilt.name);
    % start CorrelatedCluster attack and plot tainted SVM   
    myCC = CAttackerSVMCorrClusters(ker, 'C', 1, 'gamma', 0.5);
    myCC.train(Ytr, Xtr);
    myCC.flipLabels(Ytr, Xtr);
    yc = myCC.classify(Xtt);
    err = 1-perf.performance(Ytt, yc);
    ax=subplot(196);
    hold(ax, 'on');
    axis(ax, 'square');
    box(ax, 'on');
    set(ax, 'XTick', [], 'YTick', []);
    myCC.plot(Xtr, 'pointSize', 0, ...
                         'svSize',0, ...
                         'background', 1, ...
                         'title', [num2str(100*err),'%'], ...
                         'xLabel', myCC.name);
    % start nearest first ALFA attack and plot tainted SVM
    myAlfaNear = CAttackerSVMDist('near', 'ker', ker, 'C', 1, 'gamma', 0.5);
    myAlfaNear.train(Ytr, Xtr);
    myAlfaNear.flipLabels(Ytr, Xtr);
    yc = myAlfaNear.classify(Xtt);
    err = 1-perf.performance(Ytt, yc);
    ax=subplot(197);
    hold(ax, 'on');
    axis(ax, 'square');
    box(ax, 'on');
    set(ax, 'XTick', [], 'YTick', []);
    myAlfaNear.plot(Xtr, 'pointSize', 0, ...
                         'svSize',0, ...
                         'background', 1, ...
                         'title', [num2str(100*err),'%'], ...
                         'xLabel', myAlfaNear.name);
    % start far first ALFA attack and plot tainted SVM
    myAlfaFar = CAttackerSVMDist('far', 'ker', ker, 'C', 1, 'gamma', 0.5);
    myAlfaFar.train(Ytr, Xtr);
    myAlfaFar.flipLabels(Ytr, Xtr);
    yc = myAlfaFar.classify(Xtt);
    err = 1-perf.performance(Ytt, yc);
    ax=subplot(198);
    hold(ax, 'on');
    axis(ax, 'square');
    box(ax, 'on');
    set(ax, 'XTick', [], 'YTick', []);
    myAlfaFar.plot(Xtr, 'pointSize', 0, ...
                         'svSize',0, ...
                         'background', 1, ...
                         'title', [num2str(100*err),'%'], ...
                         'xLabel', myAlfaFar.name);
    % start random ALFA attack and plot tainted SVM
    myAlfaRand = CAttackerSVMDist('rand', 'ker', ker, 'C', 1, 'gamma', 0.5);
    myAlfaRand.train(Ytr, Xtr);
    myAlfaRand.flipLabels(Ytr, Xtr);
    yc = myAlfaRand.classify(Xtt);
    err = 1-perf.performance(Ytt, yc);
    ax=subplot(199);
    hold(ax, 'on');
    axis(ax, 'square');
    box(ax, 'on');
    set(ax, 'XTick', [], 'YTick', []);
    myAlfaRand.plot(Xtr, 'pointSize', 0, ...
                         'svSize',0, ...
                         'background', 1, ...
                         'title', [num2str(100*err),'%'], ...
                         'xLabel', myAlfaRand.name);               
end

% e.g., paper_exp1('parabolic', 'linear'), paper_exp1('linear', 'rbf')