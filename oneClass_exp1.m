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

function oneClass_exp1(ker)
    n=800; m=200;
    %genarate the data
    X = genRingData(n, m);
    Y = ones(n+m,1);
    Y(1:n) = -1;

    Xtr = [X(1:400, :); X(801:900, :)];
    Ytr = [Y(1:400); Y(801:900)];
    Xtt = [X(401:800, :); X(901:1000, :)];
    Ytt = [Y(401:800); Y(901:1000)];

    perf = CPerfEval();
    perf.setCriterion('f_measure');
    % plot original data
    ax=subplot(2, 1, 1); 
    hold(ax, 'on');
    axis(ax, 'square');
    box(ax, 'on');
    set(ax, 'XTick', [], 'YTick', []);
    plot(ax, X(Y==1, 1), X(Y==1, 2), 'r.','MarkerSize',8);
    plot(ax, X(Y==-1, 1), X(Y==-1, 2), 'b.','MarkerSize',8);
    title(ax, 'Original Data')
    % plot SVM classifier on untainted data
    mySVM = CClassifierSVM(ker, 'C', 1, 'gamma', 0.05, 'one_class', true, 'nu', 0.001);
    mySVM.train(Ytr, Xtr);
    yc = mySVM.classify(Xtt);
    err = perf.performance(Ytt, yc);
    c = sum(yc==-1)
    o = sum(yc==1)
    ax=subplot(2, 1, 2);
    hold(ax, 'on');
    axis(ax, 'square');
    box(ax, 'on');
    set(ax, 'XTick', [], 'YTick', []);
    mySVM.plot(Ytr, Xtr, 'pointSize', 0, ...
                         'svSize',0, ...
                         'background', 1, ...
                         'title', [num2str(err),'%'], ...
                         'xLabel', mySVM.name);
           
end

% e.g., paper_exp1('rbf')