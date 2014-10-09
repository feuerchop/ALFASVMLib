% Script for experiment in Sec. 4.2 of the paper [1]:
%
% [1] Xiao, Huang, Battista Biggio, Blaine Nelson, Han Xiao, Claudia Eckert, and Fabio Roli.
%    "Support Vector Machines under Adversarial Label Contamination."
%    Journal of Neurocomputing, Special Issue on Advances in Learning with Label Noise, 2014.
%
% You can use this script to test the performances of differnet adversarial
% label flip attacks on SVMs, against five real-world data sets reported.
% Please refer to the paper for more details.
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


function res=paper_exp2(set_name, ker, tr_size, tt_size)
%   Function description: test errors on test set after adversarial label flip attacks
    addpath(genpath(pwd));

    % get the dataset, training sampels 500, test samples 500;
    data_path = ['datasets/libsvm-dataset/', set_name];
    data = loaddata_nobias(data_path, tr_size+tt_size);
    [n, d] = size(data);
    
    % manually generate 5-folder cross-validation indices
    cvp.NumTestSets = 5;
    for f=1:5
        cvp.training{f} = randsample(n, tr_size);
        cvp.test{f} = setdiff(1:n, cvp.training{f});
    end
    
    % evaluation performance by accuracy;
    perf = CPerfEval();
    perf.setCriterion('accuracy');

    % compute the optimal C/gamma values via 5-folder CV on the full dataset
    mySVM = CClassifierSVM(ker);
    mySVM.verbose = 1;
    params = {'C', 'gamma'};
    values = {2.^[-7,10], 2.^[-7,5]};
    best = mySVM.crossval(data(:, end), data(:, 1:end-1), ...
        params, values, 'cperf', 'accuracy', 'kfold', 5);
    best_c = best{1};       % best C for SVM
    best_g = best{2};       % best gamma for SVM
    fprintf('Optimal parameters for SVM on dataset %s: C=%f, gamma=%f\n', set_name, mySVM.C, mySVM.gamma);
    mySVM.verbose = 0;
    
    % label flip from 0% to 20%
    flip_len = 10;
    flips = linspace(0, 0.2, flip_len);
    % initiate results vector, 7 attacks + 1 SVM
    res = zeros(3, 8*flip_len);
    % first 10 results are for SVM without label flips
    res(1,1:flip_len) = Inf;         % minimal value
    res(3,1:flip_len) = -Inf;        % maximal value    

    % Train SVM and test without any label flips
    for i=1:cvp.NumTestSets
        mySVM.train(data(cvp.training{i}, end), data(cvp.training{i}, 1:end-1));
        yc = mySVM.classify(data(cvp.test{i}, 1:end-1));
        err = 1-perf.performance(data(cvp.test{i}, end), yc);
        if err < res(1,1:flip_len)
            res(1,1:flip_len) = err;
        else
            if err > res(3,1:flip_len)
                res(3,1:flip_len) = err;
            end
        end
        res(2,1:flip_len) = ((i-1)*res(2,1:flip_len) + err)/i;
    end
    
    % Start attacks for each attack strategy against each flip rate
    % !!PLEASE ENABLE PARALLEL COMPUTING  
    parfor i=1:7*flip_len  % we have 7 kinds of attacks
        atk_index = ceil(i/flip_len);
        if mod(i, flip_len)>0
            flip_index=mod(i, flip_len);
        else
            flip_index=flip_len;
        end
        min_err = Inf;
        max_err = -Inf;
        mean_err = 0;
        flip = flips(flip_index);
        switch atk_index
            case 1
                % ALFA
                atk_alfa = CAttackerSVMAlfa(ker, 'C', best_c, 'gamma', best_g);
                atk_alfa.flip_rate = flip;
                for j=1:cvp.NumTestSets
                    atk_alfa.train(data(cvp.training{j}, end), data(cvp.training{j}, 1:end-1));
                    atk_alfa.flipLabels(data(cvp.training{j}, end), data(cvp.training{j}, 1:end-1));
                    atk_alfa_yc = atk_alfa.classify(data(cvp.test{j}, 1:end-1));
                    atk_alfa_yc_err = 1-perf.performance(data(cvp.test{j}, end), atk_alfa_yc);
                    if atk_alfa_yc_err < min_err
                        min_err = atk_alfa_yc_err;
                    else
                        if atk_alfa_yc_err > max_err
                            max_err = atk_alfa_yc_err;
                        end
                    end
                    mean_err = ((j-1)*mean_err + atk_alfa_yc_err)/j;
                end
            case 2
                % ALFA-Cr
                atk_alfacr = CAttackerSVMAlfaCr(ker, 'C', best_c, 'gamma', best_g);
                atk_alfacr.flip_rate = flip;
                for j=1:cvp.NumTestSets
                    atk_alfacr.train(data(cvp.training{j}, end), data(cvp.training{j}, 1:end-1));
                    atk_alfacr.flipLabels(data(cvp.training{j}, end), data(cvp.training{j}, 1:end-1));
                    atk_alfacr_yc = atk_alfacr.classify(data(cvp.test{j}, 1:end-1));
                    atk_alfacr_yc_err = 1-perf.performance(data(cvp.test{j}, end), atk_alfacr_yc);
                    if atk_alfacr_yc_err < min_err
                        min_err = atk_alfacr_yc_err;
                    else
                        if atk_alfacr_yc_err > max_err
                            max_err = atk_alfacr_yc_err;
                        end
                    end
                    mean_err = ((j-1)*mean_err + atk_alfacr_yc_err)/j;
                end
            case 3
                % ALFA-tilt
                atk_alfatilt = CAttackerSVMAlfaTilt(ker, 'C', best_c, 'gamma', best_g);
                atk_alfatilt.flip_rate = flip;
                for j=1:cvp.NumTestSets
                    atk_alfatilt.train(data(cvp.training{j}, end), data(cvp.training{j}, 1:end-1));
                    atk_alfatilt.flipLabels(data(cvp.training{j}, end), data(cvp.training{j}, 1:end-1));
                    atk_alfatilt_yc = atk_alfatilt.classify(data(cvp.test{j}, 1:end-1));
                    atk_alfatilt_yc_err = 1-perf.performance(data(cvp.test{j}, end), atk_alfatilt_yc);
                    if atk_alfatilt_yc_err < min_err
                        min_err = atk_alfatilt_yc_err;
                    else
                        if atk_alfatilt_yc_err > max_err
                            max_err = atk_alfatilt_yc_err;
                        end
                    end
                    mean_err = ((j-1)*mean_err + atk_alfatilt_yc_err)/j;
                end   
            case 4
                % ALFA-far
                atk_far = CAttackerSVMDist('far', 'ker', ker, 'C', best_c, 'gamma', best_g);
                atk_far.flip_rate = flip;
                for j=1:cvp.NumTestSets
                    atk_far.train(data(cvp.training{j}, end), data(cvp.training{j}, 1:end-1));
                    atk_far.flipLabels(data(cvp.training{j}, end), data(cvp.training{j}, 1:end-1));
                    atk_far_yc = atk_far.classify(data(cvp.test{j}, 1:end-1));
                    atk_far_yc_err = 1-perf.performance(data(cvp.test{j}, end), atk_far_yc);
                    if atk_far_yc_err < min_err
                        min_err = atk_far_yc_err;
                    else
                        if atk_far_yc_err > max_err
                            max_err = atk_far_yc_err;
                        end
                    end
                    mean_err = ((j-1)*mean_err + atk_far_yc_err)/j;
                end
            case 5
                % ALFA-near
                atk_near = CAttackerSVMDist('near', 'ker', ker, 'C', best_c, 'gamma', best_g);
                atk_near.flip_rate = flip;
                for j=1:cvp.NumTestSets
                    atk_near.train(data(cvp.training{j}, end), data(cvp.training{j}, 1:end-1));
                    atk_near.flipLabels(data(cvp.training{j}, end), data(cvp.training{j}, 1:end-1));
                    atk_near_yc = atk_near.classify(data(cvp.test{j}, 1:end-1));
                    atk_near_yc_err = 1-perf.performance(data(cvp.test{j}, end), atk_near_yc);
                    if atk_near_yc_err < min_err
                        min_err = atk_near_yc_err;
                    else
                        if atk_near_yc_err > max_err
                            max_err = atk_near_yc_err;
                        end
                    end
                    mean_err = ((j-1)*mean_err + atk_near_yc_err)/j;
                end
            case 6
                % ALFA-rand
                atk_rand = CAttackerSVMDist('rand', 'ker', ker, 'C', best_c, 'gamma', best_g);
                atk_rand.flip_rate = flip;
                for j=1:cvp.NumTestSets
                    atk_rand.train(data(cvp.training{j}, end), data(cvp.training{j}, 1:end-1));
                    atk_rand.flipLabels(data(cvp.training{j}, end), data(cvp.training{j}, 1:end-1));
                    atk_rand_yc = atk_rand.classify(data(cvp.test{j}, 1:end-1));
                    atk_rand_yc_err = 1-perf.performance(data(cvp.test{j}, end), atk_rand_yc);
                    if atk_rand_yc_err < min_err
                        min_err = atk_rand_yc_err;
                    else
                        if atk_rand_yc_err > max_err
                            max_err = atk_rand_yc_err;
                        end
                    end
                    mean_err = ((j-1)*mean_err + atk_rand_yc_err)/j;
                end 
            case 7
                % Correlated cluster
                atk_corr = CAttackerSVMCorrClusters(ker, 'C', best_c, 'gamma', best_g);
                atk_corr.flip_rate = flip;
                for j=1:cvp.NumTestSets
                    atk_corr.train(data(cvp.training{j}, end), data(cvp.training{j}, 1:end-1));
                    atk_corr.flipLabels(data(cvp.training{j}, end), data(cvp.training{j}, 1:end-1));
                    atk_corr_yc = atk_corr.classify(data(cvp.test{j}, 1:end-1));
                    atk_corr_yc_err = 1-perf.performance(data(cvp.test{j}, end), atk_corr_yc);
                    if atk_corr_yc_err < min_err
                        min_err = atk_corr_yc_err;
                    else
                        if atk_corr_yc_err > max_err
                            max_err = atk_corr_yc_err;
                        end
                    end
                    mean_err = ((j-1)*mean_err + atk_corr_yc_err)/j;
                end    
        end
        res(:,i+flip_len) = [min_err mean_err max_err]';
    end
    
    % plot the error-bar results 
    res = 100*res;
    axis('square');
    hold('on');
    box('on');
    title([set_name, ': ', ker]);
    xlabel(['Flip rate on training set: ', 'C=', num2str(best_c), ', g=', num2str(best_g)]);
    ylabel('Error rate on test set (%)');
    xlim([0 0.2]);
    ylim([0, 60]);
    line_set={'-.ko','-r+','-b*','-c^','-ms','-gv','-k*','-.r^','-.cs','-.mv','-.y+'};
    for j = 1:8
        errorbar(flips, res(2,(j-1)*flip_len+1:j*flip_len),...
            res(3,(j-1)*flip_len+1:j*flip_len)-res(2,(j-1)*flip_len+1:j*flip_len),...
            res(2,(j-1)*flip_len+1:j*flip_len)-res(1,(j-1)*flip_len+1:j*flip_len),...
            line_set{j},'MarkerSize',8,'LineWidth',1.1);
    end
    legend([{'SVM'}, {'ALFA'}, {'ALFACr'}, ...
                {'ALFATilt'},  {'Far first'}, {'Near first'},...
                {'Random'}, {'Correlated Cluster'}]);

