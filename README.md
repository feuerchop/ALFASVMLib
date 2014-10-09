=========================================================
ALFASVMLib - Adversarial Label Flip Attack on SVM Library
=========================================================
Author     : Xiao, Huang
Mail       : xiaohu@in.tum.de
Institute  : Technical University of Munich
Department : Computer Science

Author     : Biggio, Battista
Mail       : battista.biggio@diee.unica.it
Institute  : University of Cagliari
Department : Electronic and Electrical Eng., DIEE

Copyright 2014-07 Huang Xiao and Battista Biggio.
ALFASVMLib: A Matlab library for adversarial label flip attacks against SVMs

Website:	 http://pralab.diee.unica.it/ALFASVMLib

== Introduction ==
ALFASVMLib is an open source Matlab Library for researchers in the domain of adversarial machine learning,
and, in particular, for who is interested in understanding the vulnerability of SVM learning algorithms
to adversarial label flips (i.e., worst-case label noise).
It library relies on Libsvm (ver>3.17) and on the CVX solver.
The goal of the included label flip attack algorithms is to maximally decrease the SVM's classification accuracy on unseen data,
by flipping a number of labels in the training data.
We publish this library for researchers who are interested in exploring possible weaknesses of machine learning algorithms
and in designing more robust learning-based systems.
If you download and use this library, especially for research purposes, please cite our related work as follows:

	Xiao, Huang, Battista Biggio, Blaine Nelson, Han Xiao, Claudia Eckert, and Fabio Roli. 
    Support Vector Machines under Adversarial Label Contamination. 
    Neurocomputing, Special Issue on Advances in Learning with Label Noise, 2014.
    
More updated details on the publication, along with the paper itself and bibtex citation, can be found here:
    http://pralab.diee.unica.it/biggio14-neurocomputing
        
== System Requirements ==
You need at least following programs/libs installed on your machine.
--- 
- Matlab 2013a or above
- Libsvm >3.17 (http://www.csie.ntu.edu.tw/~cjlin/libsvm/)
- CVX version 2.0 (http://cvxr.com/cvx/)
- Parallel computing toolbox in Matlab enabled
--- 
Add pathes of libsvm and cvx in your matlab path, no installation required.

== File list ==
../
 ./@CPerfEval/                          % Class for performance evaluation
 ./datasets/                            % Folder of toy data sets
 ./CAttacker.m                          % Abstract class of classifier attacker 
 ./CAttackerSVM.m                       % Parent class of SVM attacker 
 ./CAttackerSVMAlfa.m                   % ALFA class of SVM attacker
 ./CAttackerSVMAlfaCr.m                 % Continuous label relaxation attacker
 ./CAttackerSVMCorrClusters.m           % Correlated cluster label attacker
 ./CAttackerSVMDist.m                   % Distance based ALFA attacker 
 ./CClassifier.m                        % Abstract class of classifier
 ./CClassifierSVM.m                     % Class of SVM
 ./CKernel.m                            % Class of Kernel
 ./README.txt                           % README file
 ./COPYRIGHT.txt                        % Copyright disclaim 
 ./LICENSE.txt                          % License information
 ./demo.m                               % Demo file of HOWTO
 ./paper_exp1.m                         % Script for the experiment in Sec. 4.1 of the paper [1]
 ./paper_exp2.m                         % Script for the experiment in Sec. 4.2 of the paper [1]

== Usage ==
See file demo.m, paper_exp1.m, paper_exp2.m

== References == 
[1] Xiao, Huang, Battista Biggio, Blaine Nelson, Han Xiao, Claudia Eckert, and Fabio Roli. 
    Support Vector Machines under Adversarial Label Contamination. 
    Journal of Neurocomputing, Special Issue on Advances in Learning with Label Noise, 2014.
