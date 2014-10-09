classdef CAttackerSVMCorrClusters < CAttackerSVM
% CAttackerSVMCorrClusters - Adv. Label Flip Attack with correlated
% clusters
%
% Superclass: SVM and Attacker
%   See also: 
%
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
% Copyright 2014-07  Huang Xiao and Battista Biggio.
% --
    % TODO: Add description
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
     
    properties
       mode; %'merge-n-delete', 'merge-n-keep', 'xor-n-keep'
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    
    methods
        % Default Constructor
        function obj = CAttackerSVMCorrClusters(varargin)
        % CAttackerSVMCorrClusters.CAttackerSVMCorrClusters
        % Constructor for CAttackerSVMCorrClusters class
        % ----------
        % Usage:
        % CAttackerSVMCorrClusters():
        %    Constructor without argument will create a default CAttackerSVMCorrClusters object
        %    with mode as merge-n-delete, and RBF kernel by setting C=1, gamma=1
        %    see also class CClassifierSVM.
        % CAttackerSVMAlfa(kernel_str):
        %    Constructor with a input kernel string will create a CAttackerSVMCorrClusters
        %    object with given kernel type in {'linear','rbf','poly','precomputed'}
        %    using default C=1, and default kernel parameters. 
        %    e.g., CAttackerSVMCorrClusters('poly')
        % CAttackerSVMCorrClusters(param, value)
        %    Construtor with optional param-value pair arguments.
        %    Params:
        %       'ker':    - kernel string in {'linear','rbf','poly','precomputed'}
        %       'C':      - regularization coefficient, default as 1,
        %                   numeric, positive
        %       'gamma'   - parameter gamma for RBF kernel, numeric
        %                   default: 1
        %       'coef'    - parameter coef for polynomial kernel, numeric 
        %                   default: 0
        %       'degree'  - parameter degree for polynomial kernel, numeric 
        %                   default: 2
        %       'verbose' - set verbose while learning, true or false
        % Returns:
        %   A CAttackerSVMCorrClusters object
        % See also:
        %   CAttackerSVM, CAttackerSVMCorrClusters.flipLabels
        % -------
            obj = obj@CAttackerSVM(varargin{:});
            obj.name = 'alfa-cc';
            obj.mode = 'merge-n-delete';
        end
            
        function set.mode(obj, m)
            if ~ismember(m, {'merge-n-delete', 'merge-n-keep', 'xor-n-keep'})
                display('You should set the mode as one of {''merge-n-delete'', ''merge-n-keep'', ''xor-n-keep''}!');
            else
                obj.mode = m;
            end
        end        
        
        function flipLabels(obj, tr_labels, tr)
            % CAttackerSVMCorrClusters.flipLabels
            % Adversarial label attack flip will flip a certain amount of
            % labels in training set, the SVM is retrained on the
            % contaminated data set to incur a degration of final SVM
            % classification performance. 
            % ----------
            % Usage:
            % flipLabels(tr_labels, tr):
            %    Inputs:
            %       tr_labels: N*1 vectors containing classes
            %                  e.g., binary classification, [-1,+1]
            %       tr:        N*d training data set
            % Returns:
            %   Properties:
            %       obj.flipped_labels
            %       obj.flip_idx 
            %   are both set in the object.
            % See also: CAttackerSVMCorrClusters, CClassifierSVM.CClassifierSVM, CAttacker,
            %           CAttackerSVM.
            % -------
            
            if nargin < 3
                error('Not enough inputs, please input training labels and training set.');
            end
            
            if obj.flip_rate == 0 
                warning('Nothing happend, no label is flipped!');
                obj.flipped_labels = tr_labels;
                return;
            end
            
            num_flips = floor(length(tr_labels)*obj.flip_rate);
            
            % Call Correlated Clusters            
            obj.flipped_labels = obj.pcc_attack(tr, tr_labels, num_flips, obj.mode);
            
            % indices of flipped labels
            idx = find(obj.flipped_labels~=tr_labels); 
            obj.flip_idx = idx(1:min(num_flips, length(idx)));
            
            % train the SVM on contaminated dataset
            obj.train(obj.flipped_labels, tr);
        end
        
    end   
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    methods %These should be private    
        
        function [err score] = errorFunc(obj,ts_labels,tr_labels,tr)  
            %Trains object and evaluates error
            obj.train(tr_labels, tr);
            [yclass, score] = obj.classify(tr);
            err = sum(yclass~=ts_labels)/size(tr,1);
        end
       
        
        function y_prime =pcc_attack(obj, x_tr, y_tr, num_flips, mode)
            %Adversarial label flip attack (Correlated Clusters)

                [N,~] = size(x_tr);

                switch mode
                    case 'merge-n-delete'
                    mergeFunc = @(x,y) xor(x,y);
                    deleteAfterMerge = 1;

                    case 'merge-n-keep'
                    mergeFunc = @(x,y) max(x,y);
                    deleteAfterMerge = 0;

                    case 'xor-n-keep'
                    mergeFunc = @(x,y) xor(x,y);
                    deleteAfterMerge = 0;

                    otherwise
                        disp('ERROR: unknown mode');
                    return;
                end

                % This matrix represents the current clustering: rows are the data points & columns are the cluster they belong to
                clusterMat = eye(N);
                % This matrix represents the errors achieved by each cluster.
                clusterError = zeros(N,1);

                % NOTE: DO NOT Record these candidate clusterings; since
                % all singlets are considered, we can implicitly rule them
                % out in the future
                candidateClusters = [];

                % Compute the error of the classifier on unflipped data as a baseline
                %[~,baseError,~] = errorFunc(learner(x_tr,y_tr),x_tr,y_tr);
                baseError  = obj.errorFunc(y_tr,y_tr,x_tr);

                % Compute the error of flipping each point (cluster) individually &
                % record the best
                bestError = -inf;
                bestLabeling = [];
                for i = 1:size(clusterMat,2)
                    % Perturb the label for cluster i.
                    flippedLbl = flipClusterLabel( y_tr, clusterMat(:,i) );
                    % Learn on the flipped labels & assess their resulting error
                    E  = obj.errorFunc(y_tr,flippedLbl,x_tr);
                    error = (E - baseError);
                    % Record the error of this singleton cluster
                    clusterError(i) = error;
                    % Check to see if it achieved the best error thus far seen.
                    if error > bestError
                        bestError = error;
                        bestLabeling = flippedLbl;
                    end
                end
                
                if(~obj.quiet)
                    disp(['Initial best error ' num2str(bestError) ' with cluster size 1.'])
                end
                
                if num_flips == 1
                    y_prime = bestLabeling;
                    return;
                end

                avgPairErrs = -inf(size(clusterMat,2),size(clusterMat,2));
                for i = 1:size(clusterMat,2)
                    parfor j = (i+1):size(clusterMat,2)
                        % compute the merger proposal
                        mergeCandidate = mergeFunc( clusterMat(:,i), clusterMat(:,j) );

                        % if the candidate has no flips, skip it
                        if ~any(mergeCandidate)
                            continue;
                        end

                        % Note: Here we assume all mergers of singlets are unique

                        % Perturb the label for (disjoint) clusters i & j.
                        flippedLbl = flipClusterLabel( y_tr, mergeCandidate);

                        % Learn on the flipped labels & assess their resulting error
                        E  = obj.errorFunc(y_tr,flippedLbl,x_tr); %#ok
                        
                        % NOTE: DO NOT Record these candidate clusterings; since
                        % all pairs of size 2 are used, we can implicitly rule them
                        % out in the future

                        avgPairErrs(i,j) = (E - baseError);% / sum(mergeCandidate);
                    end
                end

                clsz = cluster_size(clusterMat);
                while max( clsz ) < num_flips
                    % Find the best pair of clusters to merge
                    [ row_max col_argmaxes ] = max( avgPairErrs, [], 2 ); % max for each row - max over the 2nd dimension
                    [ total_max row_argmax ] = max( row_max );  % max for total matrix - max over the 1st dimension
                    col_argmax = col_argmaxes(row_argmax);
                    assert(row_argmax ~= col_argmax);

                    % Merge the two best clusters
                    mergedCluster = mergeFunc( clusterMat(:,row_argmax), clusterMat(:,col_argmax) );

                    %if(~obj.quiet)
                    %    disp(total_max);
                    %    disp(find(mergedCluster==1));
                    %end

                    if deleteAfterMerge
                        mergeTarget = min(row_argmax,col_argmax);
                        delTarget = max(row_argmax,col_argmax);

                        % Replace the 1st cluster with the new cluster & erase the old
                        % entries for its errors.
                        clusterMat(:,mergeTarget)  = mergedCluster;
                        avgPairErrs(mergeTarget,:) = -inf(1,size(clusterMat,2));
                        avgPairErrs(:,mergeTarget) = -inf(size(clusterMat,2),1);

                        % Delete the 2nd cluster & its entries
                        clusterMat(:,delTarget)  = [];
                        clusterError(delTarget) = [];
                        avgPairErrs(delTarget,:) = [];
                        avgPairErrs(:,delTarget) = [];
                    else
                        % Delete the entry for the merged pair of clusters so that that
                        % entry will not be reconsidered in the future.
                        avgPairErrs(row_argmax,col_argmax) = -inf;
                        avgPairErrs(col_argmax,row_argmax) = -inf;

                        mergeTarget = size(clusterMat,2) + 1;

                        % Add the new cluster to the cluster matrix & add entries for
                        % its errors
                        clusterMat(:,mergeTarget)  = mergedCluster;
                        avgPairErrs(mergeTarget,:) = -inf(1,size(clusterMat,2)-1);
                        avgPairErrs(:,mergeTarget) = -inf(size(clusterMat,2),1);
                    end

                    clsz = cluster_size(clusterMat);

                    % Check to see if the new cluster has a better error then
                    % our best previous cluster
                    candidateLbl = flipClusterLabel( y_tr, mergedCluster );
                    E  = obj.errorFunc(y_tr,candidateLbl,x_tr);
                    if (E-baseError) > bestError
                        bestError = (E-baseError);
                        bestLabeling = candidateLbl;
                        if(~obj.quiet)
                            disp(['New best error ' num2str(bestError) ' with cluster size ' num2str(clsz(mergeTarget)) '.']);
                        end
                    end

                    % Recompute the pairwise errors for merging the new cluster with any
                    % other cluster
                    for i = 1:size(clusterMat,2)
                        if i == mergeTarget
                            continue;
                        end

                        % compute the merger proposal
                        mergeCandidate = mergeFunc( clusterMat(:,i), clusterMat(:,mergeTarget) );

                        % if the candidate has no flips, skip it
                        if ~any(mergeCandidate)
                            continue;
                        end

                        % If a potential clustering has already been created, there is
                        % no reason to re-create it.  Thus, if this is the case, we
                        % make the error for the new proposal -inf; otherwise, we
                        % compute the error for the proposal.
                        if (~deleteAfterMerge) && clusterRepresented(candidateClusters, mergeCandidate)
                            continue;
                        else
                            % Perturb the label for (disjoint) clusters i & j.
                            flippedLbl = flipClusterLabel( y_tr, mergeCandidate);

                            % Learn on the flipped labels & assess their resulting error
                            %[~,E,~] = errorFunc(learner(x_tr,flippedLbl),x_tr,y_tr);
                            E  = obj.errorFunc(y_tr,flippedLbl,x_tr);
                            
                            % Record the candidate clustering so it isn't retried,
                            % however, only candidates that are merged with
                            % non-singletons need to be recorded;
                            % singleton-merge-matching is done implicitly.
                            %if i > N
                            %    candidateClusters(:,end+1) = mergeCandidate;
                            %end

                            if i < mergeTarget
                                avgPairErrs(i,mergeTarget) = (E - baseError);% / sum(mergeCandidate);
                            else
                                avgPairErrs(mergeTarget,i) = (E - baseError);% / sum(mergeCandidate);
                            end
                        end
                    end

                    %candidateClusters(:,end+1) = mergedCluster;
                end

                % TEST RETURN - REPLACE ONCE ATTACK IS FULLY REALIZED
                y_prime = bestLabeling;
            end
        
        
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end





function [clMat,newCol,killCol] = merge_cluster(clMat,i,j)
    [~,M] = size(clMat);
    assert(i <= M && j <= M);
    if i > j
        [clMat,newCol,killCol] = merge_cluster(clMat,j,i);
    else
        clMat(:,i) = clMat(:,i) + clMat(:,j);
        clMat(:,j) = [];
        newCol = i;
        killCol = j;
    end
end

function repped = clusterRepresented(clMat, newCluster)
    if sum(newCluster) <= 2
        repped = 1;
    elseif isempty(clMat)
        repped = 0;
    else
        repped = (max(clMat' * newCluster) >= sum(newCluster)-1);
    end
    %repped = any( ismember(clMat',newCluster','rows') );
    %repped = any(arrayfun( @(c) all(clMat(:,c)==newCluster), 1:size(clMat,2) ));
end

function sz = cluster_size(clMat)
    [N,~] = size(clMat);
    sz = ones(1,N) * clMat;
end

function newLbl = flipClusterLabel( lbls, cluster )
    newLbl = lbls .* (1 - 2*cluster);
end


