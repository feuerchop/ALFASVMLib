classdef CAttackerSVMAlfaTilt < CAttackerSVM
% CAttackerSVMAlfaTilt - Adv. Label Flip Attack
%                       maximizing hyperplane tilt angle
%
% Superclass: CAttackerSVM
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
       maxiter;     %maximum number of iterations
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    
    methods
        % Default (empty) Constructor
        function obj = CAttackerSVMAlfaTilt(varargin)
            obj = obj@CAttackerSVM(varargin{:});
            obj.name = 'alfa-tilt';
            obj.maxiter=2000;
        end
     
        
        % setters
        function set.maxiter(obj, iter)
            if iter <= 0
                display('The number of maximal iteration must be a positive integer!');
            else
                obj.maxiter = iter;
            end
        end
        
        
        
        function flipLabels(obj, tr_labels, tr)
            % CAttackerSVMAlfaTilt.flipLabels
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
            % See also: CClassifierSVM.CClassifierSVM, CAttacker,
            %           CAttackerSVM.
            % -------
            
            if nargin < 3
                error('Not enough inputs, please input training labels and training set.');
            end
            
            % No flip at all
            if obj.flip_rate == 0 
                warning('Nothing happend, no label is flipped!');
                obj.flipped_labels = tr_labels;
                return;
            end
            
            num_flips = floor(length(tr_labels)*obj.flip_rate);
            
            %Call alfa-cr method
            obj.flipped_labels =obj.alfa_tilt(tr, tr_labels,num_flips);
            
            idx = find(obj.flipped_labels~=tr_labels); 
            obj.flip_idx = idx(1:num_flips);
            obj.train(obj.flipped_labels, tr);
        end
        
    end   
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    methods %These should be private    
        
        
        function y_prime =alfa_tilt(obj, tr, tr_labels, num_flips)

            obj.train(tr_labels, tr);
            alpha = obj.getDualCoeff();
            
            K=obj.getKernel();
            Q=K.*(tr_labels*tr_labels');
            
            score1=zeros(numel(tr_labels),1);
            for i=1:numel(tr_labels)
                score1(i,1) = abs(sum(alpha .* K(:,i)));
            end
            score1=score1/max(score1);

            f=-inf;
            for i=1:obj.maxiter

                random_alpha = obj.C*rand(numel(tr_labels),1);

                score2=zeros(numel(tr_labels),1);
                for k=1:numel(tr_labels)
                    score2(k,1) = tr_labels(k)*sum(random_alpha .* K(:,k));
                end
                score2=score2/max(score2);

                y_pois = tr_labels;
                
                %now sort alpha values in ascending order
                [~, labels_to_flip] = sort(alpha/obj.C -0.1*(score1+score2),'ascend');
                labels_to_flip = labels_to_flip(1:num_flips);
                y_pois(labels_to_flip)=-y_pois(labels_to_flip);

                
                obj.train(tr_labels, y_pois);
                alpha1 = obj.getDualCoeff();
            
                Q1=K.*(y_pois*y_pois');

                fnew = abs( (alpha.*tr_labels)' * K * (alpha1.*y_pois) / (sqrt(alpha'*Q*alpha) * sqrt(alpha1'*Q1*alpha)));
                
                if(fnew>f)
                    f=fnew;
                    lab_to_flip = labels_to_flip;
                end
            end

            y_prime = tr_labels;
            y_prime(lab_to_flip(1:num_flips))=-y_prime(lab_to_flip(1:num_flips));
        end
    
        
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end




