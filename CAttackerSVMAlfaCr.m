classdef CAttackerSVMAlfaCr < CAttackerSVM
% CAttackerSVMAlfaCr - Adv. Label Flip Attack with continuous label relaxation
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
     
    properties
       maxiter;     %maximum number of iterations
       step;        %gradient step size
       rho;         %attacker's cost of errors (as C for SVM)
    end
        
    
    methods
        % Default (empty) Constructor
        function obj = CAttackerSVMAlfaCr(varargin)
            obj = obj@CAttackerSVM(varargin{:});
            obj.name = 'alfa-cr';
            obj.maxiter=2000;
            obj.step=0.1;
            obj.rho=obj.C;
        end
     
        % setters
        function set.maxiter(obj, iter)
            if iter <= 0
                display('The number of maximal iteration must be a positive integer!');
            else
                obj.maxiter = iter;
            end
        end
        
        function set.step(obj, ss)
            if ss <= 0 
                display('The gradient step size must be positive!');
            else
                obj.step = ss;
            end
        end
        
        function set.rho(obj, r)
            if r <= 0
                display('The cost of errors must be a positive value!');
            else
                obj.rho = r;
            end
        end
        
        
        function flipLabels(obj, tr_labels, tr)
            % CAttackerSVMAlfaCr.flipLabels
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
            obj.flipped_labels =obj.alfa_cr(tr, tr_labels,num_flips);
            
            idx = find(obj.flipped_labels~=tr_labels); 
            obj.flip_idx = idx(1:num_flips);
            obj.train(obj.flipped_labels, tr);
        end
        
    end   
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    methods %These should be private    
        
        %TODO: improve these functions using the current SVM implementation
        function [y_prime err xi_obj alpha_seq]=alfa_cr(obj, tr, tr_labels,num_flips)

            obj.rho=obj.C; 
            
            obj.train(tr_labels, tr);
            
            [alpha b] = obj.getDualCoeff();
            margin_SV_idx = obj.getMarginSVs();
            
            alpha_seq(1,:) = alpha;

            % evaluation the performance of the trained SVM
            [yclass, score] = obj.classify(tr);
            err(1) = sum(yclass~=tr_labels)/size(tr,1);

            %kernel matrix for TR and VD
            K=obj.getKernel();

            y_prime = tr_labels; 
            y_prime_best=y_prime;

            flipped_back=0;

            % compute the margin conditions for validation samples
            g = tr_labels.*score -1;

            xi_obj(1) = obj.alfa_cr_obj(alpha,K,y_prime,g);

            f=sum(0.5*abs(y_prime-tr_labels));
            i=0;
            while(f < 1.05*num_flips)
                i=i+1;

                %compute attack direction and update the label flips
                dy = obj.alfa_cr_gradient(y_prime,margin_SV_idx,g,alpha,b,K,tr_labels);

                if(sum(abs(dy))>0)

                    %err_best=err(i);
                    err_best=xi_obj(i);

                    y_prime=y_prime+obj.step*dy; 
                    y_prime(y_prime>2)=2;
                    y_prime(y_prime<-2)=-2;

                    %update alpha,b (with continuous labels)
                    [alpha b]=dualSVM(K,y_prime,obj.C,alpha);

                    score_tmp = K*(y_prime.*alpha) + b;
                    g_tmp = tr_labels.*score_tmp -1;
                    err_tmp = obj.alfa_cr_obj(alpha,K,y_prime,g_tmp);

                    if(err_tmp >= err_best)
                        y_prime_best=y_prime;
                    end
                end

                if( sum(abs(dy))==0 || mod(i,floor(obj.maxiter/num_flips))==0)
                   f=f+1;
                   %flip the L candidate labels
                   y_prime = obj.alfa_cr_flip(tr_labels,y_prime_best,min(f,num_flips));

                   [y_prime flipped_back]= obj.alfa_cr_undo_flip(tr,tr_labels,y_prime,K);
                   %f=f-flipped_back;

                   obj.train(y_prime, tr);
                   [alpha b] = obj.getDualCoeff();
   
                end

                margin_SV_idx = obj.getMarginSVs();
                
                % evaluate the performance of the updated SVM
                score = K*(y_prime.*alpha) + b;

                yclass = sign(score);
                err(i+1) = sum(yclass~=tr_labels)/size(tr,1);

                % compute the influence of the attack point on the errors
                g = tr_labels.*score -1;
                xi_obj(i+1) = obj.alfa_cr_obj(alpha,K,y_prime,g);

                if(mod(i,floor(obj.maxiter/num_flips))==0 || sum(abs(dy))==0 )
                    if ~obj.quiet
                        disp(['(alfa-cr) Training error (%): ' num2str(100*err(i+1),2), ...
                            ' Obj: ' num2str(xi_obj(i+1),4), ...
                            ' num. flipped: ' num2str(sum(0.5*abs(y_prime-tr_labels))) ...
                            ' ('  num2str(sum(0.5*abs(y_prime_best-tr_labels))) ...
                            ') flipped back: ' num2str(flipped_back)  ' step: ' num2str(obj.step) ]);
                    end
                end
                alpha_seq(i+1,:) = alpha;

            end
        end
    
        function objective=alfa_cr_obj(obj,alpha,K,y_prime,g)
            objective = 0.5*alpha'*(K.*(y_prime*y_prime'))*alpha +obj.rho*sum(max(0,-g));     
        end
        
        function dy = alfa_cr_gradient(obj,y_prime,margin_SV_idx,g,alpha,b,K,y_tr)
    
            % if there are no margin support vectors, gradient is null.
            if(isempty(margin_SV_idx))
                dy=zeros(numel(y_prime),1);
                return;
            end

            R = [K(margin_SV_idx,margin_SV_idx).*(y_prime(margin_SV_idx)*y_prime(margin_SV_idx)')+ ...
                1E-12*eye(numel(margin_SV_idx)) ...
                y_prime(margin_SV_idx); y_prime(margin_SV_idx)' 0 ];
            
            S = diag(K*(alpha.*y_prime) + b);
            
            delta = -R \ [ (y_prime(margin_SV_idx)*alpha').* K(margin_SV_idx,:)+S(margin_SV_idx,:); alpha'];

            %diff alpha / y, diff b / y
            da = delta(1:numel(margin_SV_idx),:);
            db = delta(end,:)';

            Qis = K(:,margin_SV_idx).*(y_tr*y_prime(margin_SV_idx)');

            delta_gi = Qis*da + K.*(y_tr*alpha') + y_tr*db';
            delta_gi(g>=0,:)=0; %hinge loss

            %margin derivative
            delta_w = alpha.*(K*(alpha.*y_prime))+da'*(K(margin_SV_idx,:).*(y_prime(margin_SV_idx)*y_prime'))*alpha;

            gradient = delta_w'-obj.rho*sum(delta_gi); 


            gradient(isnan(gradient))=0;
            if(norm(gradient)>0)
                dy =  (gradient / norm(gradient,2))';
            else
                disp('Null gradient!')
                dy=zeros(numel(y_prime),1);
            end
        end

        function y_prime_new=alfa_cr_flip(obj,tr_labels,y_prime,num_flips) %#ok 
            
            %sort the continuous labels y_prime and flip L
            [~, idx] = sort(abs(y_prime-tr_labels),'descend');

            %flip the first L
            y_prime_new = tr_labels;
            y_prime_new(idx(1:num_flips)) = -tr_labels(idx(1:num_flips));
        end
        
        function [y_prime_new flipped]=alfa_cr_undo_flip(obj,tr,tr_labels,y_prime,K)

            
            obj.train(y_prime, tr);
            [~, score]=obj.classify(tr);
            [alpha b]=obj.getDualCoeff();
            
            g=tr_labels.*score-1;
            objective=obj.alfa_cr_obj(alpha,K,y_prime,g);

            %sort the continuous labels y_prime and flip L
            flipped_idx = find(y_prime~=tr_labels);
            flipped_idx = flipped_idx(randperm(numel(flipped_idx)));

            flipped=0;

            y_prime_new=y_prime;
            for i=1:numel(flipped_idx)

                y_prime_new(flipped_idx(i))=-y_prime_new(flipped_idx(i)); %flip one back
                %disp(y_prime_new')
       
                obj.train(y_prime_new, tr);
                [~, score]=obj.classify(tr);
                [alpha b]=obj.getDualCoeff();
                
                g=tr_labels.*score-1;
                obj_tmp=obj.alfa_cr_obj(alpha,K,y_prime_new,g);
                if(obj_tmp >= objective*0.95)
                    %disp(obj_tmp)
                    flipped=flipped+1;
                else
                    %set flip back
                    y_prime_new(flipped_idx(i))=-y_prime_new(flipped_idx(i));
                end
            end

            y_prime_new = y_prime;
        end
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end


%This is used to learn/update the SVM with soft labels...
function [alpha b]=dualSVM(K,y,C,alpha0)

    n=size(K,1);

    if(nargin < 4)
        alpha0=[];
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%
    % QUADPROG
    % 
    % min 0.5*x'*H*x + f'*x
    %  x
    % subject to:  A*x <= b 
    %              
    %%%%%%%%%%%%%%%%%%%%%%%%%              
    % x = alpha            


    % Symmetric matrix D must be positive definite. Its diagonal has 
    % to be dominant, i.e. D(i,i) > sum_{j ~= i}  D(i,j).
    Q  = (y*y').*K;
    D = Q + (1E-12) * eye(n);
    D = 0.5*(D+D'); %enforcing simmetry

    f  = -ones(1,n);
    Aeq  = y';
    beq  = 0;
    lb = zeros(n,1);
    ub = repmat(C,n,1);

    opt = optimset; opt.TolKKT=1E-6; opt.verb=0; opt.MaxIter=inf;
    alpha = libqp_gsmo(D,f,Aeq,beq,lb,ub,alpha0,opt);

    alpha(alpha < 1E-9) = 0; %rounding
    alpha(alpha > C-1E-9) = C; %rounding
    SV_idx = find(alpha>0 & alpha < C);

    if(~isempty(SV_idx))
        score_margin_SV = K(SV_idx,:)*(alpha.*y);
        b=y(SV_idx)-ones(numel(SV_idx),1).*score_margin_SV;
        b=mean(b);
    else %from libSVM doc. http://www.csie.ntu.edu.tw/~cjlin/papers/libsvm.pdf, page 14
        score = K*(alpha.*y);
        score = score-y;    

        m = min([score(alpha<=0 & y==1); score(alpha>=C & y==-1)]);
        M = max([score(alpha<=0 & y==-1); score(alpha>=C & y==+1)]);

        if(isempty(m))
            b=-M;
            return;
        end

        if(isempty(M))
            b=-m;
            return;
        end

        %else...
        b=-0.5*(m+M);


    end
end




