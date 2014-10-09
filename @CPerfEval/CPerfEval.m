% *************************************************************************
% Matlab Class for Classifier Performance Evaluation
% Copyright (2014) Battista Biggio (battista.biggio@diee.unica.it)
%
% Pattern Recognition and Applications Laboratory
% http://pralab.diee.unica.it
% Department of Electrical and Electronic Engineering, 
% University of Cagliari
%  
% Last updated: 19 Jun 2014 
% *************************************************************************


classdef CPerfEval < handle
    
    properties
       criterion;
       param; %can be max fp rate for pAUC, max_fp for TP at FP, etc.
    end
    
    properties (Constant)
        SUPPORTED_CRITERION = {'accuracy', 'auc', 'pauc', 'tp_at_fp', 'eer'};
    end
        
    methods
       
        
        function obj=CPerfEval()
            obj.criterion='accuracy';
            obj.param=[];
        end
        
    end
    
    methods
        
        function setCriterion(obj,crit)
            obj.criterion=crit;
        end
        
        function setParam(obj,p)
            obj.param=p;
        end
        
       
        function [perf th] = performance(obj,y,score) 
           
            th=0;
            switch obj.criterion
                case 'accuracy'
                    perf=1-obj.zero_one_loss(y,score);
                case 'auc'
                    perf = obj.AUC(y,score);
                case 'pauc'
                    [fp tp] = obj.ROC(y,score);
                    perf = obj.pAUC(fp,tp,obj.param);
                case 'tp_at_fp'
                    [fp tp th] = obj.ROC(y,score);
                    [fn th] = obj.FNatFP(fp,tp,th,obj.param); %param here is the desired FP rate
                    perf=1-fn;
                case 'eer'
                    [fp tp] = obj.ROC(y,score);
                    perf = obj.EER(fp,tp);
            end
           
        end
    end
    
    
    methods (Static)
        
        
        
        
        %zero-one loss (threshold = 0)
        function err=zero_one_loss(y,score)
           err=sum((2*(score>=0)-1)~=y)/numel(y); 
        end
        function [fp tp th] = ROC(y,score)
            
            n=score(y==-1);
            p=score(y==+1);
            
            if(size(n,1) > size(n,2))
                n=n';
            end

            if(size(p,1) > size(p,2))
                p=p';
            end


            th = unique([n, p]);
            %now ROC will go from (0,0) to (1,1)
            th = [min(th)-1E-3, th, max(th)+1E-3];

            %%%%%%%%%% MEX FILE %%%%%%%%%
            [fp tp] = roc_for(th,n,p);
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

            fp=fp/numel(n);
            tp=tp/numel(p);
        end
        function [fp tp th] = ROCapprox(y,score)
            %ROC approssimata ma piu' veloce. su 1000 punti
            n=score(y==-1);
            p=score(y==+1);
            if(size(n,1)>1)
                n=n';
            end
            if(size(p,1)>1)
                p=p';
            end

            %quantization step
            N=999;
            %normalizing scores in [0,1]
            maximum = max([n p]);
            minimum = min([n p]);
            n = (n-minimum)/(maximum-minimum);
            p = (p-minimum)/(maximum-minimum);
            fp=zeros(1,N+1);
            tp=zeros(1,N+1);
            for th=0:N
                fp(th+1)=sum(n>th/N);
                tp(th+1)=sum(p>th/N);
            end

            %rebulding ROC thresholds
            th=0:N;
            th=th/N*(maximum-minimum)+minimum;

            tp=tp/numel(p);
            fp=fp/numel(n);
        end
        function [fp, tp] = meanROC(fpSet, tpSet)

            fp = 0:0.001:1;

            numRep = size(fpSet,2);
            for rep=1:numRep

                x = unique(fpSet{rep});

                if(numel(x)>1)

                    for i=1:numel(x)
                        %media i tp corrispondenti allo stesso fp
                        tmp = tpSet{rep}(fpSet{rep}==x(i)); 
                        y(i)=mean(tmp);
                    end

                    %ricampiona per avere la stessa ascissa
                    tp(rep,:) = interp1(x,y,fp,'linear');

                end

                clear x y
            end

            if(exist('tp','var'))
                tp = mean(tp);
            else
                fp = [];
                tp = [];
            end


        end
        %Wilcoxon-Mann-Whitney AUC's statistic
        function auc = AUC(y,score)
            n=score(y==-1);
            p=score(y==+1);
            auc=0;
            for i=1:numel(p)
                for j=1:numel(n)
                    if(p(i)>n(j))
                        I=1;
                    elseif(p(i)==n(j))
                        I=0.5;
                    else
                        I=0;
                    end
                    auc = auc + I;
                end
            end
            auc=auc/(numel(n)*numel(p));
        end
        function auc = pAUC(fp,tp,fpRate)
            if(nargin<3)
                fpRate=1;
            end

            x=fp(fp<=fpRate);
            y=tp(fp<=fpRate);

            if(numel(x)<=1 || numel(y)<=1)
                auc=0;
                return;
            end

            if(x(1)~=fpRate)
                %linear interpolation
                t=y(1)+(fpRate-x(1))*(min(tp(fp>fpRate))-y(1))/(min(fp(fp>fpRate)-x(1)));
                y=[t y];
                x=[fpRate x];
            end

            auc=-1/fpRate*trapz(x,y);
        end
        function [f t] = FNatFP(fp,tp,th,at_fp)

%             if(sum(fp <= at_fp)==0)
%                 f=max(1-tp(fp >= at_fp));
%                 t=max(th(fp >= at_fp));
%                 return;
%             end
% 
%             if(sum(fp >= at_fp)==0)
%                 f=min(1-tp(fp <= at_fp));
%                 t=min(th(fp <= at_fp));
%                 return;
%             end

            x(1) = max(fp(fp <= at_fp));
            y(1) = min(1-tp(fp <= at_fp));
            thr(1) = min(th(fp <= at_fp));

            x(2) = min(fp(fp >= at_fp));
            y(2) = max(1-tp(fp >= at_fp));
            thr(2) = max(th(fp >= at_fp));

            if(x(1)==x(2))
                f=y(1);
                t = thr(1);
                return;
            end

%             %se un punto e' al confine della ROC,
%             %ritorna solo l'altro valore
%             if(x(1)==0 || x(1) ==1)
%                 f=y(2);
%                 t = thr(2);
%                 return;
%             end
% 
%             if(x(2)==0 || x(2) ==1)
%                 f=y(1);
%                 t = thr(1);
%                 return;
%             end

            f=interp1(x,y,at_fp);
            t=interp1(x,thr,at_fp);

        end
        function [f t] = FPatFN(fp,tp,th,at_fn)
            x(1) = max(1-tp(1-tp <= at_fn));
            y(1) = min(fp(1-tp <= at_fn));
            thr(1) = min(th(1-tp <= at_fn));

            x(2) = min(1-tp(1-tp >= at_fn));
            y(2) = max(fp(1-tp >= at_fn));
            thr(2) = max(th(1-tp >= at_fn));

            if(x(1)==x(2))
                f=y(1);
                t=thr(1);
                return;
            end

            %se un punto e' al confine della ROC,
            %ritorna solo l'altro valore
            if(x(1)==0 || x(1) ==1)
                f=y(2);
                t=thr(2);
                return;
            end

            if(x(2)==0 || x(2) ==1)
                f=y(1);
                t=thr(1);
                return;
            end


            f=interp1(x,y,at_fn);
            t=interp1(x,thr,at_fn);
        end
        function eer = EER(fp,tp)
            [~, c] = min(abs(fp+tp-1));
            eer = 0.5 *(fp(c) + 1 - tp(c));
        end
        
        %Cumulative Matching Characteristic curve.
        %y are the true labels
        %S is the set of scores of each query (rows) against all templates (columns)
        function  [rank cmc] = CMC(y,S)

            %for each query (testing sample):
            users = unique(y);
            cmc = zeros(numel(users),1);
            rank = 1:numel(users);

            for q=1:numel(y)
                [~, idx] = sort(S(q,:),'descend');
                r = find(idx == y(q));
                cmc(users(r):end)=cmc(users(r):end)+1;
            end

            cmc=cmc/cmc(end);

        end
        
        
        %% PLOT FUNCTIONS
        function plotROC(fp,tp,ptitle,c)

            if(nargin<3)
                ptitle='data';
                c='b';
            end

            if(nargin<4)
                c='b';
            end

            %Plot ROC
            %axis square
            %axis([0 1 0 1]);

            h=plot(fp,tp,c,'MarkerSize', 12);
            %h=semilogx(fp,tp,c,'MarkerSize', 12);
            grid on

            set(h,'DisplayName', ptitle);
            hold on

            xlabel('FP')
            ylabel('TP')
        end
        function plot_hists(y,score)
            n=score(y==-1);
            p=score(y==+1);
            
            % plot distribution histograms
            M = max([p; n]);
            m = min([p; n]);

            p = (p - m)./(M-m);
            n = (n - m) / (M-m);

            bins = 0:0.01:1;

            hgen = hist(p,bins) ./ numel(p);
            himp = hist(n,bins)./numel(n);

            %hgen(hgen==0)=nan;
            %himp(himp==0)=nan;

            plot(bins,hgen,'b.-');
            hold on
            plot(bins,himp,'r.-');
            grid on
            legend('pos','neg')
        end
        function plotDET(fp, tp, ptitle, color)
            %function plotDET(fp,tp, ptitle, color)
            %
            %  Plot_DET plots detection performance tradeoff on a DET plot
            %  and returns the handle for plotting.
            %
            %  fp and tp are the vectors of true and false
            %  positive rates. fn (1-tp) vs fp will be plotted.
            %
            %  The usage of plotDET is analogous to the standard matlab
            %  plot function.
            %
            %
            % ptitle: plot title
            % color: e.g., 'b.-'. Default 'b'.
            %

            Pmiss = 1-tp;
            Pfa = fp;

            Npts = max(size(Pmiss));
            if Npts ~= max(size(Pfa))
                    error ('vector size of tp and fp not equal in call to plotDET');
            end

            %------------------------------
            % plot the DET

            if (nargin < 3)
                ptitle = '';
                color = 'b';
            end

            h=plot(ppndf(Pfa), ppndf(Pmiss), color); hold on;
            set(h,'DisplayName',ptitle);


            pticks = [0.00001 0.00002 0.00005 0.0001  0.0002   0.0005 ...
                      0.001   ... %0.002  
                      0.005   0.01    0.02     0.05 ...
                      0.1     0.2     0.4     0.6     0.8      0.9 ...
                      0.95    0.98    0.99    0.995   0.998    0.999 ...
                      0.9995  0.9998  0.9999  0.99995 0.99998  0.99999];

            xlabels = [' 0.001' ; ' 0.002' ; ' 0.005' ; ' 0.01 ' ; ' 0.02 ' ; ' 0.05 ' ; ...
                       '  0.1 ' ; %'  0.2 ' ;
                       ' 0.5  ' ; '  1   ' ; '  2   ' ; '  5   ' ; ...
                       '  10  ' ; '  20  ' ; '  40  ' ; '  60  ' ; '  80  ' ; '  90  ' ; ...
                       '  95  ' ; '  98  ' ; '  99  ' ; ' 99.5 ' ; ' 99.8 ' ; ' 99.9 ' ; ...
                       ' 99.95' ; ' 99.98' ; ' 99.99' ; '99.995' ; '99.998' ; '99.999'];

            ylabels = xlabels;

            %---------------------------
            % Get the min/max values of Pmiss and Pfa to plot

            DET_limits = Set_DET_limits;


            Pmiss_min = DET_limits(1);
            Pmiss_max = DET_limits(2);
            Pfa_min   = DET_limits(3);
            Pfa_max   = DET_limits(4);

            %----------------------------
            % Find the subset of tick marks to plot

            ntick = max(size(pticks));
            for n=ntick:-1:1
                if (Pmiss_min <= pticks(n))
                    tmin_miss = n;
                end
                if (Pfa_min <= pticks(n))
                    tmin_fa = n;
                end
            end

            for n=1:ntick
                if (pticks(n) <= Pmiss_max)
                    tmax_miss = n;
                end
                if (pticks(n) <= Pfa_max)
                    tmax_fa = n;
                end
            end

            %-----------------------------
            % Plot the DET grid

            set (gca, 'xlim', ppndf([Pfa_min Pfa_max]));
            set (gca, 'xtick', ppndf(pticks(tmin_fa:tmax_fa)));
            set (gca, 'xticklabel', xlabels(tmin_fa:tmax_fa,:));
            set (gca, 'xgrid', 'on');
            xlabel ('FP (in %)');


            set (gca, 'ylim', ppndf([Pmiss_min Pmiss_max]));
            set (gca, 'ytick', ppndf(pticks(tmin_miss:tmax_miss)));
            set (gca, 'yticklabel', ylabels(tmin_miss:tmax_miss,:));
            set (gca, 'ygrid', 'on')
            ylabel ('FN (in %)')

            set (gca, 'box', 'on');
            axis('square');
            axis(axis);
        end
        function DET_limits = Set_DET_limits(Pmiss_min, Pmiss_max, Pfa_min, Pfa_max)
        % function Set_DET_limits(Pmiss_min, Pmiss_max, Pfa_min, Pfa_max)
        %
        %  Set_DET_limits initializes the min.max plotting limits for P_min and P_fa.
        %
        %  See DET_usage for an example of how to use Set_DET_limits.

        %cambia qui i limiti!
        Pmiss_min_default = 0.0005+eps;
        Pmiss_max_default = 0.99-eps;
        Pfa_min_default = 0.0005+eps;
        Pfa_max_default = 0.99-eps;


        if ~(exist('Pmiss_min')); Pmiss_min = Pmiss_min_default; end;
        if ~(exist('Pmiss_max')); Pmiss_max = Pmiss_max_default; end;
        if ~(exist('Pfa_min')); Pfa_min = Pfa_min_default; end;
        if ~(exist('Pfa_max')); Pfa_max = Pfa_max_default; end;

        %-------------------------
        % Limit bounds to reasonable values

        Pmiss_min = max(Pmiss_min,eps);
        Pmiss_max = min(Pmiss_max,1-eps);
        if Pmiss_max <= Pmiss_min
            Pmiss_min = eps;
            Pmiss_max = 1-eps;
        end

        Pfa_min = max(Pfa_min,eps);
        Pfa_max = min(Pfa_max,1-eps);
        if Pfa_max <= Pfa_min
            Pfa_min = eps;
            Pfa_max = 1-eps;
        end

        %--------------------------
        % Load DET_limits with bounds to use

        DET_limits = [Pmiss_min Pmiss_max Pfa_min Pfa_max];
        end
        function norm_dev = ppndf (cum_prob)
            %function ppndf (prob), used by plotDET
            %The input to this function is a cumulative probability.
            %The output from this function is the Normal deviate
            %that corresponds to that probability.  For example:
            %  INPUT   OUTPUT
            %  0.001   -3.090
            %  0.01    -2.326
            %  0.1     -1.282
            %  0.5      0.0
            %  0.9      1.282
            %  0.99     2.326
            %  0.999    3.090

             SPLIT =  0.42;

             A0 =   2.5066282388;
             A1 = -18.6150006252;
             A2 =  41.3911977353;
             A3 = -25.4410604963;
             B1 =  -8.4735109309;
             B2 =  23.0833674374;
             B3 = -21.0622410182;
             B4 =   3.1308290983;
             C0 =  -2.7871893113;
             C1 =  -2.2979647913;
             C2 =   4.8501412713;
             C3 =   2.3212127685;
             D1 =   3.5438892476;
             D2 =   1.6370678189;

            % the following code is matlab-tized for speed.
            % on 200000 points, time went from 76 seconds to 5 seconds!
            % original routine is included at end for reference

            [Nrows Ncols] = size(cum_prob);
            norm_dev = zeros(Nrows, Ncols); % preallocate norm_dev for speed
            cum_prob(find(cum_prob>= 1.0)) = 1-eps;
            cum_prob(find(cum_prob<= 0.0)) = eps;

            R = zeros(Nrows, Ncols); % preallocate R for speed

            % adjusted prob matrix
            adj_prob=cum_prob-0.5;

            centerindexes = find(abs(adj_prob) <= SPLIT);
            tailindexes   = find(abs(adj_prob) > SPLIT);

            % do centerstuff first
            R(centerindexes) = adj_prob(centerindexes) .* adj_prob(centerindexes);
            norm_dev(centerindexes) = adj_prob(centerindexes) .* ...
                                (((A3 .* R(centerindexes) + A2) .* R(centerindexes) + A1) .* R(centerindexes) + A0);
            norm_dev(centerindexes) = norm_dev(centerindexes) ./ ((((B4 .* R(centerindexes) + B3) .* R(centerindexes) + B2) .* ...
                                         R(centerindexes) + B1) .* R(centerindexes) + 1.0);


            % find left and right tails
            right = find(cum_prob(tailindexes)> 0.5);
            left  = find(cum_prob(tailindexes)< 0.5);

            % do tail stuff
            R(tailindexes) = cum_prob(tailindexes);
            % if prob > 0.5 then prob = 1-prob
            R(tailindexes(right)) = 1 - cum_prob(tailindexes(right));
            R(tailindexes) = sqrt ((-1.0) .* log (R(tailindexes)));
            norm_dev(tailindexes) = (((C3 .* R(tailindexes) + C2) .* R(tailindexes) + C1) .* R(tailindexes) + C0);
            norm_dev(tailindexes) = norm_dev(tailindexes) ./ ((D2 .* R(tailindexes) + D1) .* R(tailindexes) + 1.0);

            % swap sign on left tail
            norm_dev(tailindexes(left)) = norm_dev(tailindexes(left)) .* -1.0;

            %bat mod
            norm_dev(isnan(cum_prob))=nan; %otherwise it assigns zero!

        end

    end
    
    
end