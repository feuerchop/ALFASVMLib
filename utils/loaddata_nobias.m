function set = loaddata_nobias(path, N)
% Load N instances of libsvm-data from PATH and make sure the
% 1/-1 labels are balanced

    [s_label, s_data] = libsvmread(path);
    if isempty(s_label)
        return
    end

    % make sure the data is enough
    if nargin < 2
        N = length(s_label);
    else
        N=min(N,length(s_label));
    end

    allPIdx=find(s_label>0);%+1
    allNIdx=find(s_label<0);%-1

    M=min([floor(N/2), length(allPIdx), length(allNIdx)]);

    selPIdx=randsample(allPIdx,M);
    selNIdx=randsample(allNIdx,M);
    % positive samples will always on top
    selIdx=[selPIdx;selNIdx];  

    s_label=s_label(selIdx,:);
    s_data=s_data(selIdx,:);

    set=full([s_data, s_label]);


