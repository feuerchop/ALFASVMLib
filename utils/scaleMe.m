function Z = scaleMe( X, options )
%SCALEME Summary of this function goes here
%   Scale the input data features either in [0,1] or [-1,1]
%   X is a NxD matrix containing N instances of D dimensional
%   if options=0, scale it in [0,1], unless [-1,1]
    
    [n,~]=size(X);
    MaxVal=max(X);
    MinVal=min(X);
    if nargin < 2 || options == 0
        % scale it in [0, 1], might be faster in learning
        Z = (X - repmat(MinVal, n, 1))./(repmat(MaxVal, n, 1) - repmat(MinVal, n, 1));
    else
        % scale it in [-1, 1]
        Z = 2*(X - repmat(MinVal, n, 1))./(repmat(MaxVal, n, 1) - repmat(MinVal, n, 1)) - 1;
    end


