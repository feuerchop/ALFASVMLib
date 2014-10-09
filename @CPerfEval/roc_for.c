#include "mex.h"

void mexFunction(int nlhs, mxArray *plhs[], 
    int nrhs, const mxArray *prhs[])
{
     int i, j, s1, s2, numel_t, numel_n, numel_p;
     double *t, *fp, *tp, *n, *p;
     if (nrhs != 3)
     mexErrMsgTxt("Wrong inputs.");


    /* Find the number of elements of n, p, thresholds */
    s1 = mxGetM(prhs[1]);
    s2 = mxGetN(prhs[1]);
    numel_n=s1*s2;
    
    s1 = mxGetM(prhs[2]);
    s2 = mxGetN(prhs[2]);
    numel_p=s1*s2;
    
    s1 = mxGetM(prhs[0]);
    s2 = mxGetN(prhs[0]);
    numel_t=s1*s2;

    /* Create an mxArray for fp,tp of numel_t elements */
    plhs[0] = mxCreateDoubleMatrix(s1, s2, mxREAL);
    plhs[1] = mxCreateDoubleMatrix(s1, s2, mxREAL);


    /* ptr to thresholds, fp, tp */
    t = mxGetPr(prhs[0]);
    n = mxGetPr(prhs[1]);
    p = mxGetPr(prhs[2]);
    
    fp = mxGetPr(plhs[0]);
    tp = mxGetPr(plhs[1]);

    for (i = 0; i < numel_t; i++) {
        fp[i]=0;
        tp[i]=0;
        for (j = 0; j < numel_n; j++)
            fp[i] += (n[j]>=t[i]);
        for (j = 0; j < numel_p; j++)
            tp[i] += (p[j]>=t[i]);
       
    }
}



