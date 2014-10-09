% rbf
res=paper_exp2('dna','rbf', 500,1000); save('dna_rbf_500_1000.mat', 'res');
res=paper_exp2('seismic','rbf', 500,1000); save('seismic_rbf_500_1000.mat', 'res');
res=paper_exp2('ijcnn1','rbf', 500,1000); save('ijcnn1_rbf_500_1000.mat', 'res');
res=paper_exp2('acoustic','rbf', 500,1000); save('acoustic_rbf_500_1000.mat', 'res');
res=paper_exp2('splice','rbf', 500,1000); save('splice_rbf_500_1000.mat', 'res');