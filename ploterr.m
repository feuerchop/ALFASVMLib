% plot the error-bar results 
flips=linspace(0,0.2,10);
flip_len=10;
axis('square');
hold('on');
box('on');
xlabel('Flip rate on training set');
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
            {'ALFATilt'},{'Far first'}, {'Near first'},  ...
            {'Random'}, {'Correlated Cluster'}]);