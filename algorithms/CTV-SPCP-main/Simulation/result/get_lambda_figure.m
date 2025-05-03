inter = result.inter;
err = result.err;
err1 = result.err1;
number  = result.number;
figure;plot(inter:inter:inter*number,err(1:number),'rp-','LineWidth',2);
% hold on;
% plot(inter:inter:inter*number,err1,'bp-','LineWidth',2);
xlabel('constant c of :$\lambda = c/\sqrt{n_1}$','interpreter','latex');
ylabel('relative error','interpreter','latex');
legend(['3DCTV-RPCA'])
%legend(['3DCTV-RPCA'], ['RPCA'])
set(gca, 'Fontname', 'Times New Roman','FontSize',24);
axis([0,inter*number, 0, err(number)+0.2])