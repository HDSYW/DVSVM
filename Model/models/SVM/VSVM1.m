function [PredictY,model] = VSVM(ValX, Trn, Para)
gam = Para.p1;
kpar = Para.kpar;
X = Trn.X;
Y = Trn.Y;
clear DataTrain
tt = tic;
K = KerF(X, kpar, X);
[m1, m2] = size(K);
Ke = [K, ones(m1,1)];
V = Para.V;
options = optimoptions('quadprog','Display','off');

H = Ke'*V*Ke + gam*[K, zeros(m1, 1); zeros(1, m1+1)];
H = (H + H');
f = -2*Ke'*V*Y;
A = [Ke; -Ke];
b = [ones(m1,1); zeros(m1,1)];
alphb = quadprog(H, f, A, b, [], [], [], [], [], options);
tr_time = toc(tt);

% % ------ output and prediction ---------
 model.tr_time = tr_time;
 model.n_SV = 0;

model.alpha = alphb(1:end-1);
model.b = alphb(end);
Predprob= KerF(ValX, kpar, X)*alphb(1:end-1) + alphb(end);
PredictY = Predprob >= 0.5;
model.prob = Predprob;

%  if Para.plt == 1
%      plt.ds = PredictY0;
%      plt.ss1 = plt.ds - 1;
%      plt.ss2 = plt.ds + 1;
%      model.plt = plt;
%  end

