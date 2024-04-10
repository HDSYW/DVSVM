function [PredictY, model] = DVSVM_QP(ValX , X_C , Trn , Para)
V=Para.V;
kpar = Para.kpar;
itmax = Para.itmO;
X = Trn.X;
Y = Trn.Y;
[m, n] = size(X);
m_c=length(X_C);
barX = [KerF(X,kpar,X), ones(m,1)];
% barw = rand(n+1, 1); % initialization of barw
barw = zeros(m+1, 1);  % initialization of barw
options = optimoptions('quadprog','Display','off');
Para.n = n;
epsi = Para.epsi;
it=0;
%% ----- compute each item of Mij(barw), pMij(barw) ------
tt = tic;%记录时间
% ---------- 求三个已知矩阵 ----------
for i = 1:m_c
    for j = 1:m
%         C(j, i) = prod(X(i,:)>=X(j,:));  
%         C(j, i) = sparse(prod(X(i,:)>=X(j,:)));  
        C(j, i) = sum(X_C(i,:)>=X(j,:))/n; %在高维情况下加和求平均
    end
end

% for i = 1:m_c
%     for j = 1:m_c
%          CC{i,j} = C(:,i)*C(:,j)'; %按列相乘
%          g{i,j} = barX'*CC{i,j}*Y; 
%       g{i,j} = 2*barX'*CC{i,j}*Y+barX'*Cii*ones(m,1)-barX'*CC{i,j}*ones(m,1);
%     end
%     G{i} = barX'*CC{i,i}*barX;
%     h(i) = Y'*CC{i,i}*Y;
%     clear CC
% end
G = barX'*V*barX;
g = barX'*V*Y; 
h = Y'*V*Y;
% ---------- 求Mij ----------
while it<itmax
    it = it + 1;
%     for i = 1:m_c
%         for j = 1:m_c
%             M(i,j) = barw'*G{i}*barw - 2*barw'*g{i,j} + h(j);
%         end
%     end
%     M=M*V;
% ---------- 权重T ----------
%     T = sinkhornKnoppAuto2(M, Para);
%     Trsum = T*ones(m, 1);
% Trsum =V*ones(m_c, 1);
% ---------- 求解w和b ----------
%     H = zeros(m+1, m+1);
%     f = zeros(m+1, 1);
%     for i = 1:m_c
%         H = H + G{i}*Trsum(i);
%         for j = 1:m_c
%             f = f + T(i,j)*g{i,j};
%         end
%     end
    % 使矩阵一定半正定
%     H = (H + H');
    H = G + epsi*eye(size(G,1));
    H=(H+H')/2;
    f = -2*g;
    A = [barX; -barX];
    b = [ones(m_c,1); zeros(m_c,1)];
    [barw, fval(it,:)] = quadprog(H, f, A, b, [], [], [], [], [], options);
%     [barw, fval(it,:)] = quadprog(H, f, [], [], [], [], [], [], [], options);      
end
tr_time = toc(tt);
% % ------ output and prediction ---------
 model.tr_time = tr_time;
model.w = barw(1:end-1);
model.b = barw(end);
% Predprobtrain= KerF(X,kpar,X)*model.w + model.b;
Predprob= KerF(ValX,kpar,X)*model.w + model.b;
PredictY = Predprob >= 0.5;
model.prob = Predprob;
model.fval = fval;

 if Para.plt == 1
     plt.ds = PredictY0;
     plt.ss1 = plt.ds - 1;
     plt.ss2 = plt.ds + 1;
     model.plt = plt;
 end







