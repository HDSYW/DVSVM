function [PredictY, model] = DVSVMnon_QP(ValX , X_C,Trn , Para)

kpar = Para.kpar;
itmax = Para.itmO;
X = Trn.X;
[m, n] = size(X);
X_C=X_C;
m_c=size(X_C,1);%当X_C为训练集加测试集时每个Xi将与训练和测试集的每一个点进行比较
Ke = [KerF(X, kpar, X), ones(m,1)];
Y = Trn.Y;
alphb = zeros(m+1, 1); % initialization of barw
it = 0;
options = optimoptions('quadprog','Display','off');
Para.n = n;
thr = Para.thr;
% % ----- compute each item of Mij(barw), pMij(barw) ------
tt = tic;
for i = 1:m_c
    for j = 1:m
%         C(j, i) = prod(X_C(i,:)>=X(j,:))/m;  
%         C(j, i) = sparse(prod(X_C(i,:)>=X(j,:)))/m;  
        C(j, i) = (sum(X_C(i,:)>=X(j,:))/n)/m;
    end
end

for i = 1:m_c
    for j = 1:m_c
        CC{i,j} = C(:,i)*C(:,j)';
        g{i,j} = Ke'*CC{i,j}*Y;
    end
    G{i} = Ke'*CC{i,i}*Ke;
    h(i) = Y'*CC{i,i}*Y;
    clear CC
end

while it<itmax
    it = it + 1;
    for i = 1:m_c
        for j = 1:m_c
            M(i,j) = alphb'*G{i}*alphb - 2*alphb'*g{i,j} + h(j);
        end
    end
    M=M+0.01*eye(size(M));
    T = sinkhornKnoppAuto2(M, Para);
    Trsum = T*ones(m_c, 1);
    
    H = zeros(m+1, m+1);
    f = zeros(m+1, 1);
    for i = 1:m_c
        H = H + G{i}*Trsum(i);
        for j = 1:m_c
            f = f + T(i,j)*g{i,j};
        end
    end
    H = H + 0.00001*eye(size(H,1));
    H = (H + H');
    f = -2*f;
    A = [Ke; -Ke];
    b = [ones(m,1); zeros(m,1)];
    [alphb, fval(it,:)] = quadprog(H, f, A, b, [], [], [], [], [], options);
%     [alphb, fval(it,:)] = quadprog(H, f, [], [], [], [], [], [], [], options);

end
tr_time = toc(tt);
% % ------ output and prediction ---------
model.tr_time = tr_time;
% model.n_SV = 0;

model.alpha = alphb(1:end-1);
model.b = alphb(end);
Predprob= KerF(ValX, kpar, X)*model.alpha + model.b;
% Predprob= sigmoid(Predprob);
% Predprob= mapminmax(Predprob',0,1)';
% 重新赋值
% for i=1:length(Predprob)
%     if Predprob(i)<=0
%         Predprob(i)=0;
%     elseif Predprob(i)>1
%         Predprob(i)=1;
%     end
% end

PredictY = Predprob >= thr;
model.prob = Predprob;
model.fval = fval;

 if Para.plt == 1
     plt.ds = PredictY0;
     plt.ss1 = plt.ds - 1;
     plt.ss2 = plt.ds + 1;
     model.plt = plt;
 end







