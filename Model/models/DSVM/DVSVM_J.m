function [PredictY, model] = DVSVM_J(ValX , X_C , Trn , Para)

itmax = Para.itmO;
X = Trn.X;
[m, n] = size(X);
m_c=length(X_C);
barX = [X, ones(m,1)];
Y = Trn.Y;
% barw = rand(n+1, 1); % initialization of barw
barw = zeros(n+1, 1);  % initialization of barw
options = optimoptions('linprog','Display','off');
Para.n = n;
epsi = Para.epsi;
it=1;
%% ----- compute each item of Mij(barw), pMij(barw) ------
tt = tic;%记录时间
% ---------- 求三个已知矩阵 ----------
for i = 1:m_c
    for j = 1:m
%         C(j, i) = prod(X(i,:)>=X(j,:));  
%         C(j, i) = sparse(prod(X(i,:)>=X(j,:)));  
        C(j, i) = (sum(X_C(i,:)>=X(j,:))/n)/(m); %在高维情况下加和求平均
    end
end

for i = 1:m_c
    for j = 1:m_c
        CC{i,j} = C(:,i)*C(:,j)'; %按列相乘
        g{i,j} = barX'*CC{i,j}*Y; 
%       g{i,j} = 2*barX'*CC{i,j}*Y+barX'*Cii*ones(m,1)-barX'*CC{i,j}*ones(m,1);
    end
    G{i} = barX'*CC{i,i}*barX;
%     G{i} = barX'*Cii*barX;
    h(i) = Y'*CC{i,i}*Y;
end
%---------M和T的初始值--------- 
for j = 1:m_c
   M(i,j) = h(j);
end
T = sinkhornKnoppAuto2(M, Para);
Trsum = T*ones(m_c, 1);

%---------迭代M和T--------- 
val(1)=0;
while it<itmax
    it=it+1;
    H = zeros(n+1, n+1);
    f = zeros(n+1, 1);
    for i = 1:m_c
        H = H + G{i}*Trsum(i);
        for j = 1:m_c
            f = f + T(i,j)*g{i,j};
        end
    end
%     [L,U]=lu(H');
%     barw_new=U\(L\f);
    barw_new=(f'*inv(H'))';
%     [barw_new,~]= linprog([],[],[],H',f',[],[],options);
    for i = 1:m_c
        for j = 1:m_c
            M_new(i,j) = barw_new'*G{i}*barw_new - 2*barw_new'*g{i,j} + h(j);
        end
    end

    T = sinkhornKnoppAuto2(M_new, Para);
    Trsum = T*ones(m_c,1);

    val(it)=sum(sum(T.*M_new));
    threshold(it)=abs(val(it)-val(it-1));
end
tr_time = toc(tt);
% % ------ output and prediction ---------
model.tr_time = tr_time;
model.n_SV = 0;
barw_new=barw_new';
model.w = barw_new(1:end-1);
model.b = barw_new(end);
Predprob= ValX*model.w + model.b;

for i=1:length(Predprob)
    if Predprob(i)<0
        Predprob(i)=0;
    elseif Predprob(i)>=1
        Predprob(i)=1;
    end
end

PredictY = (ValX*model.w +model.b) >= 0.5;
model.prob = Predprob;
% model.fval = fval;
`
 if Para.plt == 1
     plt.ds = PredictY0;
     plt.ss1 = plt.ds - 1;
     plt.ss2 = plt.ds + 1;
     model.plt = plt;
 end







