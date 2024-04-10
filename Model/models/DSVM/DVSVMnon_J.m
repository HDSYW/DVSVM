function [PredictY, model] = DVSVMnon_J(ValX , X_C,Trn , Para)

kpar = Para.kpar;
itmax = Para.itmO;
X = Trn.X;
[m, n] = size(X);
m_c=size(X_C,1);%褰揦_C涓鸿缁冮泦鍔犳祴璇曢泦鏃舵瘡涓猉i灏嗕笌璁粌鍜屾祴璇曢泦鐨勬瘡涓涓偣杩涜姣旇緝
Ke = [KerF(X, kpar, X), ones(m,1)];
Y = Trn.Y;
alphb = zeros(m+1, 1); % initialization of barw
% alphb = ones(m+1, 1); % initialization of barw
it = 1;
options = optimoptions('quadprog','Display','off');
Para.n = n;
epsi = Para.epsi;
tt = tic;
C=zeros(m,m_c);
for i = 1:m_c
    for j = 1:m
        C(j, i) = prod(X_C(i,:)>=X(j,:))/m;  
%         C(j, i) = sparse(prod(X_C(i,:)>=X(j,:)))/m;  
%         C(j, i) = (sum(X_C(i,:)>=X(j,:))/n)/m;
    end
end
%% --------------------------- Algorithm 2 ---------------------------
val=[];
YY = Y*Y';
barw=zeros(m+1,1);
% M=zeros(m_c,m_c);
M = zeros(m_c, m_c);
for i =1:m_c    
    M(:,i)=Y'*C(:,i)*C(:,i)'*Y;
end
T=sinkhornKnoppAuto2(M, Para);
MijC=zeros(m_c,m_c);
while it<itmax
    H = zeros(m+1, m+1);
    f = zeros(m, m);
    for i = 1:m_c
        CCii = C(:,i)*C(:,i)';
        Gii = Ke'*CCii*Ke;
        MijQ = barw'*Gii*barw;%浜屾椤?
        hjj = C(:,i)*C(:,i)';
        WK=-2*barw'*Ke';
        MC=YY.*hjj;
        sumMC=sum(MC(:));
        for j = 1:m_c
			gij = C(:,i)*C(:,j)';
            f = f + T(i,j)*gij;
            H = H + T(i,j)*Gii;
            M(i,j) = WK*gij*Y;%涓娆￠」
			MijC(i,j) = sumMC;%甯告暟椤?
        end
        M(i,:) = M(i,:) + MijQ;
    end
    H=H+0.00001*eye(size(H,1));
    f=Ke'*f*Y;
    M= M+MijC;
    0.01*eye(size(M,1));
    barw=(f'/H')';
	T=sinkhornKnoppAuto2(M, Para);
	J=T.*M;
	val(it)=sum(J(:));
    it=it+1;
end 
barw_new=barw;
%% --------------------------- Algorithm 3 ---------------------------

tr_time = toc(tt);
%% --------- output and prediction ---------
model.tr_time = tr_time;
model.n_SV = 0;
barw_new=barw_new';
model.alpha = barw_new(1:end-1);
model.b = barw_new(end);
Predprob= KerF(ValX, kpar, X)*model.alpha' + model.b;
Predprob= mapminmax(Predprob',0,1)';
% 閲嶆柊璧嬪?
% for i=1:length(Predprob)
%     if Predprob(i)<0
%         Predprob(i)=0;
%     elseif Predprob(i)>=1
%         Predprob(i)=1;
%     end
% end
PredictY = Predprob>= Para.thr;
% PredictY = Predprob>= Para.thr;
model.prob = Predprob;
% model.fval = fval;
model.val=val;
 if Para.plt == 1
     plt.ds = PredictY0;
     plt.ss1 = plt.ds - 1;
     plt.ss2 = plt.ds + 1;
     model.plt = plt;
 end







