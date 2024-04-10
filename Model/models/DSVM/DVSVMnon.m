function [PredictY, model] = DVSVMnon_J(ValX , X_C,Trn , Para)

kpar = Para.kpar;
itmax = Para.itmO;
X = Trn.X;
[m, n] = size(X);
m_c=length(X_C);%当X_C为训练集加测试集时每个Xi将与训练和测试集的每一个点进行比较
Ke = [KerF(X, kpar, X), ones(m,1)];
Y = Trn.Y;
alphb = zeros(m+1, 1); % initialization of barw
% alphb = ones(m+1, 1); % initialization of barw
it = 1;
options = optimoptions('quadprog','Display','off');
Para.n = n;
epsi = Para.epsi;
%% --------- compute each item of Mij(barw), pMij(barw) ---------
tt = tic;
for i = 1:m_c
    for j = 1:m
%         C(j, i) = prod(X_C(i,:)>=X(j,:))/m;  
%         C(j, i) = sparse(prod(X_C(i,:)>=X(j,:)))/m;  
        C(j, i) = (sum(X_C(i,:)>=X(j,:))/n)/m;
    end
end
% % --------- 矩阵算法 ---------
% c_c=repmat(reshape(C,size(C,2)*size(C,1),1),1,size(C,2)*size(C,1));
% CC=c_c.*c_c';
% % CC=cell(m_c,m_c);
% g=cell(m_c,m_c);
% G=cell(1,m_c);
% h=zeros(1,m_c);
% for i = 1:m_c
%     for j = 1:m_c
% %         CC{i,j} = C(:,i)*C(:,j)';
% %         g{i,j} = Ke'*CC{i,j}*Y;
%         g{i,j} = Ke'*CC((i-1)*m+1:i*m,(j-1)*m+1:j*m)*Y;
%     end
% %     G{i} = Ke'*CC{i,i}*Ke;
% %     h(i) = Y'*CC{i,i}*Y;
%     G{i} = Ke'*CC((i-1)*m+1:i*m,(i-1)*m+1:i*m)*Ke;
%     h(i) = Y'*CC((i-1)*m+1:i*m,(i-1)*m+1:i*m)*Y;
% end
% %---------M和T的初始值--------- 
% for j = 1:m_c
%    M(i,j) = h(j);
% end
% T = sinkhornKnoppAuto2(M, Para);
% Trsum = T*ones(m_c, 1);
% %---------迭代M和T--------- 
% val(1)=0;
% while it<itmax
%     it = it + 1;
%     H = zeros(m+1, m+1);
%     f = zeros(m+1, 1);
%     for i = 1:m_c
%         H = H + G{i}*Trsum(i);
%         for j = 1:m_c
%             f = f + T(i,j)*g{i,j};
%         end
%     end
%     H = H + 0.00001*eye(size(H,1));
%     barw_new=(f'/H')';
% 
%     for i = 1:m_c
%         for j = 1:m_c
%             M_new(i,j) = barw_new'*G{i}*barw_new - 2*barw_new'*g{i,j}+ h(j);
%         end
%     end
%     T = sinkhornKnoppAuto2(M_new, Para);
%     Trsum = T*ones(m_c, 1);
% 
%     val(it)=sum(sum(T.*M_new));
%     threshold(it)=abs(val(it)-val(it-1));
% end
val=[];
barw=zeros(m+1,1);
M=ones(m_c,m_c);
T=sinkhornKnoppAuto2(M, Para);
while it<itmax
    H = zeros(m+1, m+1);
    f = zeros(m+1, 1);
    for i = 1:m_c
        CCii = C(:,i)*C(:,i)';
        Gii=Ke'*CCii*Ke;
		for j = 1:m_c
			CCij = C(:,i)*C(:,j)';
            CCjj = C(:,j)*C(:,j)';
			gij=Ke'*CCij*Y;
			hjj=Y'*CCjj*Y;
            f = f + (T(i,j)*gij')';
            H = H + (T(i,j)*Gii')'; 
			M(i,j) = barw'*Gii*barw- 2*barw'*gij+ hjj;
		end
    end
    H=H+0.00001*eye(size(H,1));
    barw=(f'/H')';
	T=sinkhornKnoppAuto2(M, Para);
	J=T.*M;
	val(it)=sum(J(:));
    it=it+1;
end  
barw_new=barw;
tr_time = toc(tt);
%% --------- output and prediction ---------
model.tr_time = tr_time;
model.n_SV = 0;
barw_new=barw_new';
model.alpha = barw_new(1:end-1);
model.b = barw_new(end);
Predprob= KerF(ValX, kpar, X)*model.alpha' + model.b;
% Predprob= mapminmax(Predprob',0,1)';
% 重新赋值
% for i=1:length(Predprob)
%     if Predprob(i)<0
%         Predprob(i)=0;
%     elseif Predprob(i)>=1
%         Predprob(i)=1;
%     end
% end
PredictY = Predprob>= Para.thr;
% PredictY = Predprob<= Para.thr;
model.prob = Predprob;
% model.fval = fval;

 if Para.plt == 1
     plt.ds = PredictY0;
     plt.ss1 = plt.ds - 1;
     plt.ss2 = plt.ds + 1;
     model.plt = plt;
 end







