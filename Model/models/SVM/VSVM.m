function [PredY,model] = VSVM(TestX,DataTrain,Para)
gam = Para.p1;
kpar = Para.kpar;
V1=Para.V;
X = DataTrain.X;
Y = DataTrain.Y;
Y(Y==-1)=0;
clear DataTrain
% tic;
KerX = KerF(X,kpar,X);
% toc;
Ab = (V1*KerX+gam*eye(length(Y)))\V1*Y;
Ac = (V1*KerX+gam*eye(length(Y)))\V1*ones(length(Y),1);
c = (ones(1,length(Y))*V1*KerX*Ab-ones(1,length(Y))*V1*Y)/(ones(1,length(Y))*V1*KerX*Ac-ones(1,length(Y))*V1*ones(length(Y),1));
A = Ab-c*Ac;
% toc;
[KerTstX] = KerF(TestX,kpar,X)';
% [KerTstX] = kernel(TestX',TestX',kerneltype);
% [m,~] = size(Y);
% FTrn = (A'*KerX)'+c;
% [n,~] = size(TestX);
F = (A'*KerTstX)'+c;
f = F-0.5;
PredY = sign(f);
model.alpha = A;
model.b = c;
model.prob=f;
end

