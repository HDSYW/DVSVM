clear
clc
close all
ModS = [];
% ModS = [ModS;"LIB_L1SVCprob"];
% ModS = [ModS;"VSVM"];
% ModS = [ModS;"DVSVM3"];
ModS = [ModS;"DVSVMnon"];

%% -------- Parameter setting ---------
% % --  SVM parameters ---
Para.p1 = 10;
% Para.kpar.ktype = 'lin';     
Para.kpar.ktype = 'rbf'; 
Para.kpar.kp1 = 0.5; 
% Para.kpar.kp1 = 0.0002; 
Para.kpar.kp2 = 0; 
V_Matrix = 'Vmatrix';              %G(x-xi)G(x-xj)
V_MatrixFun = str2func(V_Matrix);
Para.vmatrix = V_Matrix;
Para.CDFx = 'uniform';
%     Para.CDFx = 'empirical';
%     Para.CDFx = 'normal';
%     -------------v-kernel-----------------
Para.v_ker = 'gaussian_ker';
Para.v_sig = 1;
CDFx = Para.CDFx;%v-vector mu(x)
for ms = 1 : length(ModS)
    Mod = ModS(ms);
    SVMFun = str2func(Mod);
    Para.itmO = 30;
    Para.itmSh = 10;
    b = 0; % bias of center
    cen = 5; % center
    dnum = 50; % data number;
    dfea = 1; % noise feature number
    Para.lam = 0.01;
    Para.plt = 0;
    Para.epsi = 0;
    Para.thr = 0.5;
    %% -------- Saved data ---------
    % load data60x2.mat
    %% -------- ArtiGen_lcn1 data---------
    % for dat = 1:nnz(dnum)*nnz(dfea) 
    %     [X,Y,datName] = ArtiGen_lcn1(b, cen,dnum,dfea,dat);
    % end
    %% -------- Generate data for different distributions ---------
    % Gentype = "normal" "uniform" "weibull" "poisson" "t" "binomial" "geometric" "gamma" "f" "chisquare"
    Gentype0 = 'normal'; Gentype1 = 'normal'; p1_0 = -1; p2_0 = 3; p1_1 = 1; p2_1 = 3;
%     Gentype0 = 'uniform'; Gentype1 = 'uniform'; p1_0 = 0.0; p2_0 = 0.1; p1_1 = 0.1; p2_1 = 0.3;
%     Gentype0 = 'uniform'; Gentype1 = 'normal'; p1_0 = -0.3; p2_0 = 0;  p1_1 = 0.3; p2_1 = 1;
    % Gentype0 = 'normal'; Gentype1 = 'uniform'; p1_0 = -0.2; p2_0 = 1; p1_1 = 0; p2_1 = 0.2;
%     Gentype0 = 'uniform'; Gentype1 = 'gamma'; p1_0 = -0.2; p2_0 = 1; p1_1 = 0; p2_1 = 0.2;

    % ---------  The 0-th class ------------
%     m0 = 30;  m1 = 30; m = m0+m1;
    per = 0.5; n = 1;
%     X0 = RandomGen(Gentype0, p1_0, p2_0, m0, n);
%     % ---------  The 1-th class ------------
%     X1 = RandomGen(Gentype1, p1_1, p2_1, m1, n);
%     X = [X0; X1];
    X = -15:0.2:15;
    X = X';
    m = size(X, 1); 
    X0pdf = pdf(Gentype0, X, p1_0, p2_0);
    X1pdf = pdf(Gentype1, X, p1_1, p2_1);
%     py1 = m1/(m0+m1);
    py1 = 0.5;
    allprob =  X0pdf*(1-py1) + X1pdf*py1;
    postprob = (X1pdf*py1)./allprob;
%     Y = [zeros(m0,1); ones(m1,1)];
    Y = postprob;
    Para.n = size(X, 2);
    %% --------------- Specify model ---------------
    ModDVSVM =["VSVM";"DVSVM";"DVSVM2";"DVSVM3";"DVSVMnon"];
     if sum(Mod==ModDVSVM)
         if sum((Y==-1)~=0)
             Y(Y==-1) = 0;
         end
         if sum(Mod=="VSVM")
            [V,~] = Vmatrix(X,CDFx,Para.v_sig,Para.v_ker);%V-matrix/vector
            Para.V = V;
         end
     else
         if length(unique(Y))==2
             if sum((Y==0)~=0)
                Y(Y==0) = -1;
             end
         elseif length(unique(Y))>2
             Y(Y>=Para.thr) = 1;
             Y(Y<Para.thr) = -1;
         end
     end

    %% --------------- Test and prediction -------
    indte = 1:m;
    Trn.X = X;
    Trn.Y = Y;
    ValX = X;
    ValY = Y;
    
%     indtr = randsample(m, round(per*m));
%     indte = setdiff(1:m, indtr);
%     ValX = X(indte, :);
%     ValY = Y(indte, :);
%     Trn.X = X(indtr, :);
%     Trn.Y = Y(indtr, :);
    indP = ValY>=Para.thr;
    indN = ValY<Para.thr;
    m1 = sum(indP);
    m2 = sum(indN);

    Yc = ValY;
    if sum(Mod==ModDVSVM)
        if length(unique(ValY))>2
             Yc(indP) = 1;
             Yc(indN) = 0;
        end
    else
        if length(unique(ValY))>2
             Yc(indP) = 1;
             Yc(indN) = -1;
        end
    end
    
    [PredictY , model] = SVMFun(ValX ,X, Trn , Para);
%     [PredictY , model] = LIB_L1SVCprob(ValX , Trn , Para);
%     [PredictY, model] = DVSVM3(ValX , Trn , Para);
%     [PredictY, model] = DVSVMnon(ValX , Trn , Para);
    Acc = sum(PredictY==Yc)/length(ValY)*100
    Erro = sum(abs(model.prob - ValY))
    Errorate = sum(abs(model.prob - ValY))/sum(abs(ValY))

    % ValX = X;
    % Trn.X = X;
    % Trn.Y = Y;
    % Para.n = size(X, 2);
    % [PredictY, model] = DVSVM3(ValX , Trn , Para);
    % Acc = sum(PredictY==Y)/length(Y)*100
    %     
    %% --------------- Plot distribution ---------------
    figure
    ValXP = ValX(indP);
    ValYP = ValY(indP);
    ValXN = ValX(indN);
    ValYN = ValY(indN);
    
    plot(ValXP, zeros(size(ValXP)), 'r<')
    hold on
    plot(ValXN, zeros(size(ValXN)), 'b>')
    
    plot(ValX, postprob(indte, :), 'ko-')
    hold on
    plot(ValX, model.prob, 'g*--')
    title('Probability')
    legend('Pos class', 'Neg class', 'Posterior probability', 'Predicted probability')
    %% --------------- Plot objective ---------------
%     figure
%     plot(1:length(model.fval), model.fval, '*-b')
%     title('Objetive')
    %% --------------- Plot scatter (X) ---------------
%     ValXP = ValX(indP);
%     ValYP = ValY(indP);
%     ValXN = ValX(indN);
%     ValYN = ValY(indN);
%     PreprobP = model.prob(indP);
%     PreprobN = model.prob(indN);
%     
%     figure
%     plot(ValXP, zeros(size(ValXP)), 'r<')
%     hold on
%     plot(ValXN, zeros(size(ValXN)), 'b>')
%     % --------------- Plot scatter (Prob) ---------------
% %     ValXP = ValX(indP);
% %     ValYP = ValY(indP);
% %     ValXN = ValX(indN);
% %     ValYN = ValY(indN);
% %     figure
%     plot(ValXP, ValYP, '--m')
%     hold on
%     plot(ValXN, ValYN, '--m')
%     
% %     figure
%     plot(ValXP, PreprobP, '-k')
%     hold on
%     plot(ValXN, PreprobN, '-k')

    % w = model.w;
    % b = model.b;
    % mi = min(X(:, 1)); ma = max(X(:, 1)); 
    % x = mi-1: 0.2: ma+1;
    % y = -1/w(2)*(w(1)*x + b);
    % plot(x,y)
end