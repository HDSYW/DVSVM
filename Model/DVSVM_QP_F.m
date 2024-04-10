function [result] = DVSVM_QP_F(Data,K,ktype,pa,data)
test_ac=0;
lam=pa.lam;
X_train_T=Data.X_train_T;
Y_train_T=Data.Y_train_T;
X_test_T=Data.X_test_T;
Y_test_T=Data.Y_test_T;
k=K;
Para.kpar.ktype = ktype;
indices = crossvalind('Kfold',X_train_T(:,1),k);
Para.itmO = pa.itmO; Para.itmSh = pa.itmSh; Para.plt = pa.plt;  Para.epsi = pa.epsi; Para.thr = pa.thr;
%----- X_C Self -----
if sum(data == 'train')
    X_C = X_train_T; X_C0 = X_train_T;
end
%----- X_C Test -----
if sum(data == 'test')
    X_C = X_test_T; X_C0 = X_test_T;
end
%----- X_C All -----
if sum(data == 'all')
    X_C = [X_train_T;X_test_T]; X_C0 = X_C;
end
%------------------------------------------------------------
Y_train_T(Y_train_T==-1)=0;Y_test_T(Y_test_T==-1)=0;
for lambda = 10%6 :pa.step:10
    Para.lam = 2.^lambda;
    for power= 2%pa.min:pa.step:pa.max
        Para.kpar.kp1 =2.^power; Para.kpar.kp2 =0;
        for i = 1:k
            test = (indices == i);  train = ~test;
            Trn.X = X_train_T(train,:);Trn.Y = Y_train_T(train,:); 
            ValX = X_train_T(test,:); ValY = Y_train_T(test,:);
            %---------- Model ----------
            [PredictY , model] = DVSVMnon_QP(ValX, X_C, Trn, Para);
            PredictY=double(PredictY);PredictY(PredictY==0)=-1;ValY(ValY==0)=-1;
            CM = ConfusionMatrix(PredictY,ValY);
            M_FM(i)=CM.FM;
            M_GM(i)=CM.GM;
            M_Acc(i) = sum(PredictY==ValY)/length(ValY)*100;
            M_Erro(i) = sum(abs(model.prob - ValY));
            M_Errorate(i) = sum(abs(model.prob - ValY))/sum(abs(ValY));
        end
        mean_GM=mean(M_GM)    ; mean_Acc =mean(M_Acc)       ;
        mean_Erro=mean(M_Erro)  ; mean_Errorate=mean(M_Errorate) ;
        mean_F = mean(M_FM)       ;
        if mean_Acc>test_ac %mean_Acc>test_ac%mean_F>test_F
            test_GM=mean_GM             ; test_ac=mean_Acc     ; best_Erro=mean_Erro    ;
            best_Errorate=mean_Errorate ; best_lambda=Para.lam ; best_kp1=Para.kpar.kp1 ;
            test_F=mean_F               ; 
        end
    end
    fprintf('Complete %.2f%%\n',(lambda-6)*100/6)
end
%>>>>>>>>>>>>>>>>>>>> Prediction <<<<<<<<<<<<<<<<<<<<
Trn.X=X_train_T      ; Trn.Y=Y_train_T;
Para.lam=best_lambda ; Para.kpar.kp1=best_kp1;
%---------- Model ----------
[PredictY , model] = DVSVMnon_QP(X_test_T , X_C0, Trn , Para);
PredictY=double(PredictY);PredictY(PredictY==0)=-1;Y_test_T(Y_test_T==0)=-1;
result.prob=model.prob;
CM = ConfusionMatrix(PredictY,Y_test_T);
result.ac_test=sum(PredictY==Y_test_T)/length(Y_test_T)*100;
result.F=CM.FM;
result.GM=CM.GM;
[~,~,~,AUC]=perfcurve(Y_test_T, PredictY, '1');
result.AUC=100*AUC;
result.lam=best_lambda;
result.kp1=best_kp1;
fprintf('%s\n', repmat('-', 1, 100))       ; fprintf('Test_AC=%.2f||',result.ac_test);
fprintf('Train_AC=%.2f||\n',test_ac)       ; fprintf('Bestlambda=%.2f||',log2(best_lambda))  ; 
fprintf('Best_kp1=%.2f||\n',log2(best_kp1)); fprintf('%s\n', repmat('=', 1, 100));
end

