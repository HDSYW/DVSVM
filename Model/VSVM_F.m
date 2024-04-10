function [result] = VSVM_F(Data,K,ktype,v_ker,CDFx,pa)
test_ac=0;
X_train_T=Data.X_train_T;
Y_train_T=Data.Y_train_T;
X_test_T=Data.X_test_T;
Y_test_T=Data.Y_test_T;
k=K;
Y_train_T(Y_train_T==0)=-1;Y_test_T(Y_test_T==0)=-1;
Para.kpar.ktype = ktype;

V_Matrix = 'Vmatrix';  
V_MatrixFun = str2func(V_Matrix);    
Para.vmatrix = V_Matrix;
Para.CDFx = CDFx;  
Para.v_ker = v_ker;    
CDFx = Para.CDFx;

indices = crossvalind('Kfold',X_train_T(:,1),k);
for j = pa.min:pa.step:pa.max
    Para.p1=2.^j;
    for power1 = pa.min:pa.step:pa.max
        Para.v_sig = 2.^power1;
        for power2 = pa.min:pa.step:pa.max
            Para.kpar.kp1 =2.^power2; Para.kpar.kp2 = 0;
            for i = 1:k
                test = (indices == i); train = ~test;
                Trn.X = X_train_T(train,:); Trn.Y = Y_train_T(train,:);
                ValX = X_train_T(test,:);   ValY =Y_train_T(test,:);
                % ---------- Vmatrix ----------
                [V,~] = Vmatrix(Trn.X,CDFx,Para.v_sig,Para.v_ker);  Para.V = V;
                % ---------- Model ----------
                [PredictY , model] = VSVM(ValX , Trn , Para);
                PredictY=double(PredictY);PredictY(PredictY==0)=-1;ValY(ValY==0)=-1;
                CM = ConfusionMatrix(PredictY,ValY);
                M_F(i)=CM.FM;
                M_Acc(i) = sum(PredictY==ValY)/length(ValY)*100;
                M_Erro(i) = sum(abs(model.prob - ValY));
                M_Errorate(i) = sum(abs(model.prob - ValY))/sum(abs(ValY));
                M_GM(i)=CM.GM;
            end
            mean_Acc =mean(M_Acc); mean_Erro=mean(M_Erro);mean_GM=mean(M_GM) ;
            mean_Errorate=mean(M_Errorate); mean_F=mean(M_F);
            if mean_Acc>test_ac%mean_Acc>test_ac %mean_F>test_F
                test_GM=mean_GM ;  test_ac=mean_Acc;      best_Erro=mean_Erro;
                best_Errorate=mean_Errorate;   best_v_sig=Para.v_sig;
                best_kp1=Para.kpar.kp1; best_p1=Para.p1; test_F=mean_F;
            end
        end
    end
    fprintf('Completed %s\t\n',num2str((j+8)*100/16))
end
% >>>>>>>>>>>>>>>>>>>> Test and prediction <<<<<<<<<<<<<<<<<<<<
Trn.X=X_train_T; Trn.Y=Y_train_T; Para.v_sig=best_v_sig; Para.kpar.kp1=best_kp1; Para.p1=best_p1;
% ---------- Vmatrix ----------
[V,~] = Vmatrix(Trn.X,CDFx,Para.v_sig,Para.v_ker);  Para.V = V;
% ---------- Model ----------
[PredictY , model] = VSVM(X_test_T , Trn , Para);
result.prob=model.prob+0.5;
PredictY=double(PredictY);  PredictY(PredictY==0)=-1;
CM = ConfusionMatrix(PredictY,Y_test_T);
result.ac_test=sum(PredictY==Y_test_T)/length(Y_test_T)*100;
result.F=CM.FM;
result.GM=CM.GM;
[~,~,~, AUC]=perfcurve(Y_test_T, PredictY, '1');
result.AUC=100*AUC;
result.lam=best_p1;
result.kp1=best_kp1;
result.v_sig=best_v_sig;
fprintf('%s\n', repmat('-', 1, 100))        ; fprintf('Test_AC=%.2f||',result.ac_test);
fprintf('Train_AC=%.2f||\n',test_ac)     ; fprintf('Best_v_sig=%.4f||',log2(best_v_sig));
fprintf('BestC=%.2f||',log2(best_p1))  ; fprintf('Best_kp1=%.2f||\n',log2(best_kp1));
fprintf('%s\n', repmat('=', 1, 100));
end

