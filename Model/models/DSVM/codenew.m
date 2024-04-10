clear
clc
close all
seed = 10;  
rng(seed);  
% diary 'n+u-dvsvm-rg.txt'
ttic=tic;
%% >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Selecting <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
% ---------- Model ----------
ModS = [];
ModS = [ModS;"LIB_L1SVCprob"];
% ModS = [ModS;"VSVM"];
% ModS = [ModS;"DVSVM3"];
% ModS = [ModS;"DVSVMnon"];
% ModS = [ModS;"test"];
% ---------- X_C Style ----------
datas=[];
datas=[datas;"train"];
% datas=[datas;"train+test"];
data=datas(1);

% ---------- Files ----------
name= ["normal"];
for l =1:length(name)
    f = '../datax10/';
    G = name(l);
    folder = f+G;
% ---------- Looping Data ----------
    files = dir(fullfile(folder, '*.mat'));
    ac_test=[];
    error_test =[];
    error_rate_test=[];
    for p = 1:length(files)
        filename = fullfile(folder, files(p).name);
        data_ori = load(filename);
        for ms = 1 : length(ModS)
            Mod = ModS(ms);
            SVMFun = str2func(Mod);  
            fprintf('%s\n', repmat('=', 1, 80))
            fprintf('Running File===>%s\t\t',G+p)
            fprintf('Running Model===>%s\t\t',Mod)
            fprintf('X_C===>%s\t\t\n',data)
            fprintf('%s\n', repmat('=', 1, 80))
%% >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Generate data for different distributions <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
            % Gentype = "normal"  "poisson" "Weibull" "chisquare"
            % ---------- 选择数据类型 ----------
    %         prompt0 = '请输入第一类分布类型(Normal or Uniform or Binomial or Geometric or Weibull or Chisquare)：';
    %         Gentype0 = input(prompt0,'s');
    %         promptp1_0 = '请输入第一类分布参数p1_0：';
    %         p1_0 = input(promptp1_0);
    %         promptp2_0 = '请输入第一类分布参数p2_0:';
    %         p2_0 = input(promptp2_0);
    %          
    %         prompt1 = '请输入第二类分布类型(Normal or Uniform or Binomial or Geometric or Weibull or Chisquare)：';
    %         Gentype1 = input(prompt1,'s');
    %         promptp1_1 = '请输入第二类分布参数p1_1：';
    %         p1_1 = input(promptp1_1);
    %         promptp2_1 = '请输入第二类分布参数p2_1:';
    %         p2_1 = input(promptp2_1);   
            Gentype0='normal';
            Gentype1='normal';
            Gentype2='normal';
            Gentype3='normal';    
            % ---------- Distribtion_trian ----------
            p1_0=-5;p2_0=6;
            p1_1=5;p2_1=6;
            % ---------- Distribtion_test ---------- 
            p1_2=-5;p2_2=6;
            p1_3=5;p2_3=6;
            % ---------- Train Data ----------
            X = sort(data_ori.X_train,'ascend');
            m = size(X, 1); 
            X0pdf = pdf(Gentype0, X, p1_0, p2_0);
            X1pdf = pdf(Gentype1, X, p1_1, p2_1);
            py1 = 0.5;
            allprob =  X0pdf*(1-py1) + X1pdf*py1;
            postprob = (X1pdf*py1)./allprob;
            Y = postprob;
            Para.n = size(X, 2);
            % ---------- X_C ----------
            if sum(data == 'train')
                X_C=X;
            end
            if sum(data == 'train+test')
                X_T = sort(data_ori.X_test,'ascend');
                X_C=[X;X_T];
            end                             
%% >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Base Setting <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
            % ---------- Base Para ----------
            test_ac=0;
            M_Acc=[];
            M_Erro=[];
            M_Errorate=[];
            ModDVSVM =["VSVM";"DVSVM";"DVSVM2";"DVSVM3";"DVSVMnon"];
            % ---------- SVM Para ---------- 
            % Para.kpar.ktype = 'lin';     
            Para.kpar.ktype = 'rbf'; 
        
            % ---------- DVSVM Para ----------
            Para.itmO = 30; % 算α、b的最大迭代
            Para.itmSh = 10;% 算T的最大迭代
            Para.plt = 0;
            Para.epsi = 0;  % 半正定参数
            Para.thr = 0.5; % 分类阈值
            
            % ---------- VSVM Para ----------
            V_Matrix = 'Vmatrix';              % G(x-xi)G(x-xj)
            V_MatrixFun = str2func(V_Matrix);  % 转换函数句柄
            Para.vmatrix = V_Matrix;
            Para.CDFx = 'uniform';% F(x)=x     % Para.CDFx = 'empirical'; Para.CDFx = 'normal';
            Para.v_ker = 'theta_ker';
            CDFx = Para.CDFx; 
            
            % ---------- K-fold ----------
            k = 10;
            indices = crossvalind('Kfold',X(:,1),k); %矩阵k折分类后的索引值
                    
            % ---------- Model Adaption ----------
             if sum(Mod==ModDVSVM)
                 if sum((Y==-1)~=0)
                     Y(Y==-1) = 0;
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
%% >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> SVM  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
            if sum(Mod=='LIB_L1SVCprob')
                for j = 1
                    Para.p1 = j;
                    for power=-8:8
                        Para.kpar.kp1 =2.^power;
                        Para.kpar.kp2 = 0; 
                        for i = 1:k
                            test = (indices == i); train = ~test;
                            Trn.X = X(train,:);
                            Trn.Y = Y(train,:);
                            ValX = X(test,:);
                            ValY = Y(test,:);
                            % ---------- Train Y Adaption ----------
                            indP = ValY>=Para.thr;
                            indN = ValY<Para.thr;
                            m1 = sum(indP);
                            m2 = sum(indN);
                            Yc = ValY;
                            if length(unique(ValY))>0.1
                                Yc(indP) = 1;
                                Yc(indN) = -1;
                            end
                            % ---------- Model ----------
                            [PredictY , model] = SVMFun(ValX , Trn , Para);
                            M_Acc(i) = sum(PredictY==Yc)/length(ValY)*100;
                            M_Erro(i) = sum(abs(model.prob - ValY));
                            M_Errorate(i) = sum(abs(model.prob - ValY))/sum(abs(ValY));
                        end
                        mean_Acc =mean(M_Acc);
                        mean_Erro=mean(M_Erro);
                        mean_Errorate=mean(M_Errorate);
                        if mean_Acc>test_ac
                            test_ac=mean_Acc;
                            best_Erro=mean_Erro;
                            best_Errorate=mean_Errorate;
                            best_p1=Para.p1;
                            best_kp1=Para.kpar.kp1;
                        end
                    end
                    fprintf('Completed %s\t\n',num2str(j))
                end
                fprintf('%s\n', repmat('=', 1, 80))
                fprintf('Train_AC=%.2f\t\t',test_ac)
                fprintf('Best_p1=%.2f\t\t',best_p1)
                fprintf('Best_kp1=%.2f\t\t\n',Para.kpar.kp1)  
                fprintf('%s\n', repmat('=', 1, 80))
                % >>>>>>>>>>>>>>>>>>>> Test and prediction <<<<<<<<<<<<<<<<<<<<
                % ---------- Test Data ----------
                X_test = sort(data_ori.X_test,'ascend');
                m = size(X_test, 1); 
                X0pdf_test = pdf(Gentype2, X_test, p1_2, p2_2);
                X1pdf_test = pdf(Gentype3, X_test, p1_3, p2_3);
                py1_test = 0.5;
                allprob_test =  X0pdf_test*(1-py1_test) + X1pdf_test*py1_test;
                postprob_test = (X1pdf_test*py1_test)./allprob_test;
                Y_test= postprob_test;
                Para.n = size(X_test, 2);
                 % ---------- Test Y Adaption ----------
                if length(unique(Y_test))==2
                    if sum((Y_test==0)~=0)
                        Y_test(Y_test==0) = -1;
                     end
                elseif length(unique(Y_test))>0.1
                        Y_test(Y_test>=Para.thr) = 1;
                        Y_test(Y_test<Para.thr) = -1;
                end
                indP_test = Y_test>=Para.thr;
                indN_test = Y_test<Para.thr;
                m1_ = sum(indP_test);
                m2_ = sum(indN_test);
                Yc0 = Y_test;
                if length(unique(Yc0))>0.1
                    Yc0(indP) = 1;
                    Yc0(indN) = -1;
                end        
                Trn.X=data_ori.X_train;
                Trn.Y=Y;
                % ---------- Selecting Para ----------
                Para.p1=best_p1;
                Para.kpar.kp1=best_kp1;
                % ---------- Model ----------
                [PredictY , model] = SVMFun(X_test , Trn , Para);
                ac_test(p) = sum(PredictY==Yc0)/length(Yc0)*100
                error_test(p) = sum(abs(model.prob - postprob_test))
                error_rate_test(p) = sum(abs(model.prob - postprob_test))/sum(abs(postprob_test))
            end
%% >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> DVSVM non <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
            if sum(Mod=='DVSVMnon')
                for lambda = 6:11
                    Para.lam = 2.^lambda;
                    for power= -8:8
                        Para.kpar.kp1 =2.^power;
                        Para.kpar.kp2 =0; 
                        for i = 1:k
                            test = (indices == i); 
                            train = ~test;
                            Trn.X = X(train,:);
                            Trn.Y = Y(train,:);
                            ValX = X(test,:);
                            ValY = Y(test,:);
                            % ---------- Train Y Adaption ----------
                            indP = ValY>=Para.thr;
                            indN = ValY<Para.thr;
                            m1 = sum(indP);
                            m2 = sum(indN);
                            Yc = ValY;
                            if sum(Mod==ModDVSVM)
                                if length(unique(ValY))>0.1
                                     Yc(indP) = 1;
                                     Yc(indN) = 0;
                                end
                            else
                                if length(unique(ValY))>2
                                     Yc(indP) = 1;
                                     Yc(indN) = -1;
                                end
                            end
                            % ---------- Model ----------
                            [PredictY , model] = SVMFun(ValX , X_C,Trn , Para);
                            M_Acc(i) = sum(PredictY==Yc)/length(ValY)*100;
                            M_Erro(i) = sum(abs(model.prob - ValY));
                            M_Errorate(i) = sum(abs(model.prob - ValY))/sum(abs(ValY));
                        end
                        mean_Acc =mean(M_Acc);
                        mean_Erro=mean(M_Erro);
                        mean_Errorate=mean(M_Errorate);
                        if mean_Acc>test_ac
                            test_ac=mean_Acc;
                            best_Erro=mean_Erro;
                            best_Errorate=mean_Errorate;
                            best_lambda=Para.lam;
                            best_kp1=Para.kpar.kp1;
                        end
                    end
                    fprintf('Completed %s\t\n',num2str(lambda))
                end
                fprintf('%s\n', repmat('=', 1, 80))
                fprintf('Train_AC=%.2f\t\t',test_ac)
                fprintf('Best_lam==%.2f\t\t',best_lambda)
                fprintf('Best_kp1=%.2f\t\t\n',best_kp1)
                fprintf('%s\n', repmat('=', 1, 80))
                % >>>>>>>>>>>>>>>>>>>> Test and prediction <<<<<<<<<<<<<<<<<<<<
                % ---------- Test Data ----------
                X_test = sort(data_ori.X_test,'ascend');
                m = size(X_test, 1); 
                X0pdf_test = pdf(Gentype2, X_test, p1_2, p2_2);
                X1pdf_test = pdf(Gentype3, X_test, p1_3, p2_3);
                py1_test = 0.5;
                allprob_test =  X0pdf_test*(1-py1_test) + X1pdf_test*py1_test;
                postprob_test = (X1pdf_test*py1_test)./allprob_test;
                Y_test= postprob_test;
                Para.n = size(X_test, 2);
                 % ---------- Test Y Adaption ----------
                indP_test = Y_test>=Para.thr;
                indN_test = Y_test<Para.thr;
                m1_ = sum(indP_test);
                m2_ = sum(indN_test);
                Yc0 = Y_test;
                if sum(Mod==ModDVSVM)
                    if length(unique(Yc0))>0.1
                         Yc0(indP_test) = 1;
                         Yc0(indN_test) = 0;
                    end
                else
                    if length(unique(Yc0))>2
                         Yc0(indP_test) = 1;
                         Yc0(indN_test) = -1;
                    end
                end
                Trn.X=data_ori.X_train;
                Trn.Y=Y;
                % ---------- Selecting Para ----------
                Para.lam=best_lambda;
                Para.kpar.kp1=best_kp1;
                % ---------- Model ----------
                [PredictY , model] = SVMFun(X_test ,X_C, Trn , Para);
                ac_test(p) = sum(PredictY==Yc0)/length(Y_test)*100
                error_test(p) = sum(abs(model.prob - postprob_test))
                error_rate_test(p) = sum(abs(model.prob - postprob_test))/sum(abs(postprob_test))
            end
        
%% >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> DVSVM lin <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
            if sum(Mod=='DVSVM3')
                for lambda = 5:10
                    Para.lam = 2.^lambda;
                    for i = 1:k
                        test = (indices == i); train = ~test;
                        Trn.X = X(train,:);
                        Trn.Y = Y(train,:);
                        ValX = X(test,:);
                        ValY = Y(test,:);
                        % ---------- Train Y Adaption ----------
                        indP = ValY>=Para.thr;
                        indN = ValY<Para.thr;
                        m1 = sum(indP);
                        m2 = sum(indN);
                        Yc = ValY;
                        if sum(Mod==ModDVSVM)
                            if length(unique(ValY))>0.1
                                 Yc(indP) = 1;
                                 Yc(indN) = 0;
                            end
                        else
                            if length(unique(ValY))>2
                                 Yc(indP) = 1;
                                 Yc(indN) = -1;
                            end
                        end
                        % ---------- Model ----------
                        [PredictY , model] = SVMFun(ValX ,X_C, Trn , Para);
                        M_Acc(i) = sum(PredictY==Yc)/length(ValY)*100;
                        M_Erro(i) = sum(abs(model.prob - ValY));
                        M_Errorate(i) = sum(abs(model.prob - ValY))/sum(abs(ValY));
                    end
                    mean_Acc =mean(M_Acc);
                    mean_Erro=mean(M_Erro);
                    mean_Errorate=mean(M_Errorate);
                    if mean_Acc>test_ac
                        test_ac=mean_Acc;
                        best_Erro=mean_Erro;
                        best_Errorate=mean_Errorate;
                        best_lambda=Para.lam;
                    end
                    fprintf('Completed %s\t\n',num2str(lambda))
                end
                fprintf('%s\n', repmat('=', 1, 80))
                fprintf('Train_AC=%.2f\t\t',test_ac)
                fprintf('Best_lam==%.2f\t\t\n',best_lambda)
                fprintf('%s\n', repmat('=', 1, 80))
                % >>>>>>>>>>>>>>>>>>>> Test and prediction >>>>>>>>>>>>>>>>>>>>
                % ---------- Test Data ----------
                X_test = sort(data_ori.X_test,'ascend');
                m = size(X_test, 1); 
                X0pdf_test = pdf(Gentype2, X_test, p1_2, p2_2);
                X1pdf_test = pdf(Gentype3, X_test, p1_3, p2_3);
                py1_test = 0.5;
                allprob_test =  X0pdf_test*(1-py1_test) + X1pdf_test*py1_test;
                postprob_test = (X1pdf_test*py1_test)./allprob_test;
                Y_test= postprob_test;
                Para.n = size(X_test, 2);
                 % ---------- Test Y Adaption ----------
                indP_test = Y_test>=Para.thr;
                indN_test = Y_test<Para.thr;
                m1_ = sum(indP_test);
                m2_ = sum(indN_test);
                Yc0 = Y_test;
                if sum(Mod==ModDVSVM)
                    if length(unique(Yc0))>2
                         Yc0(indP_test) = 1;
                         Yc0(indN_test) = 0;
                    end
                else
                    if length(unique(Yc0))>2
                         Yc0(indP_test) = 1;
                         Yc0(indN_test) = -1;
                    end
                end
                
                Trn.X=data_ori.X_train;
                Trn.Y=Y;
                % ---------- Selecting Para ----------
                Para.lam=best_lambda;
                % ---------- Model ----------
                [PredictY , model] = SVMFun(X_test , X_C, Trn , Para);
                ac_test(p) = sum(PredictY==Yc0)/length(Y_test)*100
                error_test(p) = sum(abs(model.prob - postprob_test))
                error_rate_test(p) = sum(abs(model.prob - postprob_test))/sum(abs(postprob_test))
            end
%% >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> VSVM <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< 
            if sum(Mod=="VSVM")
                for j = 1:10
                    Para.p1 = j;
                    for power1 = 0
                        Para.v_sig  = 2.^power1;
                        for power2= -8:8
                            Para.kpar.kp1 =2.^power2;
                            Para.kpar.kp2 = 0; 
                            for i = 1:k
                                test = (indices == i); train = ~test;
                                Trn.X = X(train,:);
                                Trn.Y = Y(train,:);
                                ValX = X(test,:);
                                ValY = Y(test,:);
                                % ---------- Vmatrix ----------
                                [V,~] = Vmatrix(Trn.X,CDFx,Para.v_sig,Para.v_ker);%V-matrix/vector
                                Para.V = V;
                                % ---------- Train Y Adaption ----------
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
                                % ---------- Model ----------
                                [PredictY , model] = SVMFun(ValX , Trn , Para);
                                M_Acc(i) = sum(PredictY==Yc)/length(ValY)*100;
                                M_Erro(i) = sum(abs(model.prob - ValY));
                                M_Errorate(i) = sum(abs(model.prob - ValY))/sum(abs(ValY));
                            end
                            mean_Acc =mean(M_Acc);
                            mean_Erro=mean(M_Erro);
                            mean_Errorate=mean(M_Errorate);
                            if mean_Acc>test_ac
                                test_ac=mean_Acc;
                                best_Erro=mean_Erro;
                                best_Errorate=mean_Errorate;
                                best_v_sig=Para.v_sig;
                                best_kp1=Para.kpar.kp1;
                            end
                        end
                    end
                fprintf('Completed %s\t\n',num2str(j))
                end
                fprintf('%s\n', repmat('=', 1, 80))
                fprintf('Train_AC=%.2f\t\t',test_ac)
                fprintf('Best_v_sig=%.2f\t\t',best_v_sig)
                fprintf('Best_kp1=%.2f\t\t\n',best_kp1)
                fprintf('%s\n', repmat('=', 1, 80))
                % >>>>>>>>>>>>>>>>>>>> Test and prediction <<<<<<<<<<<<<<<<<<<<
                % ---------- Test Data ----------
                X_test = sort(data_ori.X_test,'ascend');
                m = size(X_test, 1); 
                X0pdf_test = pdf(Gentype2, X_test, p1_2, p2_2);
                X1pdf_test = pdf(Gentype3, X_test, p1_3, p2_3);
                py1_test = 0.5;
                allprob_test =  X0pdf_test*(1-py1_test) + X1pdf_test*py1_test;
                postprob_test = (X1pdf_test*py1_test)./allprob_test;
                Y_test= postprob_test;
                Para.n = size(X_test, 2);
                % ---------- Test Y Adaption----------
                indP_test = Y_test>=Para.thr;
                indN_test = Y_test<Para.thr;
                m1_ = sum(indP_test);
                m2_ = sum(indN_test);
                Yc0 = Y_test;
                if sum(Mod==ModDVSVM)
                    if length(unique(Yc0))>2
                         Yc0(indP_test) = 1;
                         Yc0(indN_test) = 0;
                    end
                else
                    if length(unique(Yc0))>2
                         Yc0(indP_test) = 1;
                         Yc0(indN_test) = -1;
                    end
                end
                Trn.X=data_ori.X_train;
                Trn.Y=Y;
                % ---------- Selecting Para ----------
                Para.v_sig=best_v_sig;
                Para.kpar.kp1=best_kp1;
                % ---------- Vmatrix ----------
                [V,~] = Vmatrix(Trn.X,CDFx,Para.v_sig,Para.v_ker);%V-matrix/vector
                Para.V = V;
                % ---------- Model ----------
                [PredictY , model] = SVMFun(X_test , Trn , Para);
                ac_test(p) = sum(PredictY==Yc0)/length(Y_test)*100
                error_test(p) = sum(abs(model.prob - postprob_test))
                error_rate_test(p) = sum(abs(model.prob - postprob_test))/sum(abs(postprob_test))
            end
%% >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> TEST <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< 
            if sum(Mod=='test')
                X_test = sort(data_ori.X_test,'ascend');
                m = size(X_test, 1); 
                X0pdf_test = pdf(Gentype2, X_test, p1_2, p2_2);
                X1pdf_test = pdf(Gentype3, X_test, p1_3, p2_3);
                py1_test = 0.5;
                allprob_test =  X0pdf_test*(1-py1_test) + X1pdf_test*py1_test;
                postprob_test = (X1pdf_test*py1_test)./allprob_test;
                Y_test= postprob_test;
                Para.n = size(X_test, 2);
                % ---------- Test Y Adaption----------
                indP_test = Y_test>=Para.thr;
                indN_test = Y_test<Para.thr;
                m1_ = sum(indP_test);
                m2_ = sum(indN_test);
            end
%% >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Plot distribution <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
            figure
            m = size(X_test, 1);
            indte = 1:m;
            ValXP = X_test(indP_test);
            ValYP = Y_test(indP_test);
            ValXN = X_test(indN_test);
            ValYN = Y_test(indN_test);
            
            plot(ValXP, zeros(size(ValXP)), 'r<')
            hold on
            plot(ValXN, zeros(size(ValXN)), 'b>')
            
            plot(X_test, postprob_test(indte, :), 'ko-')
            hold on
            plot(X_test, model.prob, 'g*--')
            title('Probability')
            legend('Pos class', 'Neg class', 'Posterior probability', 'Predicted probability')
            name = sprintf('result_%d.png', p);
            saveas(gcf, name, 'png')
            %--------------- 画分布图 ---------------
% figure
% m = size(X_test, 1);
% indte = 1:m;
% plot(X_test, X0pdf_test(indte, :), 'ko-')
% hold on
% plot(X_test, X1pdf_test(indte, :), 'g*--')
%% >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Plot objective <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        %     figure
        %     plot(1:length(model.fval), model.fval, '*-b')
        %     title('Objetive')
%% >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Plot scatter (X) <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
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
    end
    final_ac=mean(ac_test);
    V_ac=std(ac_test);
    final_error=mean(error_test);
    V_err=std(error_test);
    final_error_rate=mean(error_rate_test);
    V_rate=std(error_rate_test);
    %--------------- Print Result ---------------
    fprintf('%s\n', repmat('=', 1, 80))
    fprintf('AC=%s\t\t\t\t\t',num2str(final_ac))
    fprintf('Error=%s\t\t',num2str(final_error))
    fprintf('Errorrate=%s\t\t\n',num2str(final_error_rate))
    fprintf('AC_Std=%s\t\t',num2str(V_ac))
    fprintf('Error_Std=%s\t\t',num2str(V_err))
    fprintf('Errorrate_Std==%s\t\t\n',num2str(V_rate))
end     
diary off
toc(ttic)
fprintf('%s\n', repmat('=', 1, 80))