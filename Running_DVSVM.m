clear
clc
close all
diary 'Diary1.txt'
ttic=tic;
for xunhuan=2
    fprintf('!!!SyntheticSeed=%s!!!\n',num2str(xunhuan))
    Para. AutoRec = "ON";
%     Figure="ON";
    Figure="OFF";
    Figurename='normalChisquare13';
    Synthetic="ON";
%     Synthetic="OFF";
    %% ============================== Ⅰ Running Setting ==============================
    % ----------◆ -1- Model Selecting ◆----------
    ModS = [];
%     ModS = [ModS;"LIB_L1SVCprob"];
    ModS = [ModS;"LIB_L1SVC"];
    % ModS = [ModS;"LIB_lin"];
%     ModS = [ModS;"VSVM"];
    % ModS = [ModS;"DVSVMnon_J"];
    ModS = [ModS;"DVSVMnon_QP"];
    % ModS = [ModS;"LSSVM"];
    % ----------◆ -2- ◆----------
    datas=[];
    datas=[datas;"train"];
%     datas=[datas;"test"];
    % datas=[datas;"all"];
    data=datas(1);
    % ----------◆ -3- Feature Kernel types ◆----------
    Para.kpar.ktype = 'rbf'; % poly or lin or rbf;
    % ----------◆ -4- V Kernel types ◆----------
    V_Matrix = 'Vmatrix'  ; V_MatrixFun = str2func(V_Matrix) ; Para.vmatrix = V_Matrix;
    Para.CDFx = 'uniform' ; Para.v_ker = 'gaussian_ker'      ; CDFx = Para.CDFx;
    % ----------◆ -5- Files ◆----------
    name= ["TNB"];Para.name=name;%echo ecoli heart fire Hepatitis Parkinsons
    % ----------◆ -6- Repeat ◆----------
    Repeat=10;
    % ----------◆ -7- K-Fold ◆----------
    k=3;
    % ----------◆ -8- Para Range ◆----------
    pa.min = -8  ;  pa.step =  2 ;  pa.max = 8;
    % ----------◆ -9- Synthetic Gen ◆----------
    if sum(Synthetic=="ON")
        DData = Gen_Distri("normal",1 ,2,"normal" ,2,4,100,xunhuan);
        DData=sortrows(DData,1);
        Probsvm=zeros(size(59,1),1);Probvsvm=zeros(size(59,1),1);Probdvsvm=zeros(size(59,1),1);
    end
    %% ============================== Ⅱ Running Procedure ==============================
    for ms = 1 : length(ModS)
        Mod = ModS(ms); res.Mod=Mod; Error=[];Errorrate=[];
        for chongfu=1:Repeat
            randomindex=randperm(1000,100); seed=randomindex(chongfu); res.seed=seed; rng(seed); res.chongfu=chongfu;
            for l = 1:length(name)
                f = 'data/';  G = name(l) ;  folder = f+G;  files = dir(fullfile(folder, '*.mat'));
                for p = 1:length(files)
                    % ----------◆ -1- Base Setting ◆----------
                    filename = fullfile(folder, files(p).name) ; data_ori = load(filename) ; SVMFun = str2func(Mod);
                    % ----------◆ -2- Data Preprecess ◆----------
                    if sum(Synthetic=="ON")
                        [X_train,Y_train_T,X_test,Y_test_T] = TT(DData(:, 1),DData(:,2:end),0.3);
                        Dtrain=[ X_train ,Y_train_T ]; Dtrain=sortrows(Dtrain,1);
                        Dtest=[ X_test ,Y_test_T ]; Dtest=sortrows(Dtest,1);
                        X_train=Dtrain(:,1:end-1);Y_train_T=Dtrain(:,end);
                        X_test=Dtest(:,1:end-1);Y_test_T=Dtest(:,end);
                        X_train_T=X_train(:,1);
                        X_train_P=X_train(:,2);
                        X_test_T=X_test(:,1);
                        X_test_P=X_test(:,2);
                    else
                        data_ori.X=mapminmax(data_ori.X',0,1)';
                        [coeff,score,latent,tsquared,explained,mu] = pca(data_ori.X,'Centered',false);
                        [~,~,data_ori.X,data_ori.Y] = TT(data_ori.X,data_ori.Y,0.015);
                        AA=corr(data_ori.X,data_ori.Y,"type","Kendall");
                        ind=find(AA==max(AA));
                        [secondorder,secondidx]=sort(AA,"descend");
                        nancount=sum(isnan(AA));
                        idx=secondidx(nancount+2);
%                         data_ori.X=[data_ori.X(:,ind)];
                        data_ori.X=[data_ori.X(:,[ind;idx])];
                        [X_train_T,Y_train_T,X_test_T,Y_test_T] = TT(data_ori.X,data_ori.Y,0.3);
                    end
                    fprintf('%s\n', repmat('=', 1, 100)); fprintf('Proc ===>%s\t\n',num2str(chongfu));fprintf('File===>%s\t\n',G);
                    fprintf('Seed===>%s\t\n',num2str(seed)); fprintf('Mod===>%s\t\n',Mod); fprintf('X_C===>%s\t\n',data); fprintf('%s\n', repmat('-', 1, 100));
                    % ----------◆ -2- Different Models ◆----------
                    if sum(Mod=='LIB_L1SVCprob')
                        % X_train_T=data_ori.X_train_A; Y_train_T=data_ori.Y_train_A;
                        Data.X_train_T=X_train_T;
                        Data.Y_train_T=Y_train_T;
                        Data.X_test_T=X_test_T;
                        Data.Y_test_T=Y_test_T;
                        Res = SVM_prob(Data,k,Para.kpar.ktype,pa);
                        Prob=Res.prob;
                        Probsvm=Probsvm+Res.prob;
                        result=catstruct(res,Res);
                        result=rmfield(result,'prob');
                    end
                    if sum(Mod=='LIB_L1SVC')
                        %  X_train_T=data_ori.X_train_A; Y_train_T=data_ori.Y_train_A;
                        %                     X_train=[data_ori.X_train_A;X_train_T]; Y_train=[data_ori.Y_train_A;Y_train_T];
                        Data.X_train_T=X_train_T;
                        Data.Y_train_T=Y_train_T;
                        Data.X_test_T=X_test_T;
                        Data.Y_test_T=Y_test_T;
                        Res = SVM_rbf(Data,k,Para.kpar.ktype,pa);
                        result=catstruct(res,Res);
                    end
                    if sum(Mod=='DVSVMnon_J')
                        pa.itmO = 5; pa.itmSh = 10; pa.plt = 0;  pa.epsi = 0; pa.thr = 0.5;
                        Data.X_train_T=X_train_T;
                        Data.Y_train_T=Y_train_T;
                        Data.X_test_T=X_test_T;
                        Data.Y_test_T=Y_test_T;
                        Res = DVSVM_J_F(Data,k,Para.kpar.ktype,pa,data);
                        result=catstruct(res,Res);
                    end
                    if sum(Mod=='DVSVMnon_QP')
                        pa.lam=12;    pa.itmO = 20; pa.itmSh = 10; pa.plt = 0;  pa.epsi = 0; pa.thr = 0.5;
                        Data.X_train_T=X_train_T;
                        Data.Y_train_T=Y_train_T;
                        Data.X_test_T=X_test_T;
                        Data.Y_test_T=Y_test_T;
                        Res = DVSVM_QP_F(Data,k,Para.kpar.ktype,pa,data);
                        Prob=Res.prob;
                        Probdvsvm=Probdvsvm+Res.prob;
                        result=catstruct(res,Res);
                        result=rmfield(result,'prob');
                    end
                    if sum(Mod=="VSVM")
                        % X_train_T=data_ori.X_train_A; Y_train_T=data_ori.Y_train_A;
                        Data.X_train_T=X_train_T;
                        Data.Y_train_T=Y_train_T;
                        Data.X_test_T=X_test_T;
                        Data.Y_test_T=Y_test_T;
                        Res = VSVM_F(Data,k,Para.kpar.ktype,Para.v_ker,CDFx,pa);
                        Prob=Res.prob;
                        Probvsvm=Probvsvm+Res.prob;
                        result=catstruct(res,Res);
                        result=rmfield(result,'prob');
                    end
                    % ============================================================
                end
            end
            if sum(Figure=="ON")
                Error= sum(abs(Prob - X_test_P));
                Errorrate = sum(abs(Prob - X_test_P))/sum(abs(X_test_P));
                result.Error=Error;
                result.Errorrate=Errorrate;
            end
            %% ============================== Ⅲ Result Display ==============================
            if sum(Figure=="ON")
                ResultsInfolib(result,Para)
            else
                ResultsInfo(result,Para)
            end
            MAC(chongfu)=result.ac_test;
            MF(chongfu)=result.F;
            MGM(chongfu)=result.GM;
            MAUC(chongfu)=result.AUC;
            if sum(Figure=="ON")
                MError(chongfu)=Error;
                MErrorrate(chongfu)=Errorrate;
            end
        end
        fprintf('AC=%s\t\t',num2str(sprintf('%.2f', mean(MAC))))
        fprintf('AC_Std=%s\t\t\n',num2str(sprintf('%.2f', std(MAC))))
        fprintf('FM=%s\t\t',num2str(sprintf('%.2f', mean(MF))))
        fprintf('F_Std=%s\t\t\n',num2str(sprintf('%.2f', std(MF))))
        fprintf('GM=%s\t\t',num2str(sprintf('%.2f', mean(MGM))))
        fprintf('GM_Std=%s\t\t\n',num2str(sprintf('%.2f', std(MGM))))
        fprintf('AUC=%s\t\t',num2str(sprintf('%.2f', mean(MAUC))))
        fprintf('AUC_Std=%s\t\t\n',num2str(sprintf('%.2f', std(MAUC))))
        if sum(Figure=="ON")
            fprintf('Error=%s\t\t',num2str(sprintf('%.2f', mean(MError))))
            fprintf('Error_std=%s\t\t\n',num2str(sprintf('%.2f', std(MError))))
            fprintf('Errorrate=%s\t\t',num2str(sprintf('%.2f', mean(MErrorrate))))
            fprintf('Errorrate_std=%s\t\t\n',num2str(sprintf('%.2f', std(MErrorrate))))
        end
        fprintf('%s\n', repmat('=', 1, 100))
        clear result
    end
    %% ==============================  Ⅳ Figure ==============================
    if sum(Figure=="ON")
        subplot(1,2,2)
        %     plot(valxp, zeros(size(valxp)), 'r<')
        hold on
        %     plot(valxn, zeros(size(valxn)), 'b>')
        scatter(X_test_T,X_test_P , 40,"Marker","o","MarkerFaceColor","none","MarkerEdgeColor","black")
        hold on
        scatter(X_test_T, Probsvm/Repeat, 40 ,"filled","Marker","<","MarkerFaceColor",[127/255,203/255,163/255])
        hold on
        scatter(X_test_T, Probvsvm/Repeat, 40 ,"filled","Marker","square","MarkerFaceColor",[250/255,134/255,000/255])
        hold on
        scatter(X_test_T, Probdvsvm/Repeat, 40 ,"filled","Marker","o","MarkerFaceColor",[19/255,103/255,131/255])
        %     scatter(data_ori.x_test, Probdvsvm/Repeat, 20 ,"filled","Marker","diamond","MarkerFaceColor","black")
        box off
        grid on
        grid minor
        legend("Origin","SVMprob","VSVM","DVSVM")
        saveas(gcf, [Figurename,'.png']);
    end
    %% ============================== VI Email to me ==============================
    %     receiver = '2470566766@qq.com';
    %     mailtitle = 'The experiment has ended, come check it out';
    %     mailcontent = 'Finish! ! The save location is:';
    %     Mail_To_Me(receiver,mailtitle,mailcontent)
    toc(ttic)
    diary off
end