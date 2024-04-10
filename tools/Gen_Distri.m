function Data = Gen_Distri(gentype1,p1_1 ,p2_1 ,gentype0 ,p1_0 ,p2_0,num,seed)
rng(seed)
%-------------------- distribution para --------------------
% gentype1_testp='normal'; p1_tp_1=-10 ; p1_tp_2=5;
% gentype1_testn='normal'; p1_tn_1=10 ; p1_tn_1=5;
if p2_1==0
    x_p=random(gentype1,p1_1,[1,num])';
    x_n=random(gentype0,p1_0,[1,num])';
elseif p2_0==0
    x_p=random(gentype1,p1_1,p2_1,[1,num])';
    x_n=random(gentype0,p1_0,[1,num])';
else
    x_p=random(gentype1,p1_1,p2_1,[1,num])';
    x_n=random(gentype0,p1_0,p2_0,[1,num])';
end
%     x_p=random(gentype1,p1_1,[1,100])';
%     x_n=random(gentype0,p1_0,[1,100])';
x_p=sort(x_p,'ascend');
x_n=sort(x_n,'ascend');
%------对称------
%     for k=1:size(x_p)
%             x_n(k)=4+(4-x_n(k));
%     end
%-------------------- positive and negative probability density --------------------
xppdf = pdf(gentype1, x_p, p1_1, p2_1);
% xnpdf=xppdf;
xnpdf = pdf(gentype0, x_n, p1_0, p2_0);
x_pdf=[xppdf; xnpdf];
x=[x_p;x_n];
%-------------------- class probability --------------------
pyp=0.5;
xppdf_test = pdf(gentype1, x, p1_1, p2_1);
xnpdf_test = pdf(gentype0, x, p1_0, p2_0);
allprob_test =  xppdf_test*pyp + xnpdf_test*(1-pyp);
postprob_test = (xppdf_test*pyp)./allprob_test;
%-------------------- label y --------------------
y=zeros(length(x),1);
for i = 1:length(x)
    if postprob_test(i)>0.5
        y(i)=1;
    else
        y(i)=2;
    end
end
%-------------------- Output --------------------
Data=[x, postprob_test, y];
%figure
figure("Position",[100,100,1500,500])
subplot(1,2,1)
valxp = x_p;
valxn = x_n;
% pdf
plot(x_p, xppdf, 'rs-',"LineWidth",1,"Color",[196/255,050/255,063/255])
hold on
plot(x_n, xnpdf, 'bo-',"LineWidth",1)
hold on
scatter(valxp, zeros(size(valxp)), 40 ,"Marker","square","MarkerEdgeColor",[196/255,050/255,063/255])
hold on
scatter(valxn, zeros(size(valxn)), 40 ,"Marker","o","MarkerEdgeColor","blue")
box off
grid minor
xlim([0,8])
legend('Pos PDF', 'Neg PDF','Pos Point','Neg Point')
end

