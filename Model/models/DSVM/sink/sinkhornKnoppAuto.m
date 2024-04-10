function [T, pT] = sinkhornKnoppAuto(M, pM, Para, varargin)
%   Reference
%   ---------
%   Philip A. Knight (2008) The Sinkhorn–Knopp Algorithm: Convergence and
%   Applications. SIAM Journal on Matrix Analysis and Applications 30(1),
%   261-275. doi: 10.1137/060659624
maxiter = 50;
n = length(pM{1,1});
lam = Para.lam;
K = exp(-lam*M);
validateattributes(K, {'numeric'}, {'nonnegative' 'square'});
m = size(K, 1);
for i = 1:m
    pKr{i} = [];
    for j = 1:m
%         pM{i,j} = barw;
        pK{i,j}  = -lam*K(i,j)*pM{i,j};
        pKr{i} = [pKr{i}, pK{i,j}];
    end
end

for j = 1:m
    pKc{j} = [];
    for i = 1:m
        pKc{j} = [pKc{j}, pK{i,j}];
    end
end
inp = inputParser;
inp.addParameter('Tolerance', eps(m), ...
    @(x) validateattributes(x, {'numeric'}, {'positive' 'scalar'}));
inp.addParameter('MaxIter', Inf, @(x) ...
    checkattributes(x, {'numeric'}, {'positive' 'integer' 'scalar'}) ...
    || (isinf(x) && isscalar(x) && x > 0));
inp.parse(varargin{:});
tol = inp.Results.Tolerance;
% maxiter = inp.Results.MaxIter;

% first iteration - no test %   M = diag(R) * A * diag(C)
% v = 1./sum(K); % initialize u=1_n
% u = 1./(K * v.'); % length l
iter = 0;
u = ones(m,1);
pu =  zeros(n, m); % partial of u to barw

% subsequent iterations include test
while iter < maxiter
    iter = iter + 1;
    vinv = u.' * K; % row vector
    v = 1./(vinv*m);
    v =v';
    Ku = K'*u;
    Kv = K*v;
    for j = 1:m
        pv(:, j) = (-1/(m*Ku(j)^2))*(pKc{j}*u + pu*K(:,j));
%         pv(:, j) = pKc{j}*u + pu*K(:,j);
%         pv(:, j) = (-1/(m*Ku(j)^2))*pv(:, j);
    end
    for j = 1:m
        pu(:, j) = (-1/(m*Kv(j)^2))*(pKr{j}*v + pv*K(j,:)');
%         pu(:, j) = pKr{j}*v + pv*K(j,:)';
%         pu(:, j) = (-1/(m*Kv(j)^2))*pu(:, j);
    end
    u = 1./(m*K*v);
%     if  max(abs(vinv .* v - 1)) <= tol
%         break
%     end
end

T = K .* (u * v');
for i = 1:m
    for j = 1:m
        pT{i,j} = pu(:, i)*K(i,j)*v(j) +  u(i)*pK{i,j}*v(j) + u(i)*K(i,j)*pv(:, j);
    end
end

end
