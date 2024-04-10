function [T, u, v] = sinkhornKnoppAuto2(M, Para, varargin)
%   Reference
%   ---------
%   Philip A. Knight (2008) The Sinkhorn朘nopp Algorithm: Convergence and
%   Applications. SIAM Journal on Matrix Analysis and Applications 30(1),
%   261-275. doi: 10.1137/060659624
% lam = Para.lam;

n = Para.n;
lam = Para.lam;
maxiter = Para.itmSh;
K = exp(-lam*M);
validateattributes(K, {'numeric'}, {'nonnegative' 'square'});
m = size(K, 1);
validateattributes(K, {'numeric'}, {'nonnegative' 'square'});
N = size(K, 1);

inp = inputParser;
inp.addParameter('Tolerance', eps(N), ...
    @(x) validateattributes(x, {'numeric'}, {'positive' 'scalar'}));%返回从 N 到下一个较大双精度数的距离
inp.addParameter('MaxIter', Inf, @(x) ...
    checkattributes(x, {'numeric'}, {'positive' 'integer' 'scalar'}) ...
    || (isinf(x) && isscalar(x) && x > 0));
inp.parse(varargin{:});
tol = inp.Results.Tolerance;
% maxiter = inp.Results.MaxIter;

% first iteration - no test %   M = diag(R) * A * diag(C)
iter = 1;
v = 1./sum(K);
u = 1./(K * v.');

% subsequent iterations include test
while iter < maxiter
    iter = iter + 1;
    cinv = u.' * K;
    % test whether the tolerance was achieved on the last iteration
    if  max(abs(cinv .* v - 1)) <= tol  %停机条件
        break
    end
    v = 1./cinv;
    u = 1./(K * v.');
end

T = K .* (u * v);
T = T/m^2;
end
