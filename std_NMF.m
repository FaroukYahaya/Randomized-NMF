% =========================================================================
% Non-negative Matrix Factorization using Random Projections and Nesterov
% Author      : F. Yahaya
% Contact     : faroya2011@gmail.com
% Date        : 06/09/2018
%
% If you use this code, please cite the following paper:
%
%   Yahaya, F., Puigt, M., Delmaire, G., & Roussel, G. (2018).
%   Faster-than-fast NMF using random projections and Nesterov iterations.
%   arXiv preprint arXiv:1812.04315.
%   https://arxiv.org/abs/1812.04315
%
% -------------------------------------------------------------------------
% <Inputs>
%   X    : Input data matrix (m x n)
%   r    : Target low-rank
%   Tmax : CPU time in seconds
%   W, H : Initial nonnegative factor matrices (W: m x r, H: r x n)
%
% <Outputs>
%   W    : Final basis matrix (m x r)
%   H    : Final coefficient matrix (r x n)
%   RRE  : Relative reconstruction error at each iteration
%   T    : CPU time (in seconds) at each iteration
% =========================================================================


function [W,H,RRE,T] = std_NMF(X, W, H, Tmax)
MinIter = 10;
tol = 1e-5;
T = zeros(1, 301);
RRE = zeros(1, 301);
ITER_MAX = 500;
ITER_MIN = 10;

HVt = H * X'; HHt = H * H';
WtV = W' * X; WtW = W' * W;

GradW = W * HHt - HVt';
GradH = WtW * H - WtV;

init_delta = stop_rule([W', H], [GradW', GradH]);
tolH = max(tol, 1e-3) * init_delta;
tolW = tolH;

W = W';
k = 1;
tic
RRE(k) = nmf_norm_fro(X, W', H);
T(k) = 0;

while toc <= Tmax + 0.05
    [H, iterH] = NNLS(H, WtW, WtV, ITER_MIN, ITER_MAX, tolH);
    if iterH <= ITER_MIN, tolH = tolH / 10; end

    HHt = H * H'; HVt = H * X';
    [W, iterW, GradW] = NNLS(W, HHt, HVt, ITER_MIN, ITER_MAX, tolW);
    if iterW <= ITER_MIN, tolW = tolW / 10; end

    WtW = W * W'; WtV = W * X;
    GradH = WtW * H - WtV;
    delta = stop_rule([W, H], [GradW, GradH]);

    if delta <= tol * init_delta && k >= MinIter
        break;
    end

    if toc - (k - 1) * 0.05 >= 0.05
        k = k + 1;
        RRE(k) = nmf_norm_fro(X, W', H);
        T(k) = toc;
    end
end

W = W';
end

function [H, iter, Grad] = NNLS(Z, WtW, WtV, iterMin, iterMax, tol)
if ~issparse(WtW)
    L = norm(WtW);
else
    L = norm(full(WtW));
end

H = Z;
Grad = WtW * Z - WtV;
alpha1 = 1;

for iter = 1:iterMax
    H0 = H;
    H = max(Z - Grad / L, 0);
    alpha2 = 0.5 * (1 + sqrt(1 + 4 * alpha1^2));
    Z = H + ((alpha1 - 1) / alpha2) * (H - H0);
    alpha1 = alpha2;
    Grad = WtW * Z - WtV;

    if iter >= iterMin
        pgn = stop_rule(Z, Grad);
        if pgn <= tol
            break;
        end
    end
end

Grad = WtW * H - WtV;
end

function f = nmf_norm_fro(X, W, H)
% Normalized Frobenius error: ||X - WH||_F^2 / ||X||_F^2
f = norm(X - W * H, 'fro')^2 / norm(X, 'fro')^2;
end
function retVal=stop_rule(X,gradX)

pGrad=gradX(gradX<0|X>0);
retVal=norm(pGrad);

end
