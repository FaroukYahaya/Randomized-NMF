% Randomized Power Iterations NeNMF (RPINeNMF)
% Reference
%  N. Guan, D. Tao, Z. Luo, and B. Yuan, "NeNMF: An Optimal Gradient Method
%  for Non-negative Matrix Factorization", IEEE Transactions on Signal
%  Processing, Vol. 60, No. 6, PP. 2882-2898, Jun. 2012.
%  DOI: 10.1109/TSP.2012.2190406

% Modified by: F. Yahaya | Date: 06/09/2018
% Contact: farouk.yahaya@univ-littoral.fr

% Reference
%  F. Yahaya, M. Puigt, G. Delmaire, G. Roussel, Faster-than-fast NMF using
%  random projections and Nesterov iterations, iTWIST 2018, Marseille.

% <Inputs>
%  X : Input data matrix (m x n)
%  r : Target low-rank
%  MAX_ITER : Maximum number of iterations. Default is 1,000.
%  MIN_ITER : Minimum number of iterations. Default is 10.
%  TOL : Stopping tolerance. Default is 1e-5.

% <Outputs>
%  W : Basis matrix (m x r)
%  H : Coefficients matrix (r x n)
%  T : CPU time (per iteration)
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


function [W,H,RRE,T] = rand_NMF(X,W,H,r,Tmax)
MinIter = 10;
tol = 1e-5;

T = zeros(1,5000);
RRE = zeros(1,5000);
ITER_MAX = 500;
ITER_MIN = 10;

[L,R] = compression(X,r);
X_L = L * X;
X_R = X * R;

H_comp = H * R;
W_comp = L * W;

HVt = H_comp * X_R';
HHt = H_comp * H_comp';
WtV = W_comp' * X_L;
WtW = W_comp' * W_comp;

GradW = W * HHt - HVt';
GradH = WtW * H - WtV;

init_delta = stop_rule([W',H],[GradW',GradH]);
tolH = max(tol,1e-3) * init_delta;
tolW = tolH;

W = W';
k = 1;
RRE(k) = nmf_norm_fro(X, W', H);
T(k) = 0;
tic

while toc <= Tmax
    k = k + 1;

    % Update H with W fixed
    [H,iterH] = NNLS(H,WtW,WtV,ITER_MIN,ITER_MAX,tolH);
    if iterH <= ITER_MIN
        tolH = tolH / 10;
    end

    H_comp = H * R;
    HHt = H_comp * H_comp';
    HVt = H_comp * X_R';

    % Update W with H fixed
    [W,iterW,GradW] = NNLS(W,HHt,HVt,ITER_MIN,ITER_MAX,tolW);
    if iterW <= ITER_MIN
        tolW = tolW / 10;
    end

    W_comp = W * L';
    WtW = W_comp * W_comp';
    WtV = W_comp * X_L;
    GradH = WtW * H - WtV;

    delta = stop_rule([W,H],[GradW,GradH]);
    if delta <= tol * init_delta && k >= MinIter
        break;
    end

    RRE(k) = nmf_norm_fro(X, W', H);
    T(k) = toc;
end
W = W';
end

function [H,iter,Grad] = NNLS(Z,WtW,WtV,iterMin,iterMax,tol)
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
% Compute normalized Frobenius error: ||X - WH||_F^2 / ||X||_F^2
f = norm(X - W * H,'fro')^2 / norm(X,'fro')^2;
end

function [L, R] = compression(X, r)
% Compressed NMF via random projections (Tepper & Sapiro 2016)
compressionLevel = 20;
[m,n] = size(X);
l = min(n, max(compressionLevel, r + 10));

OmegaL = randn(n, l);
B = X * OmegaL;
for j = 1:4
    B = X * (X' * B);
end
[L, ~] = qr(B, 0);
L = L';

OmegaR = randn(l, m);
B = OmegaR * X;
for j = 1:4
    B = (B * X') * X;
end
[R, ~] = qr(B', 0);
end

function retVal=stop_rule(X,gradX)
 
pGrad=gradX(gradX<0|X>0);
retVal=norm(pGrad);

end
