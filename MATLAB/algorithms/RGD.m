function [X_hat, convergence_flag, iter, all_relRes] = RGD(X,omega,r,A,B,opts)
%
%   Gradient descent algorithm for inductive matrix completion
%       with option for balance reguarlization in the form of lambda*||U'*U - V'*V||_F^2
%   Written by Pini Zilber & Boaz Nadler / 2021
%
% INPUT:
% X = the observed matrix
% omega = list of pairs (i,j) of the observed entries in X
% r = target rank of the underlying matrix
% A, B = side information matrices whose spans contain the cols and rows of the underlying matrix
%   assertions: size(X,1) == size(A,1), size(X,2) == size(B,2)
% opts = options meta-variable (see opts_default below for details)
%
% OUTPUT:
% X_hat = rank-r approximation of the underlying matrix
% all_relRes = list of the residual errors of the least-squares problem throughout the iterations
% iter = final iteration number
% convergence_flag = indicating whether the algorithm converged

%% configurations: default values for option variables
opts_default.verbose = 1;                       % display intermediate results
opts_default.eta = 0.05;                        % step size
opts_default.lambda = 0.5;                      % regularization coefficient
% number of iterations and tolerance
opts_default.max_iter = 1000;                   % maximal number of iterations
% initialization
opts_default.init_option = 0;                   % 0 for SVD, 1 for random, 2 for opts.init_U, opts.init_V
opts_default.init_U = NaN;                      % if opts.init_option==2, use this initialization for U
opts_default.init_V = NaN;                      % if opts.init_option==2, use this initialization for V
% early stopping criteria (-1 to disable a criterion)
opts_default.stop_relRes = 5e-14;               % small relRes threshold (relevant to noise-free case)
opts_default.stop_relDiff = -1;                 % small relative X_hat difference threshold
opts_default.stop_relResDiff = -1;              % small relRes difference threshold
opts_default.stop_time = -1;                    % stop if this number of seconds passed

% for each unset option set its default value
fn = fieldnames(opts_default);
for k=1:numel(fn)
    if ~isfield(opts,fn{k}) || isempty(opts.(fn{k}))
        opts.(fn{k}) = opts_default.(fn{k});
        fn{k} = opts_default.(fn{k});
    end
end

%% some definitions
[n1, n2] = size(X);   % number of rows / colums
d1 = size(A,2); d2 = size(B,2); % side information dimensions
m = size(omega,1);   % number of observed entries
p = m / (n1*n2);     % sampling rate
% mask
omega_2d = zeros(m,2);
[omega_2d(:,1), omega_2d(:,2)] = ind2sub([n1,n2],omega);
omega_row = omega_2d(:,1); omega_col = omega_2d(:,2);

%% make sure A, B are orthonormal
[A, ~, ~] = svd(A,'econ');
[B, ~, ~] = svd(B,'econ');

%% initialize U and V (of sizes d1 x r and d2 x r)
if opts.init_option == 0
    % initialization by rank-r SVD of observed matrix
    [L, Sigma, R] = svds(X/p,r);
    U = A' * L * sqrt(Sigma);
    V = B' * R * sqrt(Sigma);
elseif opts.init_option == 1
    % initialization by random orthogonal matrices
    d1 = size(A,2);
    Z = randn(d1,r);
    [U, ~, ~] = svd(Z,'econ'); 
    d2 = size(B,2);
    Z = randn(d2,r);
    [V, ~, ~] = svd(Z,'econ'); 
else
    % initiazliation by user-defined matrices
    U = opts.init_U;
    V = opts.init_V; 
end

%% before iterations
X_norm = norm(X, 'fro');
U_hat = U; V_hat = V;
omega_X_hat_prev = 0;
relRes = Inf;
all_relRes = zeros(opts.max_iter,1);
best_relRes = Inf;
convergence_flag = 0;

%% iterations
iters_tStart = cputime;
for iter = 1:opts.max_iter

    % regularized gradient descent
    %L = A*U;
    %R = B*V;
    %omega_x_hat = sum((L(omega_row, :)) .* (R(omega_col, :)), 2);
    omega_x_hat = sum((A(omega_row, :)*U) .* (B(omega_col, :)*V), 2);
    omega_X_hat = sparse(omega_row, omega_col, omega_x_hat, n1, n2);
    omega_err = omega_X_hat - X;
    At_error_B_omega = A'*omega_err*B;
    nabla_U = 1/p*At_error_B_omega*V + opts.lambda*U*(U'*U - V'*V);
    nabla_V = 1/p*At_error_B_omega'*U + opts.lambda*V*(V'*V - U'*U);   
    %nabla_U = 1/p*A'*omega_err*R + opts.lambda*U*(U'*U - V'*V);
    %nabla_V = 1/p*B'*omega_err'*L + opts.lambda*V*(V'*V - U'*U);   
    U = U - opts.eta * nabla_U; 
    V = V - opts.eta * nabla_V;

    % store relRes and update X_hat if needed
    relRes = norm(omega_X_hat(omega) - X(omega)) / X_norm;
    all_relRes(iter) = relRes; 
    if relRes < best_relRes
        best_relRes = relRes;
        U_hat = U;
        V_hat = V;
    end
    
    % update X_hat_diff
    X_hat_diff = norm(omega_X_hat - omega_X_hat_prev, 'fro') / norm(omega_X_hat, 'fro');
    
    %% report
    if opts.verbose && rem(iter, 100)
        fprintf('[INSIDE RGD] iter %4d \t diff X_r %5d\t relRes %6d\n',...
            iter, X_hat_diff, relRes);
    end

    %% check early stopping criteria
    if relRes < opts.stop_relRes
        msg = '[INSIDE RGD] Early stopping: small error on observed entries\n';
        convergence_flag = 1;
    elseif X_hat_diff < opts.stop_relDiff
        msg = '[INSIDE RGD] Early stopping: X_hat does not change\n';
        convergence_flag = 1;
    elseif iter > 1 && ...
            abs(all_relRes(iter-1)/relRes-1) < opts.stop_relResDiff
        msg = '[INSIDE RGD] Early stopping: relRes does not change\n';
        convergence_flag = 1;
    elseif relRes > 1e10
        msg = '[INSIDE RGD] Early stopping: relRes diverged\n';
        omega_X_hat = 0;
        convergence_flag = 1;
    elseif (opts.stop_time > 0) && (cputime - iters_tStart > opts.stop_time)
        msg = '[INSIDE RGD] Early stopping: time over\n';
        convergence_flag = 2;
    end
    if convergence_flag
        if opts.verbose
            fprintf(msg);
        end
        break
    end
    
    %% update before next iterate
    omega_X_hat_prev = omega_X_hat;
end

%% return
if convergence_flag == 2  % exceeding time it's like exceeding num iterations
    convergence_flag = 0;
end

X_hat = A * U_hat * V_hat' * B'; 

end
