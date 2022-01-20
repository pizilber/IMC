function [X_hat, iter, convergence_flag, all_relRes] = AltMin(X,omega,r,A,B,opts)
%
%   Alternating minimization algorithm for inductive matrix completion
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
opts_default.perform_qr = 1;                    % work with QR decomposition of U,V to enhance performance
% number of iterations and tolerance
opts_default.max_outer_iter = 100;              % maximal number of outer iterations
opts_default.max_inner_iter_init = 1000;          % maximal number of inner iterations
opts_default.max_inner_iter_final = 1000;       % use this if opts.LSQR_smart_tol && relRes < opts.LSQR_smart_obj_min
opts_default.inner_init_tol = 1e-4;             % initial tolerance of the inner LSQR solver
opts_default.LSQR_smart_tol = 1;                % update inner tol and inner iters once relRes is low enough
opts_default.LSQR_smart_obj_min = 1e-3;         % relRes threshold to update inner tol and inner iters
% initialization
opts_default.init_option = 0;                   % 0 for SVD, 1 for random, 2 for opts.init_U, opts.init_V
opts_default.init_U = NaN;                      % if opts.init_option==2, use this initialization for U
opts_default.init_V = NaN;                      % if opts.init_option==2, use this initialization for V
% early stopping criteria (-1 to disable a criterion)
opts_default.stop_relRes = 5e-14;               % small relRes threshold (relevant to noise-free case)
opts_default.stop_relDiff = -1;                 % small relative X_hat difference threshold
opts_default.stop_relResDiff = -1;              % small relRes difference threshold
opts_default.stop_relResStuck_ratio = -1;       % stop if minimal relRes didn't change by a factor of stop_relResNoChange_ratio...
opts_default.stop_relResStuck_iters = -1;       % ... in the last #stop_relResNoChange_iters outer iterations
opts_default.stop_time = -1;                    % stop if this number of seconds passed
% additional configurations
opts_default.calculate_full_X = 0;              % calculate full X_hat throughout iterations (turn off to enhance performance)

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

%% initialize U (of size d1 x r)
if opts.init_option == 0
    % initialization by rank-r SVD of observed matrix
    [L, Sigma, ~] = svds(X/p,r);
    U = A' * L * sqrt(Sigma);
elseif opts.init_option == 1
    % initialization by random orthogonal matrices
    d1 = size(A,2);
    Z = randn(d1,r);
    [U, ~, ~] = svd(Z,'econ'); 
else
    % initiazliation by user-defined matrices
    U = opts.init_U;
end
if opts.calculate_full_X
    AU = A*U;
else
    AU_omega_rows = A(omega_row, :) * U;
end

%% calculate rhs (b in ||Ax-b||^2)
rhs = zeros(m,1);  % vector of visible entries in matrix X
for counter=1:m
    rhs(counter) = X(omega_row(counter),omega_col(counter));
end

%% before iterations
X_norm = norm(X, 'fro');
X_hat_previous = zeros(n1,n2);
all_relRes = zeros(opts.max_outer_iter,1);
relRes = Inf;
best_relRes = Inf;
convergence_flag = 0;

%% iterations
iters_tStart = cputime;
for iter = 1:opts.max_outer_iter

    % determine tolerance for LSQR solver
    LSQR_tol = opts.inner_init_tol;
    LSQR_iters = opts.max_inner_iter_init;
    if opts.LSQR_smart_tol && relRes < opts.LSQR_smart_obj_min
        LSQR_tol = min(LSQR_tol, relRes^2);
    end
    LSQR_tol = max(LSQR_tol, 2*eps);  % to supress warning
    
    % determine number of inner iterations for LSQR solver
    if opts.LSQR_smart_tol && relRes < opts.LSQR_smart_obj_min
        LSQR_iters = opts.max_inner_iter_final;
    end

    %% solve for V
    % construct least squares matrix
    if opts.perform_qr
        [U_Q, U_R] = qr(U,0);
        AU_for_use = A(omega_row,:)*U_Q;
    else
        AU_for_use = AU_omega_rows;
    end
    L = generate_productMatrix(AU_for_use, B(omega_col,:)'); 
    L = sparse(L);

    % solve the least squares problem
    [z, ~, ~, ~] = lsqr(L,rhs,LSQR_tol,LSQR_iters);
        % LSQR finds the minimum norm solution and is much faster than lsqminnorm
    
    % retrieve V
    V = reshape(z, [r, d2])';
    if opts.perform_qr
        V = V / (U_R');
    end
    if opts.calculate_full_X
        BV = B*V;
    else
        BV_omega_cols = B(omega_col, :)*V;
    end

    %% solve for U
    % construct least squares matrix
    if opts.perform_qr
        [V_Q, V_R] = qr(V,0);
        BV_for_use = B(omega_col,:)*V_Q;
    else
        BV_for_use = BV_omega_cols;
    end
    L = generate_productMatrix(A(omega_row,:), BV_for_use'); 
    L = sparse(L);

    % solve the least squares problem
    [z, ~, ~, LSQR_iters_done] = lsqr(L,rhs,LSQR_tol,LSQR_iters);
        % LSQR finds the minimum norm solution and is much faster than lsqminnorm
    
    % retrieve U
    U = reshape(z, [d1, r]);
    if opts.perform_qr
        U = U / (V_R');
    end
    if opts.calculate_full_X
        AU = A*U;
    else
        AU_omega_rows = A(omega_row, :)*U;
    end
    
    %% get new estimate and calculate corresponding error
    if opts.calculate_full_X
        X_hat = AU * (BV)';
    else
        %x_hat = sum(AU(omega_row, :) .* BV(omega_col, :), 2);
        x_hat = sum(AU_omega_rows .* BV_omega_cols, 2);
        X_hat = sparse(omega_row, omega_col, x_hat, n1, n2);
    end
        
    % store relRes and update X_hat if needed
    relRes = norm(X_hat(omega) - X(omega)) / X_norm;
    all_relRes(iter) = relRes; 
    if relRes < best_relRes
        best_relRes = relRes;
        if opts.calculate_full_X
            X_hat_best = X_hat;
        else
            X_hat_best.U = U;
            X_hat_best.V = V;
        end
    end
    
    % update X_hat_diff
    X_hat_diff = norm(X_hat - X_hat_previous, 'fro') / norm(X_hat, 'fro');
    
    %% report
    if opts.verbose
        fprintf('[INSIDE AltMin] iter %4d \t diff X_r %5d\t relRes %6d\n',...
            iter, X_hat_diff, relRes);
    end

    %% check early stopping criteria
    if relRes < opts.stop_relRes
        msg = '[INSIDE AltMin] Early stopping: small error on observed entries\n';
        convergence_flag = 1;
    elseif X_hat_diff < opts.stop_relDiff
        msg = '[INSIDE AltMin] Early stopping: X_hat does not change\n';
        convergence_flag = 1;
    elseif iter > 1 && ...
            abs(all_relRes(iter-1)/relRes-1) < opts.stop_relResDiff
        msg = '[INSIDE AltMin] Early stopping: relRes does not change\n';
        convergence_flag = 1;
    elseif iter > 1 && LSQR_iters_done == 0
        msg = '[INSIDE AltMin] Early stopping: no iterations of LSQR solver\n';
        convergence_flag = 1;
    elseif iter >= 3 * opts.stop_relResStuck_iters && opts.stop_relResStuck_ratio > 0 ...
            && mod(iter, opts.stop_relResStuck_iters) == 0
        % the factor of 3 is since we want to ignore the first opts.stop_relResStuck_iters
        % iteratios, as they may contain small observed relRes with large unobserved...
        last_relRes = all_relRes(iter-opts.stop_relResStuck_iters+1:iter);
        previous_relRes = all_relRes(iter-2*opts.stop_relResStuck_iters+1:iter-opts.stop_relResStuck_iters);
        if min(last_relRes) / min(previous_relRes) > opts.stop_relResStuck_ratio
            msg = sprintf('[INSIDE AltMin] Early stopping: relRes did not decrease by a factor of %f in %d iterations\n', ...
                opts.stop_relResStuck_ratio, opts.stop_relResStuck_iters);
            convergence_flag = 1;
        end
    elseif (opts.stop_time > 0) && (cputime - iters_tStart > opts.stop_time)
        msg = '[INSIDE AltMin] Early stopping: time over\n';
        convergence_flag = 2;
    end
    if convergence_flag
        if opts.verbose
            fprintf(msg);
        end
        break
    end
    
    %% update before next iterate
    X_hat_previous = X_hat;
end

%% return
if convergence_flag == 2  % exceeding time it's like exceeding num iterations
    convergence_flag = 0;
end

if opts.calculate_full_X
    X_hat = X_hat_best;
else
   X_hat = A * X_hat_best.U * (X_hat_best.V)' * B'; 
end

end


function M = generate_productMatrix(A, B)
% Returns M such that M*vec(D) = vec(A*D*B)
    assert( (size(A,1) == size(B,2) ), 'error: dimensions mismatch!');

    d2 = size(A,2); d3 = size(B,1);
    m = size(A,1);
    M = zeros(m, d2*d3);
    for counter=1:m
        AB = A(counter,:)' * B(:,counter)';
        M(counter,:) = AB(:);
    end
    %M = sparse(M);
end
