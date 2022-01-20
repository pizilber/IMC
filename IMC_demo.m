%% configurations
addpath(genpath('algorithms'));

% experiment configurations
n1 = 300;
n2 = 350;
r = 5;
d1 = 2*r;
d2 = 2*r;
condition_number = 1e1;
oversampling_ratio = 1.5;
singular_values = linspace(1, condition_number, r);

% algorithms to run
algs = {'GNIMC', 'AltMin', 'GD', 'RGD'};
    % GNIMC: Gauss-Newton
    % AltMin: alternating minimization
    % GD, RGD: vanilla and balance-regularized gradient descent

% options (see more options in the algorithms)
% let algorithms chat
opts.verbose = 0;
% early stopping criteria (-1 to disable a criterion)
opts.stop_relRes = 1e-14;   	% small relRes threshold
                                % (relRes = ||X_hat - X||_F/||X_hat||_F on the observed entires)
opts.stop_relDiff = 1e-14;      % small relative X_hat difference threshold

%% run experiment
format long;
fprintf('\n n1,n2: %4d,%4d. rank: %2d. d1,d2: %3d,%3d. condition number: %e \n oversampling ratio: %e\n\n', ...
    n1, n2, r, d1, d2, condition_number, oversampling_ratio);

rng_value = 2021;
rng('default');
rng(rng_value);

% generate low rank matrix X0
[X0, A, B] = generate_matrix(n1, n2, d1, d2, singular_values);

% generate mask
m = min(floor(r*(d1+d2-r) * oversampling_ratio), n1*n2); % number of observed entries
[H, omega, omega_2d] = generate_mask(n1,n2, m);

% compute X, the observed matrix
X = sparse(omega_2d(:,1),omega_2d(:,2),X0(omega),n1,n2);

% run algorithms
for alg_idx = 1:numel(algs)
    alg_name = algs{alg_idx};
    switch alg_name
        case 'GNIMC'
            opts_GNIMC = opts;
            opts_GNIMC.alpha = -1;
            opts_GNIMC.max_outer_iter = 100;
            [X_hat, ~, ~, ~] = GNIMC(X, omega, r, A, B, opts_GNIMC);
        case 'AltMin'
            opts_AltMin = opts;
            opts_AltMin.max_outer_iter = 100;
            [X_hat, ~, ~, ~] = AltMin(X, omega, r, A, B, opts_AltMin);
        case 'GD'
            opts_GD = opts;
            opts_GD.lambda = 0; % no regularization
            opts_GD.eta = 0.05; % step size
            opts_GD.max_iter = 3000;
            [X_hat, ~, ~] = RGD(X, omega, r, A, B, opts_GD);
        case 'RGD'
            opts_RGD = opts;
            opts_RGD.lambda = 0.5; % balance regularization
            opts_RGD.eta = 0.05; % step size
            opts_RGD.max_iter = 3000;
            [X_hat, ~, ~, ~] = RGD(X, omega, r, A, B, opts_RGD);
    end

    % report
    true_error = norm(X_hat - X0, 'fro') / norm(X0, 'fro');
    fprintf('algorithm: %s, true error: %8d\n\n', alg_name, true_error);
end

