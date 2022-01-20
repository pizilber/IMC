function [X, A, B] = generate_matrix(n1,n2,d1,d2,singular_values)
%
% INPUT:  n1,n2 = number of rows and columns in X
%         d1,d2 = number of columns in A,B
%         singular_values = list of non-zero singular values

% OUTPUT: X = n1 x n2 matrix of rank r
%         A = n1 x d1 left singular vectors
%         B = n2 x d2 right singular vectors

r = length(singular_values);

Z = randn(n1, d1);
[A, ~, ~] = svd(Z,'econ');
Z = randn(n2, d2);
[B, ~, ~] = svd(Z,'econ'); 
Z = randn(d1, r);
[U, ~, ~] = svd(Z,'econ');
Z = randn(d2, r);
[V, ~, ~] = svd(Z,'econ');

D = diag(singular_values);
X = A * U * D * V' * B';