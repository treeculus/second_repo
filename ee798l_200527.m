% Parameters

N = 20;     
M = 40;     
D0 = 7; 

% Generating Phi
Phi = randn(N,M);

% Generating sparse weight vector
w = zeros(M,1);
w(randperm(M,D0)) = randn(D0,1); % generating values in the weight vector at non-zero indices

% Generate noise entries for different noise variances
var = [-20, -15, -10, -5, 0];
len=length(var);
nmse_values = zeros(1,len);

% Generate noise
for i = 1:len
    sigma = sqrt(10^(var(i)/10));
    eps = sigma*randn(N,1);
    t = Phi*w + eps;
    
    % Run SBL for regression
    lambda = 1e-6; % Regularisation parameter
    A = 100 * ones(M,1); % Initial value of alpha
    w_old = zeros(M,1);
    tolerance = 1e-3;

    while 1
        % Updating posterior mean

        Sigma = (lambda*eye(M) + Phi.'*Phi) \ Phi.' * t; 
        m = Sigma;

        % Compute alpha and check for convergence
        A=(M - sum(A./(diag(Sigma)))) ./ m.^2;
        if(norm(w_old-m) / norm(w_old) <= tolerance)
            break;
        end
        w_old=m;
    end
    
    % Calculate NMSE
    nmse = norm(m-w)^2 / norm(w)^2;
    nmse_values(i) = nmse;

    disp(['Noise variance:' num2str(var(i)) 'dB'])
    disp('MAP estimate of weight vector:')
    disp(m')
    disp(['NMSE: ' num2str(nmse)])
end

% Plot NMSE vs Noise Variance plot
figure;
plot(var, nmse_values, '-o');
xlabel('Noise variance (dB)');
ylabel('Normalized mean squared error');
title('NMSE vs Noise Variance');
